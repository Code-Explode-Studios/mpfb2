"""Procedural, editable foot realism for the MPFB HM08 basemesh.

Combines shape keys (arch, heel, forefoot and toe soft tissue) with a
non-destructive skin material copy. Fine creases and pores are shader-only
details localized by mesh point-domain attributes, not extra topology.
Original skin material is preserved for resetting.
GPL-3.0-or-later. Copyright (C) 2026 MPFB contributors.
"""

import json
import os

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty

from ... import ClassManager
from ...services import HumanService, LocationService, LogService, ObjectService, TargetService, UiService
from ..abstractpanel import Abstract_Panel
from ..mpfboperator import MpfbOperator
from .footrealism_core import (
    CREASE_ATTRIBUTE, FOOT_KEY_NAMES, PORE_ATTRIBUTE, compute_foot_fields,
)
from .modelpanel import MODEL_PROPERTIES

_LOG = LogService.get_logger("model.footrealism")

# The existing male/female macros are intentionally not changed. These are
# illustrative artistic presets, not inferred sex-specific anatomy.
FOOT_DETAIL_PRESETS = {
    "natural": {"arch": 0.62, "heel": 0.67, "ball": 0.70, "toes": 0.58,
                "wrinkles": 0.57, "pores": 0.65},
    "athletic": {"arch": 0.96, "heel": 0.66, "ball": 0.84, "toes": 0.83,
                 "wrinkles": 0.54, "pores": 0.72},
    "mature": {"arch": 0.28, "heel": 0.94, "ball": 0.77, "toes": 0.74,
               "wrinkles": 1.0, "pores": 0.85},
}
_PRESETS = (
    ("CUSTOM", "Current sliders", "Apply the current custom foot settings"),
    ("natural", "Natural", "Balanced arch, heel and gentle skin folds"),
    ("athletic", "Athletic", "More pronounced arch and defined toe contours"),
    ("mature", "Mature skin", "More soft-tissue fullness and wrinkles"),
)

_SCENE_NAMES = {
    "arch": "mpfb_foot_arch",
    "heel": "mpfb_foot_heel",
    "ball": "mpfb_foot_ball",
    "toes": "mpfb_foot_toes",
    "wrinkles": "mpfb_foot_wrinkles",
    "pores": "mpfb_foot_pores",
}
_BUMP_NAME_CREASE = "MPFB Foot Wrinkle Bump"
_BUMP_NAME_PORE = "MPFB Foot Pore Bump"
_MATERIAL_MARK = "mpfb_foot_procedural_skin"
_ORIGINAL_MATERIAL = "mpfb_foot_original_material"


def _body_count():
    """Read the shipped HM08 body-vertex range; exclude helper/joint cubes."""
    path = os.path.join(LocationService.get_mpfb_data("mesh_metadata"), "hm08_config.json")
    with open(path, encoding="utf-8") as inp:
        conf = json.load(inp)
    group = conf["groups_by_range"]["body"]
    if len(group) != 2 or int(group[0]) != 0:
        raise ValueError("Unexpected HM08 body vertex metadata")
    return int(group[1]) + 1


def _ensure_foot_fields(basemesh):
    keys = basemesh.data.shape_keys
    if not keys or not keys.use_relative or not keys.reference_key:
        raise ValueError("Select an unbaked MPFB human with relative shape keys")
    basis = keys.reference_key
    coords = [(v.co.x, v.co.y, v.co.z) for v in basis.data]
    body_count = _body_count()
    if len(coords) < body_count:
        raise ValueError("The basemesh does not match HM08 vertex topology")
    fields = compute_foot_fields(coords, body_count)
    return basis, fields


def _write_float_attribute(mesh, name, values):
    attribute = mesh.attributes.get(name)
    if attribute is None:
        attribute = mesh.attributes.new(name=name, type='FLOAT', domain='POINT')
    if attribute.data_type != 'FLOAT' or attribute.domain != 'POINT':
        raise ValueError("Existing MPFB foot attribute has the wrong format: " + name)
    if len(values) != len(attribute.data):
        raise ValueError("Foot mask length does not match mesh vertices")
    attribute.data.foreach_set("value", values)


def _set_shape_keys(basemesh, basis, fields, scene):
    for key_id, name in FOOT_KEY_NAMES.items():
        shapekey = basemesh.data.shape_keys.key_blocks.get(name)
        if shapekey is None:
            shapekey = basemesh.shape_key_add(name=name, from_mix=False)
        # Refresh from Basis each application, so presets don't accumulate
        # offsets. Other shape keys and macro targets remain untouched.
        for i, (dx, dy, dz) in enumerate(fields["deltas"][key_id]):
            source = basis.data[i].co
            position = shapekey.data[i].co
            position.x = source.x + dx
            position.y = source.y + dy
            position.z = source.z + dz
        shapekey.value = float(getattr(scene, _SCENE_NAMES[key_id]))


def _add_bump_to_principled(tree, principled):
    """Add two masks/bumps upstream of an existing BSDF Normal input.

    The existing normal map is chained into the crease bump's Normal socket,
    so texture-authored skin maps are not discarded.
    """
    nodes, links = tree.nodes, tree.links
    normal = principled.inputs.get("Normal")
    if normal is None:
        return 0
    existing = normal.links[0].from_socket if normal.is_linked else None

    attribute1 = nodes.new("ShaderNodeAttribute")
    attribute1.name = "MPFB Foot Crease Mask"
    attribute1.attribute_name = CREASE_ATTRIBUTE
    attribute1.location = (-720, -420)
    attribute2 = nodes.new("ShaderNodeAttribute")
    attribute2.name = "MPFB Foot Pore Mask"
    attribute2.attribute_name = PORE_ATTRIBUTE
    attribute2.location = (-720, -740)

    coords = nodes.new("ShaderNodeTexCoord")
    coords.name = "MPFB Foot Object Coordinates"
    coords.location = (-1000, -240)

    wave = nodes.new("ShaderNodeTexWave")
    wave.name = "MPFB Foot Crease Bands"
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'Y'
    wave.wave_profile = 'SIN'
    wave.inputs["Scale"].default_value = 86.0
    wave.inputs["Distortion"].default_value = 7.0
    wave.inputs["Detail"].default_value = 2.0
    wave.location = (-720, -100)
    links.new(coords.outputs["Object"], wave.inputs["Vector"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "MPFB Foot Fine Skin Noise"
    noise.inputs["Scale"].default_value = 320.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.66
    noise.location = (-720, 120)
    links.new(coords.outputs["Object"], noise.inputs["Vector"])

    crease_height = nodes.new("ShaderNodeMath")
    crease_height.name = "MPFB Foot Crease Height"
    crease_height.operation = 'MULTIPLY'
    crease_height.location = (-380, -180)
    links.new(attribute1.outputs["Fac"], crease_height.inputs[0])
    links.new(wave.outputs["Fac"], crease_height.inputs[1])

    pore_height = nodes.new("ShaderNodeMath")
    pore_height.name = "MPFB Foot Pore Height"
    pore_height.operation = 'MULTIPLY'
    pore_height.location = (-380, 100)
    links.new(attribute2.outputs["Fac"], pore_height.inputs[0])
    links.new(noise.outputs["Fac"], pore_height.inputs[1])

    wrinkle_bump = nodes.new("ShaderNodeBump")
    wrinkle_bump.name = _BUMP_NAME_CREASE
    wrinkle_bump.label = "MPFB plantar/toe creases"
    wrinkle_bump.inputs["Distance"].default_value = 0.0009
    wrinkle_bump.location = (-100, -160)
    links.new(crease_height.outputs[0], wrinkle_bump.inputs["Height"])
    if existing:
        links.new(existing, wrinkle_bump.inputs["Normal"])

    pore_bump = nodes.new("ShaderNodeBump")
    pore_bump.name = _BUMP_NAME_PORE
    pore_bump.label = "MPFB skin microtexture"
    pore_bump.inputs["Distance"].default_value = 0.00017
    pore_bump.location = (90, -80)
    links.new(pore_height.outputs[0], pore_bump.inputs["Height"])
    links.new(wrinkle_bump.outputs["Normal"], pore_bump.inputs["Normal"])
    links.new(pore_bump.outputs["Normal"], normal)
    return 1


def _tree_has_principled(tree, depth=0):
    if depth > 7:
        return False
    for node in tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return True
        if node.type == 'GROUP' and node.node_tree and _tree_has_principled(node.node_tree, depth + 1):
            return True
    return False


def _inject_foot_shader(tree, depth=0):
    """Duplicate nested groups before editing; no global MPFB shader mutation."""
    if depth > 7:
        return 0
    count = 0
    for node in list(tree.nodes):
        if node.type == 'BSDF_PRINCIPLED':
            count += _add_bump_to_principled(tree, node)
        elif node.type == 'GROUP' and node.node_tree and _tree_has_principled(node.node_tree, depth + 1):
            unique_group = node.node_tree.copy()
            unique_group.name = node.node_tree.name + " - MPFB Foot"
            node.node_tree = unique_group
            count += _inject_foot_shader(unique_group, depth + 1)
    return count


def _set_bump_strengths(tree, wrinkles, pores, depth=0):
    if depth > 9:
        return
    for node in tree.nodes:
        if node.type == 'BUMP' and node.name == _BUMP_NAME_CREASE:
            node.inputs["Strength"].default_value = wrinkles
        elif node.type == 'BUMP' and node.name == _BUMP_NAME_PORE:
            node.inputs["Strength"].default_value = pores
        elif node.type == 'GROUP' and node.node_tree:
            _set_bump_strengths(node.node_tree, wrinkles, pores, depth + 1)


def _ensure_skin_material(basemesh, scene):
    """Provide working foot shading even on a fresh, untextured MPFB human.

    A new human is deliberately untextured, so a missing skin material is
    normal. Build an isolated, neutral skin preview in that case instead of
    issuing the old "requires a Principled material" warning. Existing
    material setups are never replaced without preserving their original.
    """
    mesh = basemesh.data
    had_slot = len(mesh.materials) > 0
    original = mesh.materials[0] if had_slot else None

    if original is not None and original.get(_MATERIAL_MARK):
        material = original  # Reapply with new user-adjusted slider strengths.
    else:
        created = original is None
        if original is not None and not original.use_nodes:
            # Respect authored legacy custom skin materials. The user can
            # create a node material manually rather than silently losing it.
            return False
        if created:
            material = bpy.data.materials.new(basemesh.name + " - Foot Skin Preview")
            material.use_nodes = True
            material.diffuse_color = (0.58, 0.50, 0.45, 1.0)
            principled = next((n for n in material.node_tree.nodes
                               if n.type == 'BSDF_PRINCIPLED'), None)
            if principled:
                principled.inputs["Base Color"].default_value = (0.58, 0.50, 0.45, 1.0)
                principled.inputs["Roughness"].default_value = 0.62
        else:
            material = original.copy()
            material.name = original.name + " - Foot Detail"

        try:
            if not material.node_tree or not _inject_foot_shader(material.node_tree):
                bpy.data.materials.remove(material)
                return False
            material[_MATERIAL_MARK] = True
            material[_ORIGINAL_MATERIAL] = original.name if original else ""
            material["mpfb_foot_created_skin"] = created
            material["mpfb_foot_had_slot"] = had_slot
            if original is not None:
                # Retain the old original after saving/reopening the blend.
                material["mpfb_foot_original_fake_user"] = bool(original.use_fake_user)
                original.use_fake_user = True
            if had_slot:
                mesh.materials[0] = material
            else:
                mesh.materials.append(material)
        except Exception:
            if material.users == 0:
                bpy.data.materials.remove(material)
            raise

    _set_bump_strengths(material.node_tree,
                        float(scene.mpfb_foot_wrinkles), float(scene.mpfb_foot_pores))
    return True


def _reset_foot_material(basemesh):
    mesh = basemesh.data
    if not mesh.materials or not mesh.materials[0]:
        return False
    material = mesh.materials[0]
    if not material.get(_MATERIAL_MARK):
        return False

    if material.get("mpfb_foot_created_skin", False):
        if material.get("mpfb_foot_had_slot", False):
            mesh.materials[0] = None
        else:
            mesh.materials.pop(index=0)
        if material.users == 0:
            bpy.data.materials.remove(material)
        return True

    original = bpy.data.materials.get(material.get(_ORIGINAL_MATERIAL, ""))
    if original is not None:
        mesh.materials[0] = original
        original.use_fake_user = bool(material.get("mpfb_foot_original_fake_user", False))
        if material.users == 0:
            bpy.data.materials.remove(material)
        return True

    # Source skin was manually deleted, so mute the bump rather than removing
    # the last remaining material and leaving the character unshaded.
    _set_bump_strengths(material.node_tree, 0.0, 0.0)
    return False

class MPFB_OT_Apply_Foot_Realism(MpfbOperator):
    """Add non-destructive arches, pads and procedural foot skin detail"""

    bl_idname = "mpfb.apply_foot_realism"
    bl_label = "Apply realistic foot detail"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(name="Foot preset", items=_PRESETS, default="CUSTOM")

    @classmethod
    def poll(cls, context):
        if not context.active_object or context.mode != "OBJECT":
            return False
        mesh = ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")
        return bool(mesh and TargetService.has_any_shapekey(mesh))

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        scene = context.scene
        if self.preset != "CUSTOM":
            if self.preset not in FOOT_DETAIL_PRESETS:
                self.report({'ERROR'}, "Unknown foot realism preset")
                return {'CANCELLED'}
            for name, value in FOOT_DETAIL_PRESETS[self.preset].items():
                setattr(scene, _SCENE_NAMES[name], value)
        basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")
        if not basemesh:
            return {'CANCELLED'}
        try:
            basis, fields = _ensure_foot_fields(basemesh)
        except (ValueError, OSError, KeyError) as err:
            self.report({'ERROR'}, "Foot geometry: " + str(err)[:200])
            return {'CANCELLED'}
        ObjectService.activate_blender_object(basemesh)
        _set_shape_keys(basemesh, basis, fields, scene)
        _write_float_attribute(basemesh.data, CREASE_ATTRIBUTE, fields["crease"])
        _write_float_attribute(basemesh.data, PORE_ATTRIBUTE, fields["pores"])
        try:
            shader_ready = _ensure_skin_material(basemesh, scene)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as err:
            shader_ready = False
            self.report({'WARNING'}, "Foot shader unchanged: " + str(err)[:180])
        if MODEL_PROPERTIES.get_value("refit", entity_reference=scene):
            HumanService.refit(basemesh, operator=self)
        if not shader_ready:
            self.report({'WARNING'}, "Foot shape applied; your existing skin shader cannot be modified")
        else:
            if scene.mpfb_foot_auto_preview and context.area and context.area.type == 'VIEW_3D':
                try:
                    context.space_data.shading.type = 'MATERIAL'
                except (AttributeError, TypeError):
                    pass
            self.report({'INFO'}, "Foot shape and skin detail applied; use Material Preview and zoom in")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Apply_Foot_Realism)


class MPFB_OT_Reset_Foot_Realism(MpfbOperator):
    """Revert foot realism without touching original MPFB foot targets"""

    bl_idname = "mpfb.reset_foot_realism"
    bl_label = "Reset realistic foot details"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return MPFB_OT_Apply_Foot_Realism.poll(context)

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")
        ObjectService.activate_blender_object(basemesh)
        keys = basemesh.data.shape_keys.key_blocks
        for name in FOOT_KEY_NAMES.values():
            key = keys.get(name)
            if key:
                key.value = 0.0
        material_restored = _reset_foot_material(basemesh)
        if MODEL_PROPERTIES.get_value("refit", entity_reference=context.scene):
            HumanService.refit(basemesh, operator=self)
        note = " and original skin restored" if material_restored else ""
        self.report({'INFO'}, "Foot shape details muted" + note)
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Reset_Foot_Realism)


class MPFB_OT_Preview_Foot_Skin(MpfbOperator):
    """Enable Material Preview so fine wrinkle normals can be seen"""

    bl_idname = "mpfb.preview_foot_skin"
    bl_label = "Show foot skin details"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return bool(context.area and context.area.type == 'VIEW_3D')

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        context.space_data.shading.type = 'MATERIAL'
        self.report({'INFO'}, "Material Preview enabled. Zoom into the feet to inspect skin texture")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Preview_Foot_Skin)


bpy.types.Scene.mpfb_foot_auto_preview = BoolProperty(
    name="Auto-show Material Preview",
    description="Switch the 3D viewport from Solid to Material Preview when applying foot detail",
    default=True)


for _key, _label, _desc, _default in (
        ("arch", "Plantar arch", "Raise medial arch in the foot sole", 0.58),
        ("heel", "Heel cushion", "Round the heel fat pad", 0.54),
        ("ball", "Forefoot cushion", "Round the ball of the foot", 0.65),
        ("toes", "Toe knuckles", "Slightly define dorsal toe/knuckle soft tissue", 0.40),
        ("wrinkles", "Creases / wrinkles", "Plantar folds and toe flexion lines in bump shading", 0.42),
        ("pores", "Skin microdetail", "Fine pore-level noise in the foot shader", 0.55)):
    setattr(bpy.types.Scene, _SCENE_NAMES[_key], FloatProperty(
        name=_label, description=_desc, min=0.0, max=1.0, default=_default))


class MPFB_PT_Foot_Realism_Panel(Abstract_Panel):
    """Editable foot proportions with locally masked skin microdetail."""

    bl_label = "Foot shape & skin realism"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "MPFB_PT_Experimental_Features_Panel"
    bl_category = "Experimental Features"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return MPFB_OT_Apply_Foot_Realism.poll(context)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.label(text="Anatomy-inspired foot details")
        row = layout.row(align=True)
        for preset, caption in (("natural", "Natural"), ("athletic", "Athletic"), ("mature", "Mature")):
            op = row.operator("mpfb.apply_foot_realism", text=caption)
            op.preset = preset
        for key in ("arch", "heel", "ball", "toes"):
            layout.prop(scene, _SCENE_NAMES[key])
        layout.separator()
        for key in ("wrinkles", "pores"):
            layout.prop(scene, _SCENE_NAMES[key])
        op = layout.operator("mpfb.apply_foot_realism", text="Apply custom foot detail")
        op.preset = "CUSTOM"
        layout.operator("mpfb.reset_foot_realism", text="Mute foot details", icon='LOOP_BACK')
        layout.prop(scene, "mpfb_foot_auto_preview")
        layout.operator("mpfb.preview_foot_skin", text="Show skin wrinkles in Material Preview")
        layout.label(text="Zoom to feet: Solid mode hides skin wrinkles")
        layout.label(text="Skin detail is Blender-only until baked")
        layout.label(text="Use Model > Feet for overall foot sizing")


ClassManager.add_class(MPFB_PT_Foot_Realism_Panel)
