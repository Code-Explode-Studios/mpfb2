"""Experimental HD body geometry that preserves the MPFB control topology.

Shape keys and existing rig stay on the original base mesh. Subdivision and
optional UV-image displacement are placed BEFORE the Armature modifier so that
added surface detail follows deformation. Sculptable Multires is also available;
its stored sculpt levels are never automatically deleted when changing modes.

This does not solve anatomical topology, joint correctives, or engine export.
GPL-3.0-or-later. Copyright (C) 2026 MPFB contributors.
"""

import os

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from ... import ClassManager
from ...services import LogService, ObjectService
from ..abstractpanel import Abstract_Panel
from ..mpfboperator import MpfbOperator
from .experimentalpanel import EXPERIMENTAL_CATEGORY, EXPERIMENTAL_PANEL

_LOG = LogService.get_logger("model.hdgeometry")
_HD_SUBSURF = "MPFB HD Subdivision"
_HD_MULTIRES = "MPFB HD Multires Sculpt"
_HD_DISPLACE = "MPFB HD Image Displacement"
_HD_TEXTURE = "MPFB HD Height Texture"
_OWNER = "mpfb_character_studio_hd"


def _human(context):
    if not context.active_object:
        return None
    obj = ObjectService.find_object_of_type_amongst_nearest_relatives(
        context.active_object, "Basemesh")
    return obj if obj and obj.type == 'MESH' else None


def _owned_modifier(obj, name, mod_type):
    mod = obj.modifiers.get(name)
    if mod and (mod.type != mod_type or not mod.get(_OWNER)):
        raise ValueError("Modifier name is already used by another tool: " + name)
    return mod


def _ensure_modifier(obj, name, mod_type):
    mod = _owned_modifier(obj, name, mod_type)
    if mod is None:
        mod = obj.modifiers.new(name=name, type=mod_type)
        mod[_OWNER] = True
    return mod


def _move_detail_before_armature(obj):
    """Detail is evaluated in rest space and skinned together with the body.

    Shape keys are evaluated before modifiers in Blender. The detail modifiers
    are kept in order (subdivision, displacement) immediately before the first
    Armature, without reordering user-created modifiers relative to each other.
    """
    rig_index = next((i for i, m in enumerate(obj.modifiers)
                      if m.type == 'ARMATURE'), len(obj.modifiers))
    ordered = [m for m in obj.modifiers
               if m.name in (_HD_SUBSURF, _HD_MULTIRES, _HD_DISPLACE)
               and m.get(_OWNER)]
    ordered.sort(key=lambda m: 0 if m.type in {'SUBSURF', 'MULTIRES'} else 1)
    for mod in ordered:
        current_index = obj.modifiers.find(mod.name)
        # Moving a detail modifier from before the rig shrinks the target slot.
        if current_index < rig_index:
            rig_index -= 1
        bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=rig_index)
        rig_index += 1


def _image_texture(path, modifier):
    """Keep the imported height map isolated from authored skin materials."""
    texture = modifier.texture
    if texture and texture.type == 'IMAGE' and texture.get(_OWNER):
        if texture.users > 1:
            texture = texture.copy()
    else:
        texture = bpy.data.textures.new(name=_HD_TEXTURE, type='IMAGE')
    texture[_OWNER] = True

    # Reuse the own image when unchanged; otherwise duplicate the image
    # datablock before switching it to data/non-colour interpretation.
    image = texture.image
    if not image or bpy.path.abspath(image.filepath) != path:
        image = bpy.data.images.load(path, check_existing=True).copy()
        texture.image = image
    elif image.users > 1:
        image = image.copy()
        texture.image = image
    try:
        image.colorspace_settings.name = "Non-Color"
    except (AttributeError, TypeError):
        pass
    return texture


def _audit_lines(obj):
    rigs = [mod for mod in obj.modifiers if mod.type == 'ARMATURE']
    details = [(i, mod) for i, mod in enumerate(obj.modifiers)
               if mod.get(_OWNER) and mod.name in
               (_HD_SUBSURF, _HD_MULTIRES, _HD_DISPLACE)]
    lines = ["Mesh: %s | base vertices: %s" % (obj.name, len(obj.data.vertices))]
    lines.append("Shape keys: %s" % (len(obj.data.shape_keys.key_blocks)
                                      if obj.data.shape_keys else 0))
    if not rigs:
        lines.append("No Armature modifier: pose test unavailable")
    else:
        lines.append("Armature(s): %s" % len(rigs))
        lines.append("Preserve Volume: %s" %
                     ("on" if all(m.use_deform_preserve_volume for m in rigs) else "off/mixed"))
        first_rig = next(i for i, mod in enumerate(obj.modifiers) if mod.type == 'ARMATURE')
        if any(i > first_rig for i, _ in details):
            lines.append("WARNING: HD modifier follows armature; reapply HD setup")
        else:
            lines.append("Detail modifiers precede armature: OK")
    if any(mod.type == 'MULTIRES' for _, mod in details):
        lines.append("Multires sculpt: test joint poses after each sculpt")
    lines.append("Test elbows, knees, shoulders, wrists, hips and toes")
    lines.append("Check corrective shape keys and skin weights")
    lines.append("Preserve Volume is a Blender preview, not an export guarantee")
    return lines


class MPFB_OT_Apply_HD_Geometry(MpfbOperator):
    """Configure non-destructive high-resolution subdivision and height detail"""

    bl_idname = "mpfb.apply_hd_geometry"
    bl_label = "Apply HD geometry settings"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and _human(context) is not None

    def hardened_execute(self, context):
        obj = _human(context)
        scene = context.scene
        if obj is None:
            return {'CANCELLED'}
        path = bpy.path.abspath(scene.mpfb_hd_image).strip()
        if path and not os.path.isfile(path):
            self.report({'ERROR'}, "HD image does not exist: " + path)
            return {'CANCELLED'}
        if path and not obj.data.uv_layers:
            self.report({'ERROR'}, "Displacement requires UV coordinates on the basemesh")
            return {'CANCELLED'}
        mask = scene.mpfb_hd_vertex_group.strip()
        if path and mask and mask not in obj.vertex_groups:
            self.report({'ERROR'}, "Vertex group not found: " + mask)
            return {'CANCELLED'}
        try:
            subdivision = _owned_modifier(obj, _HD_SUBSURF, 'SUBSURF')
            multires = _owned_modifier(obj, _HD_MULTIRES, 'MULTIRES')
            _owned_modifier(obj, _HD_DISPLACE, 'DISPLACE')
        except ValueError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if scene.mpfb_hd_mode == 'MULTIRES' and subdivision is not None:
            self.report({'ERROR'}, "Remove existing HD Subdivision before switching to Multires")
            return {'CANCELLED'}
        if scene.mpfb_hd_mode == 'SUBSURF' and multires is not None:
            self.report({'ERROR'}, "Multires contains sculpt data; remove it explicitly to switch modes")
            return {'CANCELLED'}

        ObjectService.activate_blender_object(obj)
        try:
            if scene.mpfb_hd_mode == 'SUBSURF':
                mod = _ensure_modifier(obj, _HD_SUBSURF, 'SUBSURF')
                mod.subdivision_type = 'CATMULL_CLARK'
                mod.levels = scene.mpfb_hd_preview
                mod.render_levels = scene.mpfb_hd_render
                mod.show_viewport = True
                mod.show_render = True
            else:
                mod = _ensure_modifier(obj, _HD_MULTIRES, 'MULTIRES')
                desired = scene.mpfb_hd_sculpt_levels
                while mod.total_levels < desired:
                    bpy.ops.object.multires_subdivide(modifier=mod.name, mode='CATMULL_CLARK')
                mod.levels = desired
                mod.sculpt_levels = desired
                mod.render_levels = desired
                mod.show_viewport = True
                mod.show_render = True

            if path:
                displace = _ensure_modifier(obj, _HD_DISPLACE, 'DISPLACE')
                displace.texture = _image_texture(path, displace)
                displace.texture_coords = 'UV'
                displace.uv_layer = obj.data.uv_layers.active.name
                displace.strength = scene.mpfb_hd_strength
                displace.mid_level = scene.mpfb_hd_midlevel
                displace.vertex_group = mask
                displace.show_viewport = True
                displace.show_render = True
            else:
                previous = _owned_modifier(obj, _HD_DISPLACE, 'DISPLACE')
                if previous:
                    obj.modifiers.remove(previous)
            _move_detail_before_armature(obj)
        except (RuntimeError, ValueError, TypeError, AttributeError) as error:
            self.report({'ERROR'}, "HD setup failed: " + str(error))
            return {'CANCELLED'}

        scene.mpfb_hd_audit = " | ".join(_audit_lines(obj))
        self.report({'INFO'}, "HD geometry configured; original body topology retained")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Apply_HD_Geometry)


class MPFB_OT_Remove_HD_Geometry(MpfbOperator):
    """Remove only owned HD modifiers; Multires sculpted changes will be lost"""

    bl_idname = "mpfb.remove_hd_geometry"
    bl_label = "Remove experimental HD modifiers"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        return MPFB_OT_Apply_HD_Geometry.poll(context)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def hardened_execute(self, context):
        obj = _human(context)
        for name, kind in ((_HD_SUBSURF, 'SUBSURF'), (_HD_MULTIRES, 'MULTIRES'),
                           (_HD_DISPLACE, 'DISPLACE')):
            try:
                mod = _owned_modifier(obj, name, kind)
            except ValueError:
                continue
            if mod:
                obj.modifiers.remove(mod)
        context.scene.mpfb_hd_audit = "HD modifiers removed; base mesh unchanged"
        self.report({'INFO'}, "HD modifiers removed (Multires sculpt data deleted if present)")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Remove_HD_Geometry)


class MPFB_OT_HD_Deformation_Audit(MpfbOperator):
    """Report armature deformation risks and modifier ordering"""

    bl_idname = "mpfb.hd_deformation_audit"
    bl_label = "Audit rig deformation"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        return _human(context) is not None

    def hardened_execute(self, context):
        obj = _human(context)
        context.scene.mpfb_hd_audit = " | ".join(_audit_lines(obj))
        self.report({'INFO'}, "HD rig diagnostic updated below")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_HD_Deformation_Audit)


class MPFB_OT_HD_Rig_Preview(MpfbOperator):
    """Toggle Blender's dual-quaternion preview for Armature modifiers"""

    bl_idname = "mpfb.hd_rig_preview"
    bl_label = "Set rig volume preview"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        return _human(context) is not None

    def hardened_execute(self, context):
        obj = _human(context)
        rigs = [mod for mod in obj.modifiers if mod.type == 'ARMATURE']
        if not rigs:
            self.report({'WARNING'}, "No Armature modifier found on this body")
            return {'CANCELLED'}
        for rig in rigs:
            rig.use_deform_preserve_volume = context.scene.mpfb_hd_preserve_volume
        context.scene.mpfb_hd_audit = " | ".join(_audit_lines(obj))
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_HD_Rig_Preview)


for name, prop in (
    ("mpfb_hd_mode", EnumProperty(
        name="Detail mode", description="Live smoothing or permanent sculptable Multires levels",
        items=(("SUBSURF", "Subdivision", "Live non-destructive subdivided surface"),
               ("MULTIRES", "Multires sculpt", "Editable high-resolution sculpt layers")),
        default="SUBSURF")),
    ("mpfb_hd_preview", IntProperty(name="Viewport levels", min=0, max=6, default=2)),
    ("mpfb_hd_render", IntProperty(name="Render levels", min=0, max=6, default=3)),
    ("mpfb_hd_sculpt_levels", IntProperty(
        name="Sculpt levels", description="Stored Multires subdivision levels (high memory use)",
        min=1, max=4, default=2)),
    ("mpfb_hd_image", StringProperty(
        name="UV height image", description="Optional 16-bit displacement map in the body's UV layout",
        subtype='FILE_PATH')),
    ("mpfb_hd_strength", FloatProperty(
        name="Displacement strength", description="Object-local units; start small",
        min=0.0, max=0.1, default=0.001, precision=5)),
    ("mpfb_hd_midlevel", FloatProperty(
        name="Midlevel / zero", description="Image gray value treated as neutral surface",
        min=0.0, max=1.0, default=0.5)),
    ("mpfb_hd_vertex_group", StringProperty(
        name="Vertex group mask", description="Optional vertex group limiting displacement (blank = entire body)",
        default="")),
    ("mpfb_hd_preserve_volume", BoolProperty(
        name="Preserve Volume (Blender)",
        description="Armature dual quaternion preview; not guaranteed in exports", default=False)),
    ("mpfb_hd_audit", StringProperty(name="HD diagnostic", default="")),
):
    setattr(bpy.types.Scene, name, prop)


class MPFB_PT_HD_Geometry_Panel(Abstract_Panel):
    """High-definition mesh settings and quick joint deformation diagnostics."""

    bl_label = "HD body geometry & deformation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = EXPERIMENTAL_PANEL
    bl_category = EXPERIMENTAL_CATEGORY
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _human(context) is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.label(text="Morphs and original topology stay editable")
        layout.prop(scene, "mpfb_hd_mode")
        if scene.mpfb_hd_mode == 'SUBSURF':
            row = layout.row(align=True)
            row.prop(scene, "mpfb_hd_preview")
            row.prop(scene, "mpfb_hd_render")
        else:
            layout.prop(scene, "mpfb_hd_sculpt_levels")
            layout.label(text="Stored sculpt data; use Sculpt mode")
        box = layout.box()
        box.label(text="Optional displacement (UV-mapped)")
        box.prop(scene, "mpfb_hd_image")
        box.prop(scene, "mpfb_hd_strength")
        box.prop(scene, "mpfb_hd_midlevel")
        box.prop(scene, "mpfb_hd_vertex_group")
        box.label(text="Maps need the MPFB body's UV layout")
        box.label(text="Foot-only maps need UV projection/masking")
        layout.operator("mpfb.apply_hd_geometry", text="Apply / update HD body", icon='MOD_SUBSURF')
        layout.operator("mpfb.remove_hd_geometry", text="Remove HD (deletes sculpt layers)", icon='TRASH')
        rigbox = layout.box()
        rigbox.label(text="Animation / joints")
        rigbox.prop(scene, "mpfb_hd_preserve_volume")
        rigbox.operator("mpfb.hd_rig_preview", text="Update Blender rig preview")
        rigbox.operator("mpfb.hd_deformation_audit", text="Audit rig / modifier stack", icon='INFO')
        if scene.mpfb_hd_audit:
            for line in scene.mpfb_hd_audit.split(" | "):
                rigbox.label(text=line[:115])
        rigbox.label(text="Test bends in Pose Mode; add corrective morphs")
        rigbox.label(text="Dual-quaternion preview is not export-safe")


ClassManager.add_class(MPFB_PT_HD_Geometry_Panel)