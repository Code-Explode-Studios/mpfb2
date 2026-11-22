"""Regression tests for MPFB foot geometry and localized wrinkle skin detail.

Part of the Blender MPFB test suite; the computational core is pure Python.
"""

import json
import os

import bpy
import pytest

from ... import ObjectService, TargetService, dynamic_import
from .._helpers import HumanFixture, BasemeshWithV2SkinFixture, MockOperatorBase

_CORE = "mpfb.ui.model.footrealism_core"
_MOD = "mpfb.ui.model.footrealism"
Compute = dynamic_import(_CORE, "compute_foot_fields")
KeyNames = dynamic_import(_CORE, "FOOT_KEY_NAMES")
CreaseName = dynamic_import(_CORE, "CREASE_ATTRIBUTE")
PoreName = dynamic_import(_CORE, "PORE_ATTRIBUTE")
Presets = dynamic_import(_MOD, "FOOT_DETAIL_PRESETS")
Operator = dynamic_import(_MOD, "MPFB_OT_Apply_Foot_Realism")
Reset = dynamic_import(_MOD, "MPFB_OT_Reset_Foot_Realism")
BodyCount = dynamic_import(_MOD, "_body_count")


def _synthetic_human():
    """Two feet with thick forefoot, both arches and elevated body vertices."""
    vertices = []
    for side in (-1, 1):
        for n in range(11):
            for m in range(12):
                x = side * (0.07 + 0.008 * m)
                y = 0.10 - n * 0.025
                z = (0.002 if m < 5 else 0.025) + 0.002 * (n % 3)
                vertices.append((x, y, z))
    vertices += [(0.2, 0, 1.7 + (i % 7) * 0.002) for i in range(80)]
    body_count = len(vertices)
    vertices += [(0.15, -0.05, 0.01) for i in range(16)]  # rigging helpers
    return vertices, body_count


def test_foot_core_localizes_detail_to_two_feet_but_never_helper_geometry():
    vertices, body_count = _synthetic_human()
    fields = Compute(vertices, body_count)
    assert fields["affected_vertices"] >= 100
    assert max(fields["crease"]) > 0.05
    assert max(fields["pores"]) > 0.1
    assert all(len(v) == len(vertices) for v in fields["deltas"].values())
    for name in KeyNames:
        assert all(delta == (0.0, 0.0, 0.0)
                   for delta in fields["deltas"][name][body_count:])
    assert not any(fields["crease"][body_count:])
    assert not any(fields["pores"][body_count:])
    assert max(delta[2] for delta in fields["deltas"]["arch"]) > 0
    assert min(delta[2] for delta in fields["deltas"]["ball"]) < 0


def test_foot_core_rejects_non_human_topology():
    vertices, count = _synthetic_human()
    with pytest.raises(ValueError):
        Compute(vertices, count + 9999)
    with pytest.raises(ValueError):
        Compute(vertices[:60], 60)
    with pytest.raises(ValueError):
        Compute([(0, 0, 0)] * len(vertices), count)


def test_foot_operator_registered_and_presets_normalized():
    assert bpy.ops.mpfb.apply_foot_realism is not None
    assert bpy.ops.mpfb.reset_foot_realism is not None
    assert len(Presets) == 3
    for preset in Presets.values():
        assert set(preset) == {"arch", "heel", "ball", "toes", "wrinkles", "pores"}
        assert all(0 <= value <= 1 for value in preset.values())
    ObjectService.deselect_and_deactivate_all()
    assert not Operator.poll(bpy.context)
    with HumanFixture():
        assert Operator.poll(bpy.context)
        assert Reset.poll(bpy.context)


def test_foot_core_real_hm08_basis_is_both_sided_and_preserves_helpers():
    with HumanFixture():
        mesh = ObjectService.find_object_of_type_amongst_nearest_relatives(
            bpy.context.active_object, "Basemesh")
        assert mesh and mesh.data.shape_keys
        basis = mesh.data.shape_keys.reference_key
        positions = [(v.co.x, v.co.y, v.co.z) for v in basis.data]
        count = BodyCount()
        assert count == 13380
        assert len(positions) >= count
        fields = Compute(positions, count)
        assert fields["affected_vertices"] > 100
        assert any(abs(v[2]) > 1e-7 for v in fields["deltas"]["arch"][:count])
        assert all(v == (0.0, 0.0, 0.0) for v in fields["deltas"]["arch"][count:])


def test_foot_preset_and_reset_keep_existing_target_values_and_skin():
    with HumanFixture():
        mesh = ObjectService.find_object_of_type_amongst_nearest_relatives(
            bpy.context.active_object, "Basemesh")
        original = mesh.data.materials[0] if mesh.data.materials else None
        # Existing geometry (including any muscle and sex morphs) is untouched.
        source_keys = {k.name: k.value for k in mesh.data.shape_keys.key_blocks}
        first = MockOperatorBase(preset="natural")
        assert Operator.hardened_execute(first, bpy.context) == {'FINISHED'}
        assert all(mesh.data.shape_keys.key_blocks.get(name) for name in KeyNames.values())
        for name in KeyNames.values():
            assert mesh.data.shape_keys.key_blocks[name].value > 0.0
        assert mesh.data.attributes.get(CreaseName)
        assert mesh.data.attributes.get(PoreName)
        assert all(mesh.data.shape_keys.key_blocks[n].value == old
                   for n, old in source_keys.items())
        first_count = len(mesh.data.shape_keys.key_blocks)
        second = MockOperatorBase(preset="mature")
        assert Operator.hardened_execute(second, bpy.context) == {'FINISHED'}
        assert len(mesh.data.shape_keys.key_blocks) == first_count  # idempotent
        if original and original.use_nodes and mesh.data.materials[0] is not original:
            assert mesh.data.materials[0].get("mpfb_foot_procedural_skin")
        assert Reset.hardened_execute(MockOperatorBase(), bpy.context) == {'FINISHED'}
        assert all(mesh.data.shape_keys.key_blocks[name].value == 0.0 for name in KeyNames.values())
        if original and original.use_nodes:
            assert mesh.data.materials[0] is original
        assert all(mesh.data.shape_keys.key_blocks[n].value == old
                   for n, old in source_keys.items())


def _count_foot_bump_nodes(tree, level=0):
    if tree is None or level > 8:
        return 0
    found = 0
    for node in tree.nodes:
        if node.type == 'BUMP' and node.name.startswith("MPFB Foot "):
            found += 1
        if node.type == 'GROUP' and node.node_tree:
            found += _count_foot_bump_nodes(node.node_tree, level + 1)
    return found


def test_foot_shader_copies_v2_skin_and_preserves_original_material():
    # Dedicated skin fixture ensures the actual default nested-group path is
    # exercised, not only a minimal human with no material slots.
    with BasemeshWithV2SkinFixture() as fixture:
        mesh = fixture.basemesh
        original = mesh.data.materials[0]
        original_node_count = _count_foot_bump_nodes(original.node_tree)
        operator = MockOperatorBase(preset="natural")
        assert Operator.hardened_execute(operator, bpy.context) == {'FINISHED'}
        modified = mesh.data.materials[0]
        assert modified is not original
        assert modified.get("mpfb_foot_procedural_skin")
        assert _count_foot_bump_nodes(modified.node_tree) >= 2
        assert _count_foot_bump_nodes(original.node_tree) == original_node_count
        assert Reset.hardened_execute(MockOperatorBase(), bpy.context) == {'FINISHED'}
        assert mesh.data.materials[0] is original


def test_fresh_default_human_gets_generated_skin_preview_and_restores_empty_slots():
    # Regression: default untextured humans used to warn
    # "Foot shape applied; skin shader needs a node-based Principled body material".
    # The extension must work out of the box on the default human, without
    # requiring the user to first install a system asset/skin pack.
    with HumanFixture():
        human = ObjectService.find_object_of_type_amongst_nearest_relatives(
            bpy.context.active_object, "Basemesh")
        assert human is not None
        had_slot = len(human.data.materials) > 0
        before = human.data.materials[0] if had_slot else None
        if before is not None:
            pytest.skip("HumanFixture now ships with a material; no-material path unavailable")

        assert Operator.hardened_execute(MockOperatorBase(preset="natural"), bpy.context) == {'FINISHED'}
        generated = human.data.materials[0]
        assert generated is not None
        assert generated.use_nodes
        assert generated.get("mpfb_foot_procedural_skin", False)
        assert generated.get("mpfb_foot_created_skin", False)
        assert _count_foot_bump_nodes(generated.node_tree) >= 2
        assert max(v.value for v in human.data.attributes[CreaseName].data) > 0.2

        # Reapplying is idempotent, even for a newly created material.
        assert Operator.hardened_execute(MockOperatorBase(preset="mature"), bpy.context) == {'FINISHED'}
        assert human.data.materials[0] is generated

        assert Reset.hardened_execute(MockOperatorBase(), bpy.context) == {'FINISHED'}
        if had_slot:
            assert human.data.materials[0] is None
        else:
            assert len(human.data.materials) == 0


def test_foot_preset_shape_offsets_are_visibly_larger_on_real_hm08():
    # A previous algorithm displaced arch by only a few millimetres at
    # maximum, which was barely visible even up close.
    with HumanFixture():
        human = ObjectService.find_object_of_type_amongst_nearest_relatives(
            bpy.context.active_object, "Basemesh")
        basis = human.data.shape_keys.reference_key
        coords = [(v.co.x, v.co.y, v.co.z) for v in basis.data]
        foot = Compute(coords, BodyCount())
        arch_peak = max(abs(d[2]) for d in foot["deltas"]["arch"])
        heel_peak = max(max(abs(x) for x in d) for d in foot["deltas"]["heel"])
        toe_peak = max(abs(d[2]) for d in foot["deltas"]["toes"])
        # Intended full-strength peaks in metres at default 1.7 m scale.
        assert arch_peak >= 0.011
        assert heel_peak >= 0.004
        assert toe_peak >= 0.006
        assert all(all(abs(x) < 1e-8 for x in d)
                   for d in foot["deltas"]["heel"][BodyCount():])
