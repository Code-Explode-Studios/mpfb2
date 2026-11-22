"""Regression tests for male/female physique presets; requires Blender test runner."""

import bpy
from pytest import approx

from ... import dynamic_import, ObjectService, TargetService

HumanObjectProperties = dynamic_import("mpfb.entities.objectproperties", "HumanObjectProperties")
from .._helpers import HumanFixture, MockOperatorBase

MOD = "mpfb.ui.model.muscularitypresets"
Operator = dynamic_import(MOD, "MPFB_OT_Apply_Muscularity_Preset")
Presets = dynamic_import(MOD, "MUSCULARITY_LEVELS")
Regions = dynamic_import(MOD, "MUSCLE_REGIONS")


def test_muscularity_preset_operator_is_registered_and_polls_safely():
    assert bpy.ops.mpfb.apply_muscularity_preset is not None
    ObjectService.deselect_and_deactivate_all()
    assert not Operator.poll(bpy.context)
    with HumanFixture():
        assert Operator.poll(bpy.context)


def test_presets_are_normalized_and_cover_both_sides():
    assert len(Presets) == 4
    for preset in Presets.values():
        assert 0.0 <= preset["muscle"] <= 1.0
        assert 0.0 <= preset["weight"] <= 1.0
        assert set(preset["regions"]) == set(Regions)
        assert all(0.0 <= value <= 1.0 for value in preset["regions"].values())
    assert len(Regions["upper_arms"][1]) == 2
    assert len(Regions["thighs"][1]) == 2


def test_female_and_male_presets_adjust_only_selected_regions_and_macros():
    with HumanFixture():
        basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(bpy.context.active_object, "Basemesh")
        assert basemesh is not None
        age = HumanObjectProperties.get_value("age", entity_reference=basemesh)
        height = HumanObjectProperties.get_value("height", entity_reference=basemesh)

        for gender, label, value in (("female", "athletic", 0.0), ("male", "bodybuilder", 1.0)):
            operator = MockOperatorBase(preset=gender + "_" + label)
            assert Operator.hardened_execute(operator, bpy.context) == {'FINISHED'}
            operator.mock_report.assert_no_errors()
            preset = Presets[label]
            assert HumanObjectProperties.get_value("gender", entity_reference=basemesh) == approx(value)
            assert HumanObjectProperties.get_value("muscle", entity_reference=basemesh) == approx(preset["muscle"])
            assert HumanObjectProperties.get_value("weight", entity_reference=basemesh) == approx(preset["weight"])
            assert HumanObjectProperties.get_value("age", entity_reference=basemesh) == approx(age)
            assert HumanObjectProperties.get_value("height", entity_reference=basemesh) == approx(height)
            for region, (_, names) in Regions.items():
                for name in names:
                    assert TargetService.get_target_value(basemesh, name + "-incr") == approx(preset["regions"][region])
                    assert TargetService.get_target_value(basemesh, name + "-decr") == approx(0.0)


def test_invalid_scripted_preset_cannot_mutate_character():
    with HumanFixture():
        basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(bpy.context.active_object, "Basemesh")
        before = HumanObjectProperties.get_value("muscle", entity_reference=basemesh)
        operator = MockOperatorBase(preset="not_real")
        assert Operator.hardened_execute(operator, bpy.context) == {'CANCELLED'}
        assert HumanObjectProperties.get_value("muscle", entity_reference=basemesh) == before
