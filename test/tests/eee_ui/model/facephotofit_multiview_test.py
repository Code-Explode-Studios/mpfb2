"""Regressions for joint front/profile fitting, including depth-only geometry.

Run under Blender's MPFB headless pytest suite; numerical solver itself is
NumPy-only and was also exercised from an ordinary Python interpreter.
"""

import json
import os

import bpy
import numpy as np
import pytest

from ... import dynamic_import, LocationService, ObjectService
from .._helpers import HumanFixture

Core = dynamic_import("mpfb.ui.model.facephotofit_multiview_core", "fit_multiview_landmarks")
Project = dynamic_import("mpfb.ui.model.facephotofit_multiview_core", "project_view")
ProfileMap = dynamic_import("mpfb.ui.model.facephotofit", "_profile_mapping")
Specs = dynamic_import("mpfb.ui.model.facephotofit", "_category_specs")
Model3D = dynamic_import("mpfb.ui.model.facephotofit", "_face_model_3d")
FitOperator = dynamic_import("mpfb.ui.model.facephotofit", "MPFB_OT_Fit_Face_Photo")
DepthCategories = dynamic_import("mpfb.ui.model.facephotofit", "PROFILE_CATEGORIES")


def test_profile_mode_registered_and_settings_available():
    assert bpy.ops.mpfb.fit_face_photo is not None
    scene = bpy.context.scene
    for name in ("mpfb_facefit_profile_photo", "mpfb_facefit_profile_landmarks_json",
                 "mpfb_facefit_profile_direction", "mpfb_facefit_profile_strength"):
        assert hasattr(scene, name), "Missing profile control " + name
    assert scene.mpfb_facefit_profile_direction in ("LEFT", "RIGHT")
    with HumanFixture():
        assert FitOperator.poll(bpy.context)


def test_profile_additional_morphs_are_only_enabled_in_multiview():
    default = Specs()
    combined = Specs(include_profile=True)
    assert len(default) == 34
    assert len(DepthCategories) == 13
    assert len(combined) == len(default) + len(DepthCategories)
    assert all(name not in set(x[0] for x in default) for name in DepthCategories)
    for _, section, positive, negative in combined:
        for target in positive + negative:
            base = os.path.join(LocationService.get_mpfb_data("targets"), section, target)
            assert os.path.isfile(base + ".target.gz") or os.path.isfile(base + ".target")


def test_profile_mapping_excludes_hidden_side_and_has_no_duplicates():
    mapping_path = os.path.join(LocationService.get_mpfb_data("face_fit"), "mp_to_mpfb.json")
    with open(mapping_path, encoding="utf-8") as handle:
        mapping = json.load(handle)["pairs"]
    left = ProfileMap(mapping, "LEFT")
    right = ProfileMap(mapping, "RIGHT")
    assert len(left) >= 12 and len(right) >= 12
    assert {int(p["mp"]) for p in left}.isdisjoint({263, 362, 300, 293})
    assert {int(p["mp"]) for p in right}.isdisjoint({33, 133, 70, 63})
    assert {int(p["mp"]) for p in left}.intersection({1, 4, 152})
    with pytest.raises(ValueError):
        ProfileMap(mapping, "FRONT")


def test_3d_projection_side_sign_and_vertical_axis():
    point = np.array([[0.3, -0.4, 1.7]])
    assert np.allclose(Project(point, "FRONT"), [[0.3, -1.7]])
    assert np.allclose(Project(point, "LEFT"), [[-0.4, -1.7]])
    assert np.allclose(Project(point, "RIGHT"), [[0.4, -1.7]])
    with pytest.raises(ValueError):
        Project(point, "UPSIDE_DOWN")


@pytest.mark.parametrize("direction", ("LEFT", "RIGHT"))
def test_joint_fit_recovers_depth_only_target_and_improves_both_views(direction):
    rng = np.random.default_rng(1)
    rest = rng.normal(0, 0.3, (57, 3))
    rest[:, 2] += np.linspace(1, 2, 57)
    rest[:12, 1] -= 0.3
    positive = rng.normal(0, 0.02, (4, 57, 3))
    negative = -positive * 0.7

    # Morph 0 ONLY changes depth: no front view can identify its amplitude.
    positive[0, :, 0] = 0
    positive[0, :, 2] = 0
    positive[0, :12, 1] -= 0.12
    negative[0] = -positive[0] * 0.7
    positive[1, :, 1] = 0
    negative[1] = -positive[1] * 0.7
    positive[2, :, 1] += np.linspace(0.12, -0.06, 57)
    positive[3, 12:25, 1] -= 0.06

    expected = np.array([0.68, 0.43, -0.45, 0.22])
    target = rest.copy()
    for i, value in enumerate(expected):
        target += value * positive[i] if value >= 0 else -value * negative[i]
    front = Project(target, "FRONT") * 300 + [800, 400]
    side_indices = np.arange(20)
    side = Project(target[side_indices], direction) * 200 + [250, 450]
    fitted, metrics = Core(
        rest, positive, negative, front, np.ones(57), side,
        side_indices, np.ones(len(side_indices)), profile_direction=direction,
        profile_strength=2, regularization=0.00001, outer_iterations=20,
    )
    assert metrics["front_after"] < metrics["front_before"]
    assert metrics["profile_after"] < metrics["profile_before"]
    assert abs(fitted[0] - expected[0]) < 0.1
    assert fitted[2] < 0
    assert np.max(np.abs(fitted - expected)) < 0.1

    untouched, _ = Core(rest, positive, negative, front, np.ones(57),
                        side, side_indices, np.ones(20), profile_direction=direction,
                        influence=0, outer_iterations=2)
    assert np.allclose(untouched, 0)


def test_joint_fit_rejects_invalid_controls_and_unmapped_indices():
    rng = np.random.default_rng(7)
    rest = rng.normal(size=(20, 3))
    p = rng.normal(size=(2, 20, 3)) * 0.03
    n = -p
    f = Project(rest, "FRONT")
    s = Project(rest[:8], "LEFT")
    args = (rest, p, n, f, np.ones(20), s, np.arange(8), np.ones(8))
    for kwargs in ({"profile_direction": "INVALID"}, {"profile_strength": -1},
                   {"regularization": -0.1}, {"influence": 1.2}):
        with pytest.raises(ValueError):
            Core(*args, **kwargs)
    with pytest.raises(ValueError):
        Core(rest, p, n, f, np.ones(20), s, [99] * 8, np.ones(8))


def test_3d_extractor_shapes_match_reference_topology():
    with HumanFixture() as fixture:
        path = os.path.join(LocationService.get_mpfb_data("face_fit"), "mp_to_mpfb.json")
        with open(path, encoding="utf-8") as handle:
            mapping = json.load(handle)["pairs"]
        # Use a single already available target so Blender memory footprint
        # stays small; the operator tests all 47 morph target paths above.
        target_specs = []
        for key in fixture.basemesh.data.shape_keys.key_blocks:
            if key.name != "Basis" and not key.name.startswith("$md"):
                target_specs.append(("existing", "test", [key.name], []))
                break
        if target_specs:
            indices = [int(pair["vertex"]) for pair in mapping]
            rest, positive, negative, initial = Model3D(fixture.basemesh, indices, target_specs)
            assert rest.shape == (len(indices), 3)
            assert positive.shape == (1, len(indices), 3)
            assert negative.shape == positive.shape
            assert initial.shape == (1,)
