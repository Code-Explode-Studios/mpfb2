"""Tests for face photo fitting; runs in the MPFB Blender test suite."""

import json
import os

import bpy
import numpy as np
from pytest import approx

from ... import dynamic_import, LocationService, ObjectService
from .._helpers import HumanFixture

Fit = dynamic_import("mpfb.ui.model.facephotofit_core", "fit_landmarks")
Align = dynamic_import("mpfb.ui.model.facephotofit_core", "similarity_align")
Specs = dynamic_import("mpfb.ui.model.facephotofit", "_category_specs")
Observed = dynamic_import("mpfb.ui.model.facephotofit", "_observed_landmarks")
Panel = dynamic_import("mpfb.ui.model.facephotofit", "MPFB_PT_Face_Photo_Fit_Panel")
Operator = dynamic_import("mpfb.ui.model.facephotofit", "MPFB_OT_Fit_Face_Photo")
Categories = dynamic_import("mpfb.ui.model.facephotofit", "FIT_CATEGORIES")


def test_photo_fit_registered_and_requires_a_human():
    assert bpy.ops.mpfb.fit_face_photo is not None
    ObjectService.deselect_and_deactivate_all()
    assert not Operator.poll(bpy.context)
    with HumanFixture():
        assert Operator.poll(bpy.context)
        assert Panel.poll(bpy.context)


def test_photo_fit_categories_exist_and_have_valid_files():
    assert len(Specs()) == len(Categories)
    for _, section, positive, negative in Specs():
        for target in positive + negative:
            base = os.path.join(LocationService.get_mpfb_data("targets"), section, target)
            assert os.path.isfile(base + ".target.gz") or os.path.isfile(base + ".target")


def test_mapping_is_compatible_with_bundled_basemesh():
    mapping_file = os.path.join(LocationService.get_mpfb_data("face_fit"), "mp_to_mpfb.json")
    with open(mapping_file, encoding="utf-8") as handle:
        mapping = json.load(handle)["pairs"]
    assert len(mapping) >= 50
    with HumanFixture() as fixture:
        assert max(int(p["vertex"]) for p in mapping) < len(fixture.basemesh.data.vertices)
    result = {"width": 640, "height": 480, "landmarks_px": [[300, 200, 0]] * 478}
    points, indices, weights = Observed(result, mapping)
    assert points.shape == (len(mapping), 2)
    assert weights.min() > 0
    assert len(indices) == len(mapping)


def test_photo_fit_rejects_malformed_landmarks():
    try:
        Observed({"width": 1, "height": 1, "landmarks_px": []}, [{"mp": 0, "vertex": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid landmarks must be rejected")


def test_weighted_similarity_align_recovers_rotation_scale_and_translation():
    points = np.array([[0, 0], [2, 0], [0, 2], [3, 1], [1, 4]], dtype=float)
    angle = np.deg2rad(25)
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    transformed = 3 * points @ matrix.T + [10, -8]
    aligned = Align(points, transformed, np.ones(len(points)))
    assert np.allclose(aligned, transformed, atol=1e-8)


def test_photo_fit_solver_improves_synthetic_landmark_error():
    rng = np.random.default_rng(42)
    # Create nonrigid deltas that cannot be explained away by 2D alignment.
    rest = rng.normal(size=(20, 2))
    plus = rng.normal(size=(2, 20, 2)) * 0.16
    minus = -plus * 0.8
    expected = np.array([0.7, -0.5])
    target = rest + expected[0] * plus[0] + (-expected[1]) * minus[1]
    # Pixel photo with arbitrary translation, scale and roll.
    photo = target * 120 + [200, 320]
    result, metrics = Fit(rest, plus, minus, photo, np.ones(20),
                          regularization=0.0001, influence=1.0, outer_iterations=9)
    assert result.shape == (2,)
    assert np.all(result <= 1) and np.all(result >= -1)
    assert metrics["after"] < metrics["before"]
    assert result[0] > 0 and result[1] < 0
    no_op, _ = Fit(rest, plus, minus, photo, np.ones(20), influence=0)
    assert np.allclose(no_op, [0, 0])
