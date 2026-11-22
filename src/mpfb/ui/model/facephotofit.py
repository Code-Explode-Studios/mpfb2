"""Fit an editable MPFB face to a portrait using local MediaPipe landmarks.

This adapts FaceFit's GPL-3.0 MediaPipe<->MPFB vertex correspondence data
(https://github.com/VGCLUB/facefit-mpfb) while implementing the fitting UI,
morph preparation and numpy-only optimizer for MPFB's native Model panel.

Copyright (C) 2026 MPFB contributors. GPL-3.0-or-later.
"""

import json
import os
import subprocess

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, StringProperty

from ... import ClassManager
from ...services import HumanService, LocationService, LogService, ObjectService, TargetService, UiService
from ..abstractpanel import Abstract_Panel
from ..mpfboperator import MpfbOperator
from .facephotofit_core import fit_landmarks
from .modelpanel import MODEL_PROPERTIES

_LOG = LogService.get_logger("model.facephotofit")

# This conservative subset has documented built-in .target(.gz) assets in
# data/targets/target.json. All sided target pairs are coupled symmetrically.
FIT_CATEGORIES = (
    "head-scale-vert-decr-incr", "head-scale-horiz-decr-incr", "head-fat-decr-incr",
    "head-oval", "head-round", "head-square",
    "chin-height-decr-incr", "chin-width-decr-incr", "chin-jaw-drop-decr-incr",
    "chin-bones-decr-incr", "chin-triangle",
    "eye-scale-decr-incr", "eye-trans-in-out", "eye-trans-down-up",
    "eye-height1-decr-incr", "eye-height2-decr-incr",
    "eyebrows-angle-down-up", "eyebrows-trans-down-up",
    "nose-scale-vert-decr-incr", "nose-scale-horiz-decr-incr",
    "nose-trans-down-up", "nose-nostrils-width-decr-incr",
    "nose-point-width-decr-incr", "nose-width1-decr-incr",
    "mouth-scale-horiz-decr-incr", "mouth-scale-vert-decr-incr",
    "mouth-trans-down-up", "mouth-upperlip-height-decr-incr",
    "mouth-lowerlip-height-decr-incr", "mouth-upperlip-width-decr-incr",
    "mouth-lowerlip-width-decr-incr", "cheek-volume-decr-incr",
    "cheek-bones-decr-incr", "forehead-scale-vert-decr-incr",
)

# Landmark confidence varies by feature. Frontal eye corners typically produce
# stable correspondences; the jaw silhouette varies with camera viewpoint.
_LANDMARK_WEIGHTS = {
    33: 1.7, 133: 1.7, 263: 1.7, 362: 1.7,
    61: 1.4, 291: 1.4, 1: 1.4, 4: 1.4,
    152: 1.4, 168: 1.3,
}


def _category_specs():
    """Construct side-aware fitting morphs from authoritative target metadata."""
    path = os.path.join(LocationService.get_mpfb_data("targets"), "target.json")
    with open(path, encoding="utf-8") as handle:
        sections = json.load(handle)
    wanted = set(FIT_CATEGORIES)
    found = {}
    for section, data in sections.items():
        for category in data.get("categories", []):
            name = category["name"]
            if name not in wanted:
                continue
            if name in found:
                raise ValueError("Ambiguous MPFB target category " + name)
            opposed = category.get("opposites")
            sides = ("left", "right") if category.get("has_left_and_right") else ("unsided",)
            if opposed:
                positive = [opposed["positive-" + side] for side in sides]
                negative = [opposed["negative-" + side] for side in sides]
            else:
                positive = list(category["targets"])
                negative = []
            if not positive or any(not x for x in positive + negative):
                raise ValueError("Invalid facial category " + name)
            found[name] = (section, positive, negative)
    missing = wanted - set(found)
    if missing:
        raise ValueError("Missing built-in facial morph categories: " + ", ".join(sorted(missing)))
    return [(name, *found[name]) for name in FIT_CATEGORIES]


def _image_landmarks(scene, use_saved_landmarks):
    if use_saved_landmarks:
        path = bpy.path.abspath(scene.mpfb_facefit_landmarks_json)
        if not path or not os.path.isfile(path):
            raise ValueError("Choose a valid MediaPipe landmarks JSON file")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    photo = bpy.path.abspath(scene.mpfb_facefit_photo)
    model = bpy.path.abspath(scene.mpfb_facefit_model)
    if not os.path.isfile(photo):
        raise ValueError("Choose an existing frontal face photo")
    if not os.path.isfile(model):
        raise ValueError("Choose the Face Landmarker .task model (see setup documentation)")
    configured_python = scene.mpfb_facefit_python.strip()
    interpreter = bpy.path.abspath(configured_python) if configured_python else ""
    if interpreter:
        if not os.path.isfile(interpreter):
            raise ValueError("Configured external Python executable was not found")
        script = os.path.join(os.path.dirname(__file__), "face_photo_landmarks.py")
        run = subprocess.run([interpreter, script, photo, model], capture_output=True,
                             text=True, timeout=120, check=False)
        if run.returncode:
            raise ValueError("Face landmark detection failed: " + (run.stderr.strip() or run.stdout.strip())[-700:])
        return json.loads(run.stdout)
    from .face_photo_landmarks import detect
    return detect(photo, model)


def _observed_landmarks(result, mapping):
    points = result.get("landmarks_px", [])
    if not isinstance(points, list) or len(points) < 468:
        raise ValueError("Expected MediaPipe Face Landmarker JSON with at least 468 landmark points")
    width, height = float(result.get("width", 0)), float(result.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Image width/height missing from landmark data")
    if len(mapping) < 20:
        raise ValueError("Insufficient face vertex correspondences")
    observed = []
    indices = []
    weights = []
    for pair in mapping:
        landmark_id = int(pair["mp"])
        vertex = int(pair["vertex"])
        p = points[landmark_id]
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise ValueError("Malformed MediaPipe landmark")
        x, y = float(p[0]), float(p[1])
        if not np.isfinite(x) or not np.isfinite(y) or not (-width <= x <= 2 * width and -height <= y <= 2 * height):
            raise ValueError("Invalid detected face landmark coordinates")
        observed.append((x, y))
        indices.append(vertex)
        weights.append(_LANDMARK_WEIGHTS.get(landmark_id, 1.0))
    return np.asarray(observed, dtype=np.float64), indices, np.asarray(weights, dtype=np.float64)


def _get_target_path(section, name):
    root = LocationService.get_mpfb_data("targets")
    for extension in (".target.gz", ".target"):
        path = os.path.join(root, section, name + extension)
        if os.path.isfile(path):
            return path
    return None


def _prepare_face_targets(basemesh, specs):
    """Preflight all files before loading shape keys; track newly created keys."""
    needed = {}
    keys = basemesh.data.shape_keys.key_blocks
    for _, section, positive, negative in specs:
        for name in positive + negative:
            if keys.get(name) is None:
                path = _get_target_path(section, name)
                if not path:
                    raise ValueError("Missing facial target: " + section + "/" + name)
                needed[name] = path
    loaded = []
    try:
        for name, path in needed.items():
            TargetService.load_target(basemesh, path, weight=0.0, name=name)
            loaded.append(name)
    except Exception:
        for name in reversed(loaded):
            key = basemesh.data.shape_keys.key_blocks.get(name)
            if key is not None:
                basemesh.shape_key_remove(key)
        raise
    return loaded


def _face_model(basemesh, indices, specs):
    """Get a rest plane and paired region deltas without evaluating mesh modifiers.

    Mesh vertex positions are combined with *every* current relative shape key,
    so unrelated facial targets and existing phenotype are held constant.
    """
    mesh = basemesh.data
    keys = mesh.shape_keys
    if not keys or not keys.use_relative or not keys.reference_key:
        raise ValueError("Requires an editable MPFB relative-shape-key basemesh")
    if any(vertex < 0 or vertex >= len(mesh.vertices) for vertex in indices):
        raise ValueError("Reference mapping is incompatible with this basemesh topology")

    def projection(coords):
        return np.asarray([(float(c.x), -float(c.z)) for c in coords], dtype=np.float64)

    basis = keys.reference_key
    current = projection([basis.data[i].co for i in indices])
    deltas = {}
    for key in keys.key_blocks:
        if key is basis:
            continue
        relative = key.relative_key or basis
        delta = projection([key.data[i].co - relative.data[i].co for i in indices])
        deltas[key.name] = delta
        if key.value:
            current += delta * float(key.value)

    positive_deltas, negative_deltas, originals = [], [], []
    for _, _, positive, negative in specs:
        p = np.sum([deltas[name] for name in positive], axis=0)
        n = np.sum([deltas[name] for name in negative], axis=0) if negative else np.zeros_like(p)
        # Signed initial is an approximation when the current state is
        # asymmetric; the optimizer establishes a symmetric result.
        plus = float(np.mean([keys.key_blocks[name].value for name in positive]))
        minus = float(np.mean([keys.key_blocks[name].value for name in negative])) if negative else 0.0
        initial = float(np.clip(plus - minus, -1.0, 1.0))
        for name in positive + negative:
            current -= deltas[name] * float(keys.key_blocks[name].value)
        positive_deltas.append(p)
        negative_deltas.append(n)
        originals.append(initial)
    return current, np.asarray(positive_deltas), np.asarray(negative_deltas), np.asarray(originals)


def _apply_face_values(basemesh, specs, fitted, loaded, prune):
    keys = basemesh.data.shape_keys.key_blocks
    for value, (_, _, positive, negative) in zip(fitted, specs):
        plus = max(0.0, float(value))
        minus = max(0.0, -float(value))
        for name in positive:
            keys[name].value = plus
        for name in negative:
            keys[name].value = minus
    if prune:
        for name in loaded:
            key = keys.get(name)
            if key is not None and key.value < 0.0001:
                basemesh.shape_key_remove(key)


class MPFB_OT_Fit_Face_Photo(MpfbOperator):
    """Fit existing facial morphology to landmarks detected from a local face photo"""

    bl_idname = "mpfb.fit_face_photo"
    bl_label = "Fit face from reference"
    bl_options = {'REGISTER', 'UNDO'}

    use_saved_landmarks: BoolProperty(name="Use landmark JSON", default=False)

    @classmethod
    def poll(cls, context):
        if not context.active_object or context.mode != "OBJECT":
            return False
        mesh = ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")
        return bool(mesh and TargetService.has_any_shapekey(mesh))

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")
        if not basemesh or not TargetService.has_any_shapekey(basemesh):
            self.report({'ERROR'}, "Select an unbaked MPFB human")
            return {'CANCELLED'}
        scene = context.scene
        try:
            detected = _image_landmarks(scene, self.use_saved_landmarks)
            mapping_path = os.path.join(LocationService.get_mpfb_data("face_fit"), "mp_to_mpfb.json")
            with open(mapping_path, encoding="utf-8") as handle:
                mapping = json.load(handle)["pairs"]
            observed, indices, weights = _observed_landmarks(detected, mapping)
            specs = _category_specs()
        except (ValueError, FileNotFoundError, OSError, RuntimeError,
                ImportError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            self.report({'ERROR'}, "Face photo fit: " + str(exc)[:220])
            return {'CANCELLED'}
        if any(i >= len(basemesh.data.vertices) or i < 0 for i in indices):
            self.report({'ERROR'}, "MPFB face vertex correspondence is incompatible with this mesh")
            return {'CANCELLED'}
        ObjectService.activate_blender_object(basemesh)
        existing = {key.name: float(key.value) for key in basemesh.data.shape_keys.key_blocks}
        loaded = []
        try:
            loaded = _prepare_face_targets(basemesh, specs)
            rest, pos, neg, initial = _face_model(basemesh, indices, specs)
            result, metrics = fit_landmarks(
                rest, pos, neg, observed, weights, initial=initial,
                influence=float(scene.mpfb_facefit_influence),
                regularization=float(scene.mpfb_facefit_regularization),
            )
            # An aligned residual should not worsen significantly: the user's
            # model may differ from the face in angle or expression.
            if not np.all(np.isfinite(result)):
                raise ValueError("Invalid face fit result")
            prune = MODEL_PROPERTIES.get_value("prune", entity_reference=scene)
            _apply_face_values(basemesh, specs, result, loaded, prune)
        except Exception:
            for name, old in existing.items():
                key = basemesh.data.shape_keys.key_blocks.get(name)
                if key is not None:
                    key.value = old
            for name in reversed(loaded):
                key = basemesh.data.shape_keys.key_blocks.get(name)
                if key is not None:
                    basemesh.shape_key_remove(key)
            raise

        if MODEL_PROPERTIES.get_value("refit", entity_reference=scene):
            HumanService.refit(basemesh, operator=self)
        basemesh["mpfb_facefit_last_rms_before"] = metrics["before"]
        basemesh["mpfb_facefit_last_rms_after"] = metrics["after"]
        self.report({'INFO'}, "Face fit applied: aligned RMS %.4f -> %.4f (normalised face units)" %
                    (metrics["before"], metrics["after"]))
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Fit_Face_Photo)


# Scene settings provide file pickers without requiring any extra add-on.
bpy.types.Scene.mpfb_facefit_photo = StringProperty(
    name="Face photo", description="Frontal reference portrait (kept local)", subtype='FILE_PATH')
bpy.types.Scene.mpfb_facefit_model = StringProperty(
    name="Landmarker model", description="MediaPipe face_landmarker.task model", subtype='FILE_PATH')
bpy.types.Scene.mpfb_facefit_python = StringProperty(
    name="External Python", description="Optional Python executable with MediaPipe installed", subtype='FILE_PATH')
bpy.types.Scene.mpfb_facefit_landmarks_json = StringProperty(
    name="Landmarks JSON", description="Landmarks from the included detection helper", subtype='FILE_PATH')
bpy.types.Scene.mpfb_facefit_influence = FloatProperty(
    name="Fit strength", description="Mix between current character face and fitted shape",
    min=0.0, max=1.0, default=0.85)
bpy.types.Scene.mpfb_facefit_regularization = FloatProperty(
    name="Shape preservation", description="Penalty on large slider changes (higher means subtler)",
    min=0.0, max=0.1, default=0.002, precision=4)


class MPFB_PT_Face_Photo_Fit_Panel(Abstract_Panel):
    """Reference-image-driven facial morphs for an existing editable MPFB human."""

    bl_label = "Face photo fit (experimental)"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "MPFB_PT_Model_Panel"
    bl_category = UiService.get_value("MODELCATEGORY")
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return MPFB_OT_Fit_Face_Photo.poll(context)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        box = layout.box()
        box.label(text="Local photo fitting / frontal neutral face")
        box.prop(scene, "mpfb_facefit_photo")
        # Thumbnail is best-effort: Blender builds Image previews asynchronously
        # and some image formats/version combinations cannot supply an icon.
        photo_path = bpy.path.abspath(scene.mpfb_facefit_photo) if scene.mpfb_facefit_photo else ""
        if photo_path and os.path.isfile(photo_path):
            try:
                image = bpy.data.images.load(photo_path, check_existing=True)
                if image.preview and image.preview.icon_id:
                    box.template_icon(icon_value=image.preview.icon_id, scale=8.0)
            except (OSError, RuntimeError, AttributeError):
                pass
        box.prop(scene, "mpfb_facefit_model")
        box.prop(scene, "mpfb_facefit_python")
        box.prop(scene, "mpfb_facefit_influence")
        box.prop(scene, "mpfb_facefit_regularization")
        button = box.operator("mpfb.fit_face_photo", text="Detect & fit photo", icon='FACESEL')
        button.use_saved_landmarks = False
        extra = layout.box()
        extra.label(text="Alternative: pre-extracted landmarks")
        extra.prop(scene, "mpfb_facefit_landmarks_json")
        button = extra.operator("mpfb.fit_face_photo", text="Fit from landmarks JSON")
        button.use_saved_landmarks = True
        layout.label(text="Adjust facial sliders after fitting")
        layout.label(text="Refit assets/rig if auto-refit is off")


ClassManager.add_class(MPFB_PT_Face_Photo_Fit_Panel)
