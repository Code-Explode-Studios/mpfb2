"""Fit an editable MPFB face to front or front+side portraits using local MediaPipe landmarks.

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
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty

from ... import ClassManager
from ...services import HumanService, LocationService, LogService, ObjectService, TargetService, UiService
from ..abstractpanel import Abstract_Panel
from ..mpfboperator import MpfbOperator
from .facephotofit_core import fit_landmarks
from .facephotofit_multiview_core import fit_multiview_landmarks
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

# These projection/depth targets are only added to the fitting set when the
# user supplies a separate side reference. The frontal solver is unchanged.
PROFILE_CATEGORIES = (
    "nose-trans-backward-forward", "nose-scale-depth-decr-incr",
    "nose-hump-decr-incr", "nose-greek-decr-incr", "nose-curve-concave-convex",
    "chin-prominent-decr-incr", "chin-prognathism-decr-incr",
    "mouth-trans-backward-forward", "mouth-scale-depth-decr-incr",
    "eyebrows-trans-backward-forward", "forehead-trans-backward-forward",
    "eye-push1-in-out", "head-scale-depth-decr-incr",
)

# Only a near-side subset of 57 calibrated MPFB vertices is used for profile
# matching. Contour landmarks are view-dependent, so downweight side jaw
# locations and skip the hidden/far eye and brow entirely.
_PROFILE_COMMON = (168, 6, 4, 1, 2, 0, 13, 14, 17, 152)
_PROFILE_VISIBLE = {
    "LEFT": (33, 133, 70, 63, 105, 98, 64, 61, 132, 172),
    "RIGHT": (263, 362, 300, 293, 334, 327, 294, 291, 361, 397),
}

# Landmark confidence varies by feature. Frontal eye corners typically produce
# stable correspondences; the jaw silhouette varies with camera viewpoint.
_LANDMARK_WEIGHTS = {
    33: 1.7, 133: 1.7, 263: 1.7, 362: 1.7,
    61: 1.4, 291: 1.4, 1: 1.4, 4: 1.4,
    152: 1.4, 168: 1.3,
}


def _category_specs(include_profile=False):
    """Construct side-aware fitting morphs from authoritative target metadata."""
    path = os.path.join(LocationService.get_mpfb_data("targets"), "target.json")
    with open(path, encoding="utf-8") as handle:
        sections = json.load(handle)
    names = FIT_CATEGORIES + (PROFILE_CATEGORIES if include_profile else ())
    wanted = set(names)
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
    return [(name, *found[name]) for name in names]


def _image_landmarks(scene, use_saved_landmarks, profile=False):
    if use_saved_landmarks:
        value = scene.mpfb_facefit_profile_landmarks_json if profile else scene.mpfb_facefit_landmarks_json
        path = bpy.path.abspath(value) if value else ""
        if not path or not os.path.isfile(path):
            raise ValueError("Choose a valid MediaPipe landmarks JSON file")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    photo_value = scene.mpfb_facefit_profile_photo if profile else scene.mpfb_facefit_photo
    photo = bpy.path.abspath(photo_value) if photo_value else ""
    model_value = scene.mpfb_facefit_model
    if not model_value:
        local_python, local_model = _detector_paths()
        if os.path.isfile(local_model):
            model_value = local_model
    model = bpy.path.abspath(model_value) if model_value else ""
    if not os.path.isfile(photo):
        raise ValueError("Choose an existing " + ("profile" if profile else "front") + " face photo")
    if not os.path.isfile(model):
        raise ValueError("Choose the Face Landmarker .task model (see setup documentation)")
    configured_python = scene.mpfb_facefit_python.strip()
    if not configured_python:
        local_python, local_model = _detector_paths()
        if os.path.isfile(local_python) and os.path.isfile(local_model):
            configured_python = local_python
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


def _observed_landmarks(result, mapping, minimum=20):
    points = result.get("landmarks_px", [])
    if not isinstance(points, list) or len(points) < 468:
        raise ValueError("Expected MediaPipe Face Landmarker JSON with at least 468 landmark points")
    width, height = float(result.get("width", 0)), float(result.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Image width/height missing from landmark data")
    if len(mapping) < minimum:
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



def _profile_mapping(mapping, direction):
    """Select approximately visible profile features from the calibrated map.

    On steep/near-90-degree images MediaPipe may hallucinate the hidden side.
    Restrict the fit to nose, chin, lips, one eye/brow and near-side jaw.
    """
    if direction not in _PROFILE_VISIBLE:
        raise ValueError("Unknown profile orientation")
    selected = set(_PROFILE_COMMON + _PROFILE_VISIBLE[direction])
    result = [pair for pair in mapping if int(pair["mp"]) in selected]
    if len(result) < 12 or len({int(p["mp"]) for p in result}) != len(result):
        raise ValueError("Insufficient unique calibrated profile landmarks")
    return result


def _face_model_3d(basemesh, indices, specs):
    """Extract the current 3D head and signed shape-key deltas.

    Non-fitted keys and all phenotype macro keys remain included in the rest
    mesh, so fitting does not overwrite the body or unrelated facial controls.
    """
    mesh = basemesh.data
    keys = mesh.shape_keys
    if not keys or not keys.use_relative or not keys.reference_key:
        raise ValueError("Requires an editable MPFB relative-shape-key basemesh")
    if any(i < 0 or i >= len(mesh.vertices) for i in indices):
        raise ValueError("Profile landmark mapping is incompatible with this MPFB mesh")

    def coords(items):
        return np.asarray([(float(v.x), float(v.y), float(v.z)) for v in items], dtype=np.float64)

    basis = keys.reference_key
    current = coords([basis.data[i].co for i in indices])
    deltas = {}
    for key in keys.key_blocks:
        if key is basis:
            continue
        relative = key.relative_key or basis
        delta = coords([key.data[i].co - relative.data[i].co for i in indices])
        deltas[key.name] = delta
        if key.value:
            current += delta * float(key.value)
    pos, neg, initial = [], [], []
    for _, _, positive, negative in specs:
        p = np.sum([deltas[n] for n in positive], axis=0)
        n = np.sum([deltas[n] for n in negative], axis=0) if negative else np.zeros_like(p)
        plus = float(np.mean([keys.key_blocks[n].value for n in positive]))
        minus = float(np.mean([keys.key_blocks[n].value for n in negative])) if negative else 0.0
        initial.append(float(np.clip(plus - minus, -1.0, 1.0)))
        for name in positive + negative:
            current -= deltas[name] * float(keys.key_blocks[name].value)
        pos.append(p)
        neg.append(n)
    return current, np.asarray(pos), np.asarray(neg), np.asarray(initial)


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
    use_profile: BoolProperty(name="Also fit side reference", default=False)

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
            profile_data = (_image_landmarks(scene, self.use_saved_landmarks, profile=True)
                            if self.use_profile else None)
            mapping_path = os.path.join(LocationService.get_mpfb_data("face_fit"), "mp_to_mpfb.json")
            with open(mapping_path, encoding="utf-8") as handle:
                mapping = json.load(handle)["pairs"]
            observed, indices, weights = _observed_landmarks(detected, mapping)
            specs = _category_specs(include_profile=self.use_profile)
            if self.use_profile:
                profile_map = _profile_mapping(mapping, scene.mpfb_facefit_profile_direction)
                side_observed, _, side_weights = _observed_landmarks(
                    profile_data, profile_map, minimum=12)
                front_positions = {int(pair["mp"]): j for j, pair in enumerate(mapping)}
                side_positions = [front_positions[int(pair["mp"])] for pair in profile_map]
                # The side view is less reliable near the cheek/jaw contour.
                for j, pair in enumerate(profile_map):
                    if int(pair["mp"]) in (132, 172, 361, 397):
                        side_weights[j] *= 0.5
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
            if self.use_profile:
                rest, pos, neg, initial = _face_model_3d(basemesh, indices, specs)
                result, metrics = fit_multiview_landmarks(
                    rest, pos, neg, observed, weights,
                    side_observed, side_positions, side_weights,
                    profile_direction=scene.mpfb_facefit_profile_direction,
                    profile_strength=float(scene.mpfb_facefit_profile_strength),
                    initial=initial, influence=float(scene.mpfb_facefit_influence),
                    regularization=float(scene.mpfb_facefit_regularization),
                )
            else:
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
        if self.use_profile:
            basemesh["mpfb_facefit_last_rms_before"] = metrics["front_before"]
            basemesh["mpfb_facefit_last_rms_after"] = metrics["front_after"]
            basemesh["mpfb_facefit_profile_rms_before"] = metrics["profile_before"]
            basemesh["mpfb_facefit_profile_rms_after"] = metrics["profile_after"]
            self.report({'INFO'}, "Front %.4f -> %.4f / profile %.4f -> %.4f (normalised)" %
                        (metrics["front_before"], metrics["front_after"],
                         metrics["profile_before"], metrics["profile_after"]))
        else:
            basemesh["mpfb_facefit_last_rms_before"] = metrics["before"]
            basemesh["mpfb_facefit_last_rms_after"] = metrics["after"]
            self.report({'INFO'}, "Face fit applied: aligned RMS %.4f -> %.4f (normalised face units)" %
                        (metrics["before"], metrics["after"]))
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Fit_Face_Photo)


_SETUP_PROCESS = None
_SETUP_LOG = None
_SETUP_DIR = None


def _face_detector_install_dir():
    # Never write inside the installed extension: Blender extensions may be
    # read-only. The isolated dependency environment lives in user config.
    path = bpy.utils.user_resource("CONFIG", path="mpfb_character_studio/face_detector",
                                   create=True)
    if not path:
        raise ValueError("Could not find Blender user config directory")
    return path


def _detector_paths():
    folder = _face_detector_install_dir()
    py = os.path.join(folder, "venv", "Scripts" if os.name == "nt" else "bin",
                      "python.exe" if os.name == "nt" else "python")
    model = os.path.join(folder, "face_landmarker.task")
    return py, model


def _detector_setup_poll():
    """Non-blocking Blender timer: process is external, no bpy use on workers."""
    global _SETUP_PROCESS, _SETUP_LOG, _SETUP_DIR
    if _SETUP_PROCESS is None:
        return None
    if _SETUP_PROCESS.poll() is None:
        return 1.0
    code = _SETUP_PROCESS.returncode
    if _SETUP_LOG is not None:
        _SETUP_LOG.close()
        _SETUP_LOG = None
    py, model = _detector_paths()
    if code == 0 and os.path.isfile(py) and os.path.isfile(model):
        for scene in bpy.data.scenes:
            scene.mpfb_facefit_python = py
            scene.mpfb_facefit_model = model
            scene.mpfb_facefit_setup_status = "Ready: local face detector installed"
        _LOG.info("Installed isolated local MediaPipe detector", py)
    else:
        log_path = os.path.join(_SETUP_DIR, "setup.log") if _SETUP_DIR else "setup.log"
        for scene in bpy.data.scenes:
            scene.mpfb_facefit_setup_status = "Setup failed; see " + log_path
        _LOG.error("Face detector setup failed", (code, log_path))
    _SETUP_PROCESS = None
    _SETUP_DIR = None
    return None


class MPFB_OT_Setup_Face_Detector(MpfbOperator):
    """One-click opt-in install: isolated MediaPipe environment + official task model"""

    bl_idname = "mpfb.setup_face_detector"
    bl_label = "Set up local face detector"
    bl_description = ("Downloads official MediaPipe Python package from PyPI and Face Landmarker "
                      "model to an isolated user directory. Requires internet once.")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return bool(context.scene is not None and _SETUP_PROCESS is None)

    def get_logger(self):
        return _LOG

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def hardened_execute(self, context):
        global _SETUP_PROCESS, _SETUP_LOG, _SETUP_DIR
        if _SETUP_PROCESS is not None:
            self.report({'WARNING'}, "Face detector setup is already running")
            return {'CANCELLED'}
        if not bpy.app.online_access:
            self.report({'ERROR'}, "Enable Online Access in Blender preferences to set up MediaPipe")
            return {'CANCELLED'}
        try:
            directory = _face_detector_install_dir()
            setup_py = os.path.join(os.path.dirname(__file__), "face_detector_setup.py")
            logpath = os.path.join(directory, "setup.log")
            _SETUP_LOG = open(logpath, "w", encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            # Blender itself is not a Python interpreter. Locate the bundled
            # python binary (or a compatible system interpreter), then launch
            # only that interpreter. The worker never touches bpy or Blender UI.
            from .face_detector_setup import _choose_interpreter
            interpreter = _choose_interpreter()
            _SETUP_PROCESS = subprocess.Popen(
                [interpreter, setup_py, "--destination", directory],
                stdout=_SETUP_LOG, stderr=subprocess.STDOUT,
                close_fds=True, creationflags=flags)
            _SETUP_DIR = directory
            context.scene.mpfb_facefit_setup_status = "Installing detector in background..."
            bpy.app.timers.register(_detector_setup_poll, first_interval=1.0)
        except Exception as exc:
            if _SETUP_LOG is not None:
                _SETUP_LOG.close()
                _SETUP_LOG = None
            _SETUP_PROCESS = None
            _SETUP_DIR = None
            self.report({'ERROR'}, "Could not start face detector setup: " + str(exc)[:180])
            return {'CANCELLED'}
        self.report({'INFO'}, "Face detector installing; Blender remains usable. Check panel status.")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Setup_Face_Detector)


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
bpy.types.Scene.mpfb_facefit_setup_status = StringProperty(
    name="Detector setup", description="Status of optional automatic face detector installation",
    default="Not installed")
bpy.types.Scene.mpfb_facefit_profile_photo = StringProperty(
    name="Side photo", description="Near-profile side photo (kept local)", subtype='FILE_PATH')
bpy.types.Scene.mpfb_facefit_profile_landmarks_json = StringProperty(
    name="Side landmarks JSON", description="Pre-extracted profile landmarks", subtype='FILE_PATH')
bpy.types.Scene.mpfb_facefit_profile_direction = EnumProperty(
    name="Nose points", description="Direction the photographed face is pointing in the image",
    items=(("LEFT", "Left", "The nose points toward image left"),
           ("RIGHT", "Right", "The nose points toward image right")),
    default="LEFT")
bpy.types.Scene.mpfb_facefit_profile_strength = FloatProperty(
    name="Side influence", description="How strongly the side-photo profile constrains depth",
    min=0.1, max=3.0, default=1.0, precision=2)


class MPFB_PT_Face_Photo_Fit_Panel(Abstract_Panel):
    """Reference-image-driven facial morphs for an existing editable MPFB human."""

    bl_label = "Face photo fit (experimental)"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "MPFB_PT_Experimental_Features_Panel"
    bl_category = "Experimental Features"
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
        local_python, local_model = _detector_paths()
        if os.path.isfile(local_python) and os.path.isfile(local_model):
            box.label(text="Local detector ready (auto-selected)", icon='CHECKMARK')
        else:
            box.operator("mpfb.setup_face_detector", text="One-time: set up face detector",
                         icon='IMPORT')
            box.label(text="Installs MediaPipe outside Blender / internet once")
        box.label(text=scene.mpfb_facefit_setup_status)
        box.prop(scene, "mpfb_facefit_influence")
        box.prop(scene, "mpfb_facefit_regularization")
        button = box.operator("mpfb.fit_face_photo", text="Detect & fit photo", icon='FACESEL')
        button.use_saved_landmarks = False
        button.use_profile = False
        extra = layout.box()
        extra.label(text="Alternative: pre-extracted landmarks")
        extra.prop(scene, "mpfb_facefit_landmarks_json")
        button = extra.operator("mpfb.fit_face_photo", text="Fit from landmarks JSON")
        button.use_saved_landmarks = True
        button.use_profile = False

        side = layout.box()
        side.label(text="Upgrade: front + near-profile")
        side.prop(scene, "mpfb_facefit_profile_photo")
        side.prop(scene, "mpfb_facefit_profile_direction")
        side.prop(scene, "mpfb_facefit_profile_strength")
        b = side.operator("mpfb.fit_face_photo", text="Detect & fit both photos", icon='FACESEL')
        b.use_saved_landmarks = False
        b.use_profile = True
        side.prop(scene, "mpfb_facefit_profile_landmarks_json")
        b = side.operator("mpfb.fit_face_photo", text="Fit both landmark JSON files")
        b.use_saved_landmarks = True
        b.use_profile = True
        side.label(text="Near-profile (about 60-85 degrees) works best")
        layout.label(text="Adjust facial sliders after fitting")
        layout.label(text="Refit assets/rig if auto-refit is off")


ClassManager.add_class(MPFB_PT_Face_Photo_Fit_Panel)
