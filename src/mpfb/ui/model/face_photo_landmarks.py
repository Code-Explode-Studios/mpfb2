"""Standalone/local MediaPipe Face Landmarker helper for MPFB photo fitting.

Can be run using an ordinary Python interpreter with mediapipe installed,
without installing native MediaPipe wheels into Blender's Python environment.

Usage:
  python face_photo_landmarks.py /path/photo.jpg /path/face_landmarker.task

Writes JSON to stdout; exits non-zero on errors. Does not contact any server.
"""

import json
import os
import sys


def detect(image_path, model_path):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("MediaPipe not installed in the configured Python interpreter. Run: python -m pip install mediapipe") from exc

    if not os.path.isfile(image_path):
        raise ValueError("Face photo does not exist: " + image_path)
    if not os.path.isfile(model_path):
        raise ValueError("Face Landmarker .task model does not exist: " + model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        img = mp.Image.create_from_file(image_path)
        result = detector.detect(img)
    if not result.face_landmarks:
        raise ValueError("No face detected. Use a well-lit, unobstructed front or near-profile portrait; very steep side views may fail.")
    landmarks = result.face_landmarks[0]
    return {
        "format": "mpfb-mediapipe-landmarks-v1",
        "image": os.path.abspath(image_path),
        "width": img.width,
        "height": img.height,
        "landmarks_px": [[float(p.x * img.width), float(p.y * img.height),
                           float(p.z * img.width)] for p in landmarks],
    }


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise ValueError("Usage: face_photo_landmarks.py PHOTO_PATH FACE_LANDMARKER_TASK")
        print(json.dumps(detect(sys.argv[1], sys.argv[2])))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
