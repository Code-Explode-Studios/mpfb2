# Face photo fit (experimental)

**Source:** `src/mpfb/ui/model/facephotofit.py`

Fit an **existing, unbaked** MPFB human's facial morph targets toward a **single frontal face photograph**. The output remains the original editable MPFB topology; its body, gender/age macro settings, rig and unrelated shape keys are not replaced. This is an art-direction starting point, **not** a photogrammetry scan, facial recognition system, guaranteed identity likeness or full 3D reconstruction.

## Setup (once)

Face landmark detection uses [Google MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker). It returns up to 478 landmarks. Its native Python dependencies are **optional** and are deliberately not bundled into the MPFB extension.

1. In your ordinary system Python or a virtual environment (preferably **outside Blender**), install MediaPipe:

   ```bash
   python -m pip install mediapipe
   ```

2. Download the official [Face Landmarker model (face_landmarker.task)](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task). Save it to your computer. The add-on does **not** download this or upload photos.
3. Install/enable the MPFB fork containing this feature, and create a human with editable shape keys.
4. In Blender's 3D viewport open the MPFB sidebar, expand **Model → Face photo fit (experimental)**.
5. Choose your **Face photo**, **Landmarker model**, and your **External Python** executable (the one where MediaPipe was installed).
6. Use **Fit strength** (default 0.85) and **Shape preservation** (default 0.002), then click **Detect & fit photo**. Fine-tune with existing MPFB face sliders afterwards.
7. If the character has clothes, eyes or a rig already, toggle **Auto refit assets** in Model settings or manually run **Refit assets to basemesh** after the fit.

When MediaPipe is already installed in Blender's embedded Python, **External Python** may be left blank. On Windows a virtual environment's interpreter usually lives at `venv\\Scripts\\python.exe`, and on Linux/macOS at `venv/bin/python`.

## Alternative: landmarks JSON without MediaPipe installed inside Blender

Run this script outside Blender and redirect its JSON output:

```bash
python src/mpfb/ui/model/face_photo_landmarks.py /path/portrait.jpg /path/face_landmarker.task > landmarks.json
```

Then select **Landmarks JSON** and click **Fit from landmarks JSON**. This mode is useful for offline/batch workflows or if Blender cannot import MediaPipe.

The expected JSON contains `width`, `height`, and `landmarks_px`: a list of 468+ `[x_px, y_px, z_px]` coordinates from the Face Landmarker. The current fitter uses the **front-view 2D coordinates**, not MediaPipe's approximate monocular depth.

## How it works

1. Detect 478 photo landmarks with MediaPipe locally.
2. Select 57 landmarks, mapping each to a MakeHuman basemesh vertex via the included calibration file `src/mpfb/data/face_fit/mp_to_mpfb.json`.
3. Load 34 existing MPFB facial morph categories (including head silhouette, chin/jaw, eyes, brows, nose, cheeks and mouth). For left/right morphs, fit both sides symmetrically.
4. Combine the neutral mesh with any existing phenotype/body morphs and **unselected** face morphs, then perform alternating 2D weighted alignment and bounded regularized morph fitting using **NumPy** (bundled with Blender, no SciPy required).
5. Write the selected shape key values back to the basemesh. **Fit strength** blends current controls toward the solution; **Shape preservation** regularizes large changes. The operator supports Blender Undo. A before/after aligned RMS diagnostic (in normalized face units) is stored on the basemesh.

The topology correspondence data came from [VGCLUB/facefit-mpfb](https://github.com/VGCLUB/facefit-mpfb) under **GPL-3.0** and is credited in the file. Their calibrations are approximate for MPFB's default mesh: if the base mesh topology changes, regenerate the map rather than reusing outdated vertex indices.

## Expected input

Prefer a single adult face that is straight-on, in soft light, with a neutral expression, visible eyebrows/eyes/mouth, minimal lens distortion, and no obstructions over the jawline. Side views, large smiles, faces partly covered by hair, and unusual camera perspectives reduce accuracy. Consent is recommended when using photos of real people.

## Limitations

- **Not pixel-perfect**: facial landmark matching is not photometric/texture matching. Hair, skin, eye colour, tattoos, wrinkles and lighting are not reconstructed.
- **Not true 3D depth**: a single frontal photo cannot uniquely recover nose projection, back-of-head shape or profile. Additional side-view photos or a dedicated 3D reconstruction model would be needed.
- MediaPipe's 3D landmarks are approximate screen-relative estimates; this version purposefully does not treat them as true metric head scans.
- Most facial geometry is symmetric here and expression-neutral; it is not designed to replicate a wink, facial scar or asymmetric grimace.
- Existing body macros are preserved; choose the desired phenotype before applying for better results.
- Needs the original MPFB mesh with shape keys, and Blender must be in Object Mode.
- **Experimental**: no end-to-end portrait test has yet been run in a Blender installation in this development environment. The MPFB test suite includes geometry/registration tests; inspect visual results before using in production.

## Related reading

- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [FaceFit MPFB research prototype](https://github.com/VGCLUB/facefit-mpfb)
- [Blender Shape Keys API](https://docs.blender.org/api/current/bpy.types.ShapeKey.html)
- [MPFB target concepts](https://www.makehumancommunity.org/mpfb/docs/assets/concept_targets.html)
