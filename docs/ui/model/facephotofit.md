# Face photo fit (experimental)

**Source:** `src/mpfb/ui/model/facephotofit.py`

Fit an **existing, unbaked** MPFB human's facial morph targets toward a **frontal portrait** or, for improved depth constraints, a **frontal portrait plus a near-profile side photograph**. The output remains the original editable MPFB topology; its body, gender/age macro settings, rig and unrelated shape keys are not replaced. This is an art-direction starting point, **not** a photogrammetry scan, facial recognition system, guaranteed identity likeness or full 3D reconstruction.

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

## New: front + side photo joint fitting

The **Model → Face photo fit (experimental) → Upgrade: front + near-profile** section now supports fitting both photos **at once**. It changes a *single shared 3D shape-key vector* and adds 13 depth-sensitive morph categories to the 34 existing facial categories (47 total). Depth controls include nose projection and hump, chin prominence/prognathism, lip projection, forehead/brow depth and head depth.

1. Shoot the **same person** twice with a relaxed, neutral expression: one straight-on view and one **near profile (about 60–85° yaw)** with visible nose, lips, chin and one eye. Extreme 90° side views may not be detected reliably by MediaPipe. Keep hair away from the face.
2. Set **Face photo** (front image), **Side photo** (profile), **Landmarker model** and optionally **External Python** (with MediaPipe installed).
3. Set **Nose points** to **Left** when the nose points left *within the profile photograph*, or **Right** when it points right. This is image orientation, not a claim about anatomical left/right.
4. Adjust **Side influence** (default 1.0; lower values prefer the frontal outline, higher values give the profile more influence). Set **Fit strength** and **Shape preservation** as usual.
5. Click **Detect & fit both photos**. The model stays editable, and both front/profile RMS diagnostics (normalised face units) are stored on the basemesh.
6. Inspect the model from the front and side. Refit attached assets/rig if necessary.

The two photographs may have different resolutions or crop positions: each view gets its own orthographic similarity alignment (rotation, translation and scale). **This is not calibrated multi-view stereo**. It assumes a near-frontal and near-side camera, and cannot reliably reproduce a precise silhouette without better camera and contour calibration.

### Two-image offline / landmarks JSON mode

You can also run the included MediaPipe helper **twice** outside Blender:

```bash
python src/mpfb/ui/model/face_photo_landmarks.py front.jpg face_landmarker.task > front.json
python src/mpfb/ui/model/face_photo_landmarks.py profile.jpg face_landmarker.task > profile.json
```

Select **Landmarks JSON** for the front image and **Side landmarks JSON** for the side image; choose the correct **Nose points** direction, then click **Fit both landmark JSON files**. There are no cloud calls.

### What makes the second image useful

- **Front view:** eye spacing, mouth width, face width, jaw width and symmetric facial placement.
- **Side view:** nose projection, lip forward/back position, chin depth and relative forehead/face profile.
- **Joint solver:** alternates fitting each photo's 2D alignment and a bounded, regularised 3D morph fit. The same MPFB mesh and morph parameters must satisfy both views.

Near-profile MediaPipe eye/cheek/jaw landmarks can be uncertain: the fitter uses a smaller, near-side landmark subset and downweights jaw contour points. Wrong-side selection, mismatched facial expressions, hair occlusion and perspective can reduce likeness. The profile constraint is experimental and is **not** a millimetre-accurate head scan.

## Alternative: landmarks JSON without MediaPipe installed inside Blender

Run this script outside Blender and redirect its JSON output:

```bash
python src/mpfb/ui/model/face_photo_landmarks.py /path/portrait.jpg /path/face_landmarker.task > landmarks.json
```

Then select **Landmarks JSON** and click **Fit from landmarks JSON**. This mode is useful for offline/batch workflows or if Blender cannot import MediaPipe.

The expected JSON contains `width`, `height`, and `landmarks_px`: a list of 468+ `[x_px, y_px, z_px]` coordinates from the Face Landmarker. The frontal mode fits **2D landmarks**. The two-photo mode uses separately detected 2D landmarks, but projects the same editable MPFB **3D mesh** into a front and a side view. It does not treat MediaPipe's monocular z as ground-truth depth.

## How it works

1. Detect 478 photo landmarks with MediaPipe locally.
2. Select 57 landmarks, mapping each to a MakeHuman basemesh vertex via the included calibration file `src/mpfb/data/face_fit/mp_to_mpfb.json`.
3. Load 34 existing MPFB facial morph categories (including head silhouette, chin/jaw, eyes, brows, nose, cheeks and mouth). **For two-image fitting, load 13 extra depth targets (47 total).** For left/right morphs, fit both sides symmetrically.
4. Combine the neutral mesh with any existing phenotype/body morphs and **unselected** face morphs. The single-image solver uses weighted 2D alignment; the multi-view solver additionally projects MPFB's 3D target deltas into the side image and jointly fits both views using bounded, regularized NumPy coordinate descent. No SciPy is required.
5. Write the selected shape key values back to the basemesh. **Fit strength** blends current controls toward the solution; **Shape preservation** regularizes large changes. The operator supports Blender Undo. A before/after aligned RMS diagnostic (in normalized face units) is stored on the basemesh.

The topology correspondence data came from [VGCLUB/facefit-mpfb](https://github.com/VGCLUB/facefit-mpfb) under **GPL-3.0** and is credited in the file. Their calibrations are approximate for MPFB's default mesh: if the base mesh topology changes, regenerate the map rather than reusing outdated vertex indices.

## Expected input

Prefer a single adult face that is straight-on, in soft light, with a neutral expression, visible eyebrows/eyes/mouth, minimal lens distortion, and no obstructions over the jawline. Side views, large smiles, faces partly covered by hair, and unusual camera perspectives reduce accuracy. Consent is recommended when using photos of real people.

## Limitations

- **Not pixel-perfect**: facial landmark matching is not photometric/texture matching. Hair, skin, eye colour, tattoos, wrinkles and lighting are not reconstructed.
- **Depth is still approximate**: a single frontal photo cannot recover it; an added side photo now improves some relative profile constraints, but unknown camera intrinsics, viewpoint yaw, hair/occlusion and silhouette correspondences prevent a metrically accurate reconstruction. Back-of-head and true surface depth are still unresolved.
- MediaPipe's 3D landmarks are approximate screen-relative estimates; this version purposefully does not treat them as true metric head scans.
- Most facial geometry is symmetric here and expression-neutral; it is not designed to replicate a wink, facial scar or asymmetric grimace.
- Existing body macros are preserved; choose the desired phenotype before applying for better results.
- Needs the original MPFB mesh with shape keys, and Blender must be in Object Mode.
- **Experimental**: the new joint solver has passed synthetic NumPy tests for both nose-left and nose-right profiles, including a morph invisible from the front. **No real front/side portrait pair has yet been run end-to-end in Blender** in this development environment. The MPFB test suite includes geometry/registration/integration tests requiring Blender; visually verify results before production.

## Related reading

- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [FaceFit MPFB research prototype](https://github.com/VGCLUB/facefit-mpfb)
- [Blender Shape Keys API](https://docs.blender.org/api/current/bpy.types.ShapeKey.html)
- [MPFB target concepts](https://www.makehumancommunity.org/mpfb/docs/assets/concept_targets.html)
