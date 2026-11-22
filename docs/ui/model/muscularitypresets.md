# Muscularity presets

**Code:** `src/mpfb/ui/model/muscularitypresets.py`

## Use

1. Create a human using **MPFB → New human → From scratch** (or select an existing MPFB character with unbaked shape keys).
2. In the viewport **MPFB → Model**, expand **Muscularity presets**.
3. Select a **Female** or **Male** button under **Lean / defined**, **Athletic**, **Bodybuilder**, or **Power / mass**.
4. Fine-tune with the existing **phenotype**, **arms**, **torso**, and **legs** sliders.
5. If clothing, proxies or an armature are fitted, enable **Auto refit assets** before applying, or click **Refit assets to basemesh** afterwards.

The 8 combinations use the same normalized strength values for either phenotype. They change exactly three macro values (`gender`, `muscle`, `weight`) and seven built-in muscle regions: shoulders, upper arms, forearms, chest, back, thighs and calves. They set both left and right regions consistently. Age, height, ancestry, breast settings and other individual shape keys remain unchanged.

All preset values are visual art-direction starting points, **not** anatomical measurements or estimates of actual body-fat percentage. `weight` is the MPFB *body-mass morph*, not a body-fat percent. Strong macro changes plus detail targets can lead to exaggerated appearances, so inspect the mesh and adjust sliders where needed.

## Technical behavior

- Values live in `MUSCULARITY_LEVELS` and `MUSCLE_REGIONS`, which makes additional physiques easy to add.
- Existing MakeHuman `.target.gz` or `.target` files are loaded only when not already present as shape keys.
- Opposing `*-decr` muscle shape keys are zeroed so a later preset never fights a previous one. Other detail targets are untouched.
- Macro targets are regenerated **once** after updating gender/muscle/weight (the normal scene-slider path would regenerate them three times).
- The operator is undoable and honors the Model panel's **Prune** and **Auto refit assets** settings. It preflights missing muscle target files before changing the human.
- Baked meshes are not supported: a retained MPFB basemesh and its shape keys are required.

**Not included:** bone-driven muscle bulging/contraction, real muscle meshes, volume-preserving tissue simulation, or automatic Unity runtime muscle dynamics. These would require dedicated rig-space deformation, corrective shape keys, or separate muscle geometry.
