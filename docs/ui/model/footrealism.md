# Foot shape and skin realism (experimental)

This feature lives under **MPFB → Model → Foot shape & skin realism**.
It adds **editable shape keys** and **localized procedural Blender skin detail**
to an existing, unbaked HM08 MakeHuman character. It preserves the original
basemesh topology and the original foot-target system, and works with all
gender/muscularity settings.

This is an **anatomy-inspired art pass**, not a replacement for a scanned
high-resolution foot or physically simulated skin. It is meant to be a
controllable starting point for close-up character art.

## Quick start

1. Create an MPFB human with shape keys, in its ordinary upright neutral pose.
2. Make sure the character has a node-based **Principled BSDF** body skin
   material in material slot 0. This is the usual MPFB material layout.
3. Expand **Model → Foot shape & skin realism**.
4. Click **Natural**, **Athletic**, or **Mature** to apply the example strengths.
   Alternatively set the six sliders and click **Apply custom foot detail**.
5. Inspect each foot from its side, top and underside, using good lighting.
   For major size changes use MPFB's existing **Model → Feet** controls.
6. Click **Mute foot details** to zero the generated geometry shape keys and
   restore the original material. MPFB's original foot morphs are not reset.

## Controls

| Slider | What it changes |
|---|---|
| Plantar arch | Lifts the **medial** underside of the midfoot |
| Heel cushion | Rounds the heel and adds shallow heel soft tissue |
| Forefoot cushion | Adds subtle tissue around the ball of the foot |
| Toe knuckles | Accentuates the dorsal tissue at the base of the toes |
| Creases / wrinkles | Shader-only fold patterns around the forefoot, toe joints and heel |
| Skin microdetail | Shader-only fine noise to approximate foot skin microstructure |

Preset values (normalized 0–1) are **art-direction choices**, not clinical
measurements. The **Athletic** preset has a slightly higher arch and defined
toes, the **Mature** preset adds more visible folds and heel volume, and the
**Natural** preset is more restrained. Any preset can be adjusted afterwards.
These presets do **not** change the character's sex or body phenotype.

### Geometry implementation

- Four relative, named shape keys: `MPFB Foot Arch`,
  `MPFB Foot Heel Pad`, `MPFB Foot Forefoot Pad`,
  `MPFB Foot Toe Knuckles`.
- Source: `src/mpfb/ui/model/footrealism_core.py`, a pure Python function
  computing deterministic anatomical masks and displacement fields.
- Foot positions are estimated from the lowest 11% of the upright **Basis**
  body's height, separately for X-positive and X-negative feet. The model's
  toes are assumed to face -Y. No third-party assets or models required.
- Only the first **13,380 body vertices** of the shipped HM08 topology
  are eligible; helper geometry, hair, joint cubes and clothes are excluded.
  Ankle influence is gradually faded and existing phenotype/muscularity shape
  keys remain untouched.
- Reapplying recomputes the four keys **from Basis**, preventing cumulative
  geometry deformation. All adjustments remain shape keys, supporting Undo,
  animation and asset refitting through MPFB.

### Procedural skin implementation

- Two **POINT-domain FLOAT attributes**, `mpfb_foot_crease_mask` and
  `mpfb_foot_pore_mask`, are generated to isolate shader detail around the
  feet. No extra subdivision or geometry is added for pores.
- The main body skin material is **copied per character** before modification,
  including any relevant nested shader node groups, preventing edits to
  globally shared skin materials.
- A set of object-space **Wave** and **Noise** textures, each multiplied by
  the local attributes, feed two **Bump** nodes. The existing BSDF Normal map
  is preserved and chained into the new bump nodes. No photographic texture
  is downloaded or uploaded.
- **Toes, nails and unrelated material slots** are left alone: the shader is
  only injected into the primary body skin material in slot 0. A custom body
  shader without any Principled BSDF is left unchanged and Blender reports a
  warning; the geometry still applies.

### Baking and exporting to Unity

The geometry shape keys can be exported as morphs/blend shapes if the chosen
export format and Unity importer retain them. However, **Blender procedural
bump nodes and named shader attributes do not automatically become Unity
normal maps**.

For a game-ready character, inspect the final details in Blender, then bake
the wrinkle + pore shading to the character's *actual* UV layout as tangent
space normal textures, import those to Unity as **Normal Map**, and verify the
material, mesh normal/tangent and quality settings. Keep the feet's fine detail
in a separate mask or texture set if the game requires custom per-region
control. This add-on currently **does not automate baking**.

## Known limitations and future work

- Body height / foot direction assumptions are tied to the shipped HM08 mesh
  in neutral standing pose. The feature explicitly reads the HM08 body vertex
  range and does not claim to work on arbitrary imported meshes.
- Procedural creases are **static**. They move with the skin, but do **not**
  yet change tension/strength dynamically when toe joints bend. For
  photorealism in motion, use pose-driven wrinkles or a dedicated deformation
  rig; [Blender Studio's procedural wrinkle write-up](https://studio.blender.org/blog/procedural-wrinkles/)
  provides useful technical context.
- The base mesh may not have sufficient topology to capture individual toe
  nail beds, tendon attachments, heel fissures or high-frequency silhouette
  changes. Shader bump cannot improve a silhouette from close range.
- Real feet differ in arch height, toe length/order, width and soft-tissue
  proportions. The supplied presets are aesthetic variations, not an
  anatomical dataset.
- Some advanced/nonnodal skin materials won't accept this shader injection.
  It deliberately avoids replacing custom materials or overwriting their
  normal maps.
- **No real Blender-render visual acceptance test has been performed in the
  development environment.** A Blender regression test and pure-Python
  synthetic geometry tests are included, but production results should be
  checked on a generated character under strong side lighting.

## Sources

- [Anatomy and foot joints (NCBI)](https://www.ncbi.nlm.nih.gov/books/NBK536941/)
- [Human foot form and function, shape variation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10777145/)
- [Blender manual: material displacement and bump](https://docs.blender.org/manual/en/5.0/render/materials/components/displacement.html)
- [Blender Studio: Procedural Wrinkles](https://studio.blender.org/blog/procedural-wrinkles/)
- [MPFB target/shape-key architecture](https://www.makehumancommunity.org/mpfb/docs/assets/concept_targets.html)
