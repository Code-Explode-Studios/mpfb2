# MPFB Character Studio — one ZIP for Blender 4.2+

This is the full MakeHuman/MPFB Blender extension with **all three** experimental tools included in the **same extension and sidebar**:

- **Model → Muscularity presets** — eight male/female physique presets.
- **Model → Face photo fit** — one or two photo/landmark references, including nose/chin depth.
- **Model → Foot shape & skin realism** — arches, heel/forefoot pads, toe shape and masked pore/wrinkle shader. Creates a neutral skin preview automatically for a brand-new untextured human, then switches to Material Preview to display the wrinkles.

## Install: ONE archive, not the GitHub repo ZIP

1. Download **MPFB-Character-Studio.zip** from the project CI build/release.
2. **DO NOT EXTRACT.** In Blender 4.2 or newer, drag that ZIP into Blender and confirm, or open **Edit → Preferences → Get Extensions → Install from Disk**, and choose the ZIP. Enable **MPFB Character Studio**.
3. Press **N** in the 3D Viewport, open the **MPFB** tab. Start Here → Create Human, then expand **Model**.
4. You have all three feature panels. No other add-on is required.

The ZIP contains the whole original MPFB extension (base mesh, rig presets, morph assets, textures and shaders), not just the three new scripts. The package uses the original extension ID `mpfb` and **updates/replaces** the existing MPFB 2.x; uninstall/deactivate any duplicate older MPFB version if Blender reports a duplicate.

### Face photos: one-time local detector setup

Face image detection uses Google's MediaPipe, which is a substantial platform-specific native dependency. It **cannot** be packaged into one portable universal ZIP on all Windows, macOS and Linux Blender/Python combinations without shipping per-platform native wheels.

For actual JPEG/PNG photo fitting, click **Set up local face detector** in **Model → Face photo fit**. The setup uses Python in a dedicated folder outside your extension and downloads the official Face Landmarker model + MediaPipe from their sources. This **requires internet once**, no manual terminal/pip commands, and **does not upload your photos**. Afterwards, pick your photo(s) and click **Detect & fit photo** or **Detect & fit both photos**.

If auto setup is unsupported by your Blender/Python distribution (or offline), the UI retains its **External Python** and **Landmarks JSON** input paths. The rest of MPFB, muscles, feet and face-shaping sliders need **no additional installation**.

### Default human / realistic foot quick test

Create a default human, open Model → Foot shape & skin realism and click Natural or Athletic. A missing skin pack is **not** an error: the extension generates a temporary neutral node-based skin material. Blender automatically switches to Material Preview (you can disable this with the Auto-show Material Preview checkbox). Zoom into the feet to see the shader wrinkles; they do **not** appear in Solid View. Use Mute foot details to restore the original no-material state or your authored skin.

### Unity/export

The geometry is still MPFB shape keys. Foot skin bump is **Blender procedural nodes** and needs **baking to tangent-space normal maps** before Unity export; it is not automatically included as a Unity-compatible material.

### Status

This is experimental development work, not a tested retail release. The ZIP builder validates files and syntax, but actual Blender use, face detection setup on each operating system and final character renders should still be tested. Keep the GitHub pull request as a draft until validated.

### Build for developers

From the repository root:

```bash
python tools/build_onedrop.py
```

This creates `dist/MPFB-Character-Studio.zip` with the `blender_manifest.toml` at its archive root. To verify source without generating a ZIP use `--check`. No other Python dependencies are required for the build itself.
