"""Non-destructive, anatomy-inspired MPFB HM08 foot detail fields.

The fields are computed from the Basis mesh (in local coordinates) so they
remain editable relative shape keys and per-point shader masks. No Blender API
is imported here; Blender integration tests and standalone synthetic tests
can both exercise these functions.

The model is an artistic prior, not scanned anatomy: foot skin/arches vary
greatly and should be checked in a front, side and sole view.

GPL-3.0-or-later. Copyright (C) 2026 MPFB contributors.
"""

import math


FOOT_KEY_NAMES = {
    "arch": "MPFB Foot Arch",
    "heel": "MPFB Foot Heel Pad",
    "ball": "MPFB Foot Forefoot Pad",
    "toes": "MPFB Foot Toe Knuckles",
}
CREASE_ATTRIBUTE = "mpfb_foot_crease_mask"
PORE_ATTRIBUTE = "mpfb_foot_pore_mask"


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, value))


def _gaussian(t, center, spread):
    return math.exp(-0.5 * ((t - center) / spread) ** 2)


def compute_foot_fields(vertices, body_count):
    """Return four XYZ displacement fields and crease/pore float masks.

    Parameters:
      vertices: iterable of (X,Y,Z) positions from HM08 Basis
      body_count: first N true body vertices; HM08 contains additional rig,
        clothes/helper cubes that MUST NOT be distorted by skin detail.

    The character is assumed to stand upright in local mesh coordinates:
    the toes point mostly -Y, Z is up and X separates left and right foot.
    The algorithm derives per-foot bounding boxes from the lowest 11% of
    the body's Z range. A smooth transition to the ankle prevents seam lines.
    """
    positions = [tuple(float(v) for v in p) for p in vertices]
    count = len(positions)
    if count < 100 or not 100 <= body_count <= count:
        raise ValueError("Expected a MakeHuman basemesh and valid body vertex count")
    if any(len(v) != 3 or not all(math.isfinite(a) for a in v) for v in positions):
        raise ValueError("Invalid MakeHuman vertex positions")

    body = positions[:body_count]
    zmin = min(v[2] for v in body)
    zmax = max(v[2] for v in body)
    height = zmax - zmin
    if height < 0.5:
        raise ValueError("Character must use a standing human scale in local coordinates")
    low_limit = zmin + 0.11 * height
    influence_limit = zmin + 0.18 * height
    bounds = {}
    for sign in (-1, 1):
        lower = [(x, y, z) for x, y, z in body
                 if z <= low_limit and (x < 0 if sign < 0 else x >= 0)]
        if len(lower) < 25:
            raise ValueError("Cannot locate both feet on this character")
        xvalues = [abs(x) for x, _, _ in lower]
        ys = [y for _, y, _ in lower]
        foot_length = max(ys) - min(ys)
        foot_width = max(xvalues) - min(xvalues)
        if foot_length < 0.025 * height or foot_width < 0.008 * height:
            raise ValueError("Detected feet have implausible proportions")
        bounds[sign] = (min(xvalues), max(xvalues), min(ys), max(ys),
                        min(z for _, _, z in lower), foot_length, foot_width)

    delta = {name: [(0.0, 0.0, 0.0)] * count for name in FOOT_KEY_NAMES}
    crease = [0.0] * count
    pores = [0.0] * count
    changed = 0

    for i, (x, y, z) in enumerate(body):
        if z > influence_limit:
            continue
        sign = -1 if x < 0 else 1
        xmin, xmax, ymin, ymax, floor, length, width = bounds[sign]
        # Expand bounds slightly to follow the true foot silhouette rather
        # than turn individual vertices on/off at its bounding rectangle.
        if y < ymin - 0.12 * length or y > ymax + 0.12 * length:
            continue
        rel_height = z - floor
        if rel_height < -0.015 * height:
            continue
        t = _clamp((ymax - y) / max(length, 1e-6))
        # The medial (inner) side points toward the body's centre line.
        u = _clamp((abs(x) - xmin) / max(width, 1e-6))
        medial = _gaussian(u, 0.23, 0.32)
        side = _clamp((abs(x) - xmin) / max(width, 1e-6))
        sole = _clamp(1.0 - rel_height / max(0.16 * length, 1e-6))
        dorsal = _gaussian(rel_height / length, 0.17, 0.11)
        ankle_fade = _clamp((influence_limit - z) / (0.065 * height))
        # Soft outline falloff at the forefoot and heel boundaries.
        edge = _clamp((y - ymin + 0.12 * length) / (0.12 * length))
        edge *= _clamp((ymax + 0.12 * length - y) / (0.12 * length))
        foot_mask = ankle_fade * edge
        if foot_mask < 1e-5:
            continue

        heel = _gaussian(t, 0.14, 0.15)
        arch = _gaussian(t, 0.46, 0.20) * medial
        ball = _gaussian(t, 0.75, 0.12)
        knuckles = _gaussian(t, 0.84, 0.075)

        # Inner plantar arch lifts the sole without compressing the ankle.
        arch_dz = 0.045 * length * arch * sole * foot_mask
        delta["arch"][i] = (0.0, 0.0, arch_dz)

        # Heel fat pad: outward, round heel silhouette with very shallow
        # contact padding, no sharp offset at the midline.
        heel_amount = heel * foot_mask * _clamp(1.0 - rel_height / (0.32 * length))
        sideways = _clamp((abs(x) - 0.5 * (xmin + xmax)) / max(width, 1e-6), -0.5, 0.5)
        delta["heel"][i] = (
            sign * 0.023 * length * heel_amount * sideways,
            0.014 * length * heel_amount,
            -0.003 * length * heel_amount * sole,
        )

        # Forefoot cushion, keeping the change comfortably under ~3mm
        # on a typical human foot; sole/ball silhouette only.
        ball_amount = ball * foot_mask * sole
        delta["ball"][i] = (
            sign * 0.008 * length * ball_amount * (u - 0.5),
            0.0,
            -0.010 * length * ball_amount,
        )
        # Subtle dorsal soft tissue over toe bases, not across the sole.
        knuckle_amount = knuckles * dorsal * foot_mask
        delta["toes"][i] = (0.0, 0.0, 0.017 * length * knuckle_amount)

        # Creases favour metatarsal / toe flexion folds and heel skin;
        # fine pores are weaker on the dorsal (thin-skin) surface.
        crease[i] = _clamp(
            foot_mask * max(
                sole * (0.65 * _gaussian(t, 0.76, 0.12)
                        + 0.40 * _gaussian(t, 0.19, 0.15)
                        + 0.35 * _gaussian(t, 0.91, 0.10)),
                dorsal * 0.55 * _gaussian(t, 0.85, 0.13),
            )
        )
        pores[i] = _clamp(foot_mask * (0.35 + 0.65 * sole))
        changed += 1

    if changed < 50:
        raise ValueError("Could not identify enough foot geometry for detail")
    return {"deltas": delta, "crease": crease, "pores": pores,
            "affected_vertices": changed, "body_count": body_count}
