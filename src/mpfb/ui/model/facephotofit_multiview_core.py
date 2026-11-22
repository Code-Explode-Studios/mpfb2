"""Joint front/profile landmark fitting of MPFB relative shape-key morphs.

Only NumPy is required. Kept bpy-free for deterministic numerical tests.
Projections are orthographic approximations: useful for profile depth cues,
not calibrated photogrammetry. GPL-3.0-or-later.
"""

import numpy as np


def project_view(coords, view):
    """Project Blender XYZ to a head-on or left/right-facing image plane.

    In the included MPFB basemesh, the nose points toward -Y and Z is up.
    LEFT/RIGHT refer to where the nose points *in the profile photograph*.
    """
    points = np.asarray(coords, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("3D points required for face photo projection")
    if view == "FRONT":
        return np.stack((points[..., 0], -points[..., 2]), axis=-1)
    if view == "LEFT":
        return np.stack((points[..., 1], -points[..., 2]), axis=-1)
    if view == "RIGHT":
        return np.stack((-points[..., 1], -points[..., 2]), axis=-1)
    raise ValueError("Profile direction must be LEFT or RIGHT")


def _weights(weights, count):
    result = np.asarray(weights, dtype=np.float64)
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise ValueError("Invalid landmark weights")
    if np.any(result < 0) or result.sum() <= 0:
        raise ValueError("Landmark weights must be nonnegative with positive sum")
    return result / result.sum()


def _align(src, dst, weights):
    """Weighted similarity alignment without a mirror/reflection."""
    sw = weights / weights.sum()
    src_center = np.sum(src * sw[:, None], axis=0)
    dst_center = np.sum(dst * sw[:, None], axis=0)
    a = src - src_center
    b = dst - dst_center
    u, sigma, vt = np.linalg.svd((b * sw[:, None]).T @ a)
    direction = -1.0 if np.linalg.det(u @ vt) < 0 else 1.0
    fix = np.diag([1.0, direction])
    rotation = u @ fix @ vt
    variance = np.sum(sw * np.sum(a * a, axis=1))
    if variance < 1e-15:
        raise ValueError("Degenerate landmarks: cannot align photo")
    scale = float(np.sum(sigma * np.diag(fix)) / variance)
    return scale * (a @ rotation.T) + dst_center


def _contrib(values, positive, negative):
    result = np.zeros_like(positive[0])
    for idx, value in enumerate(values):
        result += value * positive[idx] if value >= 0 else -value * negative[idx]
    return result


def fit_multiview_landmarks(
    rest, positive, negative, front_observed, front_weights,
    profile_observed, profile_indices, profile_weights,
    profile_direction="LEFT", profile_strength=1.0, initial=None,
    influence=1.0, regularization=0.002, outer_iterations=10,
):
    """Jointly fit signed MPFB shape-key sliders from frontal and profile landmarks.

    rest: N x 3 basemesh points with selected morphs zeroed.
    positive/negative: K x N x 3 signed-branch target displacement arrays.
    front_observed: N x 2 MediaPipe image pixels (front photo).
    profile_observed: M x 2 MediaPipe image pixels (side photo).
    profile_indices: M row indices into the same N vertex correspondences.
    profile_strength: importance of profile error vs front error (0 to 5).

    Each view has a separate similarity-alignment nuisance transform (rotation,
    uniform scale, translation). Both share a single shape-key weight vector.
    Reference images stay in memory; nothing is uploaded or saved by the solver.
    """
    b = np.asarray(rest, dtype=np.float64)
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    front = np.asarray(front_observed, dtype=np.float64)
    profile = np.asarray(profile_observed, dtype=np.float64)
    side_indices = np.asarray(profile_indices, dtype=np.int64)
    if b.ndim != 2 or b.shape[1] != 3 or len(b) < 10:
        raise ValueError("Need at least 10 mapped 3D facial landmarks")
    count = len(b)
    if p.ndim != 3 or p.shape[1:] != b.shape or n.shape != p.shape or not len(p):
        raise ValueError("Invalid three-dimensional morph displacement arrays")
    if front.shape != (count, 2) or profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 7:
        raise ValueError("Invalid front/profile landmark arrays")
    if side_indices.shape != (len(profile),) or np.any(side_indices < 0) or np.any(side_indices >= count):
        raise ValueError("Profile landmarks do not correspond to the front mapping")
    if len(np.unique(side_indices)) != len(side_indices):
        raise ValueError("Duplicate profile vertex correspondences")
    if not all(np.all(np.isfinite(a)) for a in (b, p, n, front, profile)):
        raise ValueError("Non-finite facial landmark or shape-key coordinates")
    if not 0.0 <= influence <= 1.0 or regularization < 0.0 or not 0.0 <= profile_strength <= 5.0:
        raise ValueError("Invalid face fitting controls")
    if not isinstance(outer_iterations, int) or not 1 <= outer_iterations <= 50:
        raise ValueError("Invalid iteration count")
    if profile_direction not in ("LEFT", "RIGHT"):
        raise ValueError("Invalid profile direction")
    fw = _weights(front_weights, count)
    pw = _weights(profile_weights, len(profile))
    initial = np.zeros(len(p)) if initial is None else np.asarray(initial, dtype=np.float64)
    if initial.shape != (len(p),) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial MPFB controls must match the morph count")
    initial = np.clip(initial, -1.0, 1.0)

    # Normalize all 3D displacements using the same frontal face dimension.
    base_front = project_view(b, "FRONT")
    centered = base_front - np.sum(base_front * fw[:, None], axis=0)
    scale = float(np.sqrt(np.sum(fw * np.sum(centered * centered, axis=1))))
    if scale < 1e-8:
        raise ValueError("Degenerate face proportions")
    b, p, n = b / scale, p / scale, n / scale

    views = (
        {"projection": "FRONT", "indices": np.arange(count), "obs": front,
         "weights": fw, "strength": 1.0},
        {"projection": profile_direction, "indices": side_indices, "obs": profile,
         "weights": pw, "strength": profile_strength},
    )
    # Precompute projected displacement fields; this avoids repeating NumPy
    # allocations in coordinate descent while keeping all views consistent.
    for view in views:
        idx = view["indices"]
        view["base"] = project_view(b[idx], view["projection"])
        view["pos"] = project_view(p[:, idx, :], view["projection"])
        view["neg"] = project_view(n[:, idx, :], view["projection"])

    values = initial.copy()
    current = [v["base"] + _contrib(values, v["pos"], v["neg"]) for v in views]
    for _ in range(outer_iterations):
        target = [_align(v["obs"], pts, v["weights"]) for v, pts in zip(views, current)]
        max_delta = 0.0
        for _pass in range(5):
            for i in range(len(values)):
                old = float(values[i])
                base_without = [pts - (old * v["pos"][i] if old >= 0 else -old * v["neg"][i])
                                for v, pts in zip(views, current)]
                residuals = [target[j] - base_without[j] for j in range(len(views))]
                possibilities = []
                for sign in (1.0, -1.0):
                    numer = regularization * (initial[i] * sign)
                    denom = regularization
                    for v, residual in zip(views, residuals):
                        if not v["strength"]:
                            continue
                        delta = v["pos"][i] if sign > 0 else v["neg"][i]
                        numer += v["strength"] * np.sum(v["weights"][:, None] * residual * delta)
                        denom += v["strength"] * np.sum(v["weights"][:, None] * delta * delta)
                    amplitude = float(np.clip(numer / max(denom, 1e-14), 0, 1))
                    val = sign * amplitude
                    loss = regularization * (val - initial[i]) ** 2
                    for v, residual in zip(views, residuals):
                        delta = v["pos"][i] if sign > 0 else v["neg"][i]
                        err = residual - amplitude * delta
                        loss += v["strength"] * np.sum(v["weights"] * np.sum(err * err, axis=1))
                    possibilities.append((loss, val))
                winner = min(possibilities, key=lambda z: z[0])[1]
                max_delta = max(max_delta, abs(winner - old))
                values[i] = winner
                current = [base_without[j] + (winner * v["pos"][i]
                           if winner >= 0 else -winner * v["neg"][i])
                           for j, v in enumerate(views)]
        if max_delta < 1e-5:
            break

    final = np.clip(initial + influence * (values - initial), -1.0, 1.0)

    def error(view, vals):
        points = view["base"] + _contrib(vals, view["pos"], view["neg"])
        mapped = _align(view["obs"], points, view["weights"])
        return float(np.sqrt(np.sum(view["weights"] * np.sum((points - mapped) ** 2, axis=1))))

    front_before, profile_before = [error(v, initial) for v in views]
    front_after, profile_after = [error(v, final) for v in views]
    metrics = {"front_before": front_before, "front_after": front_after,
               "profile_before": profile_before, "profile_after": profile_after,
               "face_scale": scale, "profile_count": len(profile)}
    return final, metrics
