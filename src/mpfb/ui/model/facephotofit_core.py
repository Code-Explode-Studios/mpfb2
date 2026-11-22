"""Photo landmark fitting maths for MPFB.

Only numpy is required (ships with Blender). This module deliberately has no
bpy dependency, so its numerical logic can be tested outside Blender.

Copyright (C) 2026 MPFB contributors. GPL-3.0-or-later.
"""

import numpy as np


def _normalised_weights(weights, count):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (count,) or not np.all(np.isfinite(weights)):
        raise ValueError("Invalid landmark weights")
    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("Landmark weights must be nonnegative with positive sum")
    return weights / np.sum(weights)


def similarity_align(source, destination, weights):
    """Map 2D source points into destination using weighted scale/rotation/translation.

    A proper rotation, not a reflection, is required so asymmetric features
    are not accidentally mirrored.
    """
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Expected equal N x 2 point sets")
    w = _normalised_weights(weights, len(source))
    sm = np.sum(source * w[:, None], axis=0)
    dm = np.sum(destination * w[:, None], axis=0)
    s = source - sm
    d = destination - dm
    covariance = (d * w[:, None]).T @ s
    u, values, vt = np.linalg.svd(covariance)
    reflection = np.linalg.det(u @ vt)
    fix = np.diag([1.0, -1.0 if reflection < 0 else 1.0])
    rotation = u @ fix @ vt
    scale = np.sum(values * np.diag(fix)) / max(np.sum(w * np.sum(s * s, axis=1)), 1e-15)
    return scale * (s @ rotation.T) + dm


def _contribution(values, positive, negative):
    """The signed negative slider uses the negative morph with positive weight."""
    result = np.zeros_like(positive[0])
    for i, value in enumerate(values):
        if value >= 0:
            result += value * positive[i]
        else:
            result -= value * negative[i]
    return result


def fit_landmarks(
    rest, positive, negative, observed, weights, initial=None,
    influence=1.0, regularization=0.002, outer_iterations=7,
):
    """Fit paired MPFB morphs to 2D reference points.

    rest: N x 2 model points with the fitting morphs at zero.
    positive/negative: K x N x 2 displacement arrays, all in model units.
    observed: N x 2 pixel landmarks with y growing downward.
    Initial values are signed K-vector. A regulariser limits change from the
    initial face. Bounds are [-1,1] and positive/negative are exclusive.
    This is alternating weighted Procrustes and bounded coordinate descent.
    """
    rest = np.asarray(rest, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 2 or len(rest) < 5:
        raise ValueError("Need at least five 2D mesh landmarks")
    count = len(rest)
    if observed.shape != rest.shape:
        raise ValueError("Landmark-to-mesh correspondence count mismatch")
    if positive.ndim != 3 or positive.shape[1:] != rest.shape or negative.shape != positive.shape:
        raise ValueError("Invalid morph displacement matrix")
    if not all(np.all(np.isfinite(a)) for a in (rest, observed, positive, negative)):
        raise ValueError("Landmark coordinates contain non-finite numbers")
    if not (0.0 <= influence <= 1.0) or regularization < 0 or outer_iterations < 1:
        raise ValueError("Invalid fitting options")
    count_params = len(positive)
    if count_params < 1:
        raise ValueError("No facial morphs to fit")
    w = _normalised_weights(weights, count)
    initial = np.zeros(count_params) if initial is None else np.asarray(initial, dtype=np.float64)
    if initial.shape != (count_params,) or not np.all(np.isfinite(initial)):
        raise ValueError("Invalid initial slider vector")
    initial = np.clip(initial, -1.0, 1.0)

    # Scale all vectors to a reference face dimension, making regularisation
    # stable across Blender unit scales and arbitrarily sized photos.
    centred = rest - np.average(rest, axis=0, weights=w)
    face_scale = np.sqrt(np.average(np.sum(centred ** 2, axis=1), weights=w))
    if face_scale < 1e-8:
        raise ValueError("Degenerate MPFB head landmarks")
    b = rest / face_scale
    p = positive / face_scale
    n = negative / face_scale
    current = initial.copy()
    fitted = b + _contribution(current, p, n)

    for _ in range(outer_iterations):
        # Align photo into the shape's front plane, then solve morph amplitudes.
        target = similarity_align(observed, fitted, w)

        for _pass in range(4):
            for i in range(count_params):
                old = current[i]
                existing = (old * p[i]) if old >= 0 else (-old * n[i])
                base_without = fitted - existing
                residual = target - base_without
                candidates = []
                # Solve each half of the piecewise linear target with ridge
                # penalty towards the original value. Never enable both sides.
                for direction in (1.0, -1.0):
                    delta = p[i] if direction > 0 else n[i]
                    denom = np.sum(w[:, None] * delta * delta) + regularization
                    optimum = (np.sum(w[:, None] * residual * delta)
                               + regularization * (initial[i] * direction)) / max(denom, 1e-14)
                    amplitude = float(np.clip(optimum, 0.0, 1.0))
                    candidate = direction * amplitude
                    residual_after = residual - amplitude * delta
                    loss = (np.sum(w[:, None] * residual_after * residual_after)
                            + regularization * (candidate - initial[i]) ** 2)
                    candidates.append((loss, candidate))
                candidate = min(candidates, key=lambda item: item[0])[1]
                current[i] = candidate
                fitted = base_without + (candidate * p[i] if candidate >= 0 else -candidate * n[i])
        if not np.all(np.isfinite(current)):
            raise ValueError("Fit produced invalid values")

    result = initial + influence * (current - initial)
    result = np.clip(result, -1.0, 1.0)
    before = b + _contribution(initial, p, n)
    after = b + _contribution(result, p, n)

    def error(points):
        aligned = similarity_align(observed, points, w)
        return float(np.sqrt(np.sum(w * np.sum((points - aligned) ** 2, axis=1))))

    return result, {"before": error(before), "after": error(after), "face_scale": float(face_scale)}
