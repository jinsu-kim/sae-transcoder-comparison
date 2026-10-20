"""Rotate selected encoder weights with angle- and norm-matched random controls."""

import numpy as np

from common import (edited_scores, load_arrays, parser, score_rows,
                    validate_geometry, write_csv)


def split(beta, left, k):
    top = left[:, :k] @ (left[:, :k].T @ beta)
    return top, beta - top


def reference_fraction(lam, left, k, seed, draws=1024):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((draws, len(lam)))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rbeta = u * np.sqrt(lam)[None, :]
    projected = rbeta @ left[:, :k]
    return float(np.mean(np.sum(projected**2, axis=1) /
                         np.sum(rbeta**2, axis=1)))


def weights_delta(beta, edited, lam, q):
    return (q / np.sqrt(lam)[None, :]) @ (edited - beta)


def angles(r, g):
    r, g = np.asarray(r), np.asarray(g)
    if not (np.isfinite(r).all() and np.isfinite(g).all() and
            np.all((r > 1e-12) & (r < 1-1e-12)) and np.all((g > 0) & (g < 1))):
        raise ValueError("invalid top/rest share; no cell may be silently dropped")
    return np.arccos(np.sqrt(r)), np.arccos(np.sqrt(g))


def matched_edit(beta, left, reference, dose, k=16):
    top, rest = split(beta, left, k)
    norm2 = np.sum(beta**2, axis=0)
    if not np.all(norm2 > 0):
        raise ValueError("zero encoder coefficient")
    r = np.sum(top**2, axis=0) / norm2
    eta, goal = angles(r, reference)
    dose = np.asarray(dose)
    if not np.isfinite(dose).all() or np.any(dose < 0) or np.any(dose > abs(goal-eta)+1e-12):
        raise ValueError("angle overshoots reference")
    new_eta = eta + np.sign(goal-eta)*dose
    changed = top*(np.cos(new_eta)/np.sqrt(r)) + rest*(np.sin(new_eta)/np.sqrt(1-r))
    changed[:, dose == 0] = beta[:, dose == 0]
    return changed, r, np.cos(new_eta)**2


def control(beta, ids, dose, seed):
    """Reuse the random tangent for repeated feature IDs, with per-cluster angles."""
    out = np.empty_like(beta)
    for c, fid in enumerate(ids):
        b = beta[:, c]
        norm = np.linalg.norm(b)
        noise = np.random.default_rng(seed+int(fid)*100000).normal(size=len(b))
        noise -= b*(noise@b)/(norm*norm)
        noise /= np.linalg.norm(noise)
        out[:, c] = np.cos(dose[c])*b + np.sin(dose[c])*norm*noise
    return out


def geometry_audit(beta, changed, dose):
    n2 = np.sum(beta**2, axis=0)
    cosine = np.sum(beta*changed, axis=0)/n2
    errors = dict(norm=float(np.max(abs(np.sum(changed**2, axis=0)/n2-1))),
                  cosine=float(np.max(abs(cosine-np.cos(dose)))),
                  normalized_mse=float(np.max(abs(np.sum((changed-beta)**2, axis=0)/n2
                                                  -2*(1-np.cos(dose))))))
    if max(errors.values()) > 1e-8:
        raise ValueError(f"geometry failed: {errors}")
    return errors


GEOMETRY_KEYS = ("beta", "eigenvalues", "eigenvectors", "mean", "native",
                 "pre", "thresholds", "labels", "feature_ids", "angle")


def main():
    command = parser(__doc__)
    command.add_argument("--k", type=int, default=16)
    command.add_argument("--controls", type=int, default=8)
    command.add_argument("--seed", type=int, default=14801)
    args = command.parse_args()
    data = load_arrays(args.input, (*GEOMETRY_KEYS, "basis", "reference"))
    validate_geometry(data)
    beta, left, dose = data["beta"], data["basis"], data["angle"]
    if not 0 < args.k <= left.shape[1] or left.shape[0] != beta.shape[0]:
        raise ValueError("k must index the supplied orthonormal basis")
    np.testing.assert_allclose(left[:, :args.k].T @ left[:, :args.k], np.eye(args.k), atol=1e-8)
    if dose.shape != (beta.shape[1],) or data["feature_ids"].shape != dose.shape:
        raise ValueError("angle and feature_ids must have one entry per score column")
    if args.controls < 1:
        raise ValueError("At least one matched random control is required")
    changed, _, _ = matched_edit(beta, left, data["reference"], dose, args.k)
    geometry_audit(beta, changed, dose)
    rows = score_rows("targeted", edited_scores(data, changed), data)
    for index in range(args.controls):
        random = control(beta, data["feature_ids"], dose, args.seed + index)
        geometry_audit(beta, random, dose)
        rows.extend(score_rows(f"random_{index}", edited_scores(data, random), data))
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()

