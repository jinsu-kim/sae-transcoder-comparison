"""Remove or rescale leading input-PC components of selected encoder weights."""

import numpy as np

from common import (edited_scores, load_arrays, parser, score_rows,
                    validate_geometry, write_csv)


def basis(eigenvalues, k=16):
    indices = np.argsort(eigenvalues)[::-1][:k]
    selected = np.zeros((len(eigenvalues), k))
    selected[indices, np.arange(k)] = 1
    return selected


def main():
    from rotate_encoder import GEOMETRY_KEYS

    command = parser(__doc__)
    command.add_argument("--k", type=int, default=16)
    command.add_argument("--controls", type=int, default=8)
    command.add_argument("--seed", type=int, default=15901)
    args = command.parse_args()
    data = load_arrays(args.input, GEOMETRY_KEYS)
    validate_geometry(data)
    if not 0 < args.k < len(data["eigenvalues"]):
        raise ValueError("k must leave both leading and remaining components")
    if args.controls < 1:
        raise ValueError("At least one matched random control is required")
    columns = data["beta"].shape[1]
    if data["angle"].shape != (columns,) or data["feature_ids"].shape != (columns,):
        raise ValueError("angle and feature_ids must match the score columns")
    _, _, conditions, _ = components(data["beta"], data["eigenvalues"], data["angle"], args.k)
    rows = []
    for name, coefficients in conditions.items():
        rows.extend(score_rows(name, edited_scores(data, coefficients), data))
        if name == "neither":
            continue
        for index in range(args.controls):
            random = matched_random(data["beta"], coefficients, data["feature_ids"], args.seed + index)
            rows.extend(score_rows(f"{name}_random_{index}", edited_scores(data, random), data))
    write_csv(args.output, rows)


def components(beta, eigenvalues, angle, k=16):
    pc = basis(eigenvalues, k)
    top = pc @ (pc.T @ beta)
    rest = beta - top
    top2 = np.sum(top * top, axis=0)
    rest2 = np.sum(rest * rest, axis=0)
    total = top2 + rest2
    assert np.all(top2 > 0) and np.all(rest2 > 0)
    eta = np.arctan2(np.sqrt(rest2), np.sqrt(top2))
    assert np.all((angle >= 0) & (eta + angle < np.pi / 2))
    alpha = np.cos(eta + angle) / np.cos(eta)
    gamma = np.sin(eta + angle) / np.sin(eta)
    scale = np.sqrt(total / rest2)
    conditions = {
        'top_only': top,
        'rest_only': rest,
        'neither': np.zeros_like(beta),
        'rest_scaled': rest * scale,
        'shrink_only': top * alpha + rest,
        'amplify_only': top + rest * gamma,
        'combined': top * alpha + rest * gamma,
    }
    np.testing.assert_allclose(np.sum(conditions['combined'] ** 2, axis=0), total, atol=1e-8)
    assert np.all(alpha <= 1 + 1e-12) and np.all(gamma >= 1 - 1e-12)
    detail = dict(alpha=alpha, gamma=gamma, scale=scale, top_fraction=top2 / total, angle=angle)
    return top, rest, conditions, detail


def matched_random(beta, target, ids, seed):
    """Match the target's retained norm and angle to the original encoder row."""
    out = np.empty_like(beta)
    for j, feature_id in enumerate(ids):
        original = beta[:, j]
        norm = np.linalg.norm(original)
        target_norm = np.linalg.norm(target[:, j])
        assert norm > 0 and target_norm > 0
        cosine = np.clip(original @ target[:, j] / (norm * target_norm), -1, 1)
        noise = np.random.default_rng(seed + int(feature_id) * 100000).normal(size=len(original))
        noise -= original * (noise @ original) / (norm * norm)
        noise /= np.linalg.norm(noise)
        out[:, j] = target_norm * (cosine * original / norm
                                  + np.sqrt(max(0, 1 - cosine * cosine)) * noise)
    np.testing.assert_allclose(np.linalg.norm(out, axis=0), np.linalg.norm(target, axis=0), atol=1e-8)
    numerator = np.sum(beta * out, axis=0)
    denominator = np.linalg.norm(beta, axis=0) * np.linalg.norm(out, axis=0)
    reference = np.sum(beta * target, axis=0) / (
        np.linalg.norm(beta, axis=0) * np.linalg.norm(target, axis=0))
    np.testing.assert_allclose(numerator / denominator, reference, atol=1e-8, rtol=0)
    return out


if __name__ == "__main__":
    main()

