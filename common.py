"""Array validation, JumpReLU scores, and result files shared by the analyses."""

import argparse
import csv
from pathlib import Path

import numpy as np


def parser(description):
    command = argparse.ArgumentParser(description=description)
    command.add_argument("input", type=Path, help="Prepared NumPy .npz file")
    command.add_argument("--output", type=Path, required=True)
    return command


def load_arrays(path, required):
    with np.load(path, allow_pickle=False) as packet:
        missing = set(required) - set(packet.files)
        if missing:
            raise ValueError(f"Missing arrays in {path}: {', '.join(sorted(missing))}")
        return {name: packet[name] for name in packet.files}


def write_csv(path, rows):
    if not rows:
        raise ValueError("No results to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def finite(name, values):
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")


def validate_scores(pre, thresholds):
    if pre.ndim != 2 or thresholds.shape != (pre.shape[1],):
        raise ValueError("Expected pre [tokens, features] and thresholds [features]")
    finite("pre", pre)
    finite("thresholds", thresholds)
    if np.any(thresholds < 0):
        raise ValueError("JumpReLU thresholds must be nonnegative")


def jumprelu(pre, thresholds):
    return np.where(pre > thresholds, pre, 0.)


def validate_labels(labels, rows, columns):
    if labels.shape != (rows,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected one integer cluster label per token")
    if np.any((labels < 0) | (labels >= columns)):
        raise ValueError("Cluster labels must index the score columns")
    sizes = np.bincount(labels, minlength=columns)
    if np.any((sizes == 0) | (sizes == rows)):
        raise ValueError("Each cluster must have target and nontarget tokens")


def binary_labels(labels, shape):
    if labels.shape != shape or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Expected binary labels with shape {shape}")
    return labels.astype(bool)


def auroc(positive, negative):
    """Probability that a positive score wins, counting a tie as half a win."""
    if not len(positive) or not len(negative):
        return float("nan")
    negative = np.sort(negative)
    left = np.searchsorted(negative, positive, side="left")
    right = np.searchsorted(negative, positive, side="right")
    return float(np.sum(left + 0.5 * (right - left), dtype=np.float64)
                 / (len(positive) * len(negative)))


def cluster_aurocs(scores, labels):
    return np.array([auroc(scores[labels == j, j], scores[labels != j, j])
                     for j in range(scores.shape[1])])


def validate_geometry(data):
    beta, eigenvalues, eigenvectors = (data[name] for name in
                                      ("beta", "eigenvalues", "eigenvectors"))
    if beta.ndim != 2 or eigenvalues.shape != (beta.shape[0],):
        raise ValueError("beta and eigenvalues have incompatible dimensions")
    if eigenvectors.ndim != 2 or eigenvectors.shape[1] != len(eigenvalues):
        raise ValueError("eigenvectors must have one column per retained direction")
    for name in ("beta", "eigenvalues", "eigenvectors", "mean", "native"):
        finite(name, data[name])
    if np.any(eigenvalues <= 0):
        raise ValueError("Whitening requires positive retained eigenvalues")
    width = eigenvectors.shape[0]
    if data["mean"].shape != (width,) or data["native"].shape != (len(data["pre"]), width):
        raise ValueError("native inputs and selection mean must use the original input space")
    validate_scores(data["pre"], data["thresholds"])
    if beta.shape[1] != data["pre"].shape[1]:
        raise ValueError("beta and pre must use the same selected feature columns")
    validate_labels(data["labels"], *data["pre"].shape)
    np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(len(eigenvalues)),
                               atol=1e-8, rtol=1e-8)


def edited_scores(data, coefficients, batch_size=2048):
    """Retain the frozen baseline and add the centered encoder-weight change."""
    delta = (data["eigenvectors"] / np.sqrt(data["eigenvalues"])[None, :]) @ (
        coefficients - data["beta"])
    bias_delta = -data["mean"] @ delta
    result = np.empty(data["pre"].shape, dtype=np.float64)
    for start in range(0, len(result), batch_size):
        end = start + batch_size
        result[start:end] = (data["pre"][start:end]
                             + data["native"][start:end].astype(np.float64) @ delta
                             + bias_delta)
    return result


def score_rows(name, scores, data):
    before = cluster_aurocs(jumprelu(data["pre"], data["thresholds"]), data["labels"])
    after = cluster_aurocs(jumprelu(scores, data["thresholds"]), data["labels"])
    return [dict(condition=name, cluster=j, original_post_auc=float(before[j]),
                 edited_post_auc=float(after[j]), reduction=float(before[j] - after[j]))
            for j in range(len(before))]
