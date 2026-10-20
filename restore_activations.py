"""Restore suppressed positive activations and measure reconstruction error changes."""

import numpy as np

from common import (auroc as auc, load_arrays, parser, validate_labels,
                    validate_scores, write_csv)


def restoration(pre, threshold):
    mask = (pre > 0) & (pre <= threshold)
    return np.where(mask, pre, 0), mask


def gain_formula(residual, added, decoder):
    return 2 * added * (residual @ decoder.T) - added**2 * np.sum(decoder**2, axis=1)


def summarize(stats, scale, weights=None):
    total = stats.sum(axis=0) if weights is None else np.tensordot(weights, stats, axes=(1, 0))
    n, edited, active, nonpos, before, edit_before, gain, better, worse, same = np.moveaxis(total, -1, 0)
    def div(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    return dict(total_count=n, edit_count=edited, original_active_count=active,
                nonpositive_count=nonpos, original_error_all=div(before,n),
                edited_error_all=div(before-gain,n), error_reduction_all=div(gain,n),
                original_error_edited_only=div(edit_before,edited),
                edited_error_edited_only=div(edit_before-gain,edited),
                error_reduction_edited_only=div(gain,edited),
                error_reduction_all_over_y_norm=div(gain,n)/scale,
                improved_fraction_edited_only=div(better,edited),
                worsened_fraction_edited_only=div(worse,edited),
                unchanged_fraction_edited_only=div(same,edited))


def restore(pre, thresholds, residual, decoder, labels):
    validate_scores(pre, thresholds)
    validate_labels(labels, *pre.shape)
    if residual.ndim != 2 or residual.shape[0] != len(pre):
        raise ValueError("residual must have shape [tokens, output dimensions]")
    if decoder.shape != (pre.shape[1], residual.shape[1]):
        raise ValueError("decoder must have one row per selected feature")
    if not np.isfinite(residual).all() or not np.isfinite(decoder).all():
        raise ValueError("Nonfinite residual or decoder")
    added, mask = restoration(pre, thresholds)
    before = np.where(pre > thresholds, pre, 0.)
    after = before + added
    residual = residual.astype(np.float64)
    decoder = decoder.astype(np.float64)
    error = np.square(residual).sum(axis=1)
    rows = []
    # Each column is a separate single-feature intervention, not a joint edit.
    for j in range(pre.shape[1]):
        gain = gain_formula(residual, added[:, j:j+1].astype(np.float64), decoder[j:j+1])[:, 0]
        target = labels == j
        auc_before = auc(before[target, j], before[~target, j])
        auc_after = auc(after[target, j], after[~target, j])
        for name, group in (("target", target), ("nontarget", ~target)):
            rows.append(dict(cluster=j, group=name, tokens=int(group.sum()),
                             restored_tokens=int(mask[group, j].sum()),
                             original_error=float(error[group].mean()),
                             restored_error=float((error[group] - gain[group]).mean()),
                             error_reduction=float(gain[group].mean()),
                             post_auc_before=auc_before, post_auc_after=auc_after))
    return rows


def main():
    args = parser(__doc__).parse_args()
    data = load_arrays(args.input, ("pre", "thresholds", "residual", "decoder", "labels"))
    write_csv(args.output, restore(*(data[name] for name in
                                     ("pre", "thresholds", "residual", "decoder", "labels"))))


if __name__ == "__main__":
    main()

