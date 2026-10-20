"""Evaluate fixed entity features with separate validation cutoffs before and after JumpReLU."""

import numpy as np

from common import binary_labels, finite, jumprelu, load_arrays, parser, write_csv


def auc_bank(values, labels, step=128):
    from scipy.stats import rankdata
    n, f = values.shape
    pos = labels.sum(0).astype(np.float64)
    ans = np.full((labels.shape[1], f), np.nan)
    good = (pos > 0) & (pos < n)
    yy = labels[:, good].astype(np.float64).T
    for s in range(0, f, step):
        ranks = rankdata(np.array(values[:, s:s+step]), axis=0, method='average')
        ans[good, s:s+step] = (yy @ ranks - pos[good, None]*(pos[good, None]+1)/2)/(pos[good,None]*(n-pos[good,None]))
    return ans


def cutoff(x, y):
    # Keep nextafter(min_score, -inf) distinct from a float32 minimum.
    x = np.asarray(x, dtype=np.float64)
    if not y.any() or y.all():
        return None
    order = np.argsort(-x, kind='stable')
    z, yy = x[order], y[order]
    ends = np.r_[np.flatnonzero(z[:-1] != z[1:]), len(z)-1]
    tp = np.r_[0, np.cumsum(yy, dtype=np.int64)[ends]]
    predicted = np.r_[0, ends+1]
    fp = predicted-tp
    f1 = 2*tp/(predicted+y.sum())
    # No-positive condition, strict > each next unique value, then all-positive.
    candidates = np.r_[z[0], z[ends[:-1]+1], np.nextafter(float(z[-1]), -np.inf)]
    best = min(range(len(f1)), key=lambda i: (-f1[i], fp[i], predicted[i]))
    return float(candidates[best])


def counts(y, pred):
    return np.stack([(y & pred).sum(0), ((1-y) & pred).sum(0),
                     (y & (1-pred)).sum(0), ((1-y) & (1-pred)).sum(0)], axis=-1)


def evaluate(validation_pre, test_pre, thresholds, validation_labels, test_labels):
    if validation_pre.ndim != 2 or test_pre.ndim != 2:
        raise ValueError("Sentence scores must be two-dimensional")
    if validation_pre.shape[1] != test_pre.shape[1] or thresholds.shape != (test_pre.shape[1],):
        raise ValueError("Both splits and thresholds must use the same feature columns")
    for name, values in (("validation_pre", validation_pre), ("test_pre", test_pre),
                         ("thresholds", thresholds)):
        finite(name, values)
    if np.any(thresholds < 0):
        raise ValueError("JumpReLU thresholds must be nonnegative")
    validation_labels = binary_labels(validation_labels, validation_pre.shape)
    test_labels = binary_labels(test_labels, test_pre.shape)
    rows = []
    for stage in ("pre", "post"):
        validation = validation_pre if stage == "pre" else jumprelu(validation_pre, thresholds)
        test = test_pre if stage == "pre" else jumprelu(test_pre, thresholds)
        for j in range(test.shape[1]):
            boundary = cutoff(validation[:, j], validation_labels[:, j])
            if boundary is None or not test_labels[:, j].any() or test_labels[:, j].all():
                raise ValueError(f"Type {j} is missing a class in validation or test")
            # Compare in float64 to retain a cutoff just below the smallest float32 score.
            predicted = test[:, j].astype(np.float64) > boundary
            tp, fp, fn, tn = map(int, counts(test_labels[:, j].astype(np.uint8),
                                            predicted.astype(np.uint8)))
            rows.append(dict(type_index=j, stage=stage, cutoff=boundary,
                             TP=tp, FP=fp, FN=fn, TN=tn,
                             precision=tp / (tp + fp) if tp + fp else np.nan,
                             recall=tp / (tp + fn), F1=2 * tp / (2 * tp + fp + fn)))
    return rows


def main():
    args = parser(__doc__).parse_args()
    keys = ("validation_pre", "test_pre", "thresholds", "validation_labels", "test_labels",
            "feature_ids", "validation_feature_ids", "test_feature_ids")
    data = load_arrays(args.input, keys)
    if data["feature_ids"].shape != (data["test_pre"].shape[1],):
        raise ValueError("feature_ids must match the type columns")
    for name in ("validation_feature_ids", "test_feature_ids"):
        np.testing.assert_array_equal(data["feature_ids"], data[name])
    rows = evaluate(*(data[name] for name in keys[:5]))
    for row in rows:
        row["feature_id"] = int(data["feature_ids"][row["type_index"]])
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()

