"""Separate loss from nonpositive collapse and positive thresholding."""

import numpy as np

from common import load_arrays, parser, write_csv


def interval(point, deleted):
    if not np.isfinite(point) or not np.isfinite(deleted).all():
        return np.nan, np.nan
    n = len(deleted)
    se = np.sqrt((n-1)/n * np.square(deleted-deleted.mean()).sum())
    return float(point-1.95996398454*se), float(point+1.95996398454*se)


def statistics(rates, pre, relu, post):
    pre, relu, post = map(np.asarray, (pre, relu, post))
    a, b = rates[..., 0], rates[..., 1]
    m, d = a*b, pre-relu
    q = np.full(m.shape, np.nan)
    np.divide(d, m, out=q, where=m>0)
    q += .5
    assert np.all((rates >= 0) & (rates <= 1))
    np.testing.assert_allclose(d[m == 0], 0, rtol=0, atol=1e-10)
    assert np.all((q[m>0] >= -1e-10) & (q[m>0] <= 1+1e-10))
    return np.stack([a, b, m, q, d, pre, relu, post, relu-post, pre-post], axis=-1)


def terms(s, t):
    ms, mt, qs, qt = s[..., 2], t[..., 2], s[..., 3], t[..., 3]
    amount = (ms-mt)*((qs+qt)/2-.5)
    ordering = (ms+mt)/2*(qs-qt)
    valid = np.isfinite(amount) & np.isfinite(ordering)
    np.testing.assert_allclose((amount+ordering)[valid], (s[..., 4]-t[..., 4])[valid], rtol=0, atol=1e-10)
    return amount, ordering


METRICS = ("target_nonpositive", "nontarget_nonpositive", "both_nonpositive_pairs",
           "conditional_auc", "pre_relu_loss", "pre_auc", "relu_auc", "post_auc",
           "positive_gate_loss", "total_loss")


def main():
    args = parser(__doc__).parse_args()
    required = [f"{coder}_{name}" for coder in ("sae", "tc")
                for name in ("rates", "pre", "relu", "post")]
    data = load_arrays(args.input, required)
    points, deletions = {}, {}
    for coder in ("sae", "tc"):
        values = [data[f"{coder}_{name}"] for name in ("rates", "pre", "relu", "post")]
        if values[0].ndim != 2 or values[0].shape[1] != 2:
            raise ValueError("rates must have shape [clusters, target/nontarget]")
        if any(v.shape != values[0].shape[:1] for v in values[1:]):
            raise ValueError("AUROC arrays must have one value per cluster")
        points[coder] = statistics(*values)
        names = [f"{coder}_{name}_deleted" for name in ("rates", "pre", "relu", "post")]
        if any(name in data for name in names):
            if not all(name in data for name in names):
                raise ValueError("Provide all four leave-one-document-out arrays")
            deleted_values = [data[name] for name in names]
            if (deleted_values[0].ndim != 3 or deleted_values[0].shape[::2] != values[0].shape
                    or any(v.shape != deleted_values[0].shape[:2] for v in deleted_values[1:])):
                raise ValueError("Deleted arrays must use [clusters, documents, ...]")
            deletions[coder] = statistics(*deleted_values)
    if points["sae"].shape != points["tc"].shape:
        raise ValueError("SAE and TC must use the same clusters")
    if deletions and (set(deletions) != {"sae", "tc"}
                      or deletions["sae"].shape != deletions["tc"].shape):
        raise ValueError("Paired jackknife requires the same deletions for both coders")
    points["sae_minus_tc"] = points["sae"] - points["tc"]
    if deletions:
        deletions["sae_minus_tc"] = deletions["sae"] - deletions["tc"]
    rows = []
    for coder, values in points.items():
        for j, estimates in enumerate(values):
            for m, name in enumerate(METRICS):
                low, high = (interval(estimates[m], deletions[coder][j, :, m])
                             if deletions else (np.nan, np.nan))
                rows.append(dict(coder=coder, cluster=j, metric=name,
                                 estimate=float(estimates[m]), ci_low=low, ci_high=high))
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()

