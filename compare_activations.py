"""Compare preactivation, ReLU, and JumpReLU for fixed selected features."""

import numpy as np

from common import (cluster_aurocs, jumprelu, load_arrays, parser,
                    validate_labels, validate_scores, write_csv)


def compare(pre, thresholds, labels):
    validate_scores(pre, thresholds)
    validate_labels(labels, *pre.shape)
    before = cluster_aurocs(pre, labels)
    relu = cluster_aurocs(np.maximum(pre, 0), labels)
    after = cluster_aurocs(jumprelu(pre, thresholds), labels)
    return [dict(cluster=j, pre_auc=float(before[j]), relu_auc=float(relu[j]),
                 post_auc=float(after[j]), nonpositive_loss=float(before[j] - relu[j]),
                 positive_loss=float(relu[j] - after[j]),
                 total_loss=float(before[j] - after[j])) for j in range(pre.shape[1])]


def main():
    command = parser(__doc__)
    command.add_argument("--coder", choices=("sae", "tc"),
                         help="Read coder-prefixed arrays from an existing replay file")
    args = command.parse_args()
    prefix = f"{args.coder}_" if args.coder else ""
    data = load_arrays(args.input, ("labels", prefix + "pre", prefix + "thresholds"))
    rows = compare(data[prefix + "pre"], data[prefix + "thresholds"], data["labels"])
    if prefix + "feature_ids" in data:
        ids = data[prefix + "feature_ids"]
        if ids.shape != (len(rows),):
            raise ValueError("feature_ids must match the score columns")
        for row, feature_id in zip(rows, ids):
            row["feature_id"] = int(feature_id)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
