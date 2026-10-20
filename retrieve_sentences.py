"""Retrieve a fixed number of sentences, resolving score ties by sentence ID."""

import numpy as np

from common import binary_labels, finite, load_arrays, parser, save_arrays, write_csv

K = 200


def ranking(scores, id_order):
    return id_order[np.argsort(-scores[id_order], kind='stable')]


def retrieval(scores, truth, order, k=K):
    selected = order[:k]
    assert len(selected) == len(np.unique(selected)) == k
    cutoff = float(scores[selected[-1]])
    above = scores > cutoff
    tied = scores == cutoff
    slots = k - int(above.sum())
    ties = int(tied.sum())
    tie_tp = int(truth[tied].sum())
    above_tp = int(truth[above].sum())
    tp = int(truth[selected].sum())
    return dict(TP=tp, FP=k-tp, precision=tp/k,
                positive_sentences=int(truth.sum()), candidate_sentences=len(truth),
                random_expected_TP=k*float(truth.mean()),
                cutoff_post=cutoff, positive_post_sentences=int((scores > 0).sum()),
                selected_zero=int((scores[selected] == 0).sum()),
                boundary_ties=ties, selected_boundary_ties=slots,
                tie_min_TP=above_tp+max(0, slots-(ties-tie_tp)),
                tie_max_TP=above_tp+min(slots, tie_tp),
                tie_mean_TP=above_tp+slots*tie_tp/ties)


def bootstrap_tp(order, truth, weights, k=K):
    """Equivalent to ranking np.repeat(candidate, bootstrap multiplicity)."""
    end = min(len(order), 2*k)
    while True:
        w = weights[:, order[:end]]
        total = np.cumsum(w, axis=1, dtype=np.int32)
        if np.all(total[:, -1] >= k):
            break
        if end == len(order):
            raise ValueError('bootstrap pool smaller than review budget')
        end = min(len(order), 2*end)
    used = np.minimum(w, np.maximum(0, k-(total-w)))
    assert np.all(used.sum(1) == k)
    return np.sum(used * truth[order[:end]][None, :], axis=1)


def compare(scores, labels, sentence_ids, budget=200, draws=1000, seed=141):
    """Pair sentence-bootstrap draws across coders, layers, and entity types."""
    if scores.ndim != 4 or scores.shape[0] != 2:
        raise ValueError("scores must have shape [2 coders, layers, sentences, types]")
    if budget < 1 or draws < 1 or budget > scores.shape[2]:
        raise ValueError("Invalid review budget or bootstrap count")
    finite("scores", scores)
    if np.any(scores < 0):
        raise ValueError("Retrieval expects postactivation scores")
    labels = binary_labels(labels, scores.shape[2:])
    if sentence_ids.shape != (len(labels),) or len(np.unique(sentence_ids)) != len(labels):
        raise ValueError("sentence_ids must be unique and match score order")
    eligible = np.flatnonzero(labels.sum(0) > budget)
    if not len(eligible):
        raise ValueError("No type has more positives than the review budget")
    id_order = np.argsort(sentence_ids, kind="stable")
    rng = np.random.default_rng(seed)
    # A single draw matrix is reused for every comparison.
    weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=draws)
    bootstrap = np.zeros((scores.shape[1], 2, draws), dtype=np.float64)
    points = np.zeros((scores.shape[1], 2), dtype=np.float64)
    rows = []
    for layer in range(scores.shape[1]):
        for coder, name in enumerate(("sae", "tc")):
            type_points = []
            for entity in eligible:
                values, truth = scores[coder, layer, :, entity], labels[:, entity]
                order = ranking(values, id_order)
                result = retrieval(values, truth, order, budget)
                type_points.append(result["TP"])
                rows.append(dict(layer=layer, coder=name, type_index=int(entity), **result))
                bootstrap[layer, coder] += bootstrap_tp(order, truth, weights, budget) / len(eligible)
            points[layer, coder] = np.mean(type_points)
    summary = []
    for name, point, samples in (
        ("sae", points[:, 0].mean(), bootstrap[:, 0].mean(0)),
        ("tc", points[:, 1].mean(), bootstrap[:, 1].mean(0)),
        ("tc_minus_sae", (points[:, 1] - points[:, 0]).mean(),
         (bootstrap[:, 1] - bootstrap[:, 0]).mean(0)),
    ):
        low, high = np.quantile(samples, (0.025, 0.975))
        summary.append(dict(contrast=name, mean_target_sentences=float(point),
                            ci_low=float(low), ci_high=float(high), budget=budget,
                            types=len(eligible), layers=scores.shape[1]))
    return rows, summary, bootstrap


def main():
    command = parser(__doc__)
    command.add_argument("--budget", type=int, default=200)
    command.add_argument("--draws", type=int, default=1000)
    command.add_argument("--seed", type=int, default=141)
    args = command.parse_args()
    data = load_arrays(args.input, ("scores", "labels", "sentence_ids"))
    rows, summary, draws = compare(data["scores"], data["labels"], data["sentence_ids"],
                                  args.budget, args.draws, args.seed)
    write_csv(args.output / "per_type.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    save_arrays(args.output / "bootstrap.npz", values=draws, coders=np.array(["sae", "tc"]))


if __name__ == "__main__":
    main()

