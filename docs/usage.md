# Usage

[← Overview](../README.md) · [Validation](validation.md)

## Setup

Use Python 3.9 or newer:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Only NumPy and SciPy are needed. The command-line analyses run on CPU. Full-size
activation files may need substantial memory. Use the original frozen inputs;
changing score precision, feature selection, or document order can change results.

## Analyses

| Script | Input | Output |
| --- | --- | --- |
| `compare_activations.py` | Fixed-feature token scores | Pre/ReLU/post AUROC and losses |
| `restore_activations.py` | Token scores, decoder rows, reconstruction residuals | Single-feature restoration gains |
| `decompose_loss.py` | AUROCs and nonpositive rates | Nonpositive/positive loss decomposition |
| `rotate_encoder.py` | Frozen encoder geometry and evaluation inputs | Targeted and matched-random rotation AUROCs |
| `edit_components.py` | Frozen encoder geometry and evaluation inputs | Component removal/rescaling AUROCs |
| `detect_entities.py` | Fixed-feature validation/test sentence scores | Separately selected cutoffs and test F1 |
| `retrieve_sentences.py` | Paired SAE/TC sentence scores | Top-budget retrieval and paired bootstrap intervals |

`common.py` contains shared validation and I/O. Each script supports `--help`.

```sh
python compare_activations.py data/replay.npz --coder sae --output results/sae_auc.csv
python restore_activations.py data/restoration.npz --output results/restoration.csv
python decompose_loss.py data/loss.npz --output results/loss.csv
python rotate_encoder.py data/rotation.npz --k 16 --output results/rotation.csv
python edit_components.py data/components.npz --k 16 --output results/components.csv
python detect_entities.py data/entities.npz --output results/entities.csv
python retrieve_sentences.py data/retrieval.npz --output results/retrieval
```

## Input arrays

All inputs are `.npz` archives without pickled objects. Unless specified otherwise,
run a script once per layer, coder, and sparsity setting. Keep these identities in
your filenames. Column `j` always denotes the feature selected for target `j`;
feature identities stay fixed across pre/post scores and interventions.

### Activation comparison

- `pre`: `[tokens, clusters]`, saved preactivations of postactivation-selected features.
- `thresholds`: `[clusters]`, learned JumpReLU thresholds.
- `labels`: `[tokens]`, integer cluster indices starting at zero.
- `feature_ids`: optional `[clusters]` identifiers.

With `--coder sae` or `--coder tc`, the script reads `sae_pre`,
`sae_thresholds`, `sae_feature_ids` (or the TC equivalents) from the existing replay
format. `labels` remains unprefixed. This is not preactivation-based reselection.

### Activation restoration

Use the unprefixed activation arrays above, plus:

- `residual`: `[tokens, output_dimensions]`, MLP output minus the **full original
  coder reconstruction**, not the reconstruction of the selected feature alone.
- `decoder`: `[clusters, output_dimensions]`, decoder rows in the selected-feature order.

For each feature separately, the added activation is `pre` when
`0 < pre <= threshold`, and zero otherwise. The squared-error reduction is
`2 * added * (residual @ decoder) - added**2 * ||decoder||**2`.
Group averages include unaffected tokens. These are point estimates; the CLI
does not reconstruct the original document-bootstrap confidence intervals.
The inherited `summarize` function is retained for callers with per-document
restoration statistics.

### Loss decomposition

For each prefix `sae_` and `tc_`:

- `rates`: `[clusters, 2]`, fractions of nonpositive target and nontarget scores.
- `pre`, `relu`, `post`: `[clusters]`, corresponding AUROCs.

Optional `rates_deleted`, `pre_deleted`, `relu_deleted`, `post_deleted` provide
leave-one-document-out estimates, with a document dimension after the cluster
dimension. Supply them for both coders in the same document order. These give
per-cluster paired jackknife intervals; absent inputs produce `nan` intervals.

The conditional AUROC among pairs with both scores nonpositive is undefined when
that pair mass is zero. It remains `nan`, rather than being replaced by zero.

### Encoder interventions

Both intervention scripts use:

- `native`: `[tokens, input_dimensions]`, each coder's own evaluation inputs.
- `mean`: `[input_dimensions]`, the frozen selection-set input mean.
- `eigenvalues`: `[retained_dimensions]`, positive selection covariance eigenvalues.
- `eigenvectors`: `[input_dimensions, retained_dimensions]`, matching orthonormal columns.
- `beta`: `[retained_dimensions, clusters]`, whitened encoder coefficients:
  `sqrt(eigenvalues)[:, None] * (eigenvectors.T @ selected_encoder_weights)`.
- `pre`, `thresholds`, `labels`, `feature_ids`: fixed baseline arrays as above.
- `angle`: `[clusters]`, the frozen intervention angles in radians.

`rotate_encoder.py` additionally requires `basis`, the selected orthonormal
directions in whitened coordinates, and `reference`, a scalar or per-cluster
target squared-norm fraction. Supply the original paired angles; the script does
not choose them independently for SAE and TC. `reference_fraction` retains the
original random-weight reference calculation for callers that need it.

`edit_components.py` chooses the leading PCs by descending eigenvalue. It reports
leading components only, remainder only, neither, variance-restored remainder,
partial reduction only, amplification only, and their combination. It also
evaluates norm- and angle-matched random controls, except for the zero vector.

Edits add `native @ delta_weight - mean @ delta_weight` to the frozen baseline.
This preserves the baseline contribution outside the retained whitening space.
The selection-set mean is fixed; evaluation data are not recentered or rewhitened.
These CLIs report point estimates, not the original GPU document-bootstrap run.

### Entity detection

- `validation_pre`, `test_pre`: `[sentences, types]`, sentence maxima over valid
  nonspecial, nonpadding tokens, for the same postactivation-selected features.
- `validation_labels`, `test_labels`: binary arrays matching those shapes.
- `thresholds`: `[types]`, learned feature thresholds, not classification cutoffs.
- `feature_ids`, `validation_feature_ids`, `test_feature_ids`: matching `[types]` arrays.

Each type must have both classes in each split. Select features on training
postactivation AUROC before preparing this input; `auc_bank` retains that scoring
calculation. The CLI does not perform feature selection on validation or test.
It selects a separate validation-F1-maximizing cutoff for pre and post scores,
then reports test precision, recall, and F1.

### Sentence retrieval

- `scores`: `[2, layers, sentences, types]`, postactivation sentence scores;
  coder order is **SAE, TC**, for one sparsity setting.
- `labels`: `[sentences, types]`, binary labels shared by all conditions.
- `sentence_ids`: `[sentences]`, unique string IDs in score-row order.

Defaults are budget 200, 1,000 bootstrap samples, and seed 141. Eligible types
have strictly more positives than the budget, fixed before resampling. Ties are
resolved by ascending sentence ID, without labels. Each bootstrap resamples
sentences and reruns retrieval, using common multiplicities for every coder,
layer, and type. Means weight types equally, then layers equally. Output includes
SAE, TC, and paired TC-minus-SAE intervals.
