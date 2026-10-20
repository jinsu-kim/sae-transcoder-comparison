# Validation and provenance

[← Overview](../README.md) · [Usage](usage.md)

Tests cover score ties, threshold boundaries, reconstruction-gain identities,
encoder norms and rotation angles, validation cutoff selection, materialized
bootstrap comparisons, and all seven command-line paths on synthetic inputs.
[`source_map.json`](../source_map.json) records the original source hashes and extracted functions.

To also compare the numerical functions with an original checkout:

```sh
ORIGINAL_CODE_ROOT=/path/to/original/codex/code python -m unittest discover -s tests -v
```

One numerical edge case was corrected: cutoff candidates are constructed in
float64 so `nextafter(min_score, -inf)` remains below a float32 minimum. This
retains the all-positive candidate for strict `score > cutoff` evaluation.
Full-data paper-result replay has **not** been run for this cleaned release.
CPU arithmetic is not claimed to be bitwise identical to the original GPU runs.

Data, checkpoints, server launchers, credentials, and generated results are not
included. No license has been assigned to this repository.
