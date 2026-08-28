# Evaluation Results

Evaluation and visualization scripts store generated artifacts here by default.

```text
results/
  eval/<category>/<run>/<checkpoint>/...
  plots/<plot-kind>/<checkpoint>/...              # multi-run comparisons
  plots/<plot-kind>/<run>/<checkpoint>/...        # single-run plots
```

Set `ACM_RESULTS_ROOT` to relocate all outputs. Eval scripts also accept the
Hydra override `+results_root=/path/to/results`. Explicit `+out=...`,
`+collect_save_dir=...`, and `--out ...` values take precedence.

Visualization commands and their input categories are documented in
`script/vis/README.md`.
