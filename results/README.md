# Evaluation Results

Evaluation and visualization scripts store generated artifacts here by default.

```text
results/
  eval/<category>/<run>/<checkpoint>/...
  plots/<plot-kind>/<run>/<checkpoint>/...
```

Set `ACM_RESULTS_ROOT` to relocate all outputs. Eval scripts also accept the
Hydra override `+results_root=/path/to/results`. Explicit `+out=...`,
`+collect_save_dir=...`, and `--out ...` values take precedence.
