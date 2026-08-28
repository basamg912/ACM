# Visualization Scripts

Evaluation inputs are read from `results/eval/<category>/<group>/<checkpoint>/`.
Generated figures are written to `results/plots/<category>/<checkpoint>/`.
Set `ACM_RESULTS_ROOT` or pass `--results-root` to relocate both trees.

```bash
python script/vis/corruption_viz_cur.py --list
python script/vis/corruption_viz_cur.py -g baseline v3 --checkpoint model_30000
python script/vis/dropout_viz_cur.py -g all
python script/vis/intermittent_viz_cur.py -g baseline v3
python script/vis/push_viz_cur.py -g baseline v3 --checkpoint model_30000

python script/vis/plot_latent_tsne.py \
  results/eval/latents/v3/model_30000/latents_ckpt_30000.npz
python script/vis/plot_latent_umap.py \
  results/eval/latents/v3/model_30000/latents_ckpt_30000.npz
```

Without `--checkpoint`, comparison scripts use the latest checkpoint shared by
all selected groups. This prevents a plot from silently comparing different
training steps.
