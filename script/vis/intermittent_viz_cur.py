"""Visualize selected intermittent-noise evaluation groups.

  python script/vis/intermittent_viz_cur.py
  python script/vis/intermittent_viz_cur.py --groups baseline v3
  python script/vis/intermittent_viz_cur.py --checkpoint model_30000 --list
"""
import argparse
import sys
from pathlib import Path

ACM_ROOT = Path(__file__).resolve().parents[2]
if str(ACM_ROOT) not in sys.path:
    sys.path.insert(0, str(ACM_ROOT))

from script.result_paths import configure_visualization_cache, get_results_root

import numpy as np

from script.vis.viz_group_selection import (
    GROUPS,
    add_selection_arguments,
    choose_groups,
    common_checkpoint,
    data_path,
    default_output,
    print_groups,
    ready_groups,
)


CATEGORY = "intermittent_noise"
DATA_FILE = Path("intermittent.npz")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

DUTIES = [0.02, 0.05, 0.2, 0.5, 1.0]
BURSTS = [1, 5, 20]
MAGS = [2.0, 4.0]


def load_data(root, groups, checkpoint):
    tables = {}
    command = None
    for group in groups:
        raw = np.load(
            data_path(root, group, DATA_FILE, checkpoint), allow_pickle=True)
        group_command = float(raw["cmd_vx"])
        if command is not None and not np.isclose(group_command, command):
            raise ValueError(
                f"cmd_vx mismatch: {groups[0]}={command:g}, {group}={group_command:g}")
        tables[group] = {
            (float(row[0]), int(row[1]), float(row[2])): row
            for row in raw["summary"]
        }
        command = group_command
    return tables, command


def series(tables, group, burst, magnitude, column=3):
    return [
        float(tables[group][(duty, burst, magnitude)][column])
        for duty in DUTIES
    ]


def plot(root, groups, checkpoint, output):
    configure_visualization_cache()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tables, command = load_data(root, groups, checkpoint)
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "text.color": INK2,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig = plt.figure(figsize=(13.4, 8.4), dpi=180)
    grid = fig.add_gridspec(
        2, 3, left=0.075, right=0.982, top=0.705, bottom=0.115,
        hspace=0.54, wspace=0.13)
    y_max = max(
        max(series(tables, group, burst, magnitude))
        for group in groups for burst in BURSTS for magnitude in MAGS) * 1.12
    y_max = max(y_max, 1.0)
    x_values = np.arange(len(DUTIES))

    for row_index, magnitude in enumerate(MAGS):
        for column_index, burst in enumerate(BURSTS):
            axis = fig.add_subplot(grid[row_index, column_index])
            for spine in ["top", "right"]:
                axis.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                axis.spines[spine].set_color(AXIS)
            axis.yaxis.grid(True, color=GRID, lw=0.8)
            axis.set_axisbelow(True)
            axis.tick_params(color=AXIS)
            axis.set_xticks(x_values, [f"{duty:g}" for duty in DUTIES])
            axis.set_ylim(0, y_max)
            axis.set_xlim(-0.25, len(DUTIES) - 0.75)
            if column_index:
                axis.set_yticklabels([])
            else:
                axis.set_ylabel("falls per 1k env-steps", fontsize=8, color=MUTED)
            axis.set_xlabel("duty cycle (fraction of steps corrupted)", fontsize=8, color=MUTED)
            for group in groups:
                style = GROUPS[group]
                axis.plot(
                    x_values, series(tables, group, burst, magnitude),
                    color=style["color"], marker=style["marker"], lw=2, ms=6,
                    mec=SURFACE, mew=1.5, solid_capstyle="round", zorder=3)
            heading = "isolated spikes" if burst == 1 else f"{burst}-step bursts"
            axis.set_title(
                f"{heading}  (burst = {burst})", loc="left",
                fontsize=9.5, color=INK, pad=6)

    for row_index, magnitude in enumerate(MAGS):
        top = grid[row_index, 0].get_position(fig).y1
        fig.text(
            0.075, top + 0.055, f"spike magnitude {magnitude:g}σ",
            fontsize=9, color=INK, ha="left", weight="semibold")

    handles = [
        Line2D(
            [0], [0], color=GROUPS[group]["color"], marker=GROUPS[group]["marker"],
            lw=2, ms=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=GROUPS[group]["label"])
        for group in groups
    ]
    fig.legend(
        handles=handles, frameon=False, fontsize=8, labelcolor=INK2,
        loc="upper left", bbox_to_anchor=(0.070, 0.835), ncol=min(4, len(groups)),
        handlelength=1.6, columnspacing=1.4)
    fig.text(
        0.075, 0.955, "Intermittent observation glitches",
        fontsize=15, color=INK, ha="left", weight="semibold")
    fig.text(
        0.075, 0.918,
        f"groups: {', '.join(groups)} · {checkpoint} · vx = {command:g} m/s · "
        "gaussian spikes on all four "
        "sensor groups · 200 clean warm-up steps, then 600 measured steps",
        fontsize=8.5, color=MUTED, ha="left")
    fig.text(
        0.075, 0.888,
        "Across a row the total corruption duty is fixed while its clustering changes; "
        "at duty 1 all burst columns should agree.",
        fontsize=8.5, color=MUTED, ha="left")
    fig.text(
        0.075, 0.025,
        "Within a step, current observation and history receive the same physical glitch; "
        "magnitude is a multiple of measured on-policy σ.",
        fontsize=7.5, color=MUTED, ha="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print("selected:", ", ".join(groups))
    print("saved:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    args = parser.parse_args()
    results_root = get_results_root(args.results_root)
    configure_visualization_cache(results_root)
    root = results_root / "eval" / CATEGORY
    if args.list:
        print_groups(root, DATA_FILE, args.checkpoint)
        return
    try:
        groups = choose_groups(root, DATA_FILE, args.groups, args.checkpoint)
        if not groups:
            raise ValueError(f"no ready groups under {root} for {DATA_FILE}")
        checkpoint = common_checkpoint(
            root, groups, DATA_FILE, args.checkpoint)
        available = ready_groups(root, DATA_FILE, checkpoint)
        output_dir = results_root / "plots" / CATEGORY / checkpoint
        output = args.output or default_output(
            output_dir, "intermittent_robustness.png", groups, available)
        plot(root, groups, checkpoint, output.resolve())
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
