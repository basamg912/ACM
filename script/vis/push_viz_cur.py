"""Visualize selected external-push robustness evaluation groups.

  python script/vis/push_viz_cur.py
  python script/vis/push_viz_cur.py --groups baseline v3 v4
  python script/vis/push_viz_cur.py --checkpoint model_30000 --list

Directions and magnitudes are read from each ``push.npz``. Selected
groups must have identical condition sets, preventing accidental comparison of
old four-direction sweeps with new eight-direction sweeps.
"""
import argparse
import math
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
    default_output as selected_output,
    print_groups,
    ready_groups,
)


CATEGORY = "push_robustness"
DATA_FILE = Path("push.npz")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
TRAIN_BAND = "#eceae2"

DIRECTION_LABELS = {
    "front": "front (+x)",
    "back": "back (−x)",
    "left": "left (+y)",
    "right": "right (−y)",
    "FL": "front-left (+x,+y)",
    "FR": "front-right (+x,−y)",
    "BL": "back-left (−x,+y)",
    "BR": "back-right (−x,−y)",
}
RECOVERY_MAGNITUDE = 1.0


def push_path(root, group, checkpoint):
    return data_path(root, group, DATA_FILE, checkpoint)


def unique_in_order(values):
    return list(dict.fromkeys(values))


def npz_conditions(raw):
    pairs = [
        (str(direction), float(magnitude))
        for direction, magnitude in zip(raw["cond_dir"], raw["cond_mag"])
        if str(direction) != "clean"
    ]
    directions = unique_in_order(direction for direction, _ in pairs)
    magnitudes = sorted(set(magnitude for _, magnitude in pairs))
    return directions, magnitudes, set(pairs)


def push_condition_summary(root, group, checkpoint):
    raw = np.load(push_path(root, group, checkpoint), allow_pickle=True)
    directions, magnitudes, _ = npz_conditions(raw)
    return f"{len(directions)} dirs × {len(magnitudes)} mags: {', '.join(directions)}"


def print_push_groups(root, checkpoint):
    print_groups(root, DATA_FILE, checkpoint)
    available = ready_groups(root, DATA_FILE, checkpoint)
    if available:
        print("\nStored push conditions:")
        for group in available:
            resolved = data_path(root, group, DATA_FILE, checkpoint).parent.name
            print(
                f"  {group:<16} {resolved:<16} "
                f"{push_condition_summary(root, group, resolved)}")


def load_data(root, groups, checkpoint):
    data = {}
    command = None
    push_every = None
    reference_pairs = None
    reference_group = None
    directions = None
    magnitudes = None

    for group in groups:
        raw = np.load(push_path(root, group, checkpoint), allow_pickle=True)
        group_command = float(raw["cmd_vx"])
        group_push_every = int(raw["push_every"])
        group_directions, group_magnitudes, group_pairs = npz_conditions(raw)
        if command is not None and not np.isclose(group_command, command):
            raise ValueError(
                f"cmd_vx mismatch: {groups[0]}={command:g}, {group}={group_command:g}")
        if push_every is not None and group_push_every != push_every:
            raise ValueError(
                f"push_every mismatch: {groups[0]}={push_every}, {group}={group_push_every}")
        if reference_pairs is not None and group_pairs != reference_pairs:
            raise ValueError(
                f"push condition mismatch: {reference_group} has "
                f"{len(directions)} directions ({', '.join(directions)}), but {group} has "
                f"{len(group_directions)} ({', '.join(group_directions)}). "
                "Re-run eval_push_robustness.py with the same DIRS/MAGS before comparing.")

        data[group] = {
            "tab": {(str(row[0]), float(row[1])): row for row in raw["summary"]},
            "rec": raw["recovery_vx"],
            "rec_cnt": raw["recovery_cnt"],
            "condition_dirs": [str(value) for value in raw["cond_dir"]],
            "condition_mags": raw["cond_mag"],
        }
        if reference_pairs is None:
            reference_pairs = group_pairs
            reference_group = group
            directions = group_directions
            magnitudes = group_magnitudes
        command = group_command
        push_every = group_push_every

    if not any(np.isclose(magnitude, RECOVERY_MAGNITUDE) for magnitude in magnitudes):
        raise ValueError(
            f"recovery magnitude {RECOVERY_MAGNITUDE:g} is absent; stored magnitudes={magnitudes}")
    return data, command, push_every, directions, magnitudes


def series(data, group, direction, magnitudes, column):
    return [
        float(data[group]["tab"][(direction, magnitude)][column])
        for magnitude in magnitudes
    ]


def recovery_curve(data, group, direction, magnitude):
    for index, (candidate_direction, candidate_magnitude) in enumerate(zip(
            data[group]["condition_dirs"], data[group]["condition_mags"])):
        if candidate_direction == direction and np.isclose(
                float(candidate_magnitude), magnitude):
            return data[group]["rec"][index], data[group]["rec_cnt"][index]
    raise KeyError((group, direction, magnitude))


def default_output(output_dir, root, groups, checkpoint):
    return selected_output(
        output_dir, "push_robustness.png", groups,
        ready_groups(root, DATA_FILE, checkpoint))


def configure_axis(axis):
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        axis.spines[spine].set_color(AXIS)
    axis.yaxis.grid(True, color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(color=AXIS)


def plot(root, groups, checkpoint, output):
    configure_visualization_cache()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data, command, push_every, directions, magnitudes = load_data(
        root, groups, checkpoint)
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

    direction_columns = min(4, len(directions))
    direction_rows = math.ceil(len(directions) / direction_columns)
    metric_count = 3
    total_rows = direction_rows * metric_count
    figure_height = 10.8 if direction_rows == 1 else 18.0
    fig = plt.figure(figsize=(15.0, figure_height), dpi=180)
    grid = fig.add_gridspec(
        total_rows, direction_columns, left=0.062, right=0.985,
        top=0.865, bottom=0.065, hspace=0.58, wspace=0.16)

    x_values = np.arange(len(magnitudes))
    error_max = max(
        max(series(data, group, direction, magnitudes, 5))
        for group in groups for direction in directions) * 1.15
    error_max = max(error_max, 1e-6)
    training_indices = [
        index for index, magnitude in enumerate(magnitudes) if magnitude <= 0.5
    ]
    training_end = (max(training_indices) + 0.35) if training_indices else -0.35

    metric_specs = [
        {
            "heading": f"P(fall | push) — fall before the next push ({push_every} steps)",
            "column": 2,
            "limits": (0, 1.0),
            "ylabel": "P(fall | push)",
        },
        {
            "heading": "post-push |vx − command| over 50 steps (m/s)",
            "column": 5,
            "limits": (0, error_max),
            "ylabel": "post-push |vx − command| (m/s)",
        },
        {
            "heading": (
                f"recovery — vx after a {RECOVERY_MAGNITUDE:g} m/s configured push "
                "(surviving envs)"),
            "column": None,
            "limits": None,
            "ylabel": "measured vx (m/s)",
        },
    ]

    for metric_index, metric in enumerate(metric_specs):
        metric_start_row = metric_index * direction_rows
        heading_top = grid[metric_start_row, 0].get_position(fig).y1
        fig.text(
            0.062, heading_top + 0.018, metric["heading"], fontsize=9,
            color=INK, ha="left", weight="semibold")

        for direction_index, direction in enumerate(directions):
            local_row, column_index = divmod(direction_index, direction_columns)
            row_index = metric_start_row + local_row
            axis = fig.add_subplot(grid[row_index, column_index])
            configure_axis(axis)

            if metric["column"] is not None:
                axis.set_xticks(
                    x_values, [f"{magnitude:g}" for magnitude in magnitudes])
                axis.set_xlim(-0.35, len(magnitudes) - 0.65)
                axis.set_ylim(*metric["limits"])
                axis.axvspan(-0.35, training_end, color=TRAIN_BAND, zorder=0)
                for group in groups:
                    style = GROUPS[group]
                    axis.plot(
                        x_values,
                        series(data, group, direction, magnitudes, metric["column"]),
                        color=style["color"], marker=style["marker"], lw=2, ms=6,
                        mec=SURFACE, mew=1.5, solid_capstyle="round", zorder=3)
                if local_row == direction_rows - 1:
                    axis.set_xlabel("configured push Δv (m/s)", fontsize=8, color=MUTED)
                if metric_index == 0 and direction_index == 0:
                    axis.text(
                        0.38, 0.035, "training range", transform=axis.transAxes,
                        fontsize=7, color=MUTED, ha="center")
            else:
                time_steps = np.arange(push_every)
                axis.axhline(command, color=AXIS, lw=1.2, ls=(0, (4, 3)))
                for group in groups:
                    values, counts = recovery_curve(
                        data, group, direction, RECOVERY_MAGNITUDE)
                    valid = counts > 0
                    axis.plot(
                        time_steps[valid], values[valid], color=GROUPS[group]["color"],
                        lw=1.8, solid_capstyle="round", zorder=3)
                axis.set_xlim(0, push_every - 1)
                if local_row == direction_rows - 1:
                    axis.set_xlabel("steps after push", fontsize=8, color=MUTED)
                if direction_index == 0:
                    axis.text(
                        0.98, 0.9, f"command {command:g}", transform=axis.transAxes,
                        fontsize=7, color=MUTED, ha="right")

            if column_index == 0:
                axis.set_ylabel(metric["ylabel"], fontsize=8, color=MUTED)
            else:
                axis.set_yticklabels([])
            axis.set_title(
                DIRECTION_LABELS.get(direction, direction), loc="left",
                fontsize=9.5, color=INK, pad=6)

        used_last_row = (len(directions) - 1) % direction_columns + 1
        if used_last_row < direction_columns:
            last_row = metric_start_row + direction_rows - 1
            for column_index in range(used_last_row, direction_columns):
                axis = fig.add_subplot(grid[last_row, column_index])
                axis.set_visible(False)

    handles = [
        Line2D(
            [0], [0], color=GROUPS[group]["color"], marker=GROUPS[group]["marker"],
            lw=2, ms=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=GROUPS[group]["label"])
        for group in groups
    ]
    fig.legend(
        handles=handles, frameon=False, fontsize=8, labelcolor=INK2,
        loc="upper left", bbox_to_anchor=(0.057, 0.918), ncol=min(4, len(groups)),
        handlelength=1.6, columnspacing=1.4)
    fig.text(
        0.062, 0.982, "External push robustness",
        fontsize=15, color=INK, ha="left", weight="semibold")
    fig.text(
        0.062, 0.955,
        f"groups: {', '.join(groups)} · {checkpoint} · "
        f"{len(directions)} heading-frame directions × "
        f"{len(magnitudes)} magnitudes · vx = {command:g} m/s · push interval = {push_every} steps",
        fontsize=8.5, color=MUTED, ha="left")
    fig.text(
        0.062, 0.022,
        "Shaded region: configured training scale ≤ 0.5 m/s · observations stay clean · "
        "diagonal directions currently apply (±mag, ±mag), so resultant |Δv| = √2 × displayed mag.",
        fontsize=7.5, color=MUTED, ha="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print("selected:", ", ".join(groups))
    print("conditions:", f"{len(directions)} directions × {len(magnitudes)} magnitudes")
    print("saved:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    args = parser.parse_args()
    results_root = get_results_root(args.results_root)
    configure_visualization_cache(results_root)
    root = results_root / "eval" / CATEGORY
    if args.list:
        print_push_groups(root, args.checkpoint)
        return
    try:
        groups = choose_groups(root, DATA_FILE, args.groups, args.checkpoint)
        if not groups:
            raise ValueError(f"no ready groups under {root}")
        checkpoint = common_checkpoint(
            root, groups, DATA_FILE, args.checkpoint)
        output_dir = results_root / "plots" / CATEGORY / checkpoint
        output = args.output or default_output(
            output_dir, root, groups, checkpoint)
        plot(root, groups, checkpoint, output.resolve())
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
