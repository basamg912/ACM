"""Visualize selected single-joint dropout evaluation groups.

  python script/vis/dropout_viz_cur.py
  python script/vis/dropout_viz_cur.py --groups baseline v3 v4
  python script/vis/dropout_viz_cur.py --checkpoint model_30000 --list
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


CATEGORY = "joint_dropout"
DATA_FILE = Path("dropout.npz")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS = "#c3c2b7"
JOINTS = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch",
    "L_ankle_roll", "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee",
    "R_ankle_pitch", "R_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    # "L_sh_pitch",
]
CONDITIONS = [
    # ("zero", "pos+vel"),
    # ("freeze", "pos+vel"),
    ("zero", "pos"), ("zero", "vel"),
    ("freeze", "pos"), ("freeze", "vel"),
]
CONDITION_LABELS = [channel for _, channel in CONDITIONS]


def condition_mode_sections():
    """Return contiguous mode blocks as (mode, start, end-exclusive)."""
    sections = []
    for index, (mode, _) in enumerate(CONDITIONS):
        if not sections or sections[-1][0] != mode:
            sections.append([mode, index, index + 1])
        else:
            sections[-1][2] = index + 1
    return sections


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
            (row[0], row[1], row[2]): row for row in raw["summary"]
        }
        command = group_command
    return tables, command


def metric_grid(table, column):
    return np.array([
        [float(table[(mode, channel, joint)][column]) for mode, channel in CONDITIONS]
        for joint in JOINTS
    ])


def plot(root, groups, checkpoint, output):
    configure_visualization_cache()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list(
        "severity", ["#f6efe8", "#f3b48f", "#eb6834", "#8f2f0d"])
    tables, command = load_data(root, groups, checkpoint)
    falls = {group: metric_grid(tables[group], 3) for group in groups}
    speed = {
        group: np.abs(metric_grid(tables[group], 4) - command)
        for group in groups
    }
    clean = {
        group: (
            float(tables[group][("clean", "none", "none")][3]),
            abs(float(tables[group][("clean", "none", "none")][4]) - command),
        )
        for group in groups
    }

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
    })
    figure_width = max(9.0, 5.2 * len(groups))
    fig = plt.figure(figsize=(figure_width, 12.2), dpi=180)
    grid_spec = fig.add_gridspec(
        2, len(groups), left=0.10, right=0.92, top=0.835, bottom=0.075,
        hspace=0.20, wspace=0.10)
    metrics = [
        ("falls per 1k env-steps", falls, "{:.0f}", 0.6),
        ("|measured vx − command| (m/s)", speed, "{:.2f}", 0.25),
    ]

    for row_index, (metric_label, data, number_format, hide_below) in enumerate(metrics):
        value_max = max(matrix.max() for matrix in data.values())
        value_max = max(value_max, 1e-6)
        row_axes = []
        for column_index, group in enumerate(groups):
            axis = fig.add_subplot(grid_spec[row_index, column_index])
            row_axes.append(axis)
            matrix = data[group]
            axis.imshow(matrix, cmap=ramp, vmin=0, vmax=value_max, aspect="auto")
            axis.set_xticks(range(len(CONDITIONS)), CONDITION_LABELS, fontsize=8)
            axis.set_yticks(range(len(JOINTS)))
            axis.set_yticklabels(JOINTS if column_index == 0 else [], fontsize=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.tick_params(length=0)
            axis.set_xticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(JOINTS), 1), minor=True)
            axis.grid(which="minor", color=SURFACE, lw=1.6)
            axis.tick_params(which="minor", length=0)
            for joint_index in range(matrix.shape[0]):
                for condition_index in range(matrix.shape[1]):
                    value = matrix[joint_index, condition_index]
                    if value < hide_below:
                        continue
                    axis.text(
                        condition_index, joint_index, number_format.format(value),
                        ha="center", va="center", fontsize=6.8,
                        color=SURFACE if value > value_max * 0.55 else INK2)
            sections = condition_mode_sections()
            for _, _, end in sections[:-1]:
                axis.axvline(end - 0.5, color=SURFACE, lw=3)
            for mode, start, end in sections:
                center = (start + end - 1) / 2
                axis.text(
                    center, -0.95, mode, ha="center", fontsize=8.5,
                    color=INK2, weight="semibold")
            axis.set_title(
                f"{GROUPS[group]['label']} · {metric_label}", loc="left",
                fontsize=10.5, color=INK, pad=22)
            axis.text(
                1.0, 1.006, f"no failure: {number_format.format(clean[group][row_index])}",
                transform=axis.transAxes, fontsize=7.5, color=MUTED,
                ha="right", va="bottom")

        right_position = row_axes[-1].get_position(fig)
        color_axis = fig.add_axes([
            0.935, right_position.y0, 0.010, right_position.height
        ])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=ramp, norm=plt.Normalize(0, value_max)),
            cax=color_axis)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=7.5, color=MUTED)

    fig.text(
        0.10, 0.955, "Single-joint sensor dropout robustness",
        fontsize=15, color=INK, ha="left", weight="semibold")
    fig.text(
        0.10, 0.925,
        f"groups: {', '.join(groups)} · {checkpoint} · vx = {command:g} m/s · "
        "one joint observation is broken "
        "after a clean warm-up",
        fontsize=8.5, color=MUTED, ha="left")
    fig.text(
        0.10, 0.895,
        "zero = reading fixed at 0 · freeze = reading held at its failure-time value · "
        "all panels in a metric row share one color scale",
        fontsize=8.5, color=MUTED, ha="left")
    fig.text(
        0.10, 0.022,
        "Falls alone understate damage; the lower row reports command-speed error. "
        "Combined pos+vel failures and the L_sh_pitch control joint are excluded from this view.",
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
            output_dir, "dropout_robustness.png", groups, available)
        plot(root, groups, checkpoint, output.resolve())
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
