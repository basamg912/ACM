"""Shared result discovery and group selection for visualization scripts."""
import sys
from pathlib import Path


# Keep this order stable: terminal selection numbers refer to it.
GROUPS = {
    "baseline": {"label": "Baseline", "color": "#2a78d6", "marker": "o"},
    "v2": {"label": "V2 · HistEnc MSE", "color": "#27945b", "marker": "s"},
    "v3": {"label": "V3 · HistEnc Contrastive", "color": "#eb6834", "marker": "^"},
    "v3_wodetach": {"label": "V3 · no detach", "color": "#c83e4d", "marker": "v"},
    "v4": {"label": "V4 · Privileged", "color": "#616161", "marker": "D"},
    "v4_cont": {"label": "V4 · Privileged Contrastive", "color": "#0095c2", "marker": "P"},
    "v4_wodetach": {"label": "V4 · no detach", "color": "#8055a6", "marker": "X"},
    "v4_wodetach_cont": {"label": "V4 · no detach · cont", "color": "#bc70ff", "marker": "X"},
}


def _checkpoint_name(checkpoint):
    path = Path(checkpoint)
    return path.stem if path.suffix else path.name


def _checkpoint_sort_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return (1, int(suffix)) if suffix.isdigit() else (0, name)


def available_checkpoints(root, group, relative_path):
    group_root = Path(root) / group
    if not group_root.is_dir():
        return []
    return sorted(
        (
            child.name
            for child in group_root.iterdir()
            if child.is_dir() and (child / relative_path).is_file()
        ),
        key=_checkpoint_sort_key,
    )


def data_path(root, group, relative_path, checkpoint=None):
    group_root = Path(root) / group
    if checkpoint is not None:
        return group_root / _checkpoint_name(checkpoint) / relative_path
    checkpoints = available_checkpoints(root, group, relative_path)
    checkpoint_name = checkpoints[-1] if checkpoints else "<latest>"
    return group_root / checkpoint_name / relative_path


def ready_groups(root, relative_path, checkpoint=None):
    return [
        group for group in GROUPS
        if data_path(root, group, relative_path, checkpoint).is_file()
    ]


def print_groups(root, relative_path, checkpoint=None):
    checkpoint_label = _checkpoint_name(checkpoint) if checkpoint else "latest available"
    print(f"Available visualization groups ({relative_path}, {checkpoint_label}):")
    for index, (group, style) in enumerate(GROUPS.items(), start=1):
        path = data_path(root, group, relative_path, checkpoint)
        status = f"ready: {path.parent.name}" if path.is_file() else "missing"
        print(f"  {index}. {group:<16} {style['label']:<30} [{status}]")


def parse_group_selection(tokens, root, relative_path, checkpoint=None):
    parts = []
    for token in tokens:
        parts.extend(part for part in token.replace(",", " ").split() if part)

    available = ready_groups(root, relative_path, checkpoint)
    if not parts or (len(parts) == 1 and parts[0].lower() == "all"):
        return available

    ordered_groups = list(GROUPS)
    selected = []
    for part in parts:
        if part.isdigit():
            index = int(part) - 1
            if not 0 <= index < len(ordered_groups):
                raise ValueError(f"group number out of range: {part}")
            group = ordered_groups[index]
        else:
            group = part.lower()
            if group not in GROUPS:
                raise ValueError(
                    f"unknown group '{part}' (choose from: {', '.join(ordered_groups)})")
        path = data_path(root, group, relative_path, checkpoint)
        if not path.is_file():
            raise ValueError(f"{group}: {path} does not exist")
        if group not in selected:
            selected.append(group)
    return selected


def choose_groups(root, relative_path, cli_groups, checkpoint=None):
    if cli_groups is not None:
        return parse_group_selection(
            cli_groups, root, relative_path, checkpoint)

    if not sys.stdin.isatty():
        return ready_groups(root, relative_path, checkpoint)

    print_groups(root, relative_path, checkpoint)
    raw = input("Select groups by number/name (comma separated, Enter=all ready): ").strip()
    return parse_group_selection(
        [raw] if raw else [], root, relative_path, checkpoint)


def common_checkpoint(root, groups, relative_path, checkpoint=None):
    """Return one checkpoint shared by every selected group."""
    if checkpoint is not None:
        checkpoint_name = _checkpoint_name(checkpoint)
        missing = [
            group for group in groups
            if not data_path(root, group, relative_path, checkpoint_name).is_file()
        ]
        if missing:
            raise ValueError(
                f"{checkpoint_name} is missing for: {', '.join(missing)}")
        return checkpoint_name

    shared = None
    for group in groups:
        current = set(available_checkpoints(root, group, relative_path))
        shared = current if shared is None else shared.intersection(current)
    if not shared:
        raise ValueError(
            "selected groups have no common checkpoint; use --list and "
            "--checkpoint to choose matching results")
    return max(shared, key=_checkpoint_sort_key)


def default_output(directory, base_name, groups, available):
    base = Path(base_name)
    if groups == available:
        filename = base.name
    else:
        filename = f"{base.stem}_{'-'.join(groups)}{base.suffix}"
    return Path(directory) / filename


def add_selection_arguments(parser):
    parser.add_argument(
        "-g", "--groups", nargs="+", metavar="GROUP",
        help="group names or menu numbers; comma-separated values and 'all' are supported")
    parser.add_argument("-o", "--output", type=Path, help="output PNG path")
    parser.add_argument(
        "--checkpoint",
        help="checkpoint directory/name (for example model_30000); default: latest common")
    parser.add_argument(
        "--results-root", type=Path,
        help="result root; default: ACM_RESULTS_ROOT or <ACM>/results")
    parser.add_argument("--list", action="store_true", help="list groups and exit")
