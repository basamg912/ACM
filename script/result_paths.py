import os
import re
from pathlib import Path


ACM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ACM_ROOT / "results"


def get_results_root(override=None):
    """Return and create the configured ACM result root."""
    value = override or os.environ.get("ACM_RESULTS_ROOT")
    root = Path(value).expanduser() if value else DEFAULT_RESULTS_ROOT
    if not root.is_absolute():
        root = ACM_ROOT / root
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_component(value):
    component = re.sub(r"[^\w.-]+", "_", str(value).strip())
    return component.strip("._") or "unnamed"


def checkpoint_result_dir(checkpoint, category, results_root=None, timestamp=None):
    """Create a result directory keyed by category, run, and checkpoint."""
    checkpoint = Path(checkpoint).expanduser()
    root = get_results_root(results_root)
    output = (
        root
        / "eval"
        / _safe_component(category)
        / _safe_component(checkpoint.parent.name)
        / _safe_component(checkpoint.stem)
    )
    if timestamp is not None:
        output /= _safe_component(timestamp)
    output.mkdir(parents=True, exist_ok=True)
    return output


def checkpoint_result_path(
    checkpoint, category, filename, results_root=None, timestamp=None
):
    return checkpoint_result_dir(
        checkpoint,
        category,
        results_root=results_root,
        timestamp=timestamp,
    ) / filename


def plot_result_path(input_path, plot_kind, filename, results_root=None):
    """Place plots under the same run/checkpoint identity as eval input data."""
    input_path = Path(input_path).expanduser().resolve()
    root = get_results_root(results_root)
    eval_root = (root / "eval").resolve()

    try:
        relative = input_path.relative_to(eval_root)
    except ValueError:
        relative = None

    if relative is not None and len(relative.parts) >= 4:
        run_name = relative.parts[1]
        checkpoint_name = relative.parts[2]
    else:
        parent = input_path.parent
        run_name = parent.parent.name if parent.name in {"latents", "obs_stats"} else parent.name
        checkpoint_name = input_path.stem

    output = (
        root
        / "plots"
        / _safe_component(plot_kind)
        / _safe_component(run_name)
        / _safe_component(checkpoint_name)
        / filename
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
