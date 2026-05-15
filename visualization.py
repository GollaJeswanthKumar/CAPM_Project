"""Visualization utilities for the CAPM manifold comparison project."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ELLIPSOID_AXES, PROJECT_DIR, RESULTS_DIR, SPHERE_RADIUS
from experiment import iter_experiment_cases, run_experiment_suite


METHOD_STYLES = {
    "euclidean": {
        "label": "Euclidean pull",
        "color": "#d62828",
        "marker": "o",
    },
    "capm": {
        "label": "CAPM pull",
        "color": "#0077b6",
        "marker": "^",
    },
}


def _resolve_output_dir(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _surface_mesh(manifold, resolution=80):
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)

    if manifold == "sphere":
        a = b = c = SPHERE_RADIUS
    elif manifold == "ellipsoid":
        a, b, c = ELLIPSOID_AXES
    else:
        raise ValueError(f"Unsupported manifold: {manifold}")

    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _equator_curve(manifold, samples=240):
    theta = np.linspace(0.0, 2.0 * np.pi, samples)

    if manifold == "sphere":
        a = b = SPHERE_RADIUS
    elif manifold == "ellipsoid":
        a, b, _ = ELLIPSOID_AXES
    else:
        raise ValueError(f"Unsupported manifold: {manifold}")

    return a * np.cos(theta), b * np.sin(theta), np.zeros_like(theta)


def _surface_style(manifold):
    if manifold == "sphere":
        return {"color": "#8ecae6", "alpha": 0.28}
    return {"color": "#90be6d", "alpha": 0.32}


def _set_manifold_axes(ax, manifold):
    if manifold == "sphere":
        radius = SPHERE_RADIUS * 1.15
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
    elif manifold == "ellipsoid":
        a, b, c = ELLIPSOID_AXES
        ax.set_xlim(-1.15 * a, 1.15 * a)
        ax.set_ylim(-1.15 * b, 1.15 * b)
        ax.set_zlim(-1.15 * c, 1.15 * c)
        ax.set_box_aspect((a, b, c))
    else:
        raise ValueError(f"Unsupported manifold: {manifold}")

    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.25)


def plot_trajectory_comparison(case_result, output_path):
    """Create one comparison plot for a manifold/start-point pair."""
    manifold = case_result["manifold"]
    start_label = case_result["start_label"]
    output_path = Path(output_path)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    x_surf, y_surf, z_surf = _surface_mesh(manifold)
    ax.plot_surface(
        x_surf,
        y_surf,
        z_surf,
        linewidth=0,
        shade=True,
        **_surface_style(manifold),
    )

    x_eq, y_eq, z_eq = _equator_curve(manifold)
    ax.plot(x_eq, y_eq, z_eq, color="black", linewidth=2.6, label="Target equator")

    start = case_result["start"]
    ax.scatter(
        start[0],
        start[1],
        start[2],
        color="#ffb703",
        edgecolor="black",
        s=90,
        marker="*",
        label="Start",
    )

    for method in ("euclidean", "capm"):
        style = METHOD_STYLES[method]
        trajectory = np.asarray(case_result[method]["trajectory"], dtype=float)
        marker_interval = max(1, len(trajectory) // 8)

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=style["color"],
            linewidth=2.0,
            marker=style["marker"],
            markevery=marker_interval,
            markersize=4,
            label=style["label"],
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            trajectory[-1, 2],
            color=style["color"],
            edgecolor="black",
            s=65,
            marker="X",
        )

    _set_manifold_axes(ax, manifold)
    ax.set_title(f"{manifold.title()} manifold: Euclidean vs CAPM pull (Start {start_label})")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_summary(results, output_path):
    """Create a compact convergence summary for every experiment case."""
    cases = list(iter_experiment_cases(results))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = np.asarray(axes).ravel()

    for ax, case_result in zip(axes, cases):
        for method in ("euclidean", "capm"):
            style = METHOD_STYLES[method]
            distances = np.abs(np.asarray(case_result[method]["distances"], dtype=float))
            steps = np.arange(len(distances))
            ax.plot(
                steps,
                distances,
                color=style["color"],
                marker=style["marker"],
                markersize=3,
                linewidth=1.8,
                label=style["label"],
            )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"{case_result['manifold'].title()} - Start {case_result['start_label']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|signed distance to equator|")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Convergence Toward the Equator", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_visualizations(results=None, output_dir=RESULTS_DIR):
    """Generate polished trajectory and convergence visualizations."""
    if results is None:
        results = run_experiment_suite()

    output_dir = _resolve_output_dir(output_dir)
    generated_files = []

    for case_result in iter_experiment_cases(results):
        output_path = output_dir / (
            f"{case_result['manifold']}_start_{case_result['start_label']}_euclidean_vs_capm.png"
        )
        plot_trajectory_comparison(case_result, output_path)
        generated_files.append(output_path)
        print(f"Generated: {output_path.relative_to(PROJECT_DIR)}")

    convergence_path = output_dir / "distance_convergence_summary.png"
    plot_convergence_summary(results, convergence_path)
    generated_files.append(convergence_path)
    print(f"Generated: {convergence_path.relative_to(PROJECT_DIR)}")

    return generated_files


if __name__ == "__main__":
    run_visualizations()
