"""Main entry point for the CAPM sphere manifold experiment.

This script runs the full comparison between naive Euclidean pull and
curvature-aware CAPM pull on the unit sphere, then visualizes the results.
"""

from surface import generate_near_equator_points, signed_distance_to_equator
from euclidean_pull import run_euclidean_pull
from capm_pull import run_capm_pull
from visualization import plot_sphere_and_trajectories, plot_convergence


def main():
    print("\n=== CAPM Sphere Manifold Experiment ===\n")

    print("[1] Generating initial near-equator point")
    x0 = generate_near_equator_points(1, noise_level=0.2)[0]
    initial_distance = signed_distance_to_equator(x0)
    print(f"Initial point: {x0}")
    print(f"Initial signed distance to equator: {initial_distance:.6f}\n")

    print("[2] Running Euclidean pull baseline")
    num_steps = 20
    step_size = 0.1
    euclidean_result = run_euclidean_pull(x0, num_steps, step_size)
    euclidean_final = euclidean_result['distances'][-1]
    print(f"Final Euclidean signed distance: {euclidean_final:.6f}\n")

    print("[3] Running CAPM geodesic pull")
    capm_result = run_capm_pull(x0, num_steps, step_size)
    capm_final = capm_result['distances'][-1]
    print(f"Final CAPM signed distance: {capm_final:.6f}\n")

    print("[4] Comparing convergence behavior")
    print(f"Euclidean distance change: {abs(initial_distance - euclidean_final):.6f}")
    print(f"CAPM distance change:      {abs(initial_distance - capm_final):.6f}\n")

    print("[5] Launching visualization")
    plot_sphere_and_trajectories(euclidean_result, capm_result, x0)
    plot_convergence(euclidean_result, capm_result)

    print("\nExperiment complete. Two plot files have been generated:")
    print("- sphere_trajectories.png")
    print("- convergence_comparison.png")
    print("\nThe 3D plot shows how each method moves on the sphere, and the convergence plot")
    print("shows how the absolute distance to the equator evolves over iterations.")


if __name__ == "__main__":
    main()
