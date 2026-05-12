"""Visualization for CAPM sphere manifold project.

This module creates 3D and convergence plots to compare Euclidean pull vs CAPM pull.
It demonstrates the geometric differences between naive and curvature-aware methods.

What the plots mean:
- The 3D plot shows trajectories on the sphere: Euclidean pull may zigzag or leave the surface,
  while CAPM follows smooth geodesic paths toward the equator.
- The convergence plot shows how the absolute distance to the equator decreases over iterations.
  CAPM typically converges faster and more stably due to respecting curvature.

How curvature-aware movement differs:
- Euclidean pull treats the sphere as flat space, leading to inefficient, oscillatory paths.
- CAPM uses tangent projections and exponential maps for geometrically correct motion.

Why geodesic motion is important:
- Geodesics are the shortest paths on the manifold, ensuring efficient convergence.
- They prevent drifting off the surface and provide stable, predictable behavior.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from surface import generate_near_equator_points
from euclidean_pull import run_euclidean_pull
from capm_pull import run_capm_pull


def plot_sphere_and_trajectories(euclidean_result, capm_result, x0):
    """Create 3D plot of sphere with trajectories."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the unit sphere.
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.3)

    # Draw the equator (z=0).
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 'k-', linewidth=2, label='Equator')

    # Plot trajectories.
    euclidean_traj = np.array(euclidean_result['trajectory'])
    capm_traj = np.array(capm_result['trajectory'])
    ax.plot(euclidean_traj[:, 0], euclidean_traj[:, 1], euclidean_traj[:, 2],
            'r-o', markersize=4, label='Euclidean Pull')
    ax.plot(capm_traj[:, 0], capm_traj[:, 1], capm_traj[:, 2],
            'g-o', markersize=4, label='CAPM Pull')

    # Plot starting point.
    ax.scatter(x0[0], x0[1], x0[2], color='blue', s=100, marker='*', label='Start')

    # Plot final points.
    euclidean_final = euclidean_traj[-1]
    capm_final = capm_traj[-1]
    ax.scatter(euclidean_final[0], euclidean_final[1], euclidean_final[2],
               color='red', s=100, marker='X', label='Euclidean Final')
    ax.scatter(capm_final[0], capm_final[1], capm_final[2],
               color='green', s=100, marker='X', label='CAPM Final')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sphere Trajectories: Euclidean vs CAPM Pull')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.savefig('sphere_trajectories.png')


def plot_convergence(euclidean_result, capm_result):
    """Create convergence plot comparing distances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    steps = range(len(euclidean_result['distances']))
    euclidean_abs_dist = np.abs(euclidean_result['distances'])
    capm_abs_dist = np.abs(capm_result['distances'])

    ax.plot(steps, euclidean_abs_dist, 'r-o', label='Euclidean Pull')
    ax.plot(steps, capm_abs_dist, 'g-o', label='CAPM Pull')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Absolute Signed Distance to Equator')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True)
    plt.savefig('convergence_comparison.png')


if __name__ == "__main__":
    print("Starting visualization...")
    # Generate one random near-equator point.
    initial_points = generate_near_equator_points(1, noise_level=0.2)
    x0 = initial_points[0]
    print("Starting point:", x0)

    # Run both pulls.
    num_steps = 20
    step_size = 0.1
    euclidean_result = run_euclidean_pull(x0, num_steps, step_size)
    capm_result = run_capm_pull(x0, num_steps, step_size)

    # Create plots.
    plot_sphere_and_trajectories(euclidean_result, capm_result, x0)
    plot_convergence(euclidean_result, capm_result)

    print("Plots generated. Euclidean final distance:", np.abs(euclidean_result['distances'][-1]))
    print("CAPM final distance:", np.abs(capm_result['distances'][-1]))
