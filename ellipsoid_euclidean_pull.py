"""Euclidean pull baseline for CAPM ellipsoid manifold project.

This module implements a naive Euclidean gradient descent approach
for pulling points toward the equator on the ellipsoid manifold.
"""

import numpy as np
from ellipsoid_manifold import normalize_point as ellipsoid_normalize
from ellipsoid_surface import signed_distance_to_equator, gradient_signed_distance


def euclidean_pull_step(x, step_size, a=1.5, b=1.0, c=0.7):
    """Perform one step of Euclidean gradient descent toward the equator."""
    x = np.asarray(x, dtype=float)
    grad = gradient_signed_distance(x, a, b, c)
    x_new = x - step_size * grad
    return ellipsoid_normalize(x_new, a, b, c)


def run_euclidean_pull(x0, num_steps, step_size, a=1.5, b=1.0, c=0.7):
    """Run Euclidean pull for multiple steps."""
    x = ellipsoid_normalize(x0, a, b, c)
    trajectory = [x.copy()]
    distances = [signed_distance_to_equator(x, a, b, c)]

    for _ in range(num_steps):
        x = euclidean_pull_step(x, step_size, a, b, c)
        trajectory.append(x.copy())
        distances.append(signed_distance_to_equator(x, a, b, c))

    return {
        'trajectory': trajectory,
        'distances': distances,
    }


if __name__ == "__main__":
    from ellipsoid_surface import generate_near_equator_points

    initial_points = generate_near_equator_points(1, noise_level=0.2)
    x0 = initial_points[0]
    print("Initial point:", x0)
    print("Initial signed distance:", signed_distance_to_equator(x0))

    num_steps = 10
    step_size = 0.1
    result = run_euclidean_pull(x0, num_steps, step_size)

    print("\nStep | Signed Distance")
    print("-" * 20)
    for i, dist in enumerate(result['distances']):
        print(f"{i:4} | {dist:.6f}")

    final_point = result['trajectory'][-1]
    final_dist = result['distances'][-1]
    print(f"\nFinal point: {final_point}")
    print(f"Final signed distance: {final_dist}")
    print(f"Distance reduction: {result['distances'][0] - final_dist}")
