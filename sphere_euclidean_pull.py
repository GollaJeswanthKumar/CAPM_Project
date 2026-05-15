"""Euclidean pull baseline for CAPM sphere manifold project.

This module implements a naive Euclidean gradient descent approach
for pulling points toward the equator on the unit sphere. This serves
as a baseline comparison to geometrically correct manifold methods.

Why this is NOT geometrically correct:
- Euclidean gradient descent treats the sphere as flat space.
- Moving along the gradient in R^3 can leave the sphere's surface.
- The resulting path is not a geodesic and may not converge properly.
- On curved manifolds, straight-line motion in embedding space is incorrect.

Why normalization is needed:
- After Euclidean update, the point may no longer have unit length.
- Normalization projects the point back onto the sphere.
- Without it, the point drifts away from the manifold.

Potential errors on curved spaces:
- Points can oscillate or diverge instead of converging.
- The method ignores the sphere's curvature, leading to inefficient paths.
- Convergence may be slow or unstable near the equator.
"""

import numpy as np
from sphere_manifold import normalize_point
from sphere_surface import signed_distance_to_equator, gradient_signed_distance


def euclidean_pull_step(x, step_size):
    """Perform one step of Euclidean gradient descent toward the equator.

    This computes the gradient of the signed distance function and moves
    the point in the opposite direction in R^3, then renormalizes.

    Args:
        x: array-like of shape (3,), current point on the sphere.
        step_size: float, learning rate for the gradient descent step.

    Returns:
        np.ndarray of shape (3,), updated point on the sphere.
    """
    x = np.asarray(x, dtype=float)
    grad = gradient_signed_distance(x)
    # Move in the direction opposite to the gradient (toward equator).
    x_new = x - step_size * grad
    # Renormalize to stay on the sphere.
    return normalize_point(x_new)


def run_euclidean_pull(x0, num_steps, step_size):
    """Run Euclidean pull for multiple steps and track the trajectory.

    This iteratively applies the Euclidean pull step and records
    the position and signed distance at each step.

    Args:
        x0: array-like of shape (3,), initial point on the sphere.
        num_steps: int, number of gradient descent steps.
        step_size: float, learning rate.

    Returns:
        dict with keys:
            'trajectory': list of points (each shape (3,)),
            'distances': list of signed distances (floats).
    """
    x = normalize_point(x0)
    trajectory = [x.copy()]
    distances = [signed_distance_to_equator(x)]

    for _ in range(num_steps):
        x = euclidean_pull_step(x, step_size)
        trajectory.append(x.copy())
        distances.append(signed_distance_to_equator(x))

    return {
        'trajectory': trajectory,
        'distances': distances,
    }


if __name__ == "__main__":
    # Test the Euclidean pull on a random near-equator point.
    from sphere_surface import generate_near_equator_points

    # Generate a single random point near the equator.
    initial_points = generate_near_equator_points(1, noise_level=0.2)
    x0 = initial_points[0]
    print("Initial point:", x0)
    print("Initial signed distance:", signed_distance_to_equator(x0))

    # Run Euclidean pull for 10 steps with step size 0.1.
    num_steps = 10
    step_size = 0.1
    result = run_euclidean_pull(x0, num_steps, step_size)

    # Print distances over time.
    print("\nStep | Signed Distance")
    print("-" * 20)
    for i, dist in enumerate(result['distances']):
        print(f"{i:4} | {dist:.6f}")


    # Show final point and convergence.
    final_point = result['trajectory'][-1]
    final_dist = result['distances'][-1]
    print(f"\nFinal point: {final_point}")
    print(f"Final signed distance: {final_dist}")
    print(f"Distance reduction: {result['distances'][0] - final_dist}")
