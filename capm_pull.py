"""CAPM geodesic pull for sphere manifold project.

This module implements curvature-aware geodesic pulling using the exponential map.
It provides a geometrically correct way to move points toward the equator on the sphere.

Why exponential map matters:
- The exponential map moves points along geodesics, which are the shortest paths on the sphere.
- Unlike Euclidean updates, it respects the manifold's curvature and keeps points on the surface.
- This leads to more efficient and stable convergence.

Why tangent projection matters:
- Directions must lie in the tangent space to be valid for movement on the surface.
- Projecting onto the tangent plane ensures the update direction is perpendicular to the normal.
- Without projection, the direction might not be tangent, leading to incorrect paths.

Why this is more geometrically correct than Euclidean pull:
- Euclidean pull ignores curvature and can cause points to oscillate or diverge.
- CAPM uses manifold-aware operations: tangent projection and exponential map.
- This results in smoother trajectories and better convergence to the target surface.
"""

import numpy as np
from manifold import normalize_point, project_to_tangent, exp_map_sphere
from surface import signed_distance_to_equator, gradient_signed_distance


def capm_pull_step(x, step_size, tau=1e-6):
    """Perform one step of CAPM geodesic pull toward the equator.

    This computes a tangent direction based on the signed distance and gradient,
    then moves using the exponential map to stay on the sphere.

    Args:
        x: array-like of shape (3,), current point on the sphere.
        step_size: float, learning rate for the pull step.
        tau: float, small regularization parameter to avoid division by zero.

    Returns:
        np.ndarray of shape (3,), updated point on the sphere.
    """
    x = normalize_point(x)
    f = signed_distance_to_equator(x)
    grad_f = gradient_signed_distance(x)
    norm_grad = np.linalg.norm(grad_f)
    mu = max(norm_grad, tau)
    # Compute the tangent update direction.
    v = -step_size * (f / mu) * grad_f
    # Project onto tangent space.
    v_tangent = project_to_tangent(x, v)
    # Move using exponential map.
    return exp_map_sphere(x, v_tangent)


def run_capm_pull(x0, num_steps, step_size):
    """Run CAPM pull for multiple steps and track the trajectory.

    This iteratively applies the CAPM pull step and records
    the position and signed distance at each step.

    Args:
        x0: array-like of shape (3,), initial point on the sphere.
        num_steps: int, number of pull steps.
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
        x = capm_pull_step(x, step_size)
        trajectory.append(x.copy())
        distances.append(signed_distance_to_equator(x))

    return {
        'trajectory': trajectory,
        'distances': distances,
    }


if __name__ == "__main__":
    # Test the CAPM pull on a random near-equator point.
    from surface import generate_near_equator_points

    # Generate a single random point near the equator.
    initial_points = generate_near_equator_points(1, noise_level=0.2)
    x0 = initial_points[0]
    print("Initial point:", x0)
    print("Initial signed distance:", signed_distance_to_equator(x0))

    # Run CAPM pull for 10 steps with step size 0.1.
    num_steps = 10
    step_size = 0.1
    result = run_capm_pull(x0, num_steps, step_size)

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
