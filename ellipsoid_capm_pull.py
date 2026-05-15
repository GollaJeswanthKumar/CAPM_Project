"""CAPM geodesic pull for ellipsoid manifold project.

This module implements curvature-aware geodesic pulling using the exponential map.
"""

import numpy as np
from ellipsoid_manifold import normalize_point as ellipsoid_normalize, project_to_tangent, exp_map_ellipsoid
from ellipsoid_surface import signed_distance_to_equator, project_to_equator, gradient_signed_distance


def capm_pull_step(x, step_size, tau=1e-6, a=1.5, b=1.0, c=0.7):
    """Perform one step of CAPM geodesic pull toward the equator."""
    x = ellipsoid_normalize(x, a, b, c)
    f = signed_distance_to_equator(x, a, b, c)
    grad_f = gradient_signed_distance(x, a, b, c)
    norm_grad = np.linalg.norm(grad_f)
    mu = max(norm_grad, tau)
    v = -step_size * (f / mu) * grad_f
    v_tangent = project_to_tangent(x, v, a, b, c)
    return exp_map_ellipsoid(x, v_tangent, a, b, c)


def run_capm_pull(x0, num_steps, step_size, a=1.5, b=1.0, c=0.7, convergence_tolerance=None):
    """Run CAPM pull for multiple steps."""
    x = ellipsoid_normalize(x0, a, b, c)
    trajectory = [x.copy()]
    distances = [signed_distance_to_equator(x, a, b, c)]

    for _ in range(num_steps):
        x = capm_pull_step(x, step_size, a=a, b=b, c=c)
        distance = signed_distance_to_equator(x, a, b, c)
        if convergence_tolerance is not None and abs(distance) <= convergence_tolerance:
            x = project_to_equator(x, a, b, c)
            distance = signed_distance_to_equator(x, a, b, c)

        trajectory.append(x.copy())
        distances.append(distance)

        if convergence_tolerance is not None and abs(distance) <= convergence_tolerance:
            break

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
    result = run_capm_pull(x0, num_steps, step_size)

    print("\nStep | Signed Distance")
    print("-" * 20)
    for i, dist in enumerate(result['distances']):
        print(f"{i:4} | {dist:.6f}")

    final_point = result['trajectory'][-1]
    final_dist = result['distances'][-1]
    print(f"\nFinal point: {final_point}")
    print(f"Final signed distance: {final_dist}")
    print(f"Distance reduction: {result['distances'][0] - final_dist}")
