"""Simple sphere surface utilities for CAPM projects.

This module defines a target surface on the unit sphere and generates
random points near this surface for educational experiments.

The target surface is the equator of the unit sphere: z = 0.
This is one of the simplest geodesic surfaces on the sphere and provides
an easy geometric example for signed distance, gradients, and perturbations.
"""

import numpy as np


def signed_distance_to_equator(x):
    """Approximate a signed distance from x to the equator on the unit sphere.

    On the unit sphere, the equator is the set of points with z = 0.
    The simplest signed distance approximation is the z coordinate itself.

    Geometric intuition:
    - Points with z > 0 are above the equator, so the signed distance is positive.
    - Points with z < 0 are below the equator, so the signed distance is negative.
    - For unit sphere points, z is proportional to the vertical offset from the equatorial plane.

    Args:
        x: array-like of shape (3,), a point on the sphere.

    Returns:
        float, signed distance approximation from x to the equator.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("Input x must be a 3-dimensional vector.")
    return x[2]


def gradient_signed_distance(x):
    """Return the gradient direction of the signed distance function.

    For the equator signed distance function f(x) = z, the gradient in R^3 is [0, 0, 1].
    This gradient points in the direction that increases the signed distance.

    Geometric intuition:
    - The gradient tells us which direction moves a point upward faster.
    - In this simple case, the gradient is the vertical axis.
    - For pulling a point toward the equator, we want the opposite direction if the point is above.

    Args:
        x: array-like of shape (3,), a point in R^3.

    Returns:
        np.ndarray of shape (3,), gradient of the signed distance.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("Input x must be a 3-dimensional vector.")
    return np.array([0.0, 0.0, 1.0])


def generate_random_sphere_points(n):
    """Generate n random normalized points on the unit sphere.

    Uses a simple sampling technique based on normal distributions.
    Each point is drawn from a 3D Gaussian and then normalized to unit length.

    Args:
        n: int, the number of random points to generate.

    Returns:
        np.ndarray of shape (n, 3), points on the unit sphere.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    points = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def generate_near_equator_points(n, noise_level=0.1):
    """Generate random points near the equator with small perturbations.

    This creates points that are mostly on the equator but perturbed slightly
    in the vertical direction and around the circle. The points remain normalized.

    Args:
        n: int, number of points to generate.
        noise_level: float, maximum magnitude of vertical perturbation.

    Returns:
        np.ndarray of shape (n, 3), points near the equator.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative.")

    angles = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    x = np.cos(angles)
    y = np.sin(angles)
    z = np.random.uniform(-noise_level, noise_level, size=n)
    points = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def equator_visualization_data(n=100, noise_level=0.05):
    """Prepare points for visualization of the equator and nearby samples.

    Returns a dictionary that can be used by plotting code.
    """
    equator_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    equator_points = np.stack([np.cos(equator_angles), np.sin(equator_angles), np.zeros(n)], axis=1)
    near_equator = generate_near_equator_points(n, noise_level=noise_level)
    random_points = generate_random_sphere_points(n)
    return {
        "equator": equator_points,
        "random": random_points,
        "near_equator": near_equator,
    }


if __name__ == "__main__":
    # Basic tests and demonstration of the equator utilities.
    point_above = np.array([1.0, 0.0, 0.5])
    point_below = np.array([0.0, 1.0, -0.3])
    print("Signed distance above equator:", signed_distance_to_equator(point_above))
    print("Signed distance below equator:", signed_distance_to_equator(point_below))
    print("Gradient of signed distance:", gradient_signed_distance(point_above))

    random_points = generate_random_sphere_points(5)
    print("Random sphere points:\n", random_points)

    near_equator_points = generate_near_equator_points(5, noise_level=0.1)
    print("Near-equator points:\n", near_equator_points)
    print("Near-equator z values:", near_equator_points[:, 2])

    # Visualization-ready output example.
    sample_data = {
        "random": random_points,
        "near_equator": near_equator_points,
    }
    print("Visualization-ready keys:", list(sample_data.keys()))
