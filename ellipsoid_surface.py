"""Simple ellipsoid surface utilities for CAPM projects.

This module defines a target surface on the ellipsoid and generates
random points near this surface for educational experiments.

The target surface is the 'equator' of the ellipsoid: z = 0.
This provides an easy geometric example for signed distance, gradients, and perturbations.
"""

import numpy as np
from ellipsoid_manifold import normalize_point as ellipsoid_normalize


def signed_distance_to_equator(x, a=1.5, b=1.0, c=0.7):
    """Approximate signed distance from x to the equator on the ellipsoid.

    The equator is defined as z = 0. Since points are on the ellipsoid,
    we use the z-coordinate as a simple approximation.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("Input x must be a 3-dimensional vector.")
    return x[2]


def project_to_equator(x, a=1.5, b=1.0, c=0.7):
    """Project a near-equator ellipsoid point exactly onto z = 0."""
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("Input x must be a 3-dimensional vector.")

    scale = np.sqrt((x[0] / a) ** 2 + (x[1] / b) ** 2)
    if scale == 0.0:
        raise ValueError("Cannot project a pole point onto a unique equator point.")
    return np.array([x[0] / scale, x[1] / scale, 0.0])


def gradient_signed_distance(x, a=1.5, b=1.0, c=0.7):
    """Gradient of the signed distance function on the ellipsoid.

    For f(x) = z, the gradient is [0, 0, 1].
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("Input x must be a 3-dimensional vector.")
    return np.array([0.0, 0.0, 1.0])


def generate_random_ellipsoid_points(n, a=1.5, b=1.0, c=0.7):
    """Generate n random normalized points on the ellipsoid.

    Uses rejection sampling or scaling from Gaussian.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    points = []
    while len(points) < n:
        # Sample from Gaussian and scale
        p = np.random.normal(size=3)
        p_norm = ellipsoid_normalize(p, a, b, c)
        points.append(p_norm)
    return np.array(points)


def generate_near_equator_points(n, noise_level=0.1, a=1.5, b=1.0, c=0.7):
    """Generate points near the equator with small perturbations.

    Creates points mostly on z=0 with noise, then normalizes to ellipsoid.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative.")

    angles = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    # Approximate equatorial points
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    z = np.random.uniform(-noise_level, noise_level, size=n)
    points = np.stack([x, y, z], axis=1)
    # Normalize to ellipsoid
    normalized = []
    for p in points:
        normalized.append(ellipsoid_normalize(p, a, b, c))
    return np.array(normalized)


def ellipsoid_visualization_data(n=100, noise_level=0.05, a=1.5, b=1.0, c=0.7):
    """Prepare points for visualization of the ellipsoid and nearby samples."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    equator_points = np.stack([a * np.cos(angles), b * np.sin(angles), np.zeros(n)], axis=1)
    near_equator = generate_near_equator_points(n, noise_level, a, b, c)
    random_points = generate_random_ellipsoid_points(n, a, b, c)
    return {
        "equator": equator_points,
        "random": random_points,
        "near_equator": near_equator,
    }


if __name__ == "__main__":
    # Basic tests for ellipsoid equator utilities.
    point_above = np.array([1.0, 0.0, 0.2])
    point_below = np.array([0.0, 0.5, -0.1])
    print("Signed distance above equator:", signed_distance_to_equator(point_above))
    print("Signed distance below equator:", signed_distance_to_equator(point_below))
    print("Gradient:", gradient_signed_distance(point_above))

    random_points = generate_random_ellipsoid_points(5)
    print("Random ellipsoid points:\n", random_points)

    near_equator_points = generate_near_equator_points(5, noise_level=0.1)
    print("Near-equator points:\n", near_equator_points)
    print("Near-equator z values:", near_equator_points[:, 2])
