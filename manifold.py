"""Sphere manifold geometry utilities for CAPM projects.

This module implements basic operations on the unit sphere in R^3.
The sphere is a simple example of a curved manifold, where straight-line
Euclidean motion is not the same as moving along the surface.

Key geometric ideas:
- A point on the sphere is a vector of unit length.
- The tangent space at a point is the plane that "touches" the sphere at that point.
- The exponential map moves a point along the sphere by following a geodesic.
- Geodesic distance is the angle between two normalized points.

Why these concepts matter:
- On a curved surface like a sphere, the shortest path between two points lies on the surface.
- A tangent vector gives a valid direction to move while staying on the surface.
- The exponential map is needed because naive Euclidean addition can leave the sphere.
"""

import numpy as np


def normalize_point(x):
    """Normalize a vector in R^3 to lie on the unit sphere.

    Geometric intuition:
    - Any nonzero vector in R^3 can be scaled to have length 1.
    - After normalization, the vector represents a point on the unit sphere.

    Args:
        x: array-like of shape (3,), a point or direction in R^3.

    Returns:
        np.ndarray of shape (3,), the normalized unit vector.
    """
    x = np.asarray(x, dtype=float)
    norm_x = np.linalg.norm(x)
    if norm_x == 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    return x / norm_x


def project_to_tangent(x, v):
    """Project a vector v onto the tangent space at point x on the sphere.

    Geometric intuition:
    - The tangent space at x is the plane perpendicular to x.
    - Any vector in R^3 decomposes into a part parallel to x and a part orthogonal to x.
    - The orthogonal part lies in the tangent plane and therefore is a valid tangent direction.

    Args:
        x: array-like of shape (3,), a point on the unit sphere.
        v: array-like of shape (3,), an arbitrary vector in R^3.

    Returns:
        np.ndarray of shape (3,), the projection of v into the tangent space at x.
    """
    x = normalize_point(x)
    v = np.asarray(v, dtype=float)
    inner = np.dot(x, v)
    # Remove the component of v that is parallel to x.
    tangent = v - inner * x
    return tangent


def exp_map_sphere(x, v):
    """Compute the exponential map on the unit sphere.

    Geodesic intuition:
    - The exponential map moves x along the sphere in the tangent direction v.
    - The direction v must lie in the tangent plane at x.
    - The result follows a great-circle path, which is the geodesic on the sphere.

    Why Euclidean movement is wrong:
    - Adding v directly to x yields x + v, which generally leaves the sphere.
    - The sphere is curved, so the correct motion is along the surface.
    - The exponential map reprojects the path onto the sphere without leaving the manifold.

    Args:
        x: array-like of shape (3,), a point on the unit sphere.
        v: array-like of shape (3,), a tangent vector at x.

    Returns:
        np.ndarray of shape (3,), the new point on the unit sphere.
    """
    x = normalize_point(x)
    v = np.asarray(v, dtype=float)
    tangent = project_to_tangent(x, v)
    norm_v = np.linalg.norm(tangent)
    if norm_v == 0.0:
        return x.copy()

    # Move along the great circle: cos(theta) * x + sin(theta) * (v / ||v||)
    cos_term = np.cos(norm_v) * x
    sin_term = np.sin(norm_v) * (tangent / norm_v)
    return cos_term + sin_term


def geodesic_distance(x, y):
    """Compute the geodesic distance between two points on the unit sphere.

    Geometric intuition:
    - The geodesic distance on the sphere is the angle between the two vectors.
    - Because points on the sphere are unit vectors, their dot product equals cos(theta).
    - The distance is the arc length along the great circle.

    Args:
        x: array-like of shape (3,), a point on the unit sphere.
        y: array-like of shape (3,), another point on the unit sphere.

    Returns:
        float, the geodesic distance between x and y in radians.
    """
    x = normalize_point(x)
    y = normalize_point(y)
    dot = np.dot(x, y)
    dot_clipped = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot_clipped)


if __name__ == "__main__":
    # Simple checks for sphere manifold utilities.
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])
    x_normalized = normalize_point(x)
    print("Normalized x:", x_normalized)

    v = np.array([0.5, -0.2, 0.1])
    tangent_v = project_to_tangent(x_normalized, v)
    print("Projected tangent vector:", tangent_v)
    print("Tangent dot x (should be ~0):", np.dot(x_normalized, tangent_v))

    new_point = exp_map_sphere(x_normalized, tangent_v)
    print("Point after exponential map:", new_point)
    print("New point norm (should be 1):", np.linalg.norm(new_point))

    distance = geodesic_distance(x_normalized, y)
    print("Geodesic distance from x to y (radians):", distance)
    print("Geodesic distance from x to y (degrees):", np.degrees(distance))
