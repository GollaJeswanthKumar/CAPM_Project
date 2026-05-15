"""Ellipsoid manifold geometry utilities for CAPM projects.

This module implements basic operations on an ellipsoid in R^3.
The ellipsoid is defined by x²/a² + y²/b² + z²/c² = 1.

Key geometric ideas:
- Points on the ellipsoid satisfy the quadratic equation.
- The tangent space is orthogonal to the gradient of the constraint.
- Normalization projects points onto the ellipsoid.
- Exponential map is approximated by tangent step and reprojection.
"""

import numpy as np


def ellipsoid_constraint(x, a=1.5, b=1.0, c=0.7):
    """Compute the ellipsoid constraint value: x²/a² + y²/b² + z²/c² - 1."""
    x, y, z = x[0], x[1], x[2]
    return (x/a)**2 + (y/b)**2 + (z/c)**2 - 1


def ellipsoid_gradient(x, a=1.5, b=1.0, c=0.7):
    """Gradient of the ellipsoid constraint function."""
    x, y, z = x[0], x[1], x[2]
    return np.array([2*x/a**2, 2*y/b**2, 2*z/c**2])


def normalize_point(x, a=1.5, b=1.0, c=0.7, tol=1e-6, max_iter=100):
    """Project a point onto the ellipsoid using iterative scaling.

    Uses a simple iterative method to find the point on the ellipsoid
    closest to the input point.
    """
    x = np.asarray(x, dtype=float)
    for _ in range(max_iter):
        f = ellipsoid_constraint(x, a, b, c)
        grad = ellipsoid_gradient(x, a, b, c)
        norm_grad = np.linalg.norm(grad)
        if norm_grad < tol:
            break
        # Newton step approximation
        x = x - f * grad / (grad @ grad)
    return x


def project_to_tangent(x, v, a=1.5, b=1.0, c=0.7):
    """Project vector v onto the tangent space at point x on the ellipsoid."""
    x = normalize_point(x, a, b, c)
    v = np.asarray(v, dtype=float)
    grad = ellipsoid_gradient(x, a, b, c)
    # Remove component parallel to the gradient (normal vector)
    inner = np.dot(grad, v)
    norm_grad_sq = np.dot(grad, grad)
    if norm_grad_sq > 1e-12:
        tangent = v - (inner / norm_grad_sq) * grad
    else:
        tangent = v
    return tangent


def exp_map_ellipsoid(x, v, a=1.5, b=1.0, c=0.7):
    """Approximate exponential map on the ellipsoid with a first-order retraction."""
    x = normalize_point(x, a, b, c)
    v = np.asarray(v, dtype=float)
    tangent = project_to_tangent(x, v, a, b, c)
    if np.linalg.norm(tangent) == 0.0:
        return x.copy()

    # Exact ellipsoid geodesics are complex. This retraction respects the
    # tangent step length and then projects the point back to the ellipsoid.
    return normalize_point(x + tangent, a, b, c)


def geodesic_distance(x, y, a=1.5, b=1.0, c=0.7):
    """Approximate geodesic distance between two points on the ellipsoid.

    Uses Euclidean distance as a rough approximation since exact geodesics are hard.
    """
    x = normalize_point(x, a, b, c)
    y = normalize_point(y, a, b, c)
    return np.linalg.norm(x - y)


if __name__ == "__main__":
    # Basic tests for ellipsoid manifold utilities.
    x = np.array([1.0, 0.5, 0.3])
    x_normalized = normalize_point(x)
    print("Normalized x:", x_normalized)
    print("Constraint at normalized:", ellipsoid_constraint(x_normalized))

    v = np.array([0.1, -0.2, 0.05])
    tangent_v = project_to_tangent(x_normalized, v)
    print("Projected tangent vector:", tangent_v)
    print("Tangent dot gradient (should be ~0):", np.dot(ellipsoid_gradient(x_normalized), tangent_v))

    new_point = exp_map_ellipsoid(x_normalized, tangent_v)
    print("Point after exp map:", new_point)
    print("Constraint at new point:", ellipsoid_constraint(new_point))

    distance = geodesic_distance(x_normalized, new_point)
    print("Approximate geodesic distance:", distance)
