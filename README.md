# Curvature-Aware Pulling on Riemannian Manifolds (CAPM)

A simple educational project comparing Euclidean pull updates and curvature-aware geodesic pull updates on a sphere manifold.

## 1. Project Overview

This repository demonstrates how motion on a curved manifold differs from naive Euclidean motion in ambient space. It compares two update strategies for moving points toward a target surface on the unit sphere:

- **Euclidean pull**: a baseline update using straight-line gradient descent in R³.
- **CAPM pull**: a curvature-aware update using tangent projection and the exponential map.

The goal is to show why manifold geometry matters and how geodesic motion improves behavior.

## 2. Motivation

Many optimization and learning problems involve data that live on curved spaces rather than flat Euclidean space. In those cases, updates that ignore curvature can behave poorly, drift off the manifold, or converge slowly.

This project uses the unit sphere as a concrete example to highlight:

- the difference between flat-space and manifold-aware updates
- why normalization is not enough by itself
- how tangent directions and geodesics preserve the manifold structure

## 3. Mathematical Intuition

The sphere is a simple Riemannian manifold embedded in R³. On this manifold:

- points are unit vectors
- tangent directions lie in the plane orthogonal to the point
- the shortest path between two points is a great circle

A geometry-aware update respects these properties, while a Euclidean update does not.

## 4. Sphere Manifold Geometry

Key concepts used in this project:

- **Unit sphere**: the set of all points in R³ with norm 1.
- **Tangent space**: at a point `x`, the tangent space is the plane orthogonal to `x`.
- **Projection onto tangent space**: removes any component along `x` so the update stays valid.
- **Normalization**: ensures output points remain on the sphere after an ambient-space update.

## 5. Euclidean vs CAPM

### Euclidean pull

- Uses the signed-distance gradient in R³.
- Applies a simple update `x_new = x - step_size * grad`.
- Renormalizes afterward to stay on the sphere.
- Ignores manifold curvature, so the path is not geodesic.

### CAPM pull

- Computes a tangent direction from the signed-distance gradient.
- Normalizes gradient magnitude and scales by the signed distance.
- Projects the update onto the tangent space.
- Uses the exponential map to move along the sphere.
- Preserves geometric correctness and tends to converge more smoothly.

## 6. Signed Distance Function

The target surface is defined as the sphere equator, where `z = 0`.

- The signed distance is approximated by the `z` value.
- Positive values are above the equator, negative values are below.
- This function is simple and intuitive for the sphere.

## 7. Exponential Map

The exponential map is the correct way to move from a point along a tangent direction on the sphere.

- It follows a great circle.
- It stays on the manifold without requiring repeated projection.
- It is the manifold-equivalent of a straight-line step in Euclidean space.

## 8. Visualization Results

The project includes visualizations that compare:

- trajectories on the sphere for Euclidean and CAPM pull
- convergence of absolute signed distance to the equator over iterations

The visualizations show how CAPM yields a smoother, geometry-respecting path while Euclidean pull can move inefficiently.

## 9. Project Structure

- `manifold.py` — sphere geometry utilities, tangent projection, exponential map, and geodesic distance.
- `surface.py` — defines the equator target surface, signed-distance function, and point generators.
- `euclidean_pull.py` — baseline Euclidean pull update and trajectory tracking.
- `capm_pull.py` — curvature-aware CAPM pull update and trajectory tracking.
- `visualization.py` — plots sphere trajectories and convergence comparisons.
- `main.py` — single entry point to run the full experiment.

## 10. Installation

1. Create or activate a Python environment.
2. Install required packages:

```bash
pip install numpy matplotlib
```

## 11. How to Run

Run the full experiment with:

```bash
python main.py
```

This will execute both update methods, print a summary, and generate visualization files.

## 12. Future Improvements

Possible extensions include:

- adding more general manifold targets beyond the sphere
- comparing with other manifold optimization methods
- improving the signed-distance model for more complex surfaces
- adding interactive visualization and animation
