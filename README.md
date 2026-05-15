# Curvature-Aware Pulling on Riemannian Manifolds (CAPM)

This project compares two ways of moving a point toward an equator-like target
surface on curved manifolds:

- **Euclidean pull** - a baseline update that moves in ambient 3D space and then
  projects the point back to the manifold.
- **CAPM pull** - a curvature-aware update that projects the update direction
  into the tangent space before moving on the manifold.

The experiment is run on two manifolds:

- a unit sphere
- an ellipsoid with axes `a = 1.5`, `b = 1.0`, `c = 0.7`

The target surface for both manifolds is the equator, defined by `z = 0`.

## Project Goal

The goal is to show why geometry matters when data lives on a curved space.
Euclidean updates are simple, but they ignore the local geometry of the
manifold. CAPM uses tangent projection and manifold-aware movement, so its
trajectory better respects the curved surface.

## Project Structure

```text
capm-project/
├── config.py                    # Shared experiment constants and paths
├── experiment.py                # Runs all configured experiment cases
├── main.py                      # Main entry point
├── visualization.py             # Generates polished result plots
├── sphere_manifold.py           # Sphere geometry utilities
├── sphere_surface.py            # Sphere equator target and sampling utilities
├── sphere_euclidean_pull.py     # Euclidean pull on the sphere
├── sphere_capm_pull.py          # CAPM pull on the sphere
├── ellipsoid_manifold.py        # Ellipsoid geometry utilities
├── ellipsoid_surface.py         # Ellipsoid equator target and sampling utilities
├── ellipsoid_euclidean_pull.py  # Euclidean pull on the ellipsoid
├── ellipsoid_capm_pull.py       # CAPM pull on the ellipsoid
└── results/                     # Generated PNG visualizations
```

## How It Works

For each configured starting point, the project runs both update methods on
both manifolds.

The update methods are compared using:

- final signed distance to the equator
- distance reduction after repeated updates
- trajectory plots on the manifold surface
- convergence plots across iterations

The default run allows up to `100` pull steps. CAPM stops early when the
absolute signed distance to the equator is at most `1e-3`; the final converged
point is then projected onto the exact equator so the final CAPM marker lies on
`z = 0`.

## Visualizations

Running `main.py` generates five PNG files in `results/`:

```text
results/sphere_start_A_euclidean_vs_capm.png
results/ellipsoid_start_A_euclidean_vs_capm.png
results/sphere_start_B_euclidean_vs_capm.png
results/ellipsoid_start_B_euclidean_vs_capm.png
results/distance_convergence_summary.png
```

The trajectory plots show:

- the manifold surface
- the target equator as a black curve
- the normalized starting point
- the Euclidean pull trajectory
- the CAPM pull trajectory
- the final point reached by each method

The ellipsoid plots use the ellipsoid's real axis proportions, so the shape is
visibly stretched in the `x` direction and compressed in the `z` direction.

## Installation

Create or activate a Python environment, then install:

```bash
pip install numpy matplotlib
```

## Run

```bash
python main.py
```

The console prints the numerical comparison, and all generated figures are
saved inside `results/`.

## Main Idea in One Line

Euclidean pull moves as if the space were flat; CAPM pull moves using local
manifold geometry, so the trajectory is more faithful to the curved surface.
