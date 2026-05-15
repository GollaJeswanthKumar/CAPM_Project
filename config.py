"""Shared configuration for the CAPM experiment."""

from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

STARTING_POINTS = {
    "A": np.array([0.8, 0.3, 0.5], dtype=float),
    "B": np.array([-0.4, 0.7, 0.6], dtype=float),
}

MANIFOLDS = ("sphere", "ellipsoid")
METHODS = ("euclidean", "capm")

NUM_STEPS = 100
STEP_SIZE = 0.1
CONVERGENCE_TOLERANCE = 1e-3

SPHERE_RADIUS = 1.0
ELLIPSOID_AXES = (1.5, 1.0, 0.7)
