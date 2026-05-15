"""Experiment runner for Euclidean pull and CAPM pull comparisons."""

import numpy as np

from config import (
    CONVERGENCE_TOLERANCE,
    ELLIPSOID_AXES,
    MANIFOLDS,
    NUM_STEPS,
    STARTING_POINTS,
    STEP_SIZE,
)
from ellipsoid_capm_pull import run_capm_pull as ellipsoid_capm_pull
from ellipsoid_euclidean_pull import run_euclidean_pull as ellipsoid_euclidean_pull
from ellipsoid_manifold import normalize_point as ellipsoid_normalize
from ellipsoid_surface import signed_distance_to_equator as ellipsoid_signed_distance
from sphere_capm_pull import run_capm_pull as sphere_capm_pull
from sphere_euclidean_pull import run_euclidean_pull as sphere_euclidean_pull
from sphere_manifold import normalize_point as sphere_normalize
from sphere_surface import signed_distance_to_equator as sphere_signed_distance


def run_experiment_case(
    manifold,
    start_label,
    raw_start,
    num_steps=NUM_STEPS,
    step_size=STEP_SIZE,
    convergence_tolerance=CONVERGENCE_TOLERANCE,
):
    """Run Euclidean and CAPM pull for one manifold/start-point pair."""
    raw_start = np.asarray(raw_start, dtype=float)

    if manifold == "sphere":
        start = sphere_normalize(raw_start)
        initial_distance = sphere_signed_distance(start)
        euclidean_result = sphere_euclidean_pull(start, num_steps, step_size)
        capm_result = sphere_capm_pull(
            start,
            num_steps,
            step_size,
            convergence_tolerance=convergence_tolerance,
        )
    elif manifold == "ellipsoid":
        a, b, c = ELLIPSOID_AXES
        start = ellipsoid_normalize(raw_start, a, b, c)
        initial_distance = ellipsoid_signed_distance(start, a, b, c)
        euclidean_result = ellipsoid_euclidean_pull(start, num_steps, step_size, a, b, c)
        capm_result = ellipsoid_capm_pull(
            start,
            num_steps,
            step_size,
            a,
            b,
            c,
            convergence_tolerance=convergence_tolerance,
        )
    else:
        raise ValueError(f"Unsupported manifold: {manifold}")

    return {
        "manifold": manifold,
        "start_label": start_label,
        "raw_start": raw_start,
        "start": start,
        "initial_distance": initial_distance,
        "convergence_tolerance": convergence_tolerance,
        "euclidean": euclidean_result,
        "capm": capm_result,
    }


def run_experiment_suite(
    num_steps=NUM_STEPS,
    step_size=STEP_SIZE,
    convergence_tolerance=CONVERGENCE_TOLERANCE,
):
    """Run every configured manifold/start-point comparison."""
    results = {}
    for start_label, raw_start in STARTING_POINTS.items():
        for manifold in MANIFOLDS:
            results[(manifold, start_label)] = run_experiment_case(
                manifold,
                start_label,
                raw_start,
                num_steps=num_steps,
                step_size=step_size,
                convergence_tolerance=convergence_tolerance,
            )
    return results


def iter_experiment_cases(results):
    """Yield experiment cases in the configured display order."""
    for start_label in STARTING_POINTS:
        for manifold in MANIFOLDS:
            yield results[(manifold, start_label)]
