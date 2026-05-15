"""Main entry point for the CAPM sphere and ellipsoid experiment."""

from config import PROJECT_DIR, RESULTS_DIR
from experiment import iter_experiment_cases, run_experiment_suite
from visualization import run_visualizations


def _print_case_summary(case_result):
    manifold = case_result["manifold"].title()
    start_label = case_result["start_label"]
    initial_distance = case_result["initial_distance"]
    euclidean_final = case_result["euclidean"]["distances"][-1]
    capm_final = case_result["capm"]["distances"][-1]
    capm_steps = len(case_result["capm"]["distances"]) - 1

    print(f"[{manifold} | Start {start_label}]")
    print(f"  Normalized start: {case_result['start']}")
    print(f"  Initial signed distance: {initial_distance:.6f}")
    print(f"  Euclidean final distance: {euclidean_final:.6f}")
    print(f"  CAPM final distance:      {capm_final:.6f}")
    print(f"  CAPM steps used:          {capm_steps}")
    print(f"  Euclidean reduction:      {abs(initial_distance - euclidean_final):.6f}")
    print(f"  CAPM reduction:           {abs(initial_distance - capm_final):.6f}")
    print()


def main():
    print("\n=== Curvature-Aware Pulling on Manifolds (CAPM) ===\n")

    print("[1] Running Euclidean and CAPM pull experiments")
    results = run_experiment_suite()
    for case_result in iter_experiment_cases(results):
        _print_case_summary(case_result)

    print("[2] Generating visualization files")
    generated_files = run_visualizations(results, RESULTS_DIR)

    print(f"\nExperiment complete. Generated {len(generated_files)} files in {RESULTS_DIR.name}/:")
    for output_path in generated_files:
        print(f"- {output_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
