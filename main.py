"""
main.py

Single entry point for the whole project.
Running `python main.py` reproduces every number and every figure used in the report,
for both sensitive attributes (sex, race) and both baseline models (logreg, mlp).

Note: the MLP combinations are much slower than Logistic Regression (roughly
a minute per seed, vs. a few seconds), so the full run takes several minutes.
"""

import os

from data import load_adult
from experiment import run_all, to_table, aggregate
import visualize

ATTRIBUTES = ["sex", "race"]
MODEL_TYPES = ["logreg", "mlp"]


def run_combination(df, attr, model_type, seeds=None):
    """Runs the full experiment + figure generation for one (attribute, model_type) pair."""
    print(f"\n{'='*60}\nRunning: sensitive_attr={attr}, model_type={model_type}\n{'='*60}")

    runs = run_all(sensitive_attr=attr, model_type=model_type) if seeds is None \
        else run_all(seeds=seeds, sensitive_attr=attr, model_type=model_type)
    results_df = to_table(runs)
    summary_df = aggregate(results_df)

    results_df.to_csv(f"outputs/results_{attr}_{model_type}.csv", index=False)
    summary_df.to_csv(f"outputs/summary_{attr}_{model_type}.csv")

    print(f"\nPer-seed results (saved to outputs/results_{attr}_{model_type}.csv):")
    print(results_df.to_string(index=False))
    print(f"\nSummary -- mean +/- std across seeds (saved to outputs/summary_{attr}_{model_type}.csv):")
    print(summary_df)

    # the "primary" combination (sex + logreg) keeps the original, untagged filenames
    suffix_parts = [p for p, default in [(attr, "sex"), (model_type, "logreg")] if p != default]
    tag = "_".join(suffix_parts) if suffix_parts else None

    r = runs[0]  # figures that need one representative run use the first seed

    if model_type == "logreg":
        # the distribution chart is attribute-specific, not model-specific, so only
        # generate it once per attribute
        visualize.plot_group_distribution(df, attr=attr)
    visualize.plot_confusion_matrices(r["y_test"], r["preds"], tag=tag)
    visualize.plot_group_bars(r["y_test"], r["preds"], r["sensitive_test"], tag=tag)
    visualize.plot_tradeoff(r["y_test"], r["baseline_probs"], r["sensitive_test"], r["scores"], tag=tag)
    visualize.plot_roc(r["y_test"], r["baseline_probs"], r["sensitive_test"], attr=attr,
                       tag=(model_type if model_type != "logreg" else None))
    if model_type == "logreg":
        # feature importance via linear coefficients is specifically a Logistic
        # Regression concept -- MLPClassifier has a list of weight matrices across
        # hidden layers instead, with no single per-feature coefficient
        visualize.plot_feature_importance(r["baseline_model"], r["feature_names"], tag=tag)
    visualize.plot_seed_stability(results_df, tag=tag)

    return results_df, summary_df


def run(seeds=None):
    os.makedirs("outputs", exist_ok=True)

    print("Loading data...")
    df = load_adult()

    visualize.style()
    visualize.plot_pipeline_diagram()  # same for every combination, generate once

    all_results = {}
    for attr in ATTRIBUTES:
        for model_type in MODEL_TYPES:
            all_results[(attr, model_type)] = run_combination(df, attr, model_type, seeds=seeds)

    # cross-combination comparison figures, which need every result at once
    visualize.plot_method_comparison(all_results, y_metric="accuracy")
    visualize.plot_method_comparison(all_results, y_metric="eo_diff")

    print("\nDone. Results in outputs/results_<attr>_<model>.csv and outputs/summary_<attr>_<model>.csv,")
    print("figures in outputs/figures/.")

    return all_results


if __name__ == "__main__":
    run()
