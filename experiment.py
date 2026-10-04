"""
experiment.py

Runs the full experiment across multiple random seeds, for a given sensitive
attribute and baseline model type:
- for each seed: preprocess data, train the baseline, apply each fairness
  intervention, evaluate everything
- combines results into one table and computes mean +/- std across seeds

Methods evaluated per run:
  "baseline"   -- the plain model, sensitive attribute never used
  "threshold"  -- intervention 1, post-processing threshold adjustment
  "reductions" -- intervention 2, in-processing reductions (logreg only,
                  because the MLP can't take per-sample weights)
"""

import pandas as pd

from data import load_adult, preprocess
from model import search, train
from fairness import adjust_threshold, train_reductions, evaluate

SEEDS = [0, 1, 2, 3, 4]
METHOD_ORDER = ["baseline", "threshold", "reductions"]


def run_seed(df, seed, sensitive_attr="sex", model_type="logreg"):
    """Runs the full pipeline once for a single seed and returns everything from that run."""
    split = preprocess(df, seed, sensitive_attr=sensitive_attr)
    X_train, y_train, sens_train = split["X_train"], split["y_train"], split["sensitive_train"]
    X_test, y_test, sens_test = split["X_test"], split["y_test"], split["sensitive_test"]

    best_value, search_results = search(
        X_train, y_train, split["X_val"], split["y_val"], model_type=model_type
    )
    baseline = train(X_train, y_train, best_value, seed, model_type=model_type)

    preds = {"baseline": baseline.predict(X_test)}
    baseline_probs = baseline.predict_proba(X_test)[:, 1]

    optimizer = adjust_threshold(baseline, X_train, y_train, sens_train)
    preds["threshold"] = optimizer.predict(X_test, sensitive_features=sens_test, random_state=seed)

    if model_type == "logreg":
        mitigator = train_reductions(X_train, y_train, sens_train, best_value)
        preds["reductions"] = mitigator.predict(X_test, random_state=seed)

    scores = {name: evaluate(y_test, p, sens_test) for name, p in preds.items()}

    return {
        "seed": seed,
        "sensitive_attr": sensitive_attr,
        "model_type": model_type,
        "best_value": best_value,
        "scores": scores,
        "preds": preds,
        "search_results": search_results,
        "y_test": y_test,
        "sensitive_test": sens_test,
        "baseline_probs": baseline_probs,
        "baseline_model": baseline,
        "feature_names": split["feature_names"],
    }


def run_all(seeds=SEEDS, sensitive_attr="sex", model_type="logreg"):
    """Runs run_seed() for every seed (for the given attribute + model) and returns the list of results."""
    df = load_adult()
    return [run_seed(df, seed, sensitive_attr=sensitive_attr, model_type=model_type) for seed in seeds]


def to_table(runs):
    """Turns the list of per-seed results into one flat table (one row per seed per method)."""
    rows = []
    for r in runs:
        for name, score in r["scores"].items():
            rows.append({"seed": r["seed"], "model": name, **score})
    return pd.DataFrame(rows)


def aggregate(results_df):
    """Computes mean and standard deviation of each metric, grouped by method."""
    summary = results_df.groupby("model")[["accuracy", "dp_diff", "eo_diff"]].agg(["mean", "std"])
    return summary.reindex([m for m in METHOD_ORDER if m in summary.index])


if __name__ == "__main__":
    # quick manual check with ONE seed per combination (the MLP is slow --
    # the full 5-seed run happens via main.py)
    for attr in ["sex", "race"]:
        for model_type in ["logreg", "mlp"]:
            print(f"\n=== sensitive_attr = {attr}, model_type = {model_type} ===")
            runs = run_all(seeds=[0], sensitive_attr=attr, model_type=model_type)
            print(to_table(runs).to_string(index=False))
