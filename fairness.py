"""
fairness.py

Everything to do with fairness:
- the two metrics we're measuring (demographic parity, equalized odds)
- intervention 1, post-processing: per-group threshold adjustment
- intervention 2, in-processing: the reductions approach (Agarwal et al.)
- a shared evaluate() function so every method is scored the same way

Key difference between the two interventions: the post-processing optimizer
needs the sensitive attribute at prediction time, the reductions approach only
needs it during training.
"""

from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

from model import build_model


def dp_diff(y_true, y_pred, group):
    """Demographic Parity Difference: how different the positive-prediction rate is between groups."""
    return demographic_parity_difference(y_true, y_pred, sensitive_features=group)


def eo_diff(y_true, y_pred, group):
    """Equalized Odds Difference: how different the true/false positive rates are between groups."""
    return equalized_odds_difference(y_true, y_pred, sensitive_features=group)


def adjust_threshold(model, X_train, y_train, group_train):
    """
    Intervention 1 (post-processing): wraps the already-trained baseline model with
    a per-group decision threshold, so predictions satisfy demographic parity.
    """
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        predict_method="predict_proba",
        prefit=True,
    )
    optimizer.fit(X_train, y_train, sensitive_features=group_train)
    return optimizer


def train_reductions(X_train, y_train, group_train, param_value):
    """
    Intervention 2 (in-processing): retrains a Logistic Regression many times with
    re-weighted samples until the demographic parity constraint is met.

    Only works with logreg: this method needs the model to accept per-sample
    weights, and scikit-learn's MLPClassifier (in the pinned version) does not.
    """
    mitigator = ExponentiatedGradient(
        build_model("logreg", param_value),
        constraints=DemographicParity(),
    )
    mitigator.fit(X_train, y_train, sensitive_features=group_train)
    return mitigator


def evaluate(y_true, y_pred, group):
    """Scores a set of predictions on accuracy and both fairness metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "dp_diff": dp_diff(y_true, y_pred, group),
        "eo_diff": eo_diff(y_true, y_pred, group),
    }


if __name__ == "__main__":
    # quick manual check that everything works end to end
    from data import load_adult, preprocess
    from model import search, train

    df = load_adult()
    split = preprocess(df, seed=0)

    best_C, _ = search(split["X_train"], split["y_train"], split["X_val"], split["y_val"])
    baseline = train(split["X_train"], split["y_train"], best_C, seed=0)

    baseline_preds = baseline.predict(split["X_test"])
    print("Baseline:    ", evaluate(split["y_test"], baseline_preds, split["sensitive_test"]))

    optimizer = adjust_threshold(baseline, split["X_train"], split["y_train"], split["sensitive_train"])
    threshold_preds = optimizer.predict(split["X_test"], sensitive_features=split["sensitive_test"], random_state=0)
    print("Threshold:   ", evaluate(split["y_test"], threshold_preds, split["sensitive_test"]))

    mitigator = train_reductions(split["X_train"], split["y_train"], split["sensitive_train"], best_C)
    reductions_preds = mitigator.predict(split["X_test"], random_state=0)
    print("Reductions:  ", evaluate(split["y_test"], reductions_preds, split["sensitive_test"]))
