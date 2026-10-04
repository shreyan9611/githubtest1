"""
model.py

Trains either of the two baseline models (model_type="logreg" or "mlp"):
- searches over the relevant hyperparameter using the validation set
- trains the final model with the best value

Both models support the same two functions, search() and train(), so the
rest of the pipeline (fairness.py, experiment.py) doesn't need to know or
care which model_type it's working with.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Default hyperparameter grid per model type, and what that hyperparameter is called
DEFAULT_PARAM_VALUES = {
    "logreg": [0.001, 0.01, 0.1, 1, 10, 100],       # C: inverse regularization strength
    "mlp": [0.0001, 0.001, 0.01, 0.1, 1.0],          # alpha: L2 regularization strength
}
PARAM_NAME = {"logreg": "C", "mlp": "alpha"}

# Small MLP: one hidden layer, kept deliberately simple so it's a fair
# "second baseline" comparison rather than a much bigger model
MLP_HIDDEN_LAYER_SIZES = (64,)


def build_model(model_type, param_value, seed=None):
    if model_type == "logreg":
        return LogisticRegression(C=param_value, max_iter=1000, random_state=seed)
    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
            alpha=param_value,
            max_iter=300,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def search(X_train, y_train, X_val, y_val, model_type="logreg", param_values=None):
    """
    Tries a range of values for the model's main regularization hyperparameter
    and picks the one that gives the best validation accuracy.
    """
    if param_values is None:
        param_values = DEFAULT_PARAM_VALUES[model_type]
    param_name = PARAM_NAME[model_type]

    results = []
    for value in param_values:
        model = build_model(model_type, value)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        results.append({param_name: value, "val_accuracy": val_acc})

    results_df = pd.DataFrame(results)
    best_value = results_df.loc[results_df["val_accuracy"].idxmax(), param_name]
    return best_value, results_df


def train(X_train, y_train, param_value, seed, model_type="logreg"):
    """Trains the final model of the given type using the chosen hyperparameter value."""
    model = build_model(model_type, param_value, seed=seed)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # quick manual check that everything works end to end, for both model types
    from data import load_adult, preprocess
    import time

    df = load_adult()
    split = preprocess(df, seed=0)

    for model_type in ["logreg", "mlp"]:
        print(f"\n=== model_type = {model_type} ===")
        t0 = time.time()

        best_value, results_df = search(
            split["X_train"], split["y_train"], split["X_val"], split["y_val"], model_type=model_type
        )
        print("Hyperparameter search results:")
        print(results_df.to_string(index=False))
        print(f"Best {PARAM_NAME[model_type]}: {best_value}")

        model = train(split["X_train"], split["y_train"], best_value, seed=0, model_type=model_type)
        test_acc = accuracy_score(split["y_test"], model.predict(split["X_test"]))
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Time taken: {time.time() - t0:.1f}s")
