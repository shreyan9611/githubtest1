"""
visualize.py

Makes all the figures used in the report, in a clean, consistent style
(muted colors, minimal borders/gridlines, direct labels instead of legends
where possible -- inspired by Datawrapper's house style).

Each function saves its own PNG into outputs/figures/. Pass tag="race" etc.
to keep one combination's figures from overwriting another's.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.metrics import confusion_matrix, roc_curve, auc

OUT_DIR = "outputs/figures"

# House style: the same colour always means the same method, in every chart
NAVY = "#2C4A6E"
CORAL = "#E07856"
TEAL = "#4E9A8C"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#DDDDDD"

METHOD_ORDER = ["baseline", "threshold", "reductions"]
METHOD_COLOR = {"baseline": NAVY, "threshold": CORAL, "reductions": TEAL}
METHOD_LABEL = {"baseline": "Baseline", "threshold": "Threshold adj.", "reductions": "Reductions"}
METHOD_LABEL_LONG = {
    "baseline": "Baseline",
    "threshold": "Threshold adjustment (post-processing)",
    "reductions": "Reductions (in-processing)",
}


def style():
    """Sets the shared look for every chart: clean font, no boxed border, light gridlines."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY,
        "axes.grid": True,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def titled(ax, title, subtitle=None):
    """Puts a bold left-aligned title (and optional gray subtitle) above a chart, Datawrapper-style."""
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=(18 if subtitle else 10))
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=GRAY, va="bottom")


def save(fig, name, tag=None):
    """Saves a figure. Pass tag="race" etc. to avoid overwriting another combination's version."""
    if tag:
        name = name.replace(".png", f"_{tag}.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def _methods_in(preds_or_scores):
    """The methods present, in a consistent order."""
    return [m for m in METHOD_ORDER if m in preds_or_scores]


def plot_group_distribution(df, attr="sex", tag=None):
    """Bar chart of how many records fall into each category of the given attribute."""
    counts = df[attr].value_counts()

    # wider figure and rotated labels when there are more than 2 categories or long names
    n = len(counts)
    needs_rotation = n > 2 or counts.index.astype(str).str.len().max() > 8
    fig, ax = plt.subplots(figsize=(max(5, n * 1.6), 4.5))

    colors = [NAVY, CORAL, GRAY, "#8FA6C2", "#D9A48E"][:n]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.55)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + counts.max() * 0.015, f"{val:,}", ha="center", fontsize=10)

    if needs_rotation:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Number of records")
    titled(ax, f"{attr.capitalize()} distribution in the Adult dataset", f"{len(df):,} total records")
    save(fig, f"01_{attr}_distribution.png", tag=tag)


def plot_pipeline_diagram():
    """Flow diagram of the whole pipeline, showing the branching between models and interventions."""
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    boxes = {}

    def box(key, cx, cy, w, h, label, color):
        patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                               boxstyle="round,pad=0.005,rounding_size=0.02",
                               facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(patch)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9.5, color="white", zorder=4)
        boxes[key] = (cx, cy, w, h)

    def arrow(a, b, dashed=False):
        ax_, ay, aw, ah = boxes[a]
        bx, by, bw, bh = boxes[b]
        ax.annotate("", xy=(bx - bw / 2 - 0.004, by), xytext=(ax_ + aw / 2 + 0.004, ay), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, mutation_scale=16,
                                    linestyle="--" if dashed else "-"))

    box("raw", 0.075, 0.50, 0.11, 0.17, "Raw Data\n(UCI Adult)", NAVY)
    box("prep", 0.26, 0.50, 0.14, 0.17, "Preprocessing", NAVY)
    box("logreg", 0.46, 0.72, 0.16, 0.17, "Logistic\nRegression", NAVY)
    box("mlp", 0.46, 0.28, 0.16, 0.17, "MLP\n(1 hidden layer)", NAVY)
    box("reductions", 0.68, 0.82, 0.19, 0.17, "Reductions\n(in-processing)", TEAL)
    box("threshold", 0.68, 0.40, 0.19, 0.17, "Threshold Optimizer\n(post-processing)", CORAL)
    box("eval", 0.89, 0.58, 0.15, 0.22, "Evaluation\nAccuracy\nDP diff, EO diff", NAVY)

    arrow("raw", "prep")
    arrow("prep", "logreg")
    arrow("prep", "mlp")
    arrow("logreg", "reductions")
    arrow("logreg", "threshold")
    arrow("mlp", "threshold")
    arrow("reductions", "eval")
    arrow("threshold", "eval")

    ax.text(0.26, 0.93, "Sensitive attribute: Sex or Race\n(kept out of the model's inputs)",
            ha="center", va="center", fontsize=9, color=GRAY, style="italic")
    ax.annotate("", xy=(0.26, 0.59), xytext=(0.26, 0.86),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2, linestyle="--", mutation_scale=12))
    ax.text(0.5, 0.04, "Both baselines are also evaluated directly.  Reductions needs per-sample weights, "
            "so it runs on Logistic Regression only.",
            ha="center", va="center", fontsize=9, color=GRAY, style="italic")

    fig.suptitle("Experiment pipeline", x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.98)
    save(fig, "02_pipeline_diagram.png")


def plot_confusion_matrices(y_test, preds, tag=None):
    """One confusion matrix per method, side by side. preds is a dict: method name -> predictions."""
    methods = _methods_in(preds)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, methods):
        cm = confusion_matrix(y_test, preds[name])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["<=50K", ">50K"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["<=50K", ">50K"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(METHOD_LABEL[name], loc="left", fontsize=12, fontweight="bold")
        ax.grid(False)

    fig.suptitle("Confusion matrices by method", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.05)
    save(fig, "03_confusion_matrices.png", tag=tag)


def _group_rates(y_true, y_pred, group):
    """Small helper: accuracy, true positive rate, and false positive rate for each group."""
    rows = []
    for g in np.unique(group):
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        acc = (yt == yp).mean()
        tpr = yp[yt == 1].mean() if (yt == 1).any() else 0
        fpr = yp[yt == 0].mean() if (yt == 0).any() else 0
        rows.append({"group": g, "accuracy": acc, "tpr": tpr, "fpr": fpr})
    return rows


def plot_group_bars(y_test, preds, group_test, tag=None):
    """Accuracy / TPR / FPR per group, one bar per method. preds is a dict: method name -> predictions."""
    methods = _methods_in(preds)
    rates = {m: _group_rates(y_test, preds[m], group_test) for m in methods}
    groups = [r["group"] for r in rates[methods[0]]]

    metrics = ["accuracy", "tpr", "fpr"]
    metric_labels = ["Accuracy", "True Positive Rate", "False Positive Rate"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    width = 0.8 / len(methods)
    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(groups))
        for k, m in enumerate(methods):
            offset = (k - (len(methods) - 1) / 2) * width
            vals = [r[metric] for r in rates[m]]
            ax.bar(x + offset, vals, width, label=METHOD_LABEL[m], color=METHOD_COLOR[m])
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylim(0, 1.1)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")

    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=len(methods))
    fig.suptitle("Per-group performance by method", x=0.01, ha="left", fontsize=13, fontweight="bold")
    save(fig, "04_per_group_bars.png", tag=tag)


def plot_tradeoff(y_test, baseline_probs, group_test, scores, tag=None):
    """
    Accuracy vs. demographic parity difference as a single global decision threshold varies,
    with each intervention's own operating point marked for comparison.
    scores is a dict: method name -> {"accuracy": ..., "dp_diff": ...}.
    """
    thresholds = np.linspace(0.1, 0.9, 25)
    accs, dp_diffs = [], []
    for t in thresholds:
        preds = (baseline_probs >= t).astype(int)
        rate_by_group = [preds[group_test == g].mean() for g in np.unique(group_test)]
        accs.append((preds == y_test).mean())
        dp_diffs.append(max(rate_by_group) - min(rate_by_group))

    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.plot(dp_diffs, accs, color=NAVY, lw=2, marker="o", markersize=3, label="Threshold sweep (baseline model)")
    for m in _methods_in(scores):
        if m == "baseline":
            continue
        ax.scatter([scores[m]["dp_diff"]], [scores[m]["accuracy"]], color=METHOD_COLOR[m], s=90,
                   zorder=5, label=METHOD_LABEL_LONG[m])

    ax.set_xlabel("Demographic Parity Difference (lower = fairer)")
    ax.set_ylabel("Accuracy")
    titled(ax, "Accuracy vs. fairness trade-off", "Each line point is a different single decision threshold")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "05_tradeoff_curve.png", tag=tag)


def plot_roc(y_test, baseline_probs, group_test, attr="sex", tag=None):
    """ROC curve of the baseline model, computed separately for each group of the given attribute."""
    fig, ax = plt.subplots(figsize=(6, 5))
    unique_groups = np.unique(group_test)
    palette = [NAVY, CORAL, GRAY, "#8FA6C2", "#D9A48E"]
    colors = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}

    for g in unique_groups:
        mask = group_test == g
        fpr, tpr, _ = roc_curve(y_test[mask], baseline_probs[mask])
        ax.plot(fpr, tpr, color=colors[g], lw=2, label=f"{g} (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    titled(ax, f"ROC curve by {attr}", "Baseline model")
    ax.legend(frameon=False, loc="lower right")
    save(fig, f"06_roc_by_{attr}.png", tag=tag)


def plot_feature_importance(model, feature_names, top_n=15, tag=None):
    """Horizontal bar chart of the baseline model's strongest coefficients (logreg only)."""
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_names = [feature_names[i].replace("cat__", "").replace("num__", "") for i in order]
    top_vals = coefs[order]
    colors = [CORAL if v > 0 else NAVY for v in top_vals]

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals, color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel("Coefficient (standardized)")
    titled(ax, "Strongest predictors of income > $50K", "Coral = increases predicted income, Navy = decreases")
    save(fig, "07_feature_importance.png", tag=tag)


def plot_seed_stability(results_df, tag=None):
    """
    Shows how much accuracy, DP diff, and EO diff vary across seeds, one column of dots
    per method -- a visual companion to the summary table's mean +/- std.
    """
    methods = _methods_in(set(results_df["model"]))
    metrics = ["accuracy", "dp_diff", "eo_diff"]
    metric_labels = ["Accuracy", "Demographic Parity Diff.", "Equalized Odds Diff."]
    rng = np.random.default_rng(0)  # fixed seed so the jitter looks the same every run

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    for ax, metric, label in zip(axes, metrics, metric_labels):
        for i, m in enumerate(methods):
            vals = results_df[results_df["model"] == m][metric].values
            x = np.full(len(vals), i) + rng.uniform(-0.06, 0.06, len(vals))
            ax.scatter(x, vals, color=METHOD_COLOR[m], s=55, alpha=0.85, zorder=3)
            ax.scatter([i], [vals.mean()], color=METHOD_COLOR[m], s=170, marker="_", linewidths=3, zorder=4)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([METHOD_LABEL[m] for m in methods])
        ax.set_xlim(-0.5, len(methods) - 0.5)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")

    fig.suptitle("Stability across random seeds (dots = seeds, bar = mean)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    save(fig, "08_seed_stability.png", tag=tag)


def plot_method_comparison(all_results, y_metric="accuracy"):
    """
    One panel per sensitive attribute; every (model, method) combination plotted as a point
    in fairness-vs-y space, with +/- 1 std error bars across seeds.

    y_metric="accuracy": accuracy vs. DP diff -- shows the cost of each fairness fix.
    y_metric="eo_diff":  EO diff vs. DP diff  -- shows whether fixing one fairness metric hurts the other.

    all_results is a dict: (attribute, model_type) -> (results_df, summary_df).
    """
    y_label = {"accuracy": "Accuracy", "eo_diff": "Equalized Odds Difference (lower = fairer)"}[y_metric]
    title = {"accuracy": "Accuracy cost of each fairness method",
             "eo_diff": "Fixing demographic parity vs. equalized odds"}[y_metric]
    out_name = {"accuracy": "09_method_comparison_accuracy.png", "eo_diff": "10_method_comparison_eo.png"}[y_metric]
    marker = {"logreg": "o", "mlp": "s"}

    attrs = sorted({a for a, _ in all_results}, key=lambda a: ["sex", "race"].index(a) if a in ["sex", "race"] else 99)
    fig, axes = plt.subplots(1, len(attrs), figsize=(6.2 * len(attrs), 5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, attr in zip(axes, attrs):
        for (a, model_type), (_, summary) in all_results.items():
            if a != attr:
                continue
            for m in _methods_in(set(summary.index)):
                ax.errorbar(summary.loc[m, ("dp_diff", "mean")], summary.loc[m, (y_metric, "mean")],
                            xerr=summary.loc[m, ("dp_diff", "std")], yerr=summary.loc[m, (y_metric, "std")],
                            fmt=marker[model_type], color=METHOD_COLOR[m], markersize=9,
                            capsize=3, elinewidth=1.2, zorder=3)
        ax.set_xlabel("Demographic Parity Difference (lower = fairer)")
        ax.set_title(attr.capitalize(), loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel(y_label)

    handles = [Patch(color=METHOD_COLOR[m], label=METHOD_LABEL_LONG[m]) for m in METHOD_ORDER]
    handles += [Line2D([0], [0], marker=marker["logreg"], color=GRAY, linestyle="", markersize=8, label="Logistic Regression"),
                Line2D([0], [0], marker=marker["mlp"], color=GRAY, linestyle="", markersize=8, label="MLP")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title + " (mean \u00b1 std over seeds)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    save(fig, out_name)


if __name__ == "__main__":
    # quick manual check that the pipeline diagram and the per-run figures work
    from data import load_adult
    from experiment import run_all

    style()
    plot_pipeline_diagram()
    df = load_adult()
    plot_group_distribution(df, attr="sex")
    r = run_all(seeds=[0], sensitive_attr="sex", model_type="logreg")[0]
    plot_confusion_matrices(r["y_test"], r["preds"])
    plot_group_bars(r["y_test"], r["preds"], r["sensitive_test"])
    plot_tradeoff(r["y_test"], r["baseline_probs"], r["sensitive_test"], r["scores"])
    plot_roc(r["y_test"], r["baseline_probs"], r["sensitive_test"], attr="sex")
    plot_feature_importance(r["baseline_model"], r["feature_names"])
