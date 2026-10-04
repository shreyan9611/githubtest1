# COMP813 Final Project: code specification and handoff

Drop this file in the project root (rename to CLAUDE.md if you want Claude Code to read it automatically).

## 1. What we are building

A 2-person COMP813 final project (60% of the course grade; code is 15 of 60 marks). Topic: **accuracy vs. fairness trade-offs in income prediction** on the UCI Adult dataset.

The experiment grid is 2 sensitive attributes x 2 baseline models x up to 3 methods:

- Sensitive attributes: `sex`, `race` (race grouped as White / Non-White, see section 5.1)
- Baseline models: Logistic Regression (`logreg`), small MLP (`mlp`)
- Methods per run: `baseline` (plain model), `threshold` (post-processing, Fairlearn ThresholdOptimizer), `reductions` (in-processing, Fairlearn ExponentiatedGradient; **logreg only**)
- Metrics: accuracy, Demographic Parity Difference (DP), Equalized Odds Difference (EO)
- 5 random seeds (0 to 4), report mean +/- std
- Output: CSV tables + 10 figure types that go straight into a CVPR-style report

Headline finding so far: the threshold fix cuts DP by ~97% for a ~2 point accuracy drop on Sex, but EO more than triples. The reductions method is a third data point for that tension.

## 2. Hard requirements from the assignment brief (code deliverable)

- Submit a ZIP of runnable scripts/modules. No venv, no model checkpoints, no large files.
- README with instructions and dependencies. **Must be `README.txt`** (plain text, not .md).
- Code must be documented (simple comments) and **must reproduce the numbers in the report**.
- `python main.py` must regenerate every number and figure used in the report.

## 3. Mistakes from earlier assignments that cost marks (do not repeat)

1. Report text citing numbers that did not match the final code output. All report numbers must come from ONE final full run.
2. Pasting the same pipeline code into every task. Extend functions with parameters instead of copy-pasting (this is why `model_type` and `sensitive_attr` are parameters).
3. Repeated imports, and reusing generic variable names that shadow something earlier (e.g. a results dict named `data`).
4. Redundant charts that only restate a number already in a table next to them. Every figure must show something a table cannot.

## 4. Environment (pins matter)

Python 3.12. Create `requirements.txt` with exactly:

```
pandas==3.0.2
numpy==2.4.4
scipy==1.17.1
scikit-learn==1.5.2
fairlearn==0.14.0
matplotlib==3.10.8
```

- **scikit-learn must stay below 1.6.** Newer versions renamed an argument that fairlearn 0.14.0 still calls by the old name, which crashes ThresholdOptimizer (`check_array() got an unexpected keyword argument 'force_all_finite'`).
- That same old version has no `sample_weight` support in `MLPClassifier.fit`, so ExponentiatedGradient cannot wrap the MLP. This is why reductions is logreg-only. State this in the report.
- The `ucimlrepo` package failed with SSL errors in the sandbox, so data is downloaded directly with `urllib` from the UCI archive.

## 5. Module specifications

Keep function names short and simple. Keep comments plain and understandable.

### 5.1 `data.py`

- URLs: `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data` and `adult.test`, cached in `data/`.
- Columns: age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income.
- Read with `sep=r",\s*"`, `engine="python"`, `na_values="?"`. Test file needs `skiprows=1` and a trailing `.` stripped from `income`. Concatenate train + test: **48,842 rows**.
- `download()`, `load_adult()`, `proxy_check(df, sensitive_attr="sex")`, `preprocess(df, seed, sensitive_attr="sex", train_size=0.6, val_size=0.2)`.
- `preprocess`: drop rows with any NA; `y = income == ">50K"`; the chosen sensitive attribute is returned separately and **never used as a model feature**; the other candidate attribute (race or sex) stays in as a normal feature. Split 60/20/20, stratified on `y`, using the seed. One-hot encode categoricals with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` (**must be dense**, ThresholdOptimizer rejects sparse), scale numerics with `StandardScaler`, fit on train only.
- Returns dict keys: `X_train, X_val, X_test, y_train, y_val, y_test, sensitive_train, sensitive_val, sensitive_test, feature_names`.
- **Race grouping happens only when `sensitive_attr == "race"`, inside `preprocess` and `proxy_check`, NOT in `load_adult`.** Reason: the 5 raw categories include groups of 406 and 470 rows, leaving only ~65 people in the smallest test group, which makes per-group metrics unstable. Grouping globally was a bug: it silently changed the sex pipeline's feature count from 101 to 98 and shifted every sex number.
- `proxy_check` returns Cramer's V of every other categorical column against the sensitive attribute.

### 5.2 `model.py`

- `build_model(model_type, param_value, seed=None)`, `search(X_train, y_train, X_val, y_val, model_type="logreg", param_values=None)` returning `(best_value, results_df)`, `train(X_train, y_train, param_value, seed, model_type="logreg")`.
- logreg: `LogisticRegression(C, max_iter=1000)`, grid C in [0.001, 0.01, 0.1, 1, 10, 100].
- mlp: `MLPClassifier(hidden_layer_sizes=(64,), alpha, max_iter=300)`, grid alpha in [0.0001, 0.001, 0.01, 0.1, 1.0].
- Pick the best value by validation accuracy. `results_df` column is named `C` or `alpha`, plus `val_accuracy`.
- MLP search takes ~58 s per seed and prints ConvergenceWarnings. This is known and accepted (decided not to raise `max_iter` because it would make it slower).

### 5.3 `fairness.py`

- `dp_diff`, `eo_diff` via `fairlearn.metrics.demographic_parity_difference` / `equalized_odds_difference`.
- `adjust_threshold(model, X_train, y_train, group_train)`: `ThresholdOptimizer(estimator=model, constraints="demographic_parity", predict_method="predict_proba", prefit=True)`. **Predict with `random_state=seed`**, otherwise results change on every run (this was a real reproducibility bug). Needs the sensitive attribute at prediction time.
- `train_reductions(X_train, y_train, group_train, param_value)`: `ExponentiatedGradient(build_model("logreg", param_value), constraints=DemographicParity())`. Predict with `random_state=seed`. Does NOT need the sensitive attribute at prediction time (worth a sentence in the report). ~4 s per seed.
- `evaluate(y_true, y_pred, group)` returns `{"accuracy", "dp_diff", "eo_diff"}`.

### 5.4 `experiment.py`

- `SEEDS = [0,1,2,3,4]`, `METHOD_ORDER = ["baseline", "threshold", "reductions"]`.
- `run_seed(df, seed, sensitive_attr, model_type)` returns a dict with: `seed, sensitive_attr, model_type, best_value, scores` (method -> metrics), `preds` (method -> predictions), `search_results, y_test, sensitive_test, baseline_probs, baseline_model, feature_names`. `reductions` is only present when `model_type == "logreg"`.
- `run_all(seeds, sensitive_attr, model_type)`, `to_table(runs)` (one row per seed per method, columns `seed, model, accuracy, dp_diff, eo_diff`), `aggregate(results_df)` (mean and std per method, rows in `METHOD_ORDER`).

### 5.5 `visualize.py`

House style (inspired by Datawrapper): muted colours, no box border, light horizontal gridlines, bold left-aligned title with grey subtitle, direct value labels, bars start at zero. **One colour per method everywhere**: baseline navy `#2C4A6E`, threshold coral `#E07856`, reductions teal `#4E9A8C`. All text black or dark grey. Each function saves a PNG at dpi 150 into `outputs/figures/` and takes an optional `tag` that is appended to the filename.

| File | Content |
|---|---|
| `01_{attr}_distribution.png` | Bar chart of category counts (once per attribute; rotate labels when >2 categories) |
| `02_pipeline_diagram.png` | Flow diagram with branching: Raw -> Preprocessing -> (LogReg, MLP) -> (Reductions from LogReg only, Threshold from both) -> Evaluation, plus a note about the sensitive attribute |
| `03_confusion_matrices[_tag].png` | One matrix per method, side by side |
| `04_per_group_bars[_tag].png` | Accuracy / TPR / FPR per group, one bar per method |
| `05_tradeoff_curve[_tag].png` | Accuracy vs DP while sweeping one global threshold on baseline probabilities, with each intervention's point marked. Key figure: per-group thresholding beats any single global threshold |
| `06_roc_by_{attr}[_model].png` | Baseline ROC per group with AUC |
| `07_feature_importance[_tag].png` | Top-15 coefficients, **logreg only** (MLP has no single coef vector) |
| `08_seed_stability[_tag].png` | Dots per seed per method for the three metrics |
| `09_method_comparison_accuracy.png` | One panel per attribute, every (model, method) as a point: x = DP, y = accuracy, +/- std error bars. Marker = model, colour = method |
| `10_method_comparison_eo.png` | Same but y = EO. Shows fixing DP can worsen EO |

Tag rule: join the non-default parts of (`attr` if not "sex", `model_type` if not "logreg"), giving `None`, `mlp`, `race`, `race_mlp`. The sex + logreg figures keep untagged filenames.

### 5.6 `main.py`

- `run(seeds=None)`: load data once; style; pipeline diagram once; loop attributes x model types, calling `run_combination`; then the two comparison figures.
- `run_combination` writes `outputs/results_{attr}_{model}.csv` and `outputs/summary_{attr}_{model}.csv`, prints both tables, and makes the per-combination figures from the first seed's run (`runs[0]`). Single-run figures therefore use seed 0; tables use all 5 seeds. The report must say so.

## 6. Reference numbers for regression tests (seed 0, from verified runs)

Format: accuracy / DP / EO.

| Case | baseline | threshold | reductions |
|---|---|---|---|
| sex, logreg | 0.846545 / 0.183964 / 0.079934 | 0.823770 / 0.012187 / 0.328865 | 0.826976 / 0.025821 / 0.287634 |
| race, logreg | 0.848203 / 0.091950 / 0.060494 | 0.846103 / 0.005026 / 0.175005 | 0.843560 / 0.002516 / 0.097241 |
| sex, mlp | 0.851520 / 0.180779 / 0.071431 | 0.831177 / 0.020120 / 0.306061 | n/a |
| race, mlp | 0.853621 / 0.095104 / 0.066709 | 0.851962 / 0.003604 / 0.169779 | n/a |

Other checks:
- sex + logreg, 5 seeds, baseline: accuracy 0.847474 (std 0.004598), DP 0.183210 (0.005310), EO 0.093758 (0.022426). Threshold: 0.826954 (0.004842), 0.005455 (0.004781), 0.324613 (0.005499).
- Feature counts: 101 encoded features when sex is sensitive, 98 when race is.
- Seed 0 split sizes 27132 / 9045 / 9045. Majority-class test accuracy 0.7521.
- logreg search, seed 0, sex: best C = 0.1, validation 0.857048, test accuracy 0.8465.
- Running the same seed twice must give identical numbers (this caught the ThresholdOptimizer randomness bug).

## 7. Current state (honest)

Verified by running: `data.py`, `model.py`, `fairness.py` (all three methods), `experiment.py` for race + logreg.

**Written but NOT yet run end to end:**
- `experiment.py` refactor for sex and for the mlp path
- `visualize.py` (syntax checked only). Specifically check by eye: new pipeline diagram (arrows, text fitting inside boxes, note at the bottom not clipped), `plot_method_comparison` (legend placement, shared axes, NaN error bars if only 1 seed), group bars with 3 methods (value labels may crowd)
- `main.py` (syntax checked only)

**Missing:** `requirements.txt`, `README.txt` (lost in a sandbox reset), the final run, the ZIP.

## 8. Remaining tasks, in order

1. Run each module's `__main__` check, fix errors.
2. Smoke test: `python -c "import main; main.run(seeds=[0,1])"` (several minutes, MLP is slow; run in the background).
3. Open every figure and fix layout problems.
4. Write `requirements.txt` and `README.txt`. README needs: setup, how to run everything, how to run each baseline / intervention / attribute, how to reproduce the report numbers, the real runtime, and the logreg-only and scikit-learn pin notes.
5. Lint for the lessons in section 3: no duplicate imports, no shadowed names, no copy-pasted blocks.
6. Run the full `python main.py` once, save the console output to a log. This run is the only source of numbers for the report. Expect roughly 10 to 15 minutes.
7. Run it twice and confirm the CSVs are identical.
8. ZIP: `*.py`, `requirements.txt`, `README.txt`. Exclude `data/`, `__pycache__/`, and large files.

## 9. Not code, but depends on it

- The Word report and checklist use the old single-intervention wording and old numbers. After step 6, every number in the Abstract, Introduction, Methods, Experiments and Conclusion must be rewritten from the final log, with new sections for baseline-vs-baseline, threshold-vs-reductions, and sex-vs-race.
- Figure numbering in the report follows document order; update it when the new figures are added.

## 10. Decisions already made (do not reopen)

- Race grouped White / Non-White inside `preprocess` only.
- Reductions on logreg only (technical limit, not a choice).
- Feature importance for logreg only.
- 50/50 contribution split between the two authors, no module owners.
- Short function names, plain comments.
