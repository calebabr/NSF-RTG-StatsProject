# KTH-TIPS2 SPD Geometry Project Plan

## Goal

Compare SPD manifold geometries (AI, LogE, BW) and dimension reduction methods (PCA, UMAP)
on the KTH-TIPS2 texture dataset. All work lives in a single Python script: `analysis.py`.

---

## Repository Layout

```
STATS_Project/
├── data/
│   └── KTH-TIPS2-b/
│       ├── covariance_matrices.npz
│       └── metadata.csv
├── metric_functions.py       # provided helper; one bug fix noted below
├── analysis.py               # THE script (to be written)
├── figures/                  # created by analysis.py on first run
└── results/                  # created by analysis.py on first run
```

---

## Known Bug to Fix in metric_functions.py

Line 711 calls `ai_projection_mean_stable(...)`, which is not defined — only
`ai_projection_mean` exists. The script (and this plan) will treat them as the same
function. Before running, rename that call:

```python
# line 711 in metric_functions.py
# BEFORE:  M, iters, final = ai_projection_mean_stable(...)
# AFTER:
M, iters, final = ai_projection_mean(X, tol=tol, max_iter=max_iter, verbose=verbose)
```

---

## Script Structure (`analysis.py`)

### Section 0 — Imports and Configuration

```python
# ── User-facing knobs ────────────────────────────────────────────────────────
SUBSET      = "debug"   # "debug" | "sample_a" | "sample_ab" | "full"
METRICS     = ["bw", "ai", "loge"]
LATENT_DIMS = [5, 10, 20]
N_OUTER     = 5          # outer CV folds
RANDOM_SEED = 42
N_JOBS      = -1         # joblib: -1 = use all CPU cores
DATA_DIR    = "data/KTH-TIPS2-b"
FIG_DIR     = "figures"
RES_DIR     = "results"
```

Imports needed: `numpy`, `pandas`, `matplotlib`, `sklearn` (PCA, KNN, LogReg, SVC,
StandardScaler, StratifiedKFold, GridSearchCV, NearestNeighbors, trustworthiness,
classification_report, roc_auc_score), `scipy.stats.spearmanr`, `umap`, `joblib`
(`Parallel`, `delayed`), `time`, `os`.

---

### Section 1 — Data Loading

```python
data = np.load("covariance_matrices.npz", allow_pickle=True)
X_full = data["X"]   # (4752, 22, 22)
y_full = data["y"]   # material label strings

meta = pd.read_csv("metadata.csv")
```

**Subset selection** (controlled by `SUBSET`):

| `SUBSET`    | Filter                                    | Approx n |
|-------------|-------------------------------------------|----------|
| `"debug"`   | `sample == "sample_a"` & `scale in {2,6,10}` | 396   |
| `"sample_a"`| `sample == "sample_a"`                    | 1 188    |
| `"sample_ab"`| `sample in {"sample_a","sample_b"}`      | 2 376    |
| `"full"`    | all rows                                  | 4 752    |

After filtering, encode `y` as integers for sklearn compatibility and print a class
distribution summary.

---

### Section 2 — Task 1: Manifold UMAP vs Tangent-Space UMAP

**Research question:** Does tangent projection preserve the same neighborhood structure
as the original Riemannian manifold distances?

#### Step-by-step

For each `metric` in `{bw, ai, loge}`:

**2a. Pairwise distance matrix**

```python
D = compute_pairwise_distance_matrix(X, metric=metric)  # (n, n)
```

Note: O(n²) — slow for large n. Progress is printed by the helper.

**2b. Manifold UMAP**

```python
emb_manifold = umap.UMAP(metric="precomputed", random_state=RANDOM_SEED
                          ).fit_transform(D)   # (n, 2)
```

**2c. Tangent-space features (full dataset, no CV)**

```python
X_ppn = stack_npp_to_ppn(X)
M, info = compute_mean_projection(X_ppn, metric=metric)
Z = project_stack_to_tangent_features(X, M, metric=metric)  # (n, 253)
```

**2d. Tangent-space UMAP**

```python
emb_tangent = umap.UMAP(metric="euclidean", random_state=RANDOM_SEED
                         ).fit_transform(Z)   # (n, 2)
```

**2e. Plots**

Produce a 3×2 grid (rows = geometries, columns = Manifold UMAP / Tangent UMAP),
each scatter plot colored by material label (11 classes, distinct palette).
Save as `figures/task1_umap_grid.png`.

**2f. Comparison metrics**

For each geometry, compute and print:

| Metric | What it measures | Implementation |
|--------|-----------------|----------------|
| **Neighborhood Overlap** | Fraction of k-nearest neighbors (k=15) in manifold-distance space that also appear in tangent Euclidean space | `sklearn.neighbors.NearestNeighbors` on `D` vs on `Z` |
| **Trustworthiness** (manifold UMAP) | Whether embedding neighbors were close in manifold-distance space | `sklearn.manifold.trustworthiness(D, emb_manifold, metric="precomputed")` |
| **Trustworthiness** (tangent UMAP) | Whether embedding neighbors were close in tangent space | `trustworthiness(Z, emb_tangent)` |
| **Spearman distance correlation** | Rank correlation between upper-triangle of `D` and upper-triangle of Euclidean distance matrix of `Z` | `scipy.stats.spearmanr` on flattened upper triangles |

Save a summary table to `results/task1_metrics.csv`.

---

### Section 3 — Task 2: Raw vs Standardized Tangent Features

**Research question:** Does standardizing tangent features change the latent structure,
and does it matter which geometry is used?

This task uses the full selected subset; no cross-validation (exploratory analysis only).

#### Step-by-step

For each `metric` in `{bw, ai, loge}`:

**3a. Compute raw tangent features**

```python
X_ppn = stack_npp_to_ppn(X)
M, _ = compute_mean_projection(X_ppn, metric=metric)
Z_raw = project_stack_to_tangent_features(X, M, metric=metric)  # (n, 253)
```

**3b. Standardize**

```python
scaler = StandardScaler()
Z_std = scaler.fit_transform(Z_raw)
```

**3c. Question 1 — Feature scale variation**

Compute per-feature standard deviations: `Z_raw.std(axis=0)` for each geometry.
Plot side-by-side boxplots (one box per geometry × raw/std, 6 boxes total).
Save as `figures/task2_feature_stds_boxplot.png`.

**3d. Question 2 — PCA cumulative variance explained**

Run `PCA()` on `Z_raw` and `Z_std` for each geometry.
Plot CVE curves (cumulative sum of explained variance ratio vs. number of components)
for all 6 combinations on a single figure.
Save as `figures/task2_pca_cve.png`.

**3e. Question 3 — Classification accuracy comparison**

Use a fixed outer 5-fold stratified CV (same splits for all combinations) with a single
KNN classifier (k=5) to get a fast accuracy estimate — **no nested CV here**, since this
is purely a diagnostic comparison, not the main classification experiment.

For each fold: fit mean and scaler on train only, transform test using train statistics.
Collect accuracy per fold. Report mean ± std.
Produce a grouped barplot: x-axis = geometry, groups = raw vs. std, y = mean accuracy.
Save as `figures/task2_classification_barplot.png`.
Save numeric results to `results/task2_standardization_accuracy.csv`.

---

### Section 4 — Task 3: Downstream Classification with PCA or UMAP (Nested CV)

**Research question:** Which combination of geometry × pipeline × standardization
× latent dimension produces the best classification performance?

#### Pipelines

| Pipeline | Steps after tangent projection |
|----------|-------------------------------|
| **A** | (optional StandardScaler) → Classifier |
| **B** | (optional StandardScaler) → PCA(q) → Classifier |
| **C** | (optional StandardScaler) → UMAP(q, metric="euclidean") → Classifier |

#### Classifiers

`KNeighborsClassifier`, `LogisticRegression(max_iter=1000)`, `SVC(probability=True)`

#### Nested CV Algorithm (from slides)

**Outer loop:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)`

For each outer fold:
1. Split `X` (SPD matrices) and `y` into `X_train_spd`, `X_test_spd`, `y_train`, `y_test`.
2. **Fit SPD barycenter on training matrices only:**
   ```python
   M, _ = compute_mean_projection(stack_npp_to_ppn(X_train_spd), metric=metric)
   ```
3. **Project both sets using the training barycenter:**
   ```python
   Z_train = project_stack_to_tangent_features(X_train_spd, M, metric=metric)
   Z_test  = project_stack_to_tangent_features(X_test_spd,  M, metric=metric)
   ```
4. **Optionally standardize (fit only on train):**
   ```python
   scaler = StandardScaler()
   Z_train = scaler.fit_transform(Z_train)
   Z_test  = scaler.transform(Z_test)         # NOT fit_transform
   ```
5. **Inner CV — `GridSearchCV` on training fold only:**
   - For Pipeline B: search over `q ∈ LATENT_DIMS` and classifier hyperparameters.
   - For Pipeline C: search over `q ∈ LATENT_DIMS` and classifier hyperparameters.
   - For Pipeline A: search over classifier hyperparameters only.
   - Inner CV uses `StratifiedKFold(n_splits=3)` to keep runtime manageable.
   - Hyperparameter grid:
     ```
     KNN:   k ∈ {3, 5, 11}
     LogReg: C ∈ {0.01, 0.1, 1.0, 10.0}
     SVC:   C ∈ {0.1, 1.0, 10.0}, kernel ∈ {"rbf", "linear"}
     ```
6. **Refit best pipeline on full outer-training set.**
7. **Evaluate on held-out outer-test set.**
   Collect: accuracy, precision (macro), recall (macro), AUC (OvR macro), runtime per fold.

**Outer loop averages** the 5 fold scores. Report `accuracy_mean` and `accuracy_std`
(std across outer folds) as separate columns in the results table, matching the slides.

#### Parallelism with joblib

Task 3 has many independent combinations:
`3 geometries × 3 pipelines × 2 standardization settings × 3 classifiers = 54 combinations`,
each running a full nested CV. These are **embarrassingly parallel** — no combination
depends on another's result.

Structure the work as a list of `(metric, pipeline, standardize, classifier_name)` tuples
and dispatch them with `joblib.Parallel`.

**Pre-generate fold indices once** before dispatching so that every combination is
guaranteed to use the exact same train/test splits (slides requirement):

```python
from joblib import Parallel, delayed

outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_SEED)
folds = list(outer_cv.split(X, y))   # list of (train_idx, test_idx) — generated ONCE

def run_one_combination(metric, pipeline, standardize, clf_name,
                        X, y, folds, latent_dims, random_seed):
    # ... full nested CV for this one combination, iterating over folds ...
    return {"metric": metric, "pipeline": pipeline, ...,
            "accuracy_mean": ..., "accuracy_std": ..., ...}

combos = [
    (metric, pipeline, std, clf)
    for metric   in ["bw", "ai", "loge"]
    for pipeline in ["A", "B", "C"]
    for std      in [False, True]
    for clf      in ["knn", "logreg", "svc"]
]

results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(run_one_combination)(*combo, X, y, folds, LATENT_DIMS, RANDOM_SEED)
    for combo in combos
)
```

**Important caveats for parallel safety:**
- Each worker receives its own copy of `X` and `y` (joblib serializes them once via
  memory-mapping when `n_jobs > 1`).
- `umap.UMAP` is not thread-safe but is safe under joblib's default `loky` process
  backend (separate processes, not threads).
- The SPD barycenter iterations inside each worker are fully independent.
- Set `RANDOM_SEED` explicitly in each worker call so results are reproducible.
- `GridSearchCV` itself accepts `n_jobs`; set it to `1` inside `run_one_combination`
  since outer parallelism already saturates the cores.

#### Output

Results table (rows = geometry × pipeline × standardized × classifier × q, columns =
accuracy mean/std, precision, recall, AUC, runtime):
- Printed to stdout
- Saved to `results/task3_classification_results.csv`

Summary heatmaps or bar charts (best accuracy per geometry × pipeline):
- `figures/task3_accuracy_heatmap.png`

---

## Data Size Progression

Follow this order when developing and debugging:

1. `SUBSET = "debug"` (~396 matrices) — verify the full pipeline runs end-to-end
2. `SUBSET = "sample_a"` (~1 188) — check that results are stable
3. `SUBSET = "sample_ab"` (~2 376) — validate scaling behavior
4. `SUBSET = "full"` (4 752) — final results (pairwise distance matrix will be slow)

The pairwise distance matrix (Task 1) is the main bottleneck at O(n²) matrix-distance
evaluations. At n=396, that is ~78 000 pairs per geometry; at n=4752, ~11 million.
Consider caching `D` to disk (`.npy`) so Task 1 distances don't recompute on reruns.

---

## Required Packages

```
numpy
pandas
scipy
matplotlib
scikit-learn
umap-learn          # pip install umap-learn
joblib              # bundled with scikit-learn; no separate install needed
```

---

## Deliverables Summary

| File | Contents |
|------|----------|
| `analysis.py` | Single script implementing all tasks |
| `figures/task1_umap_grid.png` | 3×2 UMAP scatter grid |
| `figures/task2_feature_stds_boxplot.png` | Feature std boxplots |
| `figures/task2_pca_cve.png` | PCA cumulative variance explained |
| `figures/task2_classification_barplot.png` | Raw vs std accuracy barplot |
| `figures/task3_accuracy_heatmap.png` | Classification summary |
| `results/task1_metrics.csv` | Neighborhood overlap, trustworthiness, Spearman ρ |
| `results/task2_standardization_accuracy.csv` | Diagnostic accuracy comparison |
| `results/task3_classification_results.csv` | Full nested CV results table |

---

## Open Questions / Future Extensions

- **Task 1 metrics**: Neighborhood overlap currently compares k-NN structure in
  manifold-distance space vs. tangent Euclidean space (before UMAP). An alternative
  is to compare the two UMAP *embeddings* to each other. The current plan uses the
  pre-UMAP spaces, which measures tangent projection fidelity more directly.
- **UMAP transform for test set**: `umap.UMAP` supports `.transform()` on new points
  in recent versions. Confirm `umap-learn >= 0.5` is installed; earlier versions do
  not support this.
- **Runtime**: If AI or BW barycenter iterations are slow on larger subsets, consider
  loosening `tol` from `1e-6` to `1e-4` as a runtime trade-off.
- **Isomap / Spectral Embedding**: Mentioned in slides as "more options." Not in the
  core plan but can be added as Pipeline D/E after the main three are validated.
