# KTH-TIPS2 SPD Geometry Analysis

Compares SPD manifold geometries (AI, LogE, BW) and dimension reduction methods (PCA, UMAP) on image covariance descriptors from the KTH-TIPS2-b texture dataset. Part of the Auburn University NSF RTG statistics project.

## Project pipeline

```
KTH-TIPS2 images
      ↓
22×22 covariance matrices  (one per image, shape (n, 22, 22))
      ↓
Choose geometry: AI | LogE | BW
      ↓
Barycenter + tangent projection  →  253-dim feature vectors
      ↓
Dimension reduction: PCA or UMAP
      ↓
Visualization, geometry metrics, classification
```

## Repository layout

```
STATS_Project/
├── analysis.py                      # Main script — all three tasks
├── metric_functions.py              # SPD geometry library
├── guide.txt                        # Task specification and data guide
├── KTH_TIPS2_dataset_description.txt
├── data/
│   └── KTH-TIPS2-b/
│       ├── covariance_matrices.npz  # X: (4752,22,22), y: labels
│       └── metadata.csv             # material, sample, scale, image_label
├── figures/                         # Output plots (generated on run)
└── results/                         # Output CSVs + cached distance matrices
```

`results/*.npy` (cached pairwise distance matrices) are excluded from git — they are recomputed automatically on first run and can be large.

## Tasks

### Task 1 — Manifold UMAP vs Tangent-Space UMAP

For each geometry, two UMAP embeddings are computed and compared:

- **Manifold UMAP** — runs `UMAP(metric="precomputed")` on the full n×n pairwise SPD distance matrix
- **Tangent UMAP** — projects matrices to tangent space first, then runs `UMAP(metric="euclidean")`

Comparison metrics: neighborhood overlap, trustworthiness, Spearman distance correlation.

Output: `figures/task1_umap_grid.png`, `results/task1_metrics.csv`

### Task 2 — Raw vs Standardized Tangent Features

Studies whether standardizing tangent features changes downstream results:

- Feature standard deviation boxplots (raw vs. standardized, all geometries)
- PCA cumulative variance explained curves
- Classification accuracy comparison (5-fold CV, KNN k=5)

Output: `figures/task2_feature_stds_boxplot.png`, `figures/task2_pca_cve.png`, `figures/task2_classification_barplot.png`, `results/task2_standardization_accuracy.csv`

### Task 3 — Downstream Classification with Nested CV

Nested 5-fold (outer) / 3-fold (inner) cross-validation over 54 combinations:

- 3 geometries × 3 pipelines × 2 standardization settings × 3 classifiers
- **Pipeline A** — full tangent features → classifier
- **Pipeline B** — tangent features → PCA(q) → classifier
- **Pipeline C** — tangent features → UMAP(q) → classifier
- Classifiers: KNN, logistic regression, SVM
- Latent dimensions q ∈ {5, 10, 20}

Data leakage is prevented: barycenter, scaler, PCA/UMAP, and classifier are all fit on the training fold only.

Output: `figures/task3_accuracy_heatmap.png`, `results/task3_classification_results.csv`

## Setup

```bash
pip install numpy pandas scipy scikit-learn umap-learn joblib matplotlib
```

Python 3.9+ recommended.

## Running

```bash
python analysis.py
```

Edit the configuration block at the top of `analysis.py` before running:

```python
SUBSET      = "sample_a"   # "debug" | "sample_a" | "sample_ab" | "full"
METRICS     = ["bw", "ai", "loge"]
LATENT_DIMS = [5, 10, 20]
N_OUTER     = 5
N_INNER     = 3
N_JOBS      = -1           # -1 = all CPU cores
```

### Recommended progression

Start small and scale up once the pipeline runs end-to-end:

| Subset | n | Notes |
|---|---|---|
| `"debug"` | ~396 | sample_a, scales 2/6/10 only |
| `"sample_a"` | ~1,188 | all of sample_a |
| `"sample_ab"` | ~2,376 | sample_a + sample_b |
| `"full"` | 4,752 | all four samples — slow |

Pairwise distance matrices are O(n²) and are the bottleneck. They are cached to `results/pairwise_D_{metric}_{subset}.npy` after the first run and loaded automatically on subsequent runs.

### Parallelism note

Task 3 uses `joblib.Parallel(n_jobs=-1)` which claims all CPU cores. If running alongside another multiprocessing script, set `N_JOBS` to half the available cores to avoid over-subscription:

```python
N_JOBS = 4   # instead of -1
```

## Dataset

**KTH-TIPS2-b** — 11 material texture classes (aluminium foil, brown bread, corduroy, cork, cotton, cracker, lettuce leaf, linen, white bread, wood, wool). Each image is represented by a 22×22 covariance matrix computed from 22 per-pixel features: spatial coordinates (x, y), color channels (R, G, B, gray), edge responses (Sobel x/y, gradient magnitude, Laplacian), and 12 Gabor texture responses (4 orientations × 3 frequencies).

After tangent projection and half-vectorization, each matrix becomes a **253-dimensional feature vector**.

## SPD geometry library (`metric_functions.py`)

| Function | Description |
|---|---|
| `compute_mean_projection(X_ppn, metric)` | Barycenter via BW/AI iteration or LogE closed form |
| `project_stack_to_tangent_features(X, M, metric)` | Tangent projection + half-vectorization → (n, 253) |
| `compute_pairwise_distance_matrix(X, metric)` | n×n distance matrix for manifold UMAP |
| `stack_npp_to_ppn(X)` | Shape conversion (n,p,p) → (p,p,n) required by mean functions |

Supported metric strings: `"bw"`, `"ai"`, `"loge"` (and common aliases).

Matrix convention: mean functions expect **(p, p, n)**; all other functions accept both **(n, p, p)** and **(p, p, n)**.
