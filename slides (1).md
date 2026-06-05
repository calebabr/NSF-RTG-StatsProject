Overall Project Pipeline 

Raw Data or Simulated SPD Matrices 

*↓* 

SPD Matrices *C*1*, . . . , CN*   
*↓* 

Choose Geometry: AI, LogE, BW 

*↓*   
Barycenter *C*¯ and Tangent Projection 

*↓* 

Dimension Reduction: PCA or UMAP 

*↓* 

Visualization, Geometry Metrics, and Classification  
Dimension Reduction and Manifold Learning 

▶ Many datasets exist in very high-dimensional spaces, making visualization and learning difficult. 

▶ Dimension reduction attempts to represent the data in a lower-dimensional space while preserving important structure. 

▶ Linear methods such as PCA assume the data lie near a linear subspace. 

▶ Manifold learning methods such as UMAP assume the data may lie on a curved nonlinear manifold and attempt to preserve local geometric structure and neighborhoods. 

▶ In this project, there are two different manifold ideas: 

1\. The original SPD matrices already lie on a Riemannian manifold defined by AI, LogE, or BW geometry. 

2\. After tangent projection, we apply manifold learning methods such as UMAP to study whether the tangent features themselves have lower-dimensional nonlinear structure.  
Dimension Reduction Methods 

▶ PCA 

▶ linear dimension reduction   
▶ useful for studying global variance structure   
▶ gives interpretable variance explained and loadings 

▶ UMAP 

▶ nonlinear dimension reduction   
▶ useful for studying local neighborhood structure   
▶ can be run on tangent features or on precomputed manifold distances 

▶ More options 

▶ Isomap   
▶ Spectral embedding   
▶ Autoencoder  
Task 1: Compare Manifold UMAP vs Tangent-Space UMAP 

A. Manifold UMAP 

B. Tangent-Space UMAP 

*D*(*g*)   
*ij* \= *dg* (*Ci, Cj*)*, g ∈ {AI, LogE,BW }* 

*X*(*g*)   
*i* \= vec   
    
log(*g*)   
*C*¯*g*(*Ci*) 


▶ Compute pairwise SPD distances. ▶ Run UMAP(metric="precomputed"). 

▶ This asks what structure is present directly in the manifold distances.   
▶ Compute the metric-specific barycenter. ▶ Project matrices to tangent space. 

▶ Run UMAP(metric="euclidean") on tangent vectors. 

Question: Does tangent projection preserve the same structure as the original manifold distances?  
Task 1: What to Plot and Report 

1\. For each geometry *g ∈ {AI, LogE,BW }*, make two UMAP plots: ▶ manifold-distance UMAP   
▶ tangent-space UMAP 

2\. Color each point by class label or group. 

Geometry Manifold UMAP Tangent UMAP 

AI plot plot 

LogE plot plot 

BW plot plot  
Metrics for Task 1 

▶ Neighborhood Overlap 

Measures how many nearest neighbors from the original manifold distances are preserved after tangent projection. 

Python: sklearn.neighbors.NearestNeighbors 

▶ Trustworthiness 

Measures whether points that appear close in the embedding were also close in the original space. 

Python: sklearn.manifold.trustworthiness 

▶ Spearman Distance Correlation 

Measures whether pairwise distance rankings are preserved between manifold distances and tangent-space distances. 

Python: scipy.stats.spearmanr  
Task 2: Dimension Reduction for Classification 

▶ Build three classification pipelines for each geometry: AI, LogE, and BW. ▶ Pipeline A: Full tangent features 

*Ci → X*(*g*)   
*i →* Classifier 

▶ Pipeline B: PCA-reduced tangent features 

*Ci → X*(*g*)   
*i →* PCA*q*(*X*(*g*)   
*i*) *→* Classifier 

▶ Pipeline C: UMAP-reduced tangent features 

*Ci → X*(*g*)   
*i →* UMAP*q*(*X*(*g*)   
*i*) *→* Classifier 

▶ Classifiers: *k*\-nearest neighbors, logistic regression, SVM, etc.  
Task 2: Classification Comparisons 

1\. Use the same train/test splits for AI, LogE, and BW. 

2\. Compare latent dimensions: 

*q ∈ {*5*,* 10*,* 20*, ...}* 

3\. Fit dimension reduction only on the training set. 

4\. Transform validation/test observations using the fitted PCA or UMAP object. 5\. Report a table of classification metrics:   
▶ Accuracy   
▶ Standard Deviation   
▶ Precision   
▶ Recall   
▶ AUC   
▶ Runtime  
Conceptual Algorithm: Nested CV Classification 

Outer loop: 5-fold cross-validation 

1\. Split the data into 5 outer folds. 

2\. For each outer fold: 

2.1 Hold out one fold as the test set. 

2.2 Use the remaining four folds as the training set. 

2.3 Compute the barycenter using only the training SPD matrices. 

2.4 Project training and test matrices to tangent space using the training barycenter. 2.5 Half-vectorize (i.e. flatten matrix and take only upper or lower triangle) ▶ *If standardizing: standardize train matrices and fit test matrices using training standarization* 

2.6 Tune dimension reduction and classifier parameters using inner CV on the training set only. 

2.7 Refit the selected pipeline on the full outer-training set. 

2.8 Evaluate on the held-out outer-test set. 

3\. Average performance across the 5 outer folds.  
Task 3: Raw vs Standardized Tangent Features 

Question 1: Do certain metrics naturally produce better-scaled tangent features? ▶ Compute the standard deviation of each tangent feature: 

*sj* \= sd(*xj*) 

where *xj*is the *j*\-th tangent-space feature across all samples. 

▶ Create boxplots or line plots of feature standard deviations for AI, LogE, and BW.  
Task 3: Raw vs Standardized Tangent Features 

Question 2: Does standardization change the latent structure? ▶ Standardize each tangent feature: 

*x*˜*ij* \=*xij − µj*   
*σj* 

▶ Run PCA on both raw and standardized tangent features. ▶ Plot cumulative variance explained: 

P*q* 

CVE(*q*) \=   
*j*\=1 *λj*   
~~P~~*d*   
*j*\=1 *λj*  
Task 3: Raw vs Standardized Tangent Features 

Question 3: Does standardization affect classification performance? ▶ Compare accuracy using: 

▶ raw tangent features   
▶ standardized tangent features 

▶ Create grouped barplots for AI, LogE, and BW.  
Datasets for the Project 

▶ Synthetic SPD matrices 

▶ controlled class structure   
▶ known source of variation   
▶ fast enough to rerun many times 

▶ ETH-80 image covariance   
descriptors 

▶ one covariance matrix per image ▶ use image features such as 

position, color, and gradients 

▶ EEG / BCI or fMRI 

▶ one covariance matrix per EEG 

trial   
▶ one covariance matrix per brain 

scan