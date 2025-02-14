![Uploading image.png…]()

# Star, Galaxy, and Quasar Classification

This repository demonstrates a step-by-step data science project for classifying astronomical objects (stars, galaxies, and quasars) using a dataset from the Sloan Digital Sky Survey (SDSS). The code follows a **10-section** structure, with each section focusing on a specific phase of the pipeline: data preprocessing, training multiple models, clustering, interpreting feature importance, and investigating redshift distributions.

## Table of Contents

1. [Data Loading and Basic Preprocessing](#1-data-loading-and-basic-preprocessing)  
2. [Encoding the Target Column](#2-encoding-the-target-column)  
3. [Train-Test Split and Scaling](#3-train-test-split-and-scaling)  
4. [RandomizedSearchCV for RandomForest](#4-randomizedsearchcv-for-randomforest)  
5. [Training CatBoost, SVM, and MLP](#5-training-catboost-svm-and-mlp)  
6. [Classification Reports and Confusion Matrices](#6-classification-reports-and-confusion-matrices)  
7. [Correlation Matrix Heatmap](#7-correlation-matrix-heatmap)  
8. [Feature Importances (RandomForest)](#8-feature-importances-randomforest)  
9. [K-Means Clustering and Cluster Analysis](#9-k-means-clustering-and-cluster-analysis)  
   9.1. [Checking Detected Clusters vs. Original Classes](#91-checking-detected-clusters-vs-original-classes)  
10. [Interpreting the Redshift Distributions by Class](#10-interpreting-the-redshift-distributions-by-class)  

---

## 1. Data Loading and Basic Preprocessing
- We read in the CSV file (`star_classification.csv`) and remove non-informative ID columns (`obj_ID`, `run_ID`, `fiber_ID`, etc.).
- This leaves us primarily with photometric magnitudes (`u, g, r, i, z`), redshift, and the sky coordinates (`alpha`, `delta`), plus the `class` column indicating Star, Galaxy, or QSO (quasar).

## 2. Encoding the Target Column
- The `class` column is label-encoded into numeric values (e.g., 0 for Star, 1 for Galaxy, 2 for QSO).
- A quick printout of `label_encoder.classes_` helps confirm which numeric label maps to which original class name.

## 3. Train-Test Split and Scaling
- We split the dataset into **training** (80%) and **testing** (20%) sets.
- We apply `StandardScaler` to the features to standardize them, which often improves model performance.

## 4. RandomizedSearchCV for RandomForest
- We define a parameter grid for RandomForest (e.g., `n_estimators`, `max_depth`, etc.).
- `RandomizedSearchCV` is used to find a good combination of hyperparameters.
- The best model is then **fitted** on the training set and **evaluated** on the test set.

## 5. Training CatBoost, SVM, and MLP
- Additional classifiers are trained:
  - **CatBoostClassifier** (gradient boosting on decision trees).
  - **SVM** (with a linear kernel).
  - **MLPClassifier** (feedforward neural network).
- We compare their accuracies on the test set.

## 6. Classification Reports and Confusion Matrices
- We use `classification_report` to see precision, recall, and F1-score for each class (Star, Galaxy, QSO).
- We also display **confusion matrices** to visualize misclassifications:
  - Typically, **Stars** are easiest to classify (near-zero redshift).
  - **Galaxies** and **Quasars** can get confused if redshifts overlap.

## 7. Correlation Matrix Heatmap
- We generate a heatmap of Pearson correlations between features:
  - Photometric magnitudes (`u, g, r, i, z`) often show strong positive pairwise correlations (e.g., `r` vs. `i`).
  - **alpha** and **delta** (sky coordinates) typically do **not** correlate linearly with brightness or class, which makes astronomical sense: the position on the sky does not inherently tell you if an object is a star, galaxy, or quasar.
  - `redshift` has moderate correlation with certain magnitudes.  
- This matrix helps identify which features move together (+ correlation) or inversely (– correlation).

## 8. Feature Importances (RandomForest)
- Using `rf_optimized.feature_importances_`, we see which features are most influential in the RandomForest model.
- Often, `redshift` and certain magnitude filters (or color indices) rank highly.

## 9. K-Means Clustering and Cluster Analysis
- We perform an **Elbow Method** and **Silhouette Analysis** to decide on the optimal number of clusters.
- Typically, around 2–3 clusters emerge strongly. We then examine how the clusters map to the original classes (Star, Galaxy, QSO).

### 9.1. Checking Detected Clusters vs. Original Classes
- We compare the number of discovered clusters to the original number of classes.
  - If there are more clusters, we investigate the potential new clusters.
  - If fewer, some classes might have merged.
- A crosstab (`pd.crosstab`) shows which class each cluster predominantly corresponds to.

## 10. Interpreting the Redshift Distributions by Class
- **Stars (Class 0)**: Usually near-zero cosmological redshift; appear as a tight peak near 0 in histograms/KDE plots.
- **Galaxies (Class 1)**: Range of distances (moderate redshifts), creating a broader distribution.
- **Quasars (Class 2)**: Typically the most distant objects, so they can have **very high** redshift, sometimes exceeding 3 or 4.  
- We limit the x-axis to the 99th percentile of `redshift` to avoid a few extreme outliers stretching the entire plot.
- The **alpha** and **delta** coordinates do not show linear correlation with redshift or brightness (magnitudes), which is expected astronomically.

---

## How to Run This Project
1. **Clone or download** the repository with the CSV file and the Jupyter Notebook (or .py script).
2. **Install requirements** (pandas, numpy, scikit-learn, catboost, tensorflow, tqdm, seaborn, matplotlib, torch, kan).
3. **Open** the notebook or run the Python script.
4. **Follow the sections** in order:
   - Data loading, cleaning, label encoding
   - Model training (RandomForest, CatBoost, SVM, MLP)
   - Evaluation (classification reports, confusion matrices)
   - Feature importances, correlation heatmap
   - K-Means clustering
   - Redshift analysis

## Results Overview
- **Model Accuracy**: RandomForest and CatBoost often perform particularly well. SVM and MLP are also strong but can vary in how they confuse galaxies and quasars.
- **Confusion Matrices**: Stars are almost always correct; main confusion is Galaxy vs. Quasar.
- **Clustering**: K-Means usually detects 2–3 meaningful clusters that align (roughly) with the 3 classes.
- **Redshift**: A crucial feature for separating stars (low redshift) from galaxies and quasars (moderate/high redshift). Quasars can be high enough redshift to stand out clearly.

## License and Credits
- **Data** sourced from the [Sloan Digital Sky Survey (SDSS)](https://www.sdss.org/).  
- **Project** developed by Perevalov Kirill 
.

---
