# Silicone-Nano-composite-Damage-Classification

This repository hosts an end-to-end machine learning and computer vision pipeline designed to analyze structural health and degradation profiles from raw spatial thermal arrays (Testo Excel output grids).

Unlike pure classification architectures, this framework employs an unsupervised-to-supervised transition design: it extracts complex physical, spatial, and textural signatures, compresses them into principal components to discover underlying degradation states via clustering, and subsequently trains classification models to predict these structural damage levels.

## Key Features

**Multimodal Feature Extraction:** Generates 7 distinct structural metrics per frame covering spatial gradients, distribution asymmetry, Newton's law cooling kinetics, and local gray-level matrix dependencies.

**Unsupervised Damage Discovery:** Utilizes Principal Component Analysis (PCA) and $K$-Means Clustering to automatically isolate and order structural health archetypes (Healthy, Moderate Damaged, Severely Damaged) based on calculated surface thermal gradient intensities.

**Supervised Benchmarking Matrix:** Pairs a highly structured Random Forest Classifier ($150$ Estimators, Max Depth = $10$) against a Support Vector Machine (SVM with an RBF kernel) to determine optimal classification stability.

**Comprehensive Validation Split:** Evaluates performance via both a global Stratified 5-Fold Cross-Validation wrapper (for dataset resilience check) and a 20% stratified holdout split (for localized confusion matrices and evaluation metrics).

