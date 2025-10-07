# 🚀 Spaceship Titanic — Optimized Weighted Ensemble (v80)

This repository contains my **top-performing machine learning solution** for the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic).  
It uses a **clean, feature-engineered pipeline** combining **Gradient Boosting** and **Logistic Regression** in a **tuned ensemble with calibrated probabilities**.

---

## 📊 Performance Overview

| Metric | Public Leaderboard | Validation (local) |
|:-------|:------------------:|:-----------------:|
| **Score (Accuracy)** | **0.80149** | ~0.80 |
| **Rank** | **#499 / ~3500** | — |
| **Percentile** | **Top ~14% globally** | — |

**Result:** This model placed comfortably in the top tier of submissions, showing strong generalisation and a well-balanced trade-off between accuracy and interpretability.

---

## 🧠 Approach

This project focuses on building a **light, interpretable, and efficient ensemble** capable of competing with deep and complex models.  
Instead of brute-force parameter search, the pipeline was built around **thoughtful feature design, calibration, and ROC-optimized blending**.

### Key Principles
- **Reproducibility:** deterministic randomness (`random_state=42`) and no external dependencies beyond scikit-learn.
- **Interpretability:** features and model architecture kept transparent.
- **Efficiency:** full runtime under 60 seconds on a standard laptop.
- **Data Integrity:** no leakage — all parameters tuned only on training/validation data.

---

## ⚙️ Pipeline Summary

### 1. Data Cleaning
- Handled all missing numerical data using **median imputation**.  
- Categorical variables (e.g., `HomePlanet`, `Destination`, `CabinDeck`) filled with **mode values**.
- Converted target `Transported` to integer (0/1) format.

### 2. Feature Engineering
- **Cabin decomposition** into `Deck`, `Num`, and `Side` for better spatial context.  
- **Spending patterns:** total, per-age, and log-transformed spending.  
- **Group-based features:** group size and solo traveler flags derived from passenger IDs.  
- **Behavioural flags:** cryo-sleep and VIP status as binary indicators.  
- **Label encoding:** compact integer encoding for small categorical fields.

### 3. Model Architecture
- **Gradient Boosting Classifier (scikit-learn)**  
  - Tuned for moderate learning rate (`0.04`) and isotonic calibration to improve probability reliability.  
- **Logistic Regression**  
  - Provides stable, interpretable linear separation on engineered features.  
- **Weighted Ensemble**  
  - Automatic blending ratio optimized by **ROC-AUC grid search** across validation folds.  
  - Ensemble probabilities further thresholded to maximize validation accuracy.

### 4. Ensemble Optimization
- The ensemble balance between GB and LR was iteratively optimized between **0.0 → 1.0** (in 0.05 increments).
- Optimal weight for GB was approximately **0.55–0.70**, depending on split.  
- Optimal probability cutoff determined via **Youden’s J statistic (TPR – FPR)**.

---

## 📦 Technologies Used

| Category | Libraries |
|-----------|------------|
| **Core ML** | `scikit-learn` |
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib` (optional) |
| **Environment** | Python ≥3.9 |

---

## 🗂️ Repository Structure

