# Machine Learning Based Network Intrusion Detection System

A Machine Learning based Network Intrusion Detection System (NIDS) developed using the **CIC-IDS2017 dataset** to detect and classify benign network traffic and multiple types of network attacks.

The project follows a complete machine-learning pipeline including dataset exploration, data cleaning, exploratory data analysis, feature selection, class balancing using SMOTE, model training, evaluation, and SHAP-based explainability.

---

## Team Members

- Iniya Srilekha B
- Rubini S K
- Dharshini S

**Institution:** Amrita Vishwa Vidyapeetham

---

## Objective

The objective of this project is to develop a machine-learning based Network Intrusion Detection System capable of classifying network traffic into benign traffic and multiple attack categories.

The project focuses on handling the severe class imbalance present in the CIC-IDS2017 dataset and evaluating model performance using **Macro-F1** in addition to accuracy.

---

## Dataset

### CIC-IDS2017

The project uses the **CIC-IDS2017** intrusion detection dataset.

The dataset contains network-flow records representing both benign traffic and different types of cyber attacks.

### Dataset Statistics

| Stage | Result |
|---|---:|
| Combined raw dataset | 2,830,743 rows × 80 columns |
| Cleaned dataset | 2,520,798 rows |
| Original usable features | 78 |
| Selected features | 32 |
| Final test set | 504,160 rows |

The raw CIC-IDS2017 CSV files are **not included in this repository** because of their large size.

---

# Project Workflow

```text
CIC-IDS2017 Dataset
        |
        v
Dataset Exploration
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Feature Selection
   78 → 32 Features
        |
        v
Stratified 80/20 Split
        |
        +----------------------+
        |                      |
        v                      v
     Training              Test Set
        |                  504,160
        v                  untouched
      SMOTE                    |
        |                      |
        v                      |
   Model Training <------------+
        |
        +---- Random Forest
        |
        +---- XGBoost
        |
        +---- LightGBM
        |
        +---- MLP
        |
        v
Model Evaluation
        |
        v
SHAP Explainability
