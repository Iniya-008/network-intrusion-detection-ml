# Machine Learning Based Network Intrusion Detection System

### Multi-Class Cyber Threat Classification using CIC-IDS2017

A Machine Learning based **Network Intrusion Detection System (NIDS)** designed to classify network traffic as **BENIGN** or one of **14 cyberattack categories** using the CIC-IDS2017 benchmark dataset.

The project implements an end-to-end machine learning pipeline covering data exploration, cleaning, exploratory data analysis, feature selection, class balancing using SMOTE, model training, evaluation, and SHAP-based explainability.

---

## 🎯 Objective

The primary objective of this project is to build a machine-learning based intrusion detection system capable of identifying different types of network attacks from network-flow characteristics.

The project focuses on:

* Processing large-scale network-flow data
* Cleaning and preparing raw network traffic
* Reducing redundant and highly correlated features
* Handling severe class imbalance
* Training and comparing multiple ML models
* Evaluating performance using class-sensitive metrics
* Understanding model predictions using SHAP explainability

Since the dataset is highly imbalanced, **Macro-F1** is considered alongside accuracy to provide a more meaningful evaluation of minority attack classes.

---

## 📊 Dataset

### CIC-IDS2017

The project uses the **CIC-IDS2017** intrusion detection benchmark dataset.

The dataset contains network-flow records representing both normal network activity and multiple categories of cyberattacks.

### Dataset Summary

| Property          |          Value |
| ----------------- | -------------: |
| Source files      |              8 |
| Raw records       |      2,830,743 |
| Raw columns       |             79 |
| Combined dataset  | 2,830,743 × 80 |
| Usable features   |             78 |
| Traffic classes   |             15 |
| Cleaned records   |      2,520,798 |
| Selected features |             32 |
| Test set          |        504,160 |

The 15 traffic classes consist of **BENIGN traffic and 14 attack categories**.

The raw dataset is not included in this repository because of its large size.

---

## 🔄 Project Pipeline

```text
CIC-IDS2017 Dataset
        │
        ▼
Dataset Exploration
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Selection
   78 → 32 Features
        │
        ▼
Stratified 80/20 Split
        │
        ├───────────────┐
        ▼               ▼
   Training          Test Set
        │            504,160
        ▼            Untouched
      SMOTE
        │
        ▼
 Model Training
        │
   ┌────┼────┬────┐
   ▼    ▼    ▼    ▼
  RF   XGB  LGBM  MLP
   │    │    │    │
   └────┼────┴────┘
        ▼
 Model Evaluation
        │
        ▼
SHAP Explainability
```

---

# 🧪 Methodology

## 1. Dataset Exploration

All eight CIC-IDS2017 source files were inspected before modification.

The exploration included:

* Dataset dimensions
* Data types
* Class distributions
* Missing values
* Infinite values
* Duplicate records
* Feature characteristics
* Label inconsistencies

### Result

```text
2,830,743 records
8 source files
15 traffic classes
78 usable features
```

---

## 2. Dataset Combination

The eight source files were combined into a single working dataset.

Column names were standardized because the original files contained inconsistent leading spaces.

A source-file identifier was also retained for traceability.

### Result

```text
2,830,743 rows × 80 columns
```

The original source files were not modified.

---

## 3. Data Cleaning

The raw dataset contained several data-quality issues.

### Duplicate Records

Duplicate records were removed both within individual files and across the combined dataset.

### Missing and Infinite Values

Rows containing missing or infinite values were removed.

These values were primarily associated with zero-duration flows and resulting divide-by-zero calculations.

### Label Correction

Inconsistent/corrupted Web Attack labels were standardized so that equivalent attack categories used consistent names.

### Cleaning Result

```text
Before cleaning : 2,830,743 rows
After cleaning  : 2,520,798 rows
Rows removed    : 309,945
```

Approximately **11% of the original records** were removed during cleaning.

---

# 📈 Exploratory Data Analysis

EDA was performed before model development to understand the characteristics of the cleaned dataset.

The analysis covered:

* Class distribution
* Feature distributions
* Feature correlations
* Outlier comparison between benign and attack traffic

### Main Findings

The dataset is highly imbalanced.

BENIGN traffic accounts for approximately **83%** of the cleaned dataset.

Several attack classes contain extremely few original samples.

Examples:

| Attack Class  | Samples |
| ------------- | ------: |
| Heartbleed    |      11 |
| SQL Injection |      21 |
| Infiltration  |      36 |

EDA visualizations are stored in the `graphs/` directory.

---

# 🧩 Feature Selection

The original dataset contained **78 usable features**.

Feature selection was performed in three stages.

| Selection Method   | Features Removed | Purpose                        |
| ------------------ | ---------------: | ------------------------------ |
| Zero Variance      |                8 | Remove constant features       |
| Correlation > 0.95 |               24 | Remove near-duplicate features |
| VIF                |               14 | Reduce multicollinearity       |

### Final Feature Set

```text
78 Features
     ↓
32 Selected Features
```

The final feature list is available in:

```text
docs/selected_features.txt
```

---

# ⚖️ Handling Class Imbalance

Class imbalance is one of the main challenges in CIC-IDS2017.

The BENIGN class represents approximately **83% of the cleaned dataset**, while some attack categories contain fewer than 40 original records.

Training directly on this distribution can cause a model to favor the majority class.

To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

---

## SMOTE Strategy

SMOTE was applied **only after the train/test split**.

The test set was never oversampled.

Instead of increasing every class to the size of the BENIGN class, a minimum target of:

```text
2,000 samples per minority class
```

was used.

This provides additional representation for minority classes without generating an excessively large synthetic dataset.

### Examples

| Class                      | Real Training Samples | After SMOTE |
| -------------------------- | --------------------: | ----------: |
| Heartbleed                 |                     9 |       2,000 |
| Web Attack - SQL Injection |                    17 |       2,000 |
| Infiltration               |                    29 |       2,000 |
| Web Attack - XSS           |                   522 |       2,000 |
| Web Attack - Brute Force   |                 1,176 |       2,000 |
| Bot                        |                 1,558 |       2,000 |

---

# 🔀 Train/Test Split

A stratified **80/20 train-test split** was used.

Stratification preserves the class distribution between the two sets.

### Test Set

```text
504,160 flows
```

The test set remained completely untouched during:

* SMOTE
* Model training
* Synthetic sample generation
* Model fitting

It was used only for final evaluation.

---

# 🤖 Machine Learning Models

Four machine-learning approaches were evaluated.

## Random Forest

An ensemble learning method based on multiple decision trees.

## XGBoost

A gradient-boosting based classification algorithm.

## LightGBM

A gradient-boosting framework designed for efficient learning on large datasets.

## Multi-Layer Perceptron

A neural-network based classifier used to compare the tree-based models with a neural approach.

---

# 🏋️ Model Training

Due to computational constraints, the tree-based models were trained using a capped, stratified subset of the balanced training data.

The training subset contained:

```text
276,039 rows
```

Large classes were capped at:

```text
60,000 samples per class
```

The complete test set remained untouched and was used for evaluation.

---

# 📏 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro-F1
* Confusion Matrix
* Per-class performance

### Why Macro-F1?

The dataset is heavily imbalanced.

Accuracy can be dominated by the large BENIGN class and therefore may not accurately represent performance on rare attack categories.

Macro-F1 calculates the F1-score for every class and gives each class equal importance.

Therefore, the project considers **Macro-F1 together with accuracy**.

---

# 🏆 Results

All four models were evaluated on the same **504,160-row untouched test set**.

| Model             |   Accuracy | Macro Precision | Macro Recall |   Macro-F1 |
| ----------------- | ---------: | --------------: | -----------: | ---------: |
| **Random Forest** | **99.78%** |      **0.8411** |       0.8870 | **0.8488** |
| XGBoost           |     99.69% |          0.7722 |   **0.9301** |     0.8188 |
| MLP               |     97.08% |          0.6318 |       0.8683 |     0.6501 |
| LightGBM          |     65.03% |          0.1453 |       0.2466 |     0.1612 |

---

# 📌 Results Analysis

### Random Forest

```text
Accuracy        : 99.78%
Macro Precision : 0.8411
Macro Recall    : 0.8870
Macro-F1        : 0.8488
```

The Random Forest experiment achieved the highest reported accuracy and Macro-F1 among the evaluated models.

### XGBoost

```text
Accuracy        : 99.69%
Macro Precision : 0.7722
Macro Recall    : 0.9301
Macro-F1        : 0.8188
```

XGBoost achieved high Macro Recall, indicating strong coverage across attack classes.

### MLP

```text
Accuracy        : 97.08%
Macro Precision : 0.6318
Macro Recall    : 0.8683
Macro-F1        : 0.6501
```

The MLP provided a neural-network comparison with the tree-based approaches.

### LightGBM

```text
Accuracy        : 65.03%
Macro Precision : 0.1453
Macro Recall    : 0.2466
Macro-F1        : 0.1612
```

LightGBM showed substantially lower performance in the current experimental setup. This result is retained for transparency and requires further investigation.

---

# 🔍 Confusion Matrix Analysis

The Random Forest confusion matrix was analyzed using a row-normalized representation.

The results showed strong classification performance for many traffic categories, including:

* BENIGN
* DDoS
* DoS variants
* FTP/SSH-Patator
* Infiltration
* PortScan

Most of these classes were positioned strongly along the diagonal of the confusion matrix.

---

# 🌐 Web Attack Classification

The major classification challenge identified in the experiments was distinguishing between:

* Web Attack - Brute Force
* Web Attack - XSS
* Web Attack - SQL Injection

These categories show overlapping traffic characteristics and have relatively small numbers of original training examples.

Consequently, confusion between these classes remains an important area for improvement.

---

# 🧠 SHAP Explainability

**SHAP (SHapley Additive exPlanations)** was applied to the Random Forest model to understand the features influencing model predictions.

The analysis was performed at both:

* Global level
* Individual attack-class level

SHAP helps answer:

> Which network-flow features contributed to a particular prediction?

---

# 🔬 SHAP Findings

The most influential features identified in the global analysis included:

1. Destination Port
2. Init_Win_bytes_backward
3. Total Length of Fwd Packets
4. Average Packet Size

The per-class analysis also showed that different attack categories rely on different network-flow characteristics.

For example:

### DDoS

Predictions were strongly influenced by packet-length and traffic-volume characteristics.

### PortScan

Predictions relied more strongly on backward-packet-rate and related flow-rate features.

---

# 💡 Key Findings

* CIC-IDS2017 contains severe class imbalance.
* BENIGN traffic represents approximately 83% of the cleaned dataset.
* Some attack classes contain fewer than 40 real examples.
* 309,945 records were removed during data cleaning.
* Feature selection reduced 78 features to 32.
* SMOTE was applied only to the training data.
* The final test set contains 504,160 untouched flows.
* Four machine-learning models were evaluated.
* Random Forest achieved 99.78% accuracy and 0.8488 Macro-F1.
* XGBoost achieved the highest Macro Recall at 0.9301.
* Web Attack subtypes remain difficult to distinguish.
* SHAP identified meaningful network-flow features influencing predictions.
* Accuracy alone is insufficient for evaluating highly imbalanced intrusion-detection data.

---

# 🗂️ Repository Structure

```text
network-intrusion-detection-ml/
│
├── dataset/
│   ├── raw/
│   │   └── MachineLearningCVE/
│   └── processed/
│
├── docs/
│   └── selected_features.txt
│
├── graphs/
│
├── notebooks/
│
├── src/
│   ├── 00_explore_single_file.py
│   ├── 00b_explore_all_files.py
│   ├── 01_combine_files.py
│   ├── 02_clean_data.py
│   ├── 02b_final_dedupe.py
│   ├── 02c_verify_cleaned.py
│   ├── 03_eda.py
│   └── 04_feature_selection.py
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **imbalanced-learn**
* **SMOTE**
* **Random Forest**
* **XGBoost**
* **LightGBM**
* **Multi-Layer Perceptron**
* **SHAP**
* **Matplotlib**

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/Iniya-008/network-intrusion-detection-ml.git
```

## Navigate to the Project

```bash
cd network-intrusion-detection-ml
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

---

# 📥 Dataset Setup

The CIC-IDS2017 dataset is not included in the repository because of its size.

Place the required CSV files inside:

```text
dataset/raw/MachineLearningCVE/
```

The source dataset should remain unchanged.

---

# ▶️ Running the Preprocessing Pipeline

Run the preprocessing scripts in order.

### Dataset Exploration

```bash
python src/00_explore_single_file.py
python src/00b_explore_all_files.py
```

### Dataset Combination

```bash
python src/01_combine_files.py
```

### Data Cleaning

```bash
python src/02_clean_data.py
python src/02b_final_dedupe.py
python src/02c_verify_cleaned.py
```

### Exploratory Data Analysis

```bash
python src/03_eda.py
```

### Feature Selection

```bash
python src/04_feature_selection.py
```

The resulting processed dataset can then be used for the train/test split, SMOTE balancing, model training, evaluation, and SHAP analysis.

---

# 🔐 Data Leakage Prevention

The project explicitly separates training and test data before applying SMOTE.

```text
Cleaned Dataset
      │
      ▼
Stratified 80/20 Split
      │
      ├───────────────┐
      ▼               ▼
  Training          Test Set
      │             504,160
      ▼             untouched
    SMOTE
      │
      ▼
Model Training
      │
      ▼
Final Evaluation
```

The test set is never used to generate synthetic samples.

---

# ⚠️ Limitations

### 1. Severe Class Imbalance

Some attack classes contain very few real examples, limiting the amount of genuine training information available.

### 2. Web Attack Confusion

Closely related Web Attack categories remain difficult to distinguish.

### 3. Dataset Dependency

The models are trained and evaluated on CIC-IDS2017. Results on newer or real-world network traffic may differ.

### 4. Synthetic Training Data

SMOTE creates synthetic minority examples. These improve training representation but are not additional real-world observations.

### 5. LightGBM Performance

The current LightGBM experiment produced substantially lower performance than the other evaluated models and requires additional investigation.

---

# 🚀 Future Work

Future development can focus on:

* Improving Web Attack subtype classification
* Improving detection of extremely rare attack categories
* Investigating LightGBM performance
* Testing additional machine-learning algorithms
* Exploring deep-learning approaches
* Evaluating additional intrusion-detection datasets
* Testing generalization on newer network traffic
* Improving per-class error analysis
* Extending SHAP-based explainability
* Developing a real-time network monitoring pipeline
* Exploring practical NIDS deployment

---

# 📋 Project Status

### Completed

* [x] Dataset exploration
* [x] Dataset combination
* [x] Data cleaning
* [x] Duplicate removal
* [x] Missing/infinite value handling
* [x] Label correction
* [x] Exploratory Data Analysis
* [x] Feature selection
* [x] Stratified train/test split
* [x] SMOTE class balancing
* [x] Random Forest training
* [x] XGBoost training
* [x] LightGBM training
* [x] MLP training
* [x] Model evaluation
* [x] Model comparison
* [x] Confusion matrix analysis
* [x] Per-class analysis
* [x] SHAP explainability

### Remaining

* [ ] Final research paper/report
* [ ] Final presentation
* [ ] Complete GitHub repository cleanup
* [ ] Add/organize final training and evaluation scripts
* [ ] Add final result artifacts

---

# 📚 Project Significance

This project demonstrates how machine learning can be applied to large-scale network-flow data for multi-class intrusion detection.

The work addresses several practical challenges:

```text
Large Dataset
     +
Data Quality Problems
     +
Feature Redundancy
     +
Severe Class Imbalance
     +
Rare Attack Categories
     +
Model Interpretability
```

The resulting pipeline combines:

**Data Engineering + Machine Learning + Class Balancing + Explainable AI**

to create a structured approach for network intrusion detection.

---

# 👥 Authors

**Iniya Srilekha B**
**Rubini S K**
**Dharshini S**

**Amrita Vishwa Vidyapeetham**

---

# 🔗 Repository

GitHub Repository:

https://github.com/Iniya-008/network-intrusion-detection-ml

---

## 📄 Academic Project

This project was developed for academic and research purposes as a study of machine-learning based network intrusion detection and multi-class cyber threat classification using the CIC-IDS2017 dataset.
