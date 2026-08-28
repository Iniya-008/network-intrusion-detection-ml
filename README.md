# network-intrusion-detection-ml
Machine Learning based Network Intrusion Detection System using CIC-IDS2017 Dataset
# Network Intrusion Detection using Machine Learning

## Team Members

- Iniya Srilekha B
- Rubini S K
- Dharshini S

## College

Amrita Vishwa Vidyapeetham

## Objective

Build a Machine Learning model that classifies different cyber attacks using the CIC IDS2017 dataset.

## Dataset

CIC IDS2017

## Technologies

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib

## Status

🟢 Project Started


---

## Project Progress Log

| Step | Script | Status | Output |
|---|---|---|---|
| 2. Dataset Exploration | `src/00_explore_single_file.py`, `src/00b_explore_all_files.py` | Done | Console summary of shape, labels, missing/infinite values, duplicates |
| 3. Combine Files | `src/01_combine_files.py` | Done | Combined raw dataset (2,830,743 rows × 80 cols) |
| 4. Data Cleaning | `src/02_clean_data.py`, `src/02b_final_dedupe.py`, `src/02c_verify_cleaned.py` | Done | Cleaned dataset (2,520,798 rows) — duplicates removed, missing/infinite values dropped, corrupted label text fixed |
| 5. EDA | `src/03_eda.py` | Done | 4 graphs in `graphs/`: class distribution, feature distributions, correlation heatmap, outlier comparison |
| 6. Feature Selection | `src/04_feature_selection.py` | Done | 78 → 32 features (removed zero-variance, highly correlated, and multicollinear columns). List in `docs/selected_features.txt` |
| 7–9. Split + SMOTE | — | Not started | — |
| 10. Model Training | — | Not started | — |
| 11–13. Evaluation | — | Not started | — |
| 14. SHAP | — | Not started | — |
| 15. Report | — | Not started | — |

### Key Findings So Far

- **Severe class imbalance**: BENIGN = 2,095,057 rows (83%); rarest classes (Heartbleed: 11, SQL Injection: 21, Infiltration: 36) are a tiny fraction. Macro-F1 will be prioritized over accuracy for this reason.
- **Data quality issues found & fixed**: ~310,000 duplicate rows (within and across the 8 source files), ~1,600 rows with missing/infinite values (from zero-duration flows), corrupted characters in "Web Attack" label text.
- **Feature redundancy**: 8 zero-variance columns, 24 columns with >0.95 correlation to another column, and 14 more removed via VIF (multicollinearity) — final feature set is 32 columns.

### How to Run

\`\`\`bash
pip install -r requirements.txt

# Place the 8 raw CIC-IDS2017 CSVs in dataset/raw/MachineLearningCVE/, then:
python src/01_combine_files.py
python src/02_clean_data.py
python src/02b_final_dedupe.py
python src/03_eda.py
python src/04_feature_selection.py
\`\`\`
