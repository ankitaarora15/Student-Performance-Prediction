# Student Performance Prediction Using Machine Learning

## Overview
Compare classifiers for identifying students at risk of a **low final mathematics grade**. This project does **not** predict actual student dropout.

## Research questions
1. How well do Logistic Regression, Random Forest, and XGBoost predict `G3 < 10`?
2. How does Logistic Regression performance change when earlier grades `G1` and `G2` are excluded?

## Dataset
- UCI Student Performance dataset, mathematics subset (`student-mat.csv`)
- 395 records, 33 columns
- Target: `low_grade = 1` when `G3 < 10`, otherwise `0`
- Target counts: 130 low-grade cases; 265 cases with `G3 >= 10`
- 32 input features after excluding `G3` and `low_grade`
- Initial missing-value check: no pandas-recognized missing values

Review the dataset's original license and terms before redistributing the CSV.

## Workflow
1. Explore the data and inspect missing values.
2. Create the target from `G3`.
3. Use an 80:20 train/test split with `random_state=42` and stratification.
4. Scale numeric features with `StandardScaler`; encode categorical features with `OneHotEncoder(handle_unknown="ignore")`.
5. Use scikit-learn pipelines to combine preprocessing and models.
6. Compare Logistic Regression, Random Forest, XGBoost, and a most-frequent baseline.
7. Evaluate accuracy, recall, F1-score, and ROC-AUC.
8. Perform five-fold stratified cross-validation on training data.
9. Compare Logistic Regression with and without `G1` and `G2`.

## Test-set results

| Model / experiment | Accuracy | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (G1/G2 included) | 93.67% | 84.62% | 89.80% | 97.68% |
| Random Forest (G1/G2 included) | 91.14% | 88.46% | 86.79% | 97.50% |
| XGBoost (G1/G2 included) | 91.14% | 84.62% | 86.27% | 97.46% |
| Logistic Regression (G1/G2 excluded) | 69.62% | 30.77% | 40.00% | 71.48% |
| Most-frequent baseline | 67.09% | 0.00% | 0.00% | Not reported |

## Five-fold cross-validation (mean ± standard deviation)

| Model | Accuracy | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.905 ± 0.042 | 0.866 ± 0.114 | 0.854 ± 0.072 | 0.970 ± 0.017 |
| Random Forest | 0.902 ± 0.044 | 0.876 ± 0.103 | 0.853 ± 0.070 | 0.960 ± 0.030 |
| XGBoost | 0.915 ± 0.028 | 0.876 ± 0.071 | 0.871 ± 0.042 | 0.967 ± 0.021 |

## Main findings
- The three models performed strongly when `G1` and `G2` were included.
- XGBoost had the highest mean cross-validation accuracy and F1-score; Logistic Regression had the highest mean ROC-AUC.
- Removing `G1` and `G2` substantially reduced performance, indicating that earlier grades carry predictive signal.
- These are predictive associations, not causal findings.

## Limitations
- Small dataset and one educational context.
- Target is low final grade, not dropout.
- G1/G2 usefulness depends on the intended prediction time.
- No external validation; results should not be generalized without further testing.

## Suggested repository structure
```text
student-performance-prediction/
├── README.md
├── Student_Performance_Prediction_Report.pdf
├── student_performance_prediction.ipynb
├── student-mat.csv        # include only if redistribution is permitted
├── requirements.txt
└── .gitignore
```
Update the notebook filename to match the actual file you upload.

## Run
Install dependencies with `pip install -r requirements.txt`, then open the notebook in JupyterLab, Jupyter Notebook, or VS Code and run cells from top to bottom.
