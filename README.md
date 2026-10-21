# Assignment 5439 Employee Attrition Prediction Models

Subject: 24UADPC401 Data Science Advanced Programming IBM

This project predicts employee attrition using Logistic Regression, Decision Tree, Random Forest, and XGBoost. It uses leakage-safe preprocessing, five-fold stratified cross-validation, hyperparameter tuning, held-out evaluation, feature interpretation, and a saved inference pipeline.

## Verified result

Logistic Regression was selected by held-out ROC-AUC.

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.772 | 0.381 | 0.681 | 0.489 | 0.812 |

These results come from the included IBM teaching dataset and are not production-performance claims.

## Repository contents

| Path | Purpose |
|---|---|
| `data/employee_attrition.csv` | Bundled public IBM sample dataset |
| `src/train.py` | Preprocessing, training, tuning, evaluation, visualization, and model-saving workflow |
| `notebooks/attrition_prediction.ipynb` | Executed assignment notebook |
| `models/attrition_model.joblib` | Saved preprocessing and Logistic Regression pipeline |
| `reports/` | Evaluation report, business interpretation report, and result tables |
| `visualizations/` | ROC curves, confusion matrix, and feature-importance chart |
| `tests/` | Lightweight repository checks |
| `docs/SUBMISSION_CHECKLIST.md` | Requirement-to-evidence mapping |

## Installation

Python 3.12 is recommended. Run from this repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Reproduce the analysis

```powershell
python src/train.py
```

The script reads `data/employee_attrition.csv` using a repository-relative path and refreshes the results, figures, and saved model.

## Run the notebook

```powershell
jupyter lab notebooks/attrition_prediction.ipynb
```

Choose **Kernel > Restart Kernel and Run All Cells**. The notebook runs the same implementation in `src/train.py`.

## Load the model

```python
import joblib
model = joblib.load("models/attrition_model.joblib")
probability = model.predict_proba(employee_dataframe)[:, 1]
```

The dataframe must contain the feature columns expected by `src/train.py`.

## Tests

```powershell
pytest -q
```

## Dataset and limitations

The bundled file is IBM's fictional HR Analytics Employee Attrition and Performance sample. Its public source is documented in `data/README.md`; it contains no real employee identities.

- The sample is small and cross-sectional.
- Model signals are associations, not causes.
- Moderate precision means many alerts are false positives.
- Scores should support voluntary, low-cost retention conversations only.
- Real use requires governed current data, fairness analysis, privacy review, calibration checks, and human oversight.

Upload the contents of this folder to its own GitHub repository.
