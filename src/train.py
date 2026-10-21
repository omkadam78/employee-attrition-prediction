from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "employee_attrition.csv"
OUT = ROOT / "reports" / "results"
VIS = ROOT / "visualizations"
MODELS = ROOT / "models"
for folder in (OUT, VIS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)


def load_data(path: Path = DATA) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    drop = ["Attrition", "EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"]
    return frame.drop(columns=drop), frame["Attrition"].map({"No": 0, "Yes": 1})


def make_preprocessor(frame: pd.DataFrame, scale: bool) -> ColumnTransformer:
    numeric = frame.select_dtypes(exclude="object").columns.tolist()
    categorical = frame.select_dtypes(include="object").columns.tolist()
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, prob),
    }


def main() -> None:
    sns.set_theme(style="whitegrid")
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = {
        "Logistic Regression": Pipeline([("pre", make_preprocessor(x, True)), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))]),
        "Decision Tree": Pipeline([("pre", make_preprocessor(x, False)), ("model", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline([("pre", make_preprocessor(x, False)), ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "XGBoost": Pipeline([("pre", make_preprocessor(x, False)), ("model", XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED, n_jobs=-1))]),
    }
    grids = {
        "Logistic Regression": {"model__C": [0.03, 0.1, 0.3, 1.0]},
        "Decision Tree": {"model__max_depth": [3, 4, 5, 7], "model__min_samples_leaf": [5, 10, 20]},
        "Random Forest": {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 3, 5]},
        "XGBoost": {"model__n_estimators": [150, 250], "model__max_depth": [3, 4], "model__learning_rate": [0.03, 0.08]},
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1": "f1", "ROC-AUC": "roc_auc"}
    cv_rows, test_rows, params, fitted = {}, {}, {}, {}
    for name, pipeline in models.items():
        cv_result = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_rows[name] = {metric: float(np.mean(cv_result[f"test_{metric}"])) for metric in scoring}
        search = GridSearchCV(pipeline, grids[name], scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(x_train, y_train)
        fitted[name] = search.best_estimator_
        params[name] = search.best_params_
        test_rows[name] = evaluate(search.best_estimator_, x_test, y_test)
    test = pd.DataFrame(test_rows).T.sort_values("ROC-AUC", ascending=False)
    cv_table = pd.DataFrame(cv_rows).T.sort_values("ROC-AUC", ascending=False)
    test.to_csv(OUT / "test_metrics.csv")
    cv_table.to_csv(OUT / "cross_validation_metrics.csv")
    (OUT / "best_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    best_name = str(test.index[0])
    best = fitted[best_name]
    joblib.dump(best, MODELS / "attrition_model.joblib")
    metadata = {
        "selected_model": best_name,
        "selection_criterion": "highest held-out ROC-AUC after cross-validated tuning",
        "test_metrics": test.loc[best_name].to_dict(),
        "rows": int(len(x)),
        "attrition_rate": float(y.mean()),
        "random_seed": SEED,
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for name, model in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} ({test.loc[name, 'ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set(title="ROC curves for tuned classifiers", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(VIS / "roc_curves.png", dpi=180); plt.close(fig)

    cm = confusion_matrix(y_test, best.predict(x_test))
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
    ax.set(title=f"Confusion matrix for {best_name}", xlabel="Predicted", ylabel="Observed")
    fig.tight_layout(); fig.savefig(VIS / "confusion_matrix.png", dpi=180); plt.close(fig)

    pre = best.named_steps["pre"]
    names = pre.get_feature_names_out()
    estimator = best.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = estimator.feature_importances_
    importance = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False).head(20)
    importance.to_csv(OUT / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    sns.barplot(data=importance.head(12), x="importance", y="feature", color="#356A8A", ax=ax)
    ax.set(title=f"Leading features in {best_name}", xlabel="Absolute model importance", ylabel="")
    fig.tight_layout(); fig.savefig(VIS / "feature_importance.png", dpi=180); plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
