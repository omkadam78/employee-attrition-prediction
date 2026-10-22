from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_present():
    assert (ROOT / 'data' / 'employee_attrition.csv').stat().st_size > 1000

def test_training_source_has_seed():
    assert 'SEED = 42' in (ROOT / 'src' / 'train.py').read_text(encoding='utf-8')

def test_mandatory_classifiers_and_metrics_are_implemented():
    text = (ROOT / 'src' / 'train.py').read_text(encoding='utf-8')
    for marker in ('LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier'):
        assert marker in text
    for marker in ('accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score'):
        assert marker in text

def test_optional_regression_outputs_are_present():
    assert (ROOT / 'reports' / 'results' / 'regression_metrics.csv').stat().st_size > 100
    assert (ROOT / 'reports' / 'results' / 'regression_summary.json').stat().st_size > 100
    assert (ROOT / 'visualizations' / 'regression_model_comparison.png').stat().st_size > 1000
