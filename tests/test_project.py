from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_present():
    assert (ROOT / 'data' / 'employee_attrition.csv').stat().st_size > 1000

def test_training_source_has_seed():
    assert 'SEED = 42' in (ROOT / 'src' / 'train.py').read_text(encoding='utf-8')
