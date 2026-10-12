"""Offline smoke tests for the locally saved prediction pipeline."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision_support import predict_risk
from local_inference import load_model, parse_request, predict_records, format_response
from scripts.local_inference_demo import EXAMPLE_STUDENT


def expect_invalid(record):
    try:
        parse_request(json.dumps({'instances': [record]}))
    except ValueError:
        return
    raise AssertionError('Expected an invalid student record to be rejected')


def main():
    model = load_model(ROOT / 'models' / 'academic_risk_pipeline.joblib')
    rows = parse_request(json.dumps({'instances': [EXAMPLE_STUDENT]}))
    results = predict_records(model, rows)
    assert len(results) == 1

    local_probability = predict_risk(model, EXAMPLE_STUDENT)
    assert 0 <= local_probability <= 1
    assert abs(results[0]['risk_probability'] - local_probability) < 1e-12

    response = json.loads(format_response(results))
    assert response['predictions'] == results
    assert response['threshold_note'].startswith('Project-defined')

    expect_invalid({})  # All required features missing
    expect_invalid({**EXAMPLE_STUDENT, 'G1': 25})  # Invalid grade
    expect_invalid({**EXAMPLE_STUDENT, 'higher': 'maybe'})  # Invalid category
    expect_invalid({**EXAMPLE_STUDENT, 'unknown': 1})  # Unexpected input

    print('All offline inference tests passed. No external services were used.')


if __name__ == '__main__':
    main()
