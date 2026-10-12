"""Offline inference demonstration using a fictional example student.

No internet, AWS services, or real-time student data are used.
This example is NOT an independent evaluation of model performance.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_inference import load_model, parse_request, predict_records, format_response

# Fictional input values matching the variables used during model training.
EXAMPLE_STUDENT = {
    'traveltime': 2,
    'studytime': 2,
    'failures': 0,
    'schoolsup': 'no',
    'famsup': 'yes',
    'paid': 'no',
    'activities': 'yes',
    'higher': 'yes',
    'G1': 9,
    'G2': 10,
}


def main():
    model = load_model(ROOT / 'models' / 'academic_risk_pipeline.joblib')
    records = parse_request(json.dumps({'instances': [EXAMPLE_STUDENT]}))
    print(format_response(predict_records(model, records)))


if __name__ == '__main__':
    main()

