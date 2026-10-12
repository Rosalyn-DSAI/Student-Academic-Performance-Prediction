"""Validate and run offline predictions with the saved research model.

No cloud services or external connections. Only load trusted model files:
joblib artifacts use pickle under the hood.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd

FEATURES = (
    'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
    'paid', 'activities', 'higher', 'G1', 'G2'
)
RANGES = {
    'traveltime': (1, 4),
    'studytime': (1, 4),
    'failures': (0, 3),
    'G1': (0, 20),
    'G2': (0, 20),
}
YES_NO_FIELDS = set(FEATURES) - set(RANGES)


def load_model(path: str | Path):
    """Load the trusted, locally saved scikit-learn pipeline."""
    return joblib.load(Path(path))


def parse_request(request_body: str | bytes) -> pd.DataFrame:
    """Validate a JSON object containing 1–100 student feature records."""
    if isinstance(request_body, bytes):
        request_body = request_body.decode('utf-8')
    try:
        payload = json.loads(request_body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Request must be valid JSON') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('instances'), list):
        raise ValueError('Expected a JSON object with an instances list')
    instances = payload['instances']
    if not 1 <= len(instances) <= 100:
        raise ValueError('The instances list must contain 1 to 100 records')

    for i, record in enumerate(instances, start=1):
        if not isinstance(record, dict):
            raise ValueError(f'Record {i} must be a JSON object')
        missing = set(FEATURES) - record.keys()
        extra = record.keys() - set(FEATURES)
        if missing or extra:
            raise ValueError(f'Record {i}: missing {sorted(missing)}, unexpected {sorted(extra)}')
        for field, (minimum, maximum) in RANGES.items():
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
                raise ValueError(f'Record {i}: {field} must be a number from {minimum} to {maximum}')
        for field in YES_NO_FIELDS:
            if record[field] not in ('yes', 'no'):
                raise ValueError(f'Record {i}: {field} must be yes or no')
    return pd.DataFrame(instances, columns=list(FEATURES))


def predict_records(model, rows: pd.DataFrame) -> list[dict]:
    """Return below-12/20 risk estimates for already-validated rows."""
    probabilities = model.predict_proba(rows)[:, 1]
    return [{'risk_probability': float(prob)} for prob in probabilities]


def format_response(predictions: list[dict]) -> str:
    """Format local predictions as JSON for reproducible testing."""
    return json.dumps({
        'predictions': predictions,
        'threshold_note': 'Project-defined outcome: final grade below 12/20',
        'warning': 'Research demonstration only; do not use for actual student decisions.',
    }, indent=2)
