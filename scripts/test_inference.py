"""Local, offline smoke tests for the saved prediction pipeline and AWS handler."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
from decision_support import STAGES,read_data,predict_risk
from aws.inference import input_fn, output_fn, predict_fn, model_fn

def main():
    data=read_data(ROOT/'data/student_data.csv')
    model=joblib.load(ROOT/'models/academic_risk_pipeline.joblib')
    record=data.iloc[0][STAGES['After second grade']].to_dict()
    local=predict_risk(model,record)
    assert 0 <= local <= 1
    try:
        predict_risk(model,{})
        raise AssertionError('Expected missing feature validation')
    except ValueError:pass
    payload=input_fn(json.dumps({'instances':[record]}),'application/json')
    preds=predict_fn(payload,model)
    assert len(preds)==1 and abs(preds[0]['risk_probability']-local)<1e-12
    decoded=json.loads(output_fn(preds,'application/json'))
    assert decoded['threshold_note'].startswith('Project-defined')
    reloaded=model_fn(str(ROOT/'models'))
    assert abs(predict_risk(reloaded,record)-local)<1e-12
    try:
        input_fn(json.dumps({'instances':[{}]}),'application/json')
        raise AssertionError('Expected invalid schema rejection')
    except ValueError:pass
    print('All local inference / validation tests passed. No cloud endpoint was created.')

if __name__=='__main__': main()
