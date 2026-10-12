"""Show JSON inference on a single existing example; no cloud services used."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_support import STAGES,read_data
from aws.inference import model_fn,input_fn,predict_fn,output_fn
model=model_fn(str(ROOT/'models'))
df=read_data(ROOT/'data/student_data.csv')
row=df.iloc[0][STAGES['After second grade']].to_dict()
payload=json.dumps({'instances':[row]})
print(output_fn(predict_fn(input_fn(payload,'application/json'),model),'application/json'))
