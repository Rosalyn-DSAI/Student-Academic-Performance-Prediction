"""Reproducible, exploratory student-risk model comparisons and inference tools.

No individual student decisions should be based on this academic demonstration.
The model's risk target is final grade < 12/20, a project-specific benchmark.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = 'G3/label_y'
# Use only plausible intake fields; exact collection timing is not known.
# Absences excluded because the CSV has only a final aggregate count.
INTAKE_FEATURES = ['traveltime','studytime','failures','schoolsup','famsup',
                   'paid','activities','higher']
STAGES = {
    'Intake survey (provisional)': INTAKE_FEATURES,
    'After first grade': INTAKE_FEATURES + ['G1'],
    'After second grade': INTAKE_FEATURES + ['G1','G2'],
}
CATEGORICAL = ['schoolsup','famsup','paid','activities','higher']
SEED = 42

def read_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(INTAKE_FEATURES + ['G1','G2',TARGET]) - set(df)
    if missing:
        raise ValueError(f'Columns missing: {sorted(missing)}')
    if df[TARGET].isna().any():
        raise ValueError('Rows with unknown final grades require separate handling')
    return df

def make_model(columns: list[str], model: str) -> Pipeline:
    categories = [c for c in columns if c in CATEGORICAL]
    numerical = [c for c in columns if c not in CATEGORICAL]
    numeric_steps = [('imputer',SimpleImputer(strategy='median'))]
    if model == 'Logistic regression':
        numeric_steps.append(('scaler',StandardScaler()))
    processor = ColumnTransformer([
        ('numeric',Pipeline(numeric_steps),numerical),
        ('categorical',Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore')),
        ]),categories),
    ])
    if model == 'Logistic regression':
        estimator = LogisticRegression(max_iter=2000, random_state=SEED)
    elif model == 'Random forest':
        estimator = RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                           random_state=SEED, n_jobs=1)
    elif model == 'Majority class':
        estimator = DummyClassifier(strategy='most_frequent')
    else:
        raise ValueError(f'Unknown model: {model}')
    return Pipeline([('preprocess',processor),('model',estimator)])

def prioritized_indices(risk_scores: np.ndarray, capacity: int) -> np.ndarray:
    # stable ties, no target labels used to rank: no outcome leakage
    return np.argsort(-np.asarray(risk_scores),kind='stable')[:capacity]

def performance(y: np.ndarray, prob: np.ndarray, cap: int) -> dict:
    pred=(prob >= .5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    selected=prioritized_indices(prob,cap)
    return {
        'accuracy':round(float(accuracy_score(y,pred)),4),
        'balanced_accuracy':round(float(balanced_accuracy_score(y,pred)),4),
        'risk_precision':round(float(precision_score(y,pred,zero_division=0)),4),
        'risk_recall':round(float(recall_score(y,pred,zero_division=0)),4),
        'risk_f1':round(float(f1_score(y,pred,zero_division=0)),4),
        'roc_auc':round(float(roc_auc_score(y,prob)),4),
        'brier_loss':round(float(brier_score_loss(y,prob)),4),
        'missed_risk':int(fn),'false_alarms':int(fp),
        'identified_in_top_20pct':int(y[selected].sum()),
        'top_20pct_precision':round(float(y[selected].mean()),4),
        'top_20pct_recall':round(float(y[selected].sum()/y.sum()),4),
    }

def analyze(df:pd.DataFrame):
    y=(df[TARGET] < 12).astype(int).to_numpy()  # 1 = risk; NOT the notebook's pass=1
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    capacity=int(np.ceil(len(df)*.20))
    rows=[]
    predictions={}
    for stage,columns in STAGES.items():
        for model_name in ['Majority class','Logistic regression','Random forest']:
            model=make_model(columns,model_name)
            oof=cross_val_predict(model,df[columns],y,cv=cv,
                                  method='predict_proba',n_jobs=1)[:,1]
            predictions[(stage,model_name)]=oof
            rows.append({'stage':stage,'method':model_name,'n':len(y),
                         'at_risk_count':int(y.sum()),'review_capacity':capacity,
                         **performance(y,oof,capacity)})
        # Predetermined, non-ML benchmark for ranking students by review priority.
        # Rules are illustrative and not tuned on the target variable.
        if stage=='Intake survey (provisional)':
            rule=10*df['failures'].to_numpy()-df['studytime'].to_numpy()
            label='Rule: previous failures, then study time'
        elif stage=='After first grade':
            rule=-df['G1'].to_numpy()
            label='Rule: lowest first-period grades'
        else:
            rule=-df['G2'].to_numpy()
            label='Rule: lowest second-period grades'
        indices=prioritized_indices(rule,capacity)
        rows.append({'stage':stage,'method':label,'n':len(y),
                     'at_risk_count':int(y.sum()),'review_capacity':capacity,
                     'identified_in_top_20pct':int(y[indices].sum()),
                     'top_20pct_precision':round(float(y[indices].mean()),4),
                     'top_20pct_recall':round(float(y[indices].sum()/y.sum()),4)})
    return pd.DataFrame(rows),predictions,y,capacity

def train_example(df:pd.DataFrame, stage='After second grade',model='Random forest'):
    cols=STAGES[stage]
    clf=make_model(cols,model)
    clf.fit(df[cols],(df[TARGET]<12).astype(int))
    return clf

def predict_risk(pipeline, record:dict, stage='After second grade') -> float:
    required=STAGES[stage]
    missing=[c for c in required if c not in record]
    if missing:
        raise ValueError(f'Missing feature(s): {missing}')
    row=pd.DataFrame([{k:record[k] for k in required}])
    return float(pipeline.predict_proba(row)[0,1])
