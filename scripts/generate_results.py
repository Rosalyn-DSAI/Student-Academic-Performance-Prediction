"""Regenerate out-of-fold decision metrics, report figures, and example pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from decision_support import read_data,analyze,train_example,STAGES

MODEL_COLORS = {
    'Decision tree': '#5DA5DA',
    'Logistic regression': '#60BD68',
    'Random forest': '#F17CB0',
    'Simple rule': '#9E9E9E',
}

def main():
    data=read_data(ROOT/'data/student_data.csv')
    metrics,predictions,y,capacity=analyze(data)
    metrics.to_csv(ROOT/'reports/decision_metrics.csv',index=False)
    ml=metrics[metrics['method'].isin(['Logistic regression','Random forest'])].copy()
    order=list(STAGES)
    plt.figure(figsize=(9,5))
    line_order=['Decision tree','Logistic regression','Random forest']
    for method in line_order:
        group=metrics[metrics['method']==method]
        if group.empty:
            continue
        vals=group.set_index('stage').reindex(order)['risk_recall'].to_numpy()
        plt.plot(
            range(3),
            vals,
            marker='o',
            linewidth=2.2,
            markersize=7,
            color=MODEL_COLORS[method],
            label=method
        )
    plt.xticks(range(3),['Intake survey\n(provisional)','After G1','After G2'])
    plt.ylim(0,1.05);plt.ylabel('Recall for students below threshold')
    plt.title('Risk detection versus information availability')
    plt.grid(axis='y',alpha=.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(ROOT/'assets/decision_stage_recall.png',dpi=170);plt.close()

    plt.figure(figsize=(10,5))
    w=.21
    labels=['Intake survey','After G1','After G2']
    bar_methods=['Logistic regression','Random forest','Simple rule']
    for j,stage in enumerate(order):
        group=metrics[metrics['stage']==stage]
        vals=[
            int(group[group['method']=='Logistic regression']['identified_in_top_20pct'].iloc[0]),
            int(group[group['method']=='Random forest']['identified_in_top_20pct'].iloc[0]),
            int(group[~group['method'].isin(['Decision tree','Logistic regression','Random forest','Majority class'])]['identified_in_top_20pct'].iloc[0])
        ]
        for k, n in enumerate(vals):
            method=bar_methods[k]
            plt.bar(
                j+(k-1)*w,
                n,
                width=w,
                color=MODEL_COLORS[method],
                label=method if j==0 else None
            )
    plt.xticks(range(3),labels)
    plt.ylabel('At-risk students found among first '+str(capacity)+' reviewed')
    plt.title('Review capacity: 20% of students (out-of-fold ML predictions)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(ROOT/'assets/review_capacity_comparison.png',dpi=170);plt.close()

    best=metrics[(metrics['stage']=='After second grade') & (metrics['method']=='Random forest')].iloc[0]
    saved=train_example(data,stage='After second grade',model='Random forest')
    joblib.dump(saved,ROOT/'models/academic_risk_pipeline.joblib')
    metadata={
      'note':'Research prototype only: trained on all data AFTER out-of-fold evaluation; no independent holdout or external validation.',
      'target':'G3/label_y < 12 (project-defined threshold; 1 = below threshold)',
      'feature_stage':'After second grade',
      'features':STAGES['After second grade'],
      'threshold':12,
      'data_rows':len(data),
      'risks_count':int(y.sum()),
      'review_capacity':capacity,
      'random_forest_stage2_oof_accuracy':float(best['accuracy']),
      'random_forest_stage2_oof_recall':float(best['risk_recall']),
      'recommended_use':'Notebook reproducibility and local inference testing only; not for student-level decisions.',
    }
    (ROOT/'models/model_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    print(metrics.fillna('').to_string(index=False))
    return metrics

if __name__=='__main__':main()
