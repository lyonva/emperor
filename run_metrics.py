import pandas as pd
import numpy as np
from evaluation import evaluate
from helper import save_metrics
import os

results_dir = "results"
prediction_file = "result-predictions-goals-0-6+grid.csv"
result_file = "result-metrics-goals-0-6+grid.csv"

df_predictions = pd.read_csv(os.path.join(results_dir, prediction_file),index_col = None)
df_predictions = df_predictions.sort_values(["goal", "dataset", "tuner", "model", "iteration"])

metric_names = ["sa", "mar", "effect_size"]

goals = np.unique( df_predictions["goal"] )
datasets = np.unique( df_predictions["dataset"] )
tuners = np.unique( df_predictions["tuner"] )
models = np.unique( df_predictions["model"] )

result_df = pd.DataFrame()

for goal in goals:
    for dataset in datasets:
        for tuner in tuners:
            for model in models:
                cond_a = df_predictions["goal"] == goal
                cond_b = df_predictions["dataset"] == dataset
                cond_c = df_predictions["tuner"] == tuner
                cond_d = df_predictions["model"] == model
                sub_predictions = df_predictions[cond_a & cond_b & cond_c & cond_d]
                
                y_true = sub_predictions["y_true"]
                y_pred = sub_predictions["y_pred"]
                
                # Calculate metrics
                metrics = evaluate(y_true, y_pred, metric_names)
                
                # Append remaining columns
                metrics["dataset"] = dataset
                metrics["goal"] = goal
                metrics["model"] = model
                metrics["tuner"] = tuner
                metrics = dict( [(key, [val]) for key, val in metrics.items()] )
                
                # Append to current results
                metrics = pd.DataFrame.from_dict(metrics)
                result_df = pd.concat([result_df, metrics])

# After all has been calculated, save
result_df.to_csv(os.path.join(results_dir, result_file), index=False)