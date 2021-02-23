import os
import pandas as pd
import numpy as np
from evaluation import evaluate

result_name = "result-%s-goal0-randomN.csv"

file_types = ["predictions"]

for file_type in file_types:
    # Load everything in directory
    files = [f for f in os.listdir(".") if f.endswith('.csv')]
    
    # Filter non-type results
    files = [f for f in files if file_type in f]
    
    frames = []
    
    for file_name in files:
        data = pd.read_csv(file_name, index_col=None)
        frames.append(data)
    
    result = pd.concat(frames)

result = result.replace("DefaultCV", "RandomRangeSearchCV")
result.to_csv(result_name % file_type, index=False)

# Calculate results
metrics_file = "result-metrics-goal0-randomN.csv"

result = result.sort_values(["goal", "dataset", "tuner", "n", "model", "iteration"])

metric_names = ["sa", "mar", "effect_size"]

goals = np.unique( result["goal"] )
datasets = np.unique( result["dataset"] )
tuners = np.unique( result["tuner"] )
models = np.unique( result["model"] )
ns = np.unique( result["n"] )

result_df = pd.DataFrame()

for goal in goals:
    for dataset in datasets:
        for tuner in tuners:
            for model in models:
                for n in ns:
                    cond_a = result["goal"] == goal
                    cond_b = result["dataset"] == dataset
                    cond_c = result["tuner"] == tuner
                    cond_d = result["model"] == model
                    cond_e = result["n"] == n
                    sub_predictions = result[cond_a & cond_b & cond_c & cond_d & cond_e]
                    
                    y_true = sub_predictions["y_true"]
                    y_pred = sub_predictions["y_pred"]
                    
                    # Calculate metrics
                    metrics = evaluate(y_true, y_pred, metric_names)
                    
                    # Append remaining columns
                    metrics["dataset"] = dataset
                    metrics["goal"] = goal
                    metrics["model"] = model
                    metrics["tuner"] = tuner
                    metrics["n"] = n
                    metrics = dict( [(key, [val]) for key, val in metrics.items()] )
                    
                    # Append to current results
                    metrics = pd.DataFrame.from_dict(metrics)
                    result_df = pd.concat([result_df, metrics])

# After all has been calculated, save
result_df.to_csv(metrics_file, index=False)
