import datetime
import pandas as pd
import numpy as np
import os

save_dir = "results"

class SingletonDataframes:
    __time__ = datetime.datetime.now()
    __metrics_df__ = pd.DataFrame()
    __prediction_df__ = pd.DataFrame()

def print_progress(dataset_name, i, model_name, tuner_name):
    print("-"*30)
    print( datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") )
    print("%10s: %15s" % ("dataset", dataset_name))
    print("%10s: %15d" % ("iteration", i))
    print("%10s: %15s" % ("model", model_name))
    print("%10s: %15s" % ("tuner", tuner_name))
    print("-"*30)
    print()

def save_prediction(dataset_name, goal, model_name, tuner_name, y_true, y_pred):
    # Convert results to dataframe format
    n = y_true.size
    predictions = {}
    predictions["dataset"] = np.repeat(dataset_name, n)
    predictions["goal"] = np.repeat(goal, n)
    predictions["model"] = np.repeat(model_name, n)
    predictions["tuner"] = np.repeat(tuner_name, n)
    predictions["iteration"] = range(n)
    predictions["y_pred"] = y_pred
    predictions["y_true"] = y_true
    df_row = pd.DataFrame.from_dict(predictions)
    
    # Load current status
    current_df = SingletonDataframes.__prediction_df__
    time = SingletonDataframes.__time__
    
    # Add new row to data and save
    prediction_df = pd.concat([current_df, df_row])
    file_path = os.path.join( save_dir, "emperor-predictions-%s.csv" % time.strftime('%Y-%m-%d_%H-%M-%S') )
    prediction_df.to_csv(file_path, index=False)
    
    # Store status
    SingletonDataframes.__prediction_df__ = prediction_df

def save_metrics(dataset_name, goal, model_name, tuner_name, metrics):
    # Convert results to dataframe format
    metrics = metrics.copy()
    metrics["dataset"] = dataset_name
    metrics["goal"] = goal
    metrics["model"] = model_name
    metrics["tuner"] = tuner_name
    metrics = dict( [(key, [val]) for key, val in metrics.items()] )
    df_row = pd.DataFrame.from_dict(metrics)
    
    # Load current status
    current_df = SingletonDataframes.__metrics_df__
    time = SingletonDataframes.__time__
    
    # Add new row to data and save
    metrics_df = pd.concat([current_df, df_row])
    file_path = os.path.join( save_dir, "emperor-metrics-%s.csv" % time.strftime('%Y-%m-%d_%H-%M-%S') )
    metrics_df.to_csv(file_path, index=False)
    
    # Store status
    SingletonDataframes.__metrics_df__ = metrics_df
