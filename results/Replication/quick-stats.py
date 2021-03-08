import numpy as np
import pandas as pd

file = "emperor-metrics-tuned-128.csv"

df = pd.read_csv(file)

res_keys = ["metric", "model", "goal", "stat", "value"]
result_dict = { k:[] for k in res_keys }

metrics = ["mmre", "sa"]
models = np.unique(df["model"])
goals = np.unique(df["goal"])

for metric in metrics:
    for goal in goals:
        for model in models:
            sub_df = df[ df["objective"] == metric ]
            sub_df = sub_df[ sub_df["model"] == model ]
            sub_df = sub_df[ sub_df["goal"] == goal ]
            
            res = sub_df[metric]
            median = np.median(res)
            q75, q25 = np.percentile(res, [75 ,25])
            iqr = q75 - q25
            
            print("-"*40)
            print("%15s: %10s" % ("Model", model))
            print("%15s: %10s" % ("Target", metric))
            print("%15s: %10d" % ("Goal", goal))
            print("%15s: %10f" % ("Median", median))
            print("%15s: %10f" % ("IQR", iqr))
        
            # Add to dict
            result_dict["model"].append( model )
            result_dict["metric"].append( metric )
            result_dict["goal"].append( goal )
            result_dict["stat"].append( "med" )
            result_dict["value"].append( median )
            
            result_dict["model"].append( model )
            result_dict["metric"].append( metric )
            result_dict["goal"].append( goal )
            result_dict["stat"].append( "iqr" )
            result_dict["value"].append( iqr )

# Export results
result_df = pd.DataFrame.from_dict(result_dict)
for metric in metrics:
    for stat in ["med", "iqr"]:
        sub_res = result_df[ np.logical_and(result_df["stat"] == stat ,
                                            result_df["metric"] == metric) ]
        sub_res = sub_res[["model", "goal", "value"]]
        sub_res = sub_res.pivot( index = "goal", columns = "model", values = "value")
        
        sub_res.to_csv("%s-%s.csv" % (metric, stat))
        
