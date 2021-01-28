import os
import pandas as pd

result_name = "result-128.csv"
files = [f for f in os.listdir(".") if f.endswith('.csv')]

# Filter non-metric results
files = [f for f in files if "metrics" in f]

frames = []

for file_name in files:
    data = pd.read_csv(file_name, index_col=None)
    frames.append(data)

result = pd.concat(frames)
result.to_csv(result_name, index=False)
