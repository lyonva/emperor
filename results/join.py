import os
import pandas as pd

result_name = "result-%s-goals-0-6.csv"
goals = [0, 1, 2, 3, 4, 5, 6]
goals = ["goal%d" % g for g in goals]

file_types = ["metrics", "predictions"]

for file_type in file_types:
    # Load everything in directory
    files = [f for f in os.listdir(".") if f.endswith('.csv')]
    
    # Filter non-type results
    files = [f for f in files if file_type in f]
    
    # Filter per goal name
    files = [f for f in files if True in [ g in f for g in goals ]]
    
    frames = []
    
    for file_name in files:
        data = pd.read_csv(file_name, index_col=None)
        frames.append(data)
    
    result = pd.concat(frames)
    result.to_csv(result_name % file_type, index=False)
