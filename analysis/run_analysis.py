import sk
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

data_dir = "../results"
data_name = "result-128.csv"
output_name = "analysis-128.txt"

df = pd.read_csv(os.path.join(data_dir, data_name), sep=',')

# Data may come disorganized, sort by dataset > model > tuner
df = df.sort_values(by = ["dataset", "model", "tuner"])

# Make one split per tuner
tuners = df["tuner"].unique()
tuners_df = [ df.loc[ df["tuner"] == tuner, : ] for tuner in tuners ]
tuner_dict = dict( zip( tuners, tuners_df ) )

# Redirect stdout to file
original_stdout = sys.stdout
sys.stdout = open(output_name, 'w')

# Now to scott-knott per metric

metrics = ["sa", "sa_md", "mar", "mdmre"]
for metric in metrics:
    # Select feature corresponding to metric
    met_list = [ list( tdf[metric] ) for tdf in tuners_df ]
    met_dict = dict( zip( tuners, met_list ) )
    
    # Run scott-knott on current metric
    print("Result for %s:" % metric)
    sk.Rx.show(sk.Rx.sk(sk.Rx.data(**met_dict)))
    print("\n")


# Return output to stdout
sys.stdout = original_stdout

# Generate box-plots
for metric in metrics:
    # Select feature corresponding to metric
    met_list = [ list( tdf[metric] ) for tdf in tuners_df ]
    plt.boxplot( met_list, labels = tuners )

# Find outliers

for tuner in tuners:
    tuner_frame = tuner_dict[tuner]
    for metric in metrics:
        distribution = tuner_frame[metric]
        q1, q3= np.percentile(distribution,[25,75])
        print(q1, q3)
        