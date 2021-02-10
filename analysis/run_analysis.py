import sk
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set_palette("gray", 3)

data_dir = "../results"
data_name = "result-metrics-goals-0-2.csv"
output_name = "analysis-goals-0-2"

goal_names = ['monthly_commits', 'monthly_contributors', 'monthly_open_PRs',
              'monthly_closed_PRs', 'monthly_open_issues', 'monthly_closed_issues']
metrics = ["sa", "mar"]

df = pd.read_csv(os.path.join(data_dir, data_name), sep=',')

# Data may come disorganized, sort by goal > dataset > model > tuner
df = df.sort_values(["goal", "dataset", "model", "tuner"])

# Remove projects
projects_remove = ["project0175.csv"]
df = df[ [ d not in projects_remove for d in df["dataset"] ] ]

# Unique lists for filtering
goals = np.unique(df["goal"])
tuners = df["tuner"].unique()

sk_df = pd.DataFrame()

# Redirect stdout to file
original_stdout = sys.stdout
sys.stdout = open("%s.txt" % output_name, 'w')

for goal in goals:
    goal_df = df[df["goal"] == goal]
    
    # Make one split per tuner
    tuners_df = [ goal_df[ goal_df["tuner"] == tuner ] for tuner in tuners ]
    tuner_dict = dict( zip( tuners, tuners_df ) )
    
    # Now to scott-knott per metric
    for metric in metrics:
        # Select feature corresponding to metric
        met_list = [ list( tdf[metric] ) for tdf in tuners_df ]
        met_dict = dict( zip( tuners, met_list ) )
        
        # Run scott-knott on current metric
        print("Goal %s:" % goal_names[goal])
        print("Result for %s:" % metric)
        sk_result = sk.Rx.sk(sk.Rx.data(**met_dict))
        sk.Rx.show(sk_result)
        print("\n")
        
        # Save results for plotting
        row = {
            "goal" : [goal for s in sk_result],
            "metric" : [metric for s in sk_result],
            "tuner" : [s.rx for s in sk_result],
            "rank" : [s.rank for s in sk_result]
            }
        row = pd.DataFrame.from_dict(row)
        sk_df = pd.concat([sk_df, row])
    
# Return output to stdout
sys.stdout = original_stdout

df_boxplot = df.melt(id_vars=["dataset", "goal", "model", "tuner"],
                     value_vars = metrics, var_name = "metric" )
df_boxplot["rank"] = df_boxplot.apply( lambda x : sk_df[ (x["goal"] == sk_df["goal"]) &
                                                (x["metric"] == sk_df["metric"]) &
                                                (x["tuner"] == sk_df["tuner"])]["rank"].iloc[0], axis = 1)
df_boxplot["goal_name"] = df_boxplot.apply( lambda x : goal_names[x["goal"]], axis = 1)
df_boxplot["tuner_short"] = df_boxplot.apply( lambda x : 
                                             {'DifferentialEvolutionCV' : "DE",
                                              'DodgeCV' : "DODGE",
                                              'RandomRangeSearchCV' : "Random"
                                              }[x["tuner"]], axis = 1)

sns.set(font_scale=1.4)
plot = sns.catplot( x = "tuner_short", y = "value", col = "goal_name", row = "metric",
                  hue = "rank", data = df_boxplot, kind = "box", sharey = "row",
                  legend = False, margin_titles = True, showfliers = False, height = 8)
plot.set_axis_labels("", "").set_titles(row_template = '{row_name}', col_template = '{col_name}').set_xticklabels(rotation=0)
plot.savefig("%s.png" % output_name)

# for goal in goals:
#     goal_df = df[df["goal"] == goal]

#     # Generate box-plots
#     for metric in metrics:
#         # Select feature corresponding to metric
#         # met_list = [ list( tdf[metric] ) for tdf in tuners_df ]
#         sns.boxplot( x = "tuner", y = metric, data = goal_df )

# Find outliers

# for tuner in tuners:
#     tuner_frame = tuner_dict[tuner]
#     for metric in metrics:
#         distribution = tuner_frame[metric]
#         q1, q3= np.percentile(distribution,[25,75])
#         print(q1, q3)
        