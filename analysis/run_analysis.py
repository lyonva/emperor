import sk
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set_palette("gray", 3)

data_dir = "../results/Replication"
data_name = "emperor-metrics-DECART-128-online.csv"
output_name = "analysis-tuned-DECART-128-online"
dimensionality_name = "health_indicator_output.csv"
# random_n_name = "RandomN/result-metrics-goal0-randomN.csv"

goal_names = ['monthly_commits', 'monthly_contributors', 'monthly_stargazer', 'monthly_open_PRs',
              'monthly_closed_PRs', 'monthly_open_issues', 'monthly_closed_issues']


df = pd.read_csv(os.path.join(data_dir, data_name), sep=',')

# Data may come disorganized, sort by goal > dataset > model > tuner
df = df.sort_values(["goal", "dataset", "model", "tuner"])

# Unique lists for filtering
goals = np.unique(df["goal"])
tuners = df["tuner"].unique()
models = df["model"].unique()
metrics = df["objective"].unique()

sk_df = pd.DataFrame()

# Redirect stdout to file
original_stdout = sys.stdout
sys.stdout = open("%s.txt" % output_name, 'w')

for goal in goals:
    goal_df = df[df["goal"] == goal]
    # Now to scott-knott per metric
    for metric in metrics:
    # Make one split per tuner and model
        metric_df = goal_df[goal_df["objective"] == metric]
        tuners_df = [ metric_df[ np.logical_and(metric_df["tuner"] == tuner, metric_df["model"] == model) ] for tuner in tuners for model in models ]
        tuners_key = [ tuner + "+" + model for tuner in tuners for model in models ]
        tuner_dict = dict( zip( tuners_key, tuners_df ) )
    
    
        # Select feature corresponding to metric
        met_list = [ list( tdf[metric] ) for tdf in tuners_df ]
        if metric == "sa": # Flip sign if metric is not an error
            met_list = [ [ -value for value in met_sublist ] for met_sublist in met_list  ]
        met_dict = dict( zip( tuners_key, met_list ) )
        
        # Drop rows with na and inf
        # met_df = pd.DataFrame.from_dict( met_dict )
        # met_df = met_df.replace([np.inf, -np.inf], np.nan)
        # met_df = met_df.dropna(axis=0)
        # met_dict = met_df.to_dict("list")
        
        # Invert metrics if they are mmre
        
        # Run scott-knott on current metric
        print("Goal: %s" % goal_names[goal])
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
df_boxplot["name"] = df_boxplot["tuner"] + "+" + df_boxplot["model"]

df_boxplot["rank"] = df_boxplot.apply( lambda x : sk_df[ (x["goal"] == sk_df["goal"]) &
                                                (x["metric"] == sk_df["metric"]) &
                                                (x["name"] == sk_df["tuner"])]["rank"].iloc[0], axis = 1)
df_boxplot["goal_name"] = df_boxplot.apply( lambda x : goal_names[x["goal"]], axis = 1)
df_boxplot["tuner_short"] = df_boxplot.apply( lambda x : 
                                             {'DifferentialEvolutionCV' : "DE",
                                              'DodgeCV' : "DODGE",
                                              'RandomRangeSearchCV' : "Random",
                                              'GridSearchCV' : "Grid",
                                              'DefaultCV' : "Default"
                                              }[x["tuner"]], axis = 1)
df_boxplot["name"] = df_boxplot["tuner_short"] + "+" + df_boxplot["model"]
df_boxplot["name"] = df_boxplot["name"].replace({'DE+DecisionTreeRegressor':"DECART",
                                                 'Default+DecisionTreeRegressor':"CART"})


sns.set(font_scale=3)
plot = sns.catplot( x = "name", y = "value", col = "goal_name", row = "metric",
                  hue = "rank", data = df_boxplot, kind = "box", sharey = "row",
                  legend = False, margin_titles = True, showfliers = False, dodge=False,
                  height = 8, aspect=0.75, palette = [sns.color_palette("Set1")[1], "white"])
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

# Intrinsic dimensionality analysis
datasets = np.unique(df["dataset"])
dimensionality = pd.read_csv(dimensionality_name, sep=',')
dimensionality["Dataset"] = dimensionality["Dataset"].str.lower() + ".csv"
dimensionality = dimensionality.rename(columns={"Dataset":"dataset"})
dimensionality = dimensionality[ dimensionality["dataset"].isin(datasets) ]

fig, ax = plt.subplots()
plot = sns.scatterplot( x = "intrinsic dim_L1", y = "intrinsic dim_L2",
                   data = dimensionality, color="b" )
plot.set(xlabel = "Intrinsic dimensionality (L1)", ylabel = "Intrinsic dimensionality (L2)")
fig.set_size_inches(10, 5)
plot.get_figure().savefig("dimensionality.png", dpi=300)


random_df = df[ df["tuner"] == "RandomRangeSearchCV" ]
random_df = random_df.replace([np.inf, -np.inf], np.nan)
random_df = random_df.dropna(axis=0)

for goal in goals:
    rs_goal_df = random_df[random_df["goal"] == goal]
    rs_goal_df = rs_goal_df.join(dimensionality.set_index('dataset'), on="dataset")
    
    fig, ax = plt.subplots()
    plot = sns.regplot( x = "intrinsic dim_L1", y = "sa", data = rs_goal_df, color="b", line_kws = {"linewidth":0.75} )
    plot.set(xlabel = "Intrinsic dimensionality (L1)", ylabel = "SA", ylim = [-3, 3])
    fig.set_size_inches(10, 5)
    plot.get_figure().savefig("%s-%s.png" % ("sa-dim-goal", goal), dpi=300)
  
    
# Amount of explored values analysis
# df_random_n = pd.read_csv(os.path.join(data_dir, random_n_name), sep=',')
# df_random_n = df_random_n[df_random_n["sa"] > -5]
# # df_random_n = df_random_n[df_random_n["n"] > 0]
# # df_random_n["n"] = df_random_n["n"].replace(0,1)

# fig, ax = plt.subplots()
# plot = sns.regplot(x = "n", y = "sa", data = df_random_n, order=2, color="b", x_estimator=np.median)
# plot.set(xlabel = "Amount of random samples (n)", ylabel = "SA", xlim = [0, 65])
# fig.set_size_inches(10, 5)
# plot.get_figure().savefig("randomN.png", dpi=300)

# fig, ax = plt.subplots()
# plot = sns.boxplot(x = "n", y = "sa", data = df_random_n, color="b")
# plot.set(xlabel = "Amount of random samples (n)", ylabel = "SA", ylim = [-1,1])
# plot.get_figure().savefig("randomN.png", dpi=300)
# fig.set_size_inches(10, 5)
# plot.get_figure().savefig("randomN-Box.png", dpi=300)

# fig, ax = plt.subplots()
# plot = sns.boxplot(x = "n", y = "mar", data = df_random_n, color="b")
# plot.set(xlabel = "Amount of random samples (n)", ylabel = "MAR")
# plot.get_figure().savefig("randomN.png", dpi=300)
# fig.set_size_inches(10, 5)
# plot.get_figure().savefig("randomN-Box-MAR.png", dpi=300)

