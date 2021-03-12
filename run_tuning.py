# Datasets
import numpy as np
import pandas as pd
from data.DataLoader import DataLoader
import os

# Cross-validation
from comp.OnlineCV import OnlineCV
from comp.MonthlyCV import MonthlyCV
from sklearn.model_selection import LeaveOneOut, GridSearchCV

# Scoring and metrics
from sklearn.metrics import make_scorer
from evaluation import sa, mmre, evaluate

# Files and auxiliary
from helper import print_progress, save_prediction, save_metrics, save_parameters
import time

# SKLearn models
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Tuners
from tuning import DefaultCV
from tuning import DifferentialEvolutionCV
from tuning import RandomRangeSearchCV
from tuning import DodgeCV

# Pre-processing
from sklearn.preprocessing import MinMaxScaler

# Dataset and goal loader
data_dir = os.path.join("data","ph_names") # Datasets you want to use
loader = DataLoader.get_loader_instance(data_dir)
goals = loader.get_objectives()
datasets = loader.get_datasets()
# goals = [0]


# cross_validation = OnlineCV()
cross_validation = MonthlyCV()

# Machine learning algorithms
models = [DecisionTreeRegressor]
model_ranges = [
    # Decision Tree Regressor
    {
      # "max_features": [0.01, 1.0],
      "max_depth": [1, 12],
      "min_samples_split": [0.00001, 0.5],
      "min_samples_leaf": [0.00001, 1]
    }
]
# For Grid Search
# model_ranges = [
#     # Decision Tree Regressor
#     {"max_features": [0.01, 0.1, 0.25, 0.5, 1.0],
#      "max_depth": [1, 6, 12],
#      "min_samples_split": [2, 20],
#      "min_samples_leaf": [1, 12]
#     }
# ]
# For default parameters
# models = [LinearRegression,
#           KNeighborsRegressor,
#           DecisionTreeRegressor,
#           SVR,
#           RandomForestRegressor]
# model_ranges = [
#     {},
#     {},
#     {},
#     {},
#     {}
# ]

# Hyper-parameter tuners
# tuners = [DefaultCV, DifferentialEvolutionCV, RandomRangeSearchCV, DodgeCV]
# tuner_params = [
#     # Default
#     {},
#     # Differential Evolution
#     {"population_size":20,
#       "mutation_rate" : 0.8,
#       "crossover_rate" : 0.7,
#       "iterations": 10
#       },
#     # Random Search
#     {"n_iter":60
#       },
#     # Dodge
#     {"epsilon": 0.01,
#       "initial_size": 12,
#       "population_size": 60
#       }
# ]

# Just DE and default for testing
tuners = [DefaultCV, DifferentialEvolutionCV]
tuner_params = [
    # Default
    {},
    # Differential Evolution
    {"population_size":20,
      "mutation_rate" : 0.8,
      "crossover_rate" : 0.7,
      "iterations": 10
      }
]

# Grid Search
# tuners = [GridSearchCV]
# tuner_params = [
#     {}
# ]
# Just Random with different N
# tuners = [RandomRangeSearchCV]
# tuner_params = [
#     # Random Search
#     {"n_iter":45
#       }
# ]
# Default parameters
# tuners = [DefaultCV]
# tuner_params = [
#     {}
# ]

# tuner_cv = LeaveOneOut()
tuner_cv = MonthlyCV()
# tuner_cv = OnlineCV(skip=0.25)

tuner_objectives = {"sa":(sa, True), "mmre":(mmre, False)}
# tuner_objectives = {"mmre":(mmre, False)}

n_jobs = 3

# Pre-processing
prep = MinMaxScaler()

# Metrics
metric_names = ["sa", "sa_md", "mar", "mdar", "mmre", "mdmre"]

for goal in goals:
    
    for ds_name in datasets:
        
        X, y = loader.load_dataset(ds_name, goal)
        
        # Normalize dataset
        X_norm = prep.fit_transform(X)
        X = pd.DataFrame(X_norm, columns = X.columns, index = X.index)
        X.name = ds_name
        
        for model_class, search_space in zip(models, model_ranges):
            for tuner_class, tuner_settings in zip(tuners, tuner_params):
                for obj_name, (obj, gib) in tuner_objectives.items():
                
                    # For storing predicted values
                    y_pred = []
                    y_true = []
                    tuning_times = []
                    fitting_times = []
                    
                    for i, (train_index, test_index) in enumerate(cross_validation.split(X, y)):
                        
                        # State current iteration
                        dataset_name = X.name
                        model_name = model_class.__name__
                        tuner_name = tuner_class.__name__
                        print_progress(dataset_name, goal, i, model_name, tuner_name, obj_name)
                        
                        # Get splits
                        X_train, y_train = X.iloc[train_index,:], y.iloc[train_index]
                        X_test, y_test = X.iloc[test_index,:], y.iloc[test_index]
                        
                        # Make scorer
                        tuner_scoring = make_scorer(lambda yy_true, yy_pred:
                                                    obj(yy_true, yy_pred, y_train=y),
                                                    greater_is_better=gib)
                        
                        model = model_class()
                        tuner = tuner_class(model, search_space, scoring=tuner_scoring, cv=tuner_cv, n_jobs = n_jobs, **tuner_settings)
                        
                        # Perform hyper-parameter tuning
                        
                        tuning_start = time.time()
                        tuner.fit(X_train, y_train)
                        tuning_time = time.time() - tuning_start
                        tuning_results = tuner.cv_results_
                        
                        # Re-fit model
                        # Redundant with refit parameter, done for sanity
                        best_params = tuner.best_params_
                        model.set_params(**best_params)
                        fit_start = time.time()
                        model.fit(X_train, y_train)
                        fit_time = time.time() - fit_start
                        
                        # Test and save metrics
                        y_pred.extend( model.predict(X_test) )
                        y_true.extend( y_test )
                        tuning_times.append(tuning_time)
                        fitting_times.append(fit_time)
                        
                        # Save results of tuning
                        save_parameters(dataset_name, goal, i, model_name,
                                        tuner_name, obj_name, tuner.cv_results_)
                
                    # Evaluate the obtained results
                    y_true = np.rint(np.array(y_true))
                    y_pred = np.array(y_pred)
                    metrics = evaluate(y_true, y_pred, metric_names, y_train)
                    
                    # Save results
                    save_prediction(dataset_name, goal, model_name, tuner_name, obj_name,
                                    y_true, y_pred, fitting_times, tuning_times)
                    save_metrics(dataset_name, goal, model_name, tuner_name, obj_name,
                                 metrics, np.median(fitting_times), np.median(tuning_times))

