import pandas as pd
import numpy as np
from comp.OnlineCV import OnlineCV
from comp.MonthlyCV import MonthlyCV
from data import load_datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from tuning import DifferentialEvolutionCV
from tuning import RandomRangeSearchCV
from tuning import DodgeCV
from tuning import DefaultCV
from sklearn.metrics import make_scorer
from evaluation import sa, evaluate
from helper import print_progress, save_prediction, save_metrics, save_parameters

# General config
goal = 0 # Prediction objective
data_dir = "data/data_selected" # Datasets you want to use


datasets = load_datasets(data_dir, goal)
# cross_validation = OnlineCV()
cross_validation = MonthlyCV()

# Machine learning algorithms
models = [DecisionTreeRegressor]
model_ranges = [
    # Decision Tree Regressor
    {"max_features": [0.01, 1.0],
      "max_depth": [1, 12],
      "min_samples_split": [2, 20],
      "min_samples_leaf": [1, 12]
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

# Hyper-parameter tuners
tuners = [DefaultCV, DifferentialEvolutionCV, RandomRangeSearchCV, DodgeCV]
tuner_params = [
    # Default
    {},
    # Differential Evolution
    {"population_size":20,
      "mutation_rate" : 0.75,
      "crossover_rate" : 0.3,
      "iterations": 10
      },
    # Random Search
    {"n_iter":60
      },
    # Dodge
    {"epsilon": 0.01,
      "initial_size": 12,
      "population_size": 60
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


tuner_scoring = make_scorer(sa, greater_is_better=True)
# tuner_cv = LeaveOneOut()
tuner_cv = MonthlyCV()
n_jobs = 1

# Metrics
metric_names = ["sa", "mar", "sdar", "mmre"]

for X, y in datasets:
    for model_class, search_space in zip(models, model_ranges):
        for tuner_class, tuner_settings in zip(tuners, tuner_params):
            
            # For storing predicted values
            y_pred = []
            y_true = []
            
            for i, (train_index, test_index) in enumerate(cross_validation.split(X, y)):
                
                # State current iteration
                dataset_name = X.name
                model_name = model_class.__name__
                tuner_name = tuner_class.__name__
                print_progress(dataset_name, i, model_name, tuner_name)
                
                # Get splits
                X_train, y_train = X.iloc[train_index,:], y.iloc[train_index]
                X_test, y_test = X.iloc[test_index,:], y.iloc[test_index]
                
                model = model_class()
                tuner = tuner_class(model, search_space, scoring=tuner_scoring, cv=tuner_cv, n_jobs = n_jobs, **tuner_settings)
                
                # Perform hyper-parameter tuning
                tuner.fit(X_train, y_train)
                tuning_results = tuner.cv_results_
                
                # Re-fit model
                # Redundant with refit parameter, done for sanity
                best_params = tuner.best_params_
                model.set_params(**best_params)
                model.fit(X_train, y_train)
                
                # Test and save metrics
                y_pred.extend( model.predict(X_test) )
                y_true.extend( y_test )
                
                # Save results of tuning
                save_parameters(dataset_name, goal, i, model_name, tuner_name, tuner.cv_results_)
            
            # Evaluate the obtained results
            y_true = np.rint(np.array(y_true))
            y_pred = np.array(y_pred)
            metrics = evaluate(y_true, y_pred, metric_names)
            
            # Save results
            save_prediction(dataset_name, goal, model_name, tuner_name, y_true, y_pred)
            save_metrics(dataset_name, goal, model_name, tuner_name, metrics)
            
            

