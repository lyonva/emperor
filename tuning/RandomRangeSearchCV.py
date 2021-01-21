from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from tuning.helper import grid_to_bounds, grid_types, cast_parameters, aggregate_dict, random_population
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Random Search
# Instead of grid, uses all posible values in the range
class RandomRangeSearchCV(BaseSearchCV):
    def __init__(self, estimator, search_space, n_iter,
                 *, scoring=None, n_jobs=None, iid='deprecated', refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.search_space = search_space
        self.n_iter = n_iter
    
    def _run_search(self, evaluate_candidates):
        bounds = grid_to_bounds(self.search_space)
        categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in bounds.keys() ] )
        types = grid_types(self.search_space)
        
        # Generate and evaluate random population
        population = random_population( bounds, list(types.values()), categories, self.n_iter )
        evaluate_candidates(population)
        