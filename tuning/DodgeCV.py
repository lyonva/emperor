from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from tuning.helper import grid_to_bounds, grid_types, cast_parameters, aggregate_dict, random_population
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Dodge
# Based on the implementation by Amritanshu Agrawal
# Dodge framework
# https://github.com/amritbhanu/Dodge
# Adapted to the scikit learn BaseSearchCV class
class DodgeCV(BaseSearchCV):
    def __init__(self, estimator, search_space, epsilon,
                 initial_size = 15, population_size = 30,
                 *, scoring=None, n_jobs=None, iid='deprecated', refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.search_space = search_space
        self.epsilon = epsilon
        self.initial_size = initial_size
        self.population_size = population_size
    
    def _run_search(self, evaluate_candidates):
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        bounds = grid_to_bounds(self.search_space)
        categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in bounds.keys() ] )
        types = grid_types(self.search_space)
        
        # Evaluate the first option: default parameters
        default_params = dict( zip( self.search_space.keys(),
           [ self.estimator.get_params()[key] for key in self.search_space.keys() ] ) )
        results = evaluate_candidates([ default_params ])
        
        # Save results in alpha, the array of interesting values
        # Dont save the default parameters, often they dont use numerical features
        alpha = list(results[self.mean_test_name_])
        
        # Generate and evaluate random population
        population = random_population( bounds, list(types.values()), categories, self.initial_size )
        results = evaluate_candidates(population)
        
        # See if random population lies within epsilon
        parameters = list(population)
        weight = [0 for i in range( self.initial_size )]
        for i in range(0, self.initial_size): # endorse or deprecate
            score = results[self.mean_test_name_][i]
            if all(abs(a - score) > self.epsilon for a in alpha):
                weight[i] += 1
                alpha.append(score)
            else:
                weight[i] -= 1
        
        
        # Now mutate individuals from tree
        for i in range( self.initial_size + 1, self.population_size ):
            # Find best setting so far (random if multiple)
            best_idx = np.random.choice( np.where( np.array(weight) == max(weight) )[0] )
            best_param = parameters[ best_idx ]
            
            # Find worst for mutation (random if multiple)
            worst_idx = np.random.choice( np.where( np.array(weight) == min(weight) )[0] )
            worst_param = parameters[ worst_idx ]
            
            # Generate new parameter ranges
            best_num = [ best_param[key] for key in bounds.keys() ]
            worst_num = [ worst_param[key] for key in bounds.keys() ]
            bounds_new = dict(zip( bounds.keys() , [ [ best, (best + worst)/2 ] for best, worst in zip( best_num, worst_num )] ) )
            categories_new = dict(zip( categories.keys(), [ [best_param[key]] for key in categories.keys() ] ))
            
            # Generate new random individual and evaluate
            # Could maybe be more than one
            individual = population = random_population( bounds_new, list(types.values()), categories_new, 1 )
            results = evaluate_candidates(individual)
            
            # See if new value lies within epsilon
            # We endorse or deprecate the new individual
            # Based on father's weight
            score = results[self.mean_test_name_][i]
            parameters.extend(individual)
            if all(abs(a - score) > self.epsilon for a in alpha):
                weight.append( weight[best_idx] + 1 ) # Endorse self
                alpha.append(score)
            else:
                weight.append( weight[best_idx] - 1 ) # Deprecate self
        
        