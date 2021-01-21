from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from tuning.helper import grid_to_bounds, grid_types, cast_parameters

# Differential Evolution
# Based on the implementation by Tianpei Xia
# OIL framework
# https://github.com/arennax/effort_oil_2019
# Adapted to the scikit learn BaseSearchCV class
class DifferentialEvolutionCV(BaseSearchCV):
    def __init__(self, estimator, search_space, mutation_rate, crossover_rate,
                 population_size, iterations, *, scoring=None, n_jobs=None,
                 iid='deprecated', refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.search_space = search_space
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.iterations = iterations
    
    def _run_search(self, evaluate_candidates):
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        bounds = grid_to_bounds(self.search_space)
        types = grid_types(self.search_space)
        types = dict( [ (key, val) for key, val in types.items() if key in bounds.keys() ] ) # Only for bounded types
        types = list(types.values())
        dimensions = len(bounds)
        
        # Our initial population is completely random (normalized)
        population = [dict(zip(bounds.keys(), np.random.rand(dimensions))) for i in range(self.population_size)]
        
        # Scale the population to the hyper-parameter values
        min_b, max_b = np.asarray(list(bounds.values()))[:,0], np.asarray(list(bounds.values()))[:,1]
        diff = np.fabs(min_b - max_b)
        population_denorm = [dict(zip(ind.keys(), cast_parameters(min_b + np.array(list(ind.values())) * diff, types) )) for ind in population]
        
        # Get evaluation values
        results = evaluate_candidates(population_denorm)
        population_index = np.arange(self.population_size, dtype=int)
        
        # best_idx = np.argmin( results["rank_test_score"] )
        # best = results["params"][best_idx]
        
        # Simulate a certain amount of generations
        for i in range(self.iterations):
            
            # Generate new population by mutating current population
            population_trial = []
            population_trial_denorm = []
            for j in range(self.population_size):
                # Select three random other individuals
                indexes = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = np.array(population)[np.random.choice(indexes, 3, replace=False)]
                a, b, c = np.array(list(a.values())), np.array(list(b.values())), np.array(list(c.values()))
                
                # Create a mutant individual
                mutant = np.clip(a + self.mutation_rate * (b - c), 0, 1)
                
                # For each parameter, determine if we will use original or mutant
                cross_points = np.random.rand( dimensions ) < self.crossover_rate
                # But we need at least one parameter to change
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                # Mutate the individual
                trial = np.where(cross_points, mutant, np.asarray(list(population[j].values())) )
                
                # Denormalize new individual and add to new generation
                trial_denorm = cast_parameters(min_b + trial * diff, types)
                trial = dict(zip( population[j].keys(), trial ))
                trial_denorm = dict(zip( population[j].keys(), trial_denorm ))
                population_trial.append(trial)
                population_trial_denorm.append(trial_denorm)
            
            # Evaluate new population candidates
            results = evaluate_candidates(population_trial_denorm)
            
            # Select new individuals
            fitness = results[self.rank_test_name_]
            for j in range(self.population_size):
                old = population_index[j]
                new = j + (i + 1) * self.population_size
                if fitness[new] > fitness[old]:
                    population[j] = population_trial[j]
                    population_index[j] = new
                    
        
