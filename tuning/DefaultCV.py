from sklearn.model_selection._search import BaseSearchCV
import numpy as np

# Default parameter CV
# Only explores the default parameter
class DefaultCV(BaseSearchCV):
    
    def __init__(self, estimator, search_space,
                 *, scoring=None, n_jobs=None, iid='deprecated', refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.search_space = search_space # Unused, for compatibility
    
    def _run_search(self, evaluate_candidates):
        default_params = dict( zip( self.search_space.keys(),
           [ self.estimator.get_params()[key] for key in self.search_space.keys() ] ) )
        evaluate_candidates([default_params])

