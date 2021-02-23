from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

# Prediction for time-based data
# Will predict the last value in the set
# Values near the end can be skipped to simulate the scenario of predicting time units in the future
class MonthlyCV(BaseCrossValidator):
    
    def __init__(self, skip = 0):
        self.skip = skip
    
    def _iter_test_indices(self, X=None, y=None, groups=None):
        yield [X.shape[1] - 1]
    
    def split(self, X, y=None, groups=None):
        train_index = np.arange(0, (X.shape[1]-self.skip-1))
        test_index  = [X.shape[1] - 1]
        yield train_index, test_index
    
    def get_n_splits(self, X, y=None, groups=None):
        return 1


if __name__ == "__main__":
    X = pd.DataFrame(np.reshape(np.zeros(100), (10, 10)))
    y =  pd.DataFrame(np.zeros(10))
    
    for i, (train_index, test_index) in enumerate(MonthlyCV().split(X, y)):
        print( "iter: %d\ntraining: %s\ntesting: %s\n" % (i, train_index, test_index) )
    
