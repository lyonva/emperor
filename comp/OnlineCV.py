from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

class OnlineCV(BaseCrossValidator):
    
    def __init__(self, skip = 0):
        self.skip = skip
    
    def _iter_test_indices(self, X=None, y=None, groups=None):
        for i in range(2, X.shape[0]):
            yield i
    
    def split(self, X, y=None, groups=None):
        # As we are online CV, our test data is only the current project
        # And training data is the previous ones up to now
        skips = np.ceil((X.shape[0] - 2) * self.skip)
        for train_index, test_index in super().split(X, y, groups):
            train_index = [ i for i in range(0, test_index[0]) ]
            if skips > 0:
                skips -= 1
            else:
                yield train_index, test_index
    
    def get_n_splits(self, X, y=None, groups=None):
        skips = np.ceil((X.shape[0] - 2) * self.skip)
        return int(X.shape[0] - 2 - skips)


if __name__ == "__main__":
    X = pd.DataFrame(np.reshape(np.zeros(100), (10, 10)))
    y =  pd.DataFrame(np.zeros(10))
    
    for i, (train_index, test_index) in enumerate(OnlineCV(skip=0).split(X, y)):
        print( "iter: %d\ntraining: %s\ntesting: %s\n" % (i, train_index, test_index) )
    
