import numpy as np

def marp0(actual, n_runs = 1000):
    res = 0
    std = 0
    n = actual.size
    actual = np.array(actual)
    samples = []
    # Depending on size of sample
    # Use original MARP0 by Shepperd and MacDonell
    # Or use unbiased version by Langdon et al.
    if n <= 2000:
        # Unbiased
        for i in range(0, n):
            for j in range(0, i):
                res += abs( actual[i] - actual[j] )
                samples.append(actual[j])
        res *= 2.0 / ( n ** 2 )
        
    else:
        # Original
        samples = []
        for i in range(0, n):
            p = [0 if x == i else 1/(actual.size - 1) for x in range(actual.size)]
            pred = np.random.choice(actual, n_runs, replace=True, p=p)
            samples.extend(pred)
            res +=  np.abs( np.average(pred) - actual[i] )
        res /= n
    
    std = np.std(samples)
        
    return res, std

def mdarp0(actual, n_runs = 1000):
    res = 0
    std = 0
    n = actual.size
    actual = np.array(actual)
    
    samples = []
    for i in range(0, n):
        p = [0 if x == i else 1/(actual.size - 1) for x in range(actual.size)]
        pred = np.random.choice(actual, n_runs, replace=True, p=p)
        samples.extend( np.abs( actual[i] - pred ) )
        
    res = np.median(samples)
    std = np.std(samples)
        
    return res, std

# Mean baseline
def mean(actual):
    med = np.mean(actual)
    return mar(actual, med), sdar(actual, med)

# Median baseline
def median(actual):
    med = np.median(actual)
    return mdar(actual, med), sdar(actual, med)

# Standarized accuracy
# Using mean
def sa(y_true, y_pred):
    return 1 - ( mar(y_true, y_pred) / marp0( y_true )[0] )

# Using median
def sa_md(y_true, y_pred):
    return 1 - ( mdar(y_true, y_pred) / median( y_true )[0] )

# Effect size of SA, also known as Glass's delta
# Using mean
def effect_size(y_true, y_pred):
    return np.abs( ( mar(y_true, y_pred) - marp0( y_true )[0] ) / marp0( y_true )[1] )

# Using median
def effect_size_md(y_true, y_pred):
    return np.abs( ( mdar(y_true, y_pred) - mdarp0( y_true )[0] ) / mdarp0( y_true )[1] )

# Standarized deviation
# Based off the stability ratio and standardized accuracy
# With respect to baseline

# Using mean
def sd(y_true, y_pred):
    return 1 - ( sdar(y_true, y_pred) / marp0( y_true )[1] )

# Using median
def sd_md(y_true, y_pred):
    return 1 - ( sdar(y_true, y_pred) / median( y_true )[1] )

# Absolute error, aka absolute resuidual or AR
def ae_i(y_true, y_pred):
    return np.abs(y_true - y_pred)

# Magnitude of relative error
# def mre(y_true, y_pred):
#     return ae_i(y_true, y_pred) / y_true

# Magnitude of relative error
# With fixes by Xia et al. to account for zeros in the data
def mre(y_true, y_pred):
    res = []
    for pred, actual in zip(y_pred, y_true):
        score = 0
        if actual == 0:
            if pred == 0:
                score = 0
            elif abs(pred) < 0:
                score = 1
            else:
                score = (abs(pred - actual) + 1) / (actual + 1)
        else:
            score = abs(pred - actual) / actual
        res.append(score)
    return np.array(res)
                

# Mean magnitude of relative error
def mmre(y_true, y_pred):
    return np.nanmean(mre(y_true, y_pred))

# Median magnitude of relative error
def mdmre(y_true, y_pred):
    return np.nanmedian(mre(y_true, y_pred))

# PRED(X), usually PRED(25). % of predictions above 25% of the MRE
def pred(y_true, y_pred, n=25):
    return np.sum( mre(y_true, y_pred) <= (n/100) ) / y_true.size

# Mean absolute residual, aka mean absolute error or mae
def mar(y_true, y_pred):
    return np.average(ae_i(y_true, y_pred))

# Median absolute residual, aka median absolute error or mdae or mdae
def mdar(y_true, y_pred):
    return np.median(ae_i(y_true, y_pred))

# Standard deviation of absolute residual
def sdar(y_true, y_pred):
    return np.std(ae_i(y_true, y_pred))

_functions_map = {
    "sa":sa,
    "sa_md":sa_md,
    "effect_size":effect_size,
    "effect_size_md":effect_size_md,
    "sd":sd,
    "sd_md":sd_md,
    "mar":mar,
    "mdar":mdar,
    "sdar":sdar,
    "mmre":mmre,
    "mdmre":mdmre,
    "pred":pred
    }

def evaluate(y_true, y_pred, metrics):
    result = {}
    
    for met_name in metrics:
        if met_name in _functions_map:
            result[met_name] = _functions_map[met_name](y_true, y_pred)
    
    return result

if __name__ == "__main__":
    y_true = np.random.random(30)*500 + 750
    y_pred = np.zeros(30) + 1000
    metrics = _functions_map.keys()
    
    result = evaluate(y_true, y_pred, metrics)
    
    for key, val in result.items():
        print("%15s: %+3.5f" % (key, val))
    
