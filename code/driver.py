import numpy as np
from generate_features import get_brands
from mrots import opt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Demonstrates how to use multiple output regression with output and task structure (MROTS)
# to make predictions on brand mention data and compares results with linear regression,
# predicting the mean of the training set, and predicting zero as baselines.

def rmse(y, pred):
    """ Calculates RMSE (root mean square error). """        
    return np.sqrt(((pred - y)**2).mean())

def mrots_pred(W, b, X):
    """
    Calculates the MROTS prediction on X given W and b.
    """
    b_dot = np.dot(np.ones((X.shape[0], 1)), b.reshape(1, b.shape[0]))
    return np.dot(X, W) + b_dot

brandcount, fmat = get_brands(70)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(fmat, brandcount)

# Hyperparameters should be altered as desired. These parameter values were chosen for
# the brand predicting problem.
lam, lam1, lam3 = [1e-3 for i in range(3)]
lam2 = 1e-5
N, D = X_train.shape
K = y_train.shape[1]
Ominv = np.identity(K)
Siginv = np.identity(K)
b = (1./ N) * np.dot(y_train.T, np.ones((N, 1)))
W, b = opt(X_train, y_train, b, Ominv, Siginv, lam, lam1, lam2, lam3, 1000, 1e-8)

# Compare with linear regression
lin = LinearRegression()
lin.fit(X_train, y_train)

# Compare results
mrots_rmse = rmse(y_test, mrots_pred(W, b, X_test))
lin_rmse = rmse(y_test, lin.predict(y_train))
mean_rmse = rmse(y_test, y_train.mean())
zero_rmse = rmse(y_test, 0)

print "RMSE with MROTS: {}\nRMSE with linear regression: {}\nRMSE predicting mean: \
{}\nRMSE predicting zero: {}".format(mrots_rmse, lin_rmse, mean_rmse, zero_rmse)
