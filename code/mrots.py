# Functions implementing multiple output regression with output and task structure (MROTS)
# See (Rai et. al., 2012) for details.

import numpy as np
from scipy.sparse import kron
from sklearn.covariance import GraphLasso, graph_lasso
import time

def opt(X, y, b, Ominv, Siginv, lam, lam1, lam2, lam3, max_iter, tol=1e-5):
    """
    Returns regression weights and bias vector for the MROTS model fitting the training
    inputs X and training outputs y.
    """
    W = 0
    N, D = X.shape
    K = y.shape[1]
    c = 0

    while True:
        c += 1
        W_prev = W
        W = opt_W(X, y, b, Ominv, Siginv, lam, lam1, N, D, K)
        b = opt_b(X, y, W, Ominv, Siginv, N, D, K)
        Siginv = opt_Siginv(X, y, W, b, Ominv, lam1, lam3, N, D, K)
        Ominv = opt_Ominv(X, y, W, b, Siginv, lam2, N, D, K) 

        if np.linalg.norm(W - W_prev) < tol or c > max_iter:
            return W, b
        
def opt_W(X, Y, b, Ominv, Siginv, lam, lam1, N, D, K):
    """ Optimizes W with everything else fixed. """
    # print np.dot(X.T, X).shape, Ominv.shape, Siginv.shape, K, D
    A = (kron(Ominv, np.dot(X.T, X)) + kron((lam1*Siginv + lam*np.identity(K)), 
        np.identity(D))).toarray()
    m = np.dot(np.ones((N, 1)), b.reshape(1, b.shape[0]))
    bb = (np.dot(np.dot(X.T, Y - m), Ominv)).flatten(order='F').reshape(A.shape[0], 1)
    Wflat = np.linalg.solve(A, bb)
    return Wflat.reshape((D, K), order='F')
    
def opt_b(X, Y, W, Ominv, Siginv, N, D, K):
    """ Optimizes b with everything else fixed. """
    return (1./N) * np.dot((Y - np.dot(X, W)).T, np.ones((N, 1)))
    
def opt_Siginv(X, Y, W, b, Ominv, lam1, lam3, N, D, K):
    """ Optimizes Sigma inverse with everything else fixed. """
    return glasso((lam1/D) * np.dot(W.T, W), lam3)
    
def opt_Ominv(X, Y, W, b, Siginv, lam2, N, D, K):
    """ Optimizes Omega inverse with everything else fixed. """
    m = np.dot(np.ones((N,1)), b.reshape(1,b.shape[0]))
    return glasso((1./N) * np.dot((Y - np.dot(X, W) - m).T, Y - np.dot(X, W) - m), lam2)

def glasso(A, rho):
    """ Applies the graphical lasso method to A with penalty rho. """
    assert(A.shape[0] == A.shape[1]), 'Mismatched dimensions of A'
    return graph_lasso(A, rho, max_iter=100)[0]