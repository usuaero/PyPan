"""Helpful math functions."""

import numpy as np
import math as m


def vec_norm(x):
    """Calculates the norm of the last dimension of x."""
    xT = x.T
    return np.sqrt(xT[0]*xT[0]+xT[1]*xT[1]+xT[2]*xT[2]).T


def norm(x):
    """Calculates the norm of of x."""
    return m.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])


def dist(x, y):
    """Calculates the Euclidean distance between x and y."""
    return norm([x[0]-y[0], x[1]-y[1], x[2]-y[2]])


def vec_inner(x, y):
    """Calculates the inner product of the last dimensions of x and y."""
    xT = x.T
    yT = y.T
    return (xT[0]*yT[0]+xT[1]*yT[1]+xT[2]*yT[2]).T


def inner(x, y):
    """Calculates the inner product of x and y."""
    return x[0]*y[0]+x[1]*y[1]+x[2]*y[2]


def vec_cross(x, y):
    """Calculates the cross product of the last dimensions of x and y."""
    xT = x.T
    yT = y.T
    return np.array([xT[1]*yT[2]-xT[2]*yT[1],
                     xT[2]*yT[0]-xT[0]*yT[2],
                     xT[0]*yT[1]-xT[1]*yT[0]]).T


def cross(x, y):
    """Calculates the cross product of the last dimensions of x and y."""
    return np.array([x[1]*y[2]-x[2]*y[1],
                     x[2]*y[0]-x[0]*y[2],
                     x[0]*y[1]-x[1]*y[0]]).T

def gauss_seidel(A, b, tolerance=1e-10, max_iterations=10000, verbose=False):
    
    x = np.zeros_like(b, dtype=np.double)

    if verbose:
        print("Running Gauss-Seidel")
        print("{0:<20}{1:<20}".format("Iteration", "Error"))
    
    #Iterate
    for k in range(max_iterations):
        
        x_old  = x.copy()
        
        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]

        # Check error
        err = np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
        if verbose:
            print("{0:<20}{1:<20.5e}".format(k, err))
            
        #Stop condition 
        if err < tolerance:
            break
            
    return x