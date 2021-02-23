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