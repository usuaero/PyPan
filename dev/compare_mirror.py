import time
import os

import pypan as pp
import numpy as np
import matplotlib.pyplot as plt

from pypan.helpers import compare_mirror


if __name__=="__main__":

    # Load meshes
    left = pp.Mesh(name='left', mesh_file='dev/meshes/half_wing_left.vtk', verbose=True)
    right = pp.Mesh(name='right', mesh_file='dev/meshes/half_wing.vtk', verbose=True)

    # Compare
    compare_mirror(left, right, 1)