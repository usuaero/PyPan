"""PyPan supersonic solver. I do not anticipate this to ever be fully functional. This is a space for me to prototype ideas within the framework of PyPan."""

import numpy as np
from pypan.mesh import Mesh
from pypan.solvers import Solver
from pypan.pp_math import vec_inner
from pypan.helpers import OneLineProgress

class SupersonicSolver(Solver):
    """A class for modelling linearized supersonic flow about a body.

    Parameters
    ----------
    mesh : Mesh
        A PyPan mesh object about which to calculate the flow.

    verbose : bool, optional
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Get number of vertices
        self._N_vert = self._mesh.vertices.shape[0]


    def set_condition(self, **kwargs):
        """Sets the condition for the supersonic flow about the body.

        Parameters
        ----------
        M : float
            Freestream Mach number. Must be greater than 1.

        alpha : float, optional
            Freestream angle of attack in degrees (assuming standard body-fixed coordinate system). Defaults to 0.

        beta : float, optional
            Freestream sideslip angle in degrees (true, not flank, assuming standard fody-fixed coordinate system). Defaults to 0.
        """

        # Get kwargs
        self._M = kwargs['M']
        self._alpha = np.radians(kwargs.get('alpha', 0.0))
        self._beta = np.radians(kwargs.get('beta', 0.0))

        # Calculate compressibility direction vector
        self._c_0 = -np.array([np.cos(self._alpha)*np.cos(self._beta), np.sin(self._beta), np.sin(self._alpha)*np.cos(self._beta)])

        # Calculate Mach parameters
        self._B = np.sqrt(self._M**2-1.0)
        self._C_mu = self._B/self._M
        self._mu = np.arccos(self._C_mu)

        # Run domain of dependence searches
        self._run_dod_recursive_search()
        self._run_dod_brute_force_search()


    def _run_dod_recursive_search(self):
        # Determines the domain of dependence for each vertex using a recursive algorithm

        if self._verbose:
            print()
            prog = OneLineProgress(4, msg="    Running recursive domain of dependence search")

        # Sort vertices in compressibility direction
        x_c = vec_inner(self._mesh.vertices, -self._c_0)
        sorted_ind = np.argsort(x_c)

        if self._verbose: prog.display()
        if self._verbose: prog.display()
        if self._verbose: prog.display()
        if self._verbose: prog.display()


    def _run_dod_brute_force_search(self):
        # Determines the domain of dependence for each vertex using the brute force method
        
        if self._verbose:
            print()
            prog = OneLineProgress(4, msg="    Running recursive domain of dependence search")
        if self._verbose: prog.display()
        if self._verbose: prog.display()
        if self._verbose: prog.display()
        if self._verbose: prog.display()
