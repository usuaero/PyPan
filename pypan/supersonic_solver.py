"""PyPan supersonic solver. I do not anticipate this to ever be fully functional. This is a space for me to prototype ideas within the framework of PyPan."""

import sys

import numpy as np

from pypan.mesh import Mesh
from pypan.solvers import Solver
from pypan.pp_math import vec_inner, inner, norm
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

        # Initialize a few things
        self._verts_in_dod = np.zeros((self._N_vert, self._N_vert), dtype=bool)
        self._verts_in_dod_brute_force = np.zeros((self._N_vert, self._N_vert), dtype=bool)
        self._dod_is_calculated = np.zeros(self._N_vert, dtype=bool)


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

        # Calculate compressibility direction vector and orthogonal complement projection
        self._c_0 = np.array([np.cos(self._alpha)*np.cos(self._beta), np.sin(self._beta), np.sin(self._alpha)*np.cos(self._beta)])
        self._P_c = np.eye(3)-np.einsum('i,j->ij', self._c_0, self._c_0)

        # Calculate Mach parameters
        self._B_2 = self._M**2-1.0
        self._B = np.sqrt(self._B_2)
        self._C_mu = self._B/self._M
        self._mu = np.arccos(self._C_mu)

        # Run domain of dependence searches
        self._run_dod_recursive_search()
        self._run_dod_brute_force_search()

        # Check dod searches got the same result
        mismatch = np.argwhere(self._verts_in_dod != self._verts_in_dod_brute_force)
        print(mismatch)


    def _run_dod_recursive_search(self):
        # Determines the domain of dependence for each vertex using a recursive algorithm

        if self._verbose:
            print()
            prog = OneLineProgress(self._N_vert, msg="Running recursive domain of dependence search")

        # Sort vertices in compressibility direction, from most downstream to most upstream
        x_c = vec_inner(self._mesh.vertices, self._c_0)
        self._sorted_ind = np.argsort(x_c)

        # Run recursive search
        for i, vert_ind in enumerate(self._sorted_ind):

            # Check if it's already been calculated
            if not self._dod_is_calculated[vert_ind]:

                # Call recursive function
                self._verts_in_dod[vert_ind,:] = self._get_dod(vert_ind, i, 0)

            if self._verbose: prog.display()


    def _get_dod(self, ind, i, depth):
        # Returns the domain of dependence for the vertex with index ind, which is the i-th sorted index

        # Check if it's already been calculated
        if not self._dod_is_calculated[ind]:

            # Loop through upstream vertices
            for j, upstream_ind in enumerate(self._sorted_ind[i+1::]):
                j = j+i+1

                # Check if it's already in dod
                if not self._verts_in_dod[ind, upstream_ind]:

                    # Check if it's in the dod
                    if self._in_dod_upstream(self._mesh.vertices[ind], self._mesh.vertices[upstream_ind]):

                        # Add to dod
                        self._verts_in_dod[ind, upstream_ind] = True

                        # Get its dod
                        self._verts_in_dod[ind,:] |= self._get_dod(upstream_ind, j, depth+1)

        return self._verts_in_dod[ind,:]


    def _in_dod_upstream(self, r0, r1):
        # Checks if r1 is in the domain of dependence for r0 knowing already that r1 is upstream

        # Get displacements
        r = r1-r0
        x_c = inner(r, self._c_0)
        r_c = r-x_c*self._c_0

        # Check
        if x_c*x_c >= self._B_2*inner(r_c, r_c):
            return True
        
        return False


    def _in_dod(self, r0, r1):
        # Checks if r1 is in the domain of dependence for r0

        # Get displacement vector
        r = r1-r0

        # Check for upstream
        x_c = inner(r, self._c_0)
        if x_c>0:

            # Get radius from compressibility axis
            r_c = r-x_c*self._c_0

            # Check
            if x_c*x_c >= self._B_2*inner(r_c, r_c):
                return True
        
        return False


    def _hyperbolic_distance(self, r0, r1):
        # Calculates the hyperbolic distance between two points

        # Get displacements
        r = r1-r0
        x_c = inner(r, self._c_0)
        r_c = norm(r-x_c*self._c_0)

        return np.sqrt(x_c**2-self._B_2*r_c**2)


    def _run_dod_brute_force_search(self):
        # Determines the domain of dependence for each vertex using the brute force method
        
        if self._verbose:
            print()
            prog = OneLineProgress(self._N_vert, msg="Running brute force domain of dependence search")

        for i in range(self._N_vert):
            for j in range(self._N_vert):

                # Check if in dod
                self._verts_in_dod_brute_force[i,j] = self._in_dod(self._mesh.vertices[i], self._mesh.vertices[j])

            if self._verbose: prog.display()
