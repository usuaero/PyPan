import time

import numpy as np
import matplotlib.pyplot as plt

from pypan.solvers import Solver
from pypan.pp_math import norm, vec_norm, vec_inner, vec_cross
from pypan.helpers import OneLineProgress

class VortexRingSolver(Solver):
    """Vortex ring (doublet sheet) solver.

    Parameters
    ----------
    mesh : Mesh
        A PyPan mesh object about which to calculate the flow.

    verbose : bool, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._verbose:
            print()
            prog = OneLineProgress(self._N_panels, msg="Calculating panel influence matrix")

        # Create panel influence matrix; first index is the influenced panel, second is the influencing panel
        self._panel_influence_matrix = np.zeros((self._N_panels, self._N_panels, 3))
        for i, panel in enumerate(self._mesh.panels):
            self._panel_influence_matrix[:,i] = panel.get_ring_influence(self._mesh.cp)
            if self._verbose:
                prog.display()

        # Determine panel part of A matrix
        self._A_panels = np.einsum('ijk,ik->ij', self._panel_influence_matrix, self._mesh.n)


    def set_condition(self, **kwargs):
        """Sets the atmospheric conditions for the computation.

        V_inf : list
            Freestream velocity vector.

        rho : float
            Freestream density.
        """

        # Get freestream
        self._v_inf = np.array(kwargs["V_inf"])
        self._V_inf = norm(self._v_inf)
        self._u_inf = self._v_inf/self._V_inf
        self._V_inf_2 = self._V_inf*self._V_inf
        self._rho = kwargs["rho"]

        # Create part of b vector dependent upon V_inf
        self._b = -vec_inner(self._v_inf, self._mesh.n)


    def solve(self, **kwargs):
        """Solves the panel equations to determine the flow field around the mesh.

        Parameters
        ----------
        method : str, optional
            Method for computing the least-squares solution to the system of equations.
            May be 'direct' or 'svd'. 'direct' solves the equation A*Ax=A*b using a standard
            linear algebra solver. 'svd' solves the equation Ax=b in a least-squares sense
            using the singular value decomposition. 'direct' is much faster but may be susceptible
            to numerical error due to a poorly conditioned system. 'svd' is more reliable at
            producing a stable solution. Defaults to 'direct'.

        verbose : bool, optional

        Returns
        -------
        F : ndarray
            Force vector in mesh coordinates.

        M : ndarray
            Moment vector in mesh coordinates.
        """
        start_time = time.time()

        # Get kwargs
        method = kwargs.get("method", "direct")
        self._verbose = kwargs.get("verbose", False)

        # Set up A matrix
        if self._verbose: print("\nSolving case...", end='', flush=True)

        # Get wake influence matrix
        self._vortex_influence_matrix = self._mesh.wake.get_influence_matrix(points=self._mesh.cp, u_inf=self._u_inf, omega=np.zeros(3))

        # Specify A matrix
        A = (self._A_panels+np.einsum('ijk,ik->ij', self._vortex_influence_matrix, self._mesh.n))[:,1:]

        # Solve system
        self._mu = np.zeros(self._N_panels)

        # Direct method
        if method=='direct':
            self._mu[1:] = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, self._b[:,np.newaxis])).flatten()

        # Singular value decomposition
        elif method == "svd":
            self._mu[1:], res, rank, s_a = np.linalg.lstsq(A, b, rcond=None)

        # Print computation results
        end_time = time.time()
        if self._verbose:
            print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)
            print("    Sum of doublet strengths: {0}".format(np.sum(self._mu)))

            if method=="svd":
                try:
                    print("    Maximum residual: {0}".format(np.max(res)))
                except:
                    pass
                print("    Rank of A matrix: {0}".format(rank))
                print("    Max singular value of A: {0}".format(np.max(s_a)))
                print("    Min singular value of A: {0}".format(np.min(s_a)))

        if self._verbose: print("\nDetermining velocities, pressure coefficients, and forces...", end='', flush=True)
        start_time = time.time()

        # Determine velocities at each control point induced by panels
        self._v = self._v_inf[np.newaxis,:]+np.einsum('ijk,j', self._panel_influence_matrix, self._mu)

        # Include vortex sheet principal value in the velocity
        self._grad_mu = self._mesh.get_gradient(self._mu)
        self._grad_mu_in_plane = np.einsum('ijk,ik->ij', self._P_surf, self._grad_mu)
        self._v -= 0.5*self._grad_mu_in_plane

        # Determine wake induced velocities
        self._v += np.sum(self._vortex_influence_matrix*self._mu[np.newaxis,:,np.newaxis], axis=1)

        # Determine coefficients of pressure
        self._V = vec_norm(self._v)
        self._C_P = 1.0-(self._V*self._V)/self._V_inf_2

        # Determine force acting on each panel
        self._dF = -(0.5*self._rho*self._V_inf_2*self._mesh.dA*self._C_P)[:,np.newaxis]*self._mesh.n

        # Sum force components (doing it component by component allows numpy to employ a more stable addition scheme)
        self._F = np.zeros(3)
        self._F[0] = np.sum(self._dF[:,0])
        self._F[1] = np.sum(self._dF[:,1])
        self._F[2] = np.sum(self._dF[:,2])

        # Determine moment contribution due to each panel
        self._dM = vec_cross(self._mesh.r_CG, self._dF)

        # Sum moment components
        self._M = np.zeros(3)
        self._M[0] = np.sum(self._dM[:,0])
        self._M[1] = np.sum(self._dM[:,1])
        self._M[2] = np.sum(self._dM[:,2])

        end_time = time.time()
        if self._verbose: print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)
        return self._F, self._M