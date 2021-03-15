import time

import numpy as np
import matplotlib.pyplot as plt

from pypan.solvers import Solver
from pypan.pp_math import norm, vec_norm, vec_inner, vec_cross
from pypan.helpers import OneLineProgress
from pypan.wake import NonIterativeWake

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
        
        angular_rate : list, optional
            Body-fixed angular rate vector (given in rad/s). Defaults to [0.0, 0.0, 0.0].
        """

        # Set solved flag
        self._solved = False

        # Get freestream
        self._v_inf = np.array(kwargs["V_inf"])
        self._V_inf = norm(self._v_inf)
        self._u_inf = self._v_inf/self._V_inf
        self._V_inf_2 = self._V_inf*self._V_inf
        self._rho = kwargs["rho"]
        self._omega = np.array(kwargs.get("angular_rate", [0.0, 0.0, 0.0]))

        # Create part of b vector dependent upon v_inf and rotation
        v_rot = vec_cross(self._omega, self._mesh.cp)
        self._b = -vec_inner(self._v_inf-v_rot, self._mesh.n)

        # Get solid body rotation
        self._omega = np.array(kwargs.get("angular_rate", [0.0, 0.0, 0.0]))

        # Update wake
        self._mesh.wake.set_filament_direction(self._v_inf, self._omega)


    def solve(self, **kwargs):
        """Solves the panel equations to determine the flow field around the mesh.

        Parameters
        ----------
        method : str, optional
            Method for computing the least-squares solution to the system of equations. May be 'direct' or 'svd'. 'direct' solves the equation A*Ax=A*b using a standard linear algebra solver. 'svd' solves the equation Ax=b in a least-squares sense using the singular value decomposition. 'direct' is much faster but may be susceptible to numerical error due to a poorly conditioned system. 'svd' is more reliable at producing a stable solution. Defaults to 'direct'.

        wake_iterations : int, optional
            How many times the shape of the wake should be iterated on. This will only be used if the mesh has been set with an iterative wake. On the first iteration, the wake shape is assumed to be simply in the direction of the freestream. Thus, if the effects of wake rollup are to be observed, at least 2 wake iterations must be allowed. Defaults to 3.

        verbose : bool, optional

        Returns
        -------
        F : ndarray
            Force vector in mesh coordinates.

        M : ndarray
            Moment vector in mesh coordinates.
        """

        # Begin timer
        self._verbose = kwargs.get("verbose", False)

        # Get kwargs
        method = kwargs.get("method", "direct")
        wake_iterations = kwargs.get("wake_iterations", 3)
        dont_iterate_on_wake = isinstance(self._mesh.wake, NonIterativeWake)
        if dont_iterate_on_wake:
            wake_iterations = 1

        # Iterate on wake
        for i in range(wake_iterations):
            if self._verbose and not dont_iterate_on_wake:
                print("\nWake Iteration {0}".format(i+1))
                print("====================")
            if self._verbose:
                print()
                prog = OneLineProgress(3, msg="    Solving case")

            # Get wake influence matrix
            self._wake_influence_matrix = self._mesh.wake.get_influence_matrix(points=self._mesh.cp, u_inf=self._u_inf, omega=self._omega, N_panels=self._N_panels)
            if self._verbose: prog.display()

            # Specify A matrix
            A = np.zeros((self._N_panels+1,self._N_panels))
            A[:-1] = self._A_panels
            if not isinstance(self._wake_influence_matrix, float):
                A[:-1] += np.einsum('ijk,ik->ij', self._wake_influence_matrix, self._mesh.n)
            A[-1] = 1.0
            if self._verbose: prog.display()

            # Specify b vector
            b = np.zeros(self._N_panels+1)
            b[:-1] = self._b

            # Direct method
            if method=='direct':
                self._mu = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, b[:,np.newaxis])).flatten()
                res = np.matmul(A, self._mu[:,np.newaxis]).flatten()-b

            # Singular value decomposition
            elif method == "svd":
                self._mu, res, rank, s_a = np.linalg.lstsq(A, b, rcond=None)

            # Print computation results
            if self._verbose:
                prog.display()
                print()
                print("    Solver Results:")
                print("        Sum of doublet strengths: {0}".format(np.sum(self._mu)))
                try:
                    print("        Maximum residual magnitude: {0}".format(np.max(np.abs(res))))
                    print("        Average residual magnitude: {0}".format(np.average(np.abs(res))))
                except:
                    pass

                if method=="svd":
                    print("        Rank of A matrix: {0}".format(rank))
                    print("        Max singular value of A: {0}".format(np.max(s_a)))
                    print("        Min singular value of A: {0}".format(np.min(s_a)))

            if self._verbose:
                print()
                prog = OneLineProgress(7, msg="    Calculating derived quantities")

            # Determine velocities at each control point induced by panels
            self._v = self._v_inf[np.newaxis,:]+np.einsum('ijk,j', self._panel_influence_matrix, self._mu)
            if self._verbose: prog.display()

            # Determine wake induced velocities
            self._v += np.sum(self._wake_influence_matrix*self._mu[np.newaxis,:,np.newaxis], axis=1)
            if self._verbose: prog.display()

            # Include doublet sheet principal value in the velocity
            self._grad_mu = self._mesh.get_gradient(self._mu)
            self._v -= 0.5*self._grad_mu
            if self._verbose: prog.display()

            # Determine coefficients of pressure
            self._V = vec_norm(self._v)
            self._C_P = 1.0-(self._V*self._V)/self._V_inf_2
            if self._verbose: prog.display()

            # Determine force acting on each panel
            self._dF = -(0.5*self._rho*self._V_inf_2*self._mesh.dA*self._C_P)[:,np.newaxis]*self._mesh.n
            if self._verbose: prog.display()

            # Sum force components (doing it component by component allows numpy to employ a more stable addition scheme)
            self._F = np.zeros(3)
            self._F[0] = np.sum(self._dF[:,0])
            self._F[1] = np.sum(self._dF[:,1])
            self._F[2] = np.sum(self._dF[:,2])
            if self._verbose: prog.display()

            # Determine moment contribution due to each panel
            self._dM = vec_cross(self._mesh.r_CG, self._dF)

            # Sum moment components
            self._M = np.zeros(3)
            self._M[0] = np.sum(self._dM[:,0])
            self._M[1] = np.sum(self._dM[:,1])
            self._M[2] = np.sum(self._dM[:,2])
            if self._verbose: prog.display()

            # Update wake
            if not dont_iterate_on_wake:
                self._mesh.wake.update(self.get_velocity_induced_by_body, self._mu, self._v_inf, self._verbose)

        # Set solved flag
        self._solved = True

        return self._F, self._M

    
    def get_velocity_off_body(self, points):
        """Determines the velocity at the given points off the body. Considers the influence of both the body and the wake. Should not be used for points close to either.

        Parameters
        ----------
        points : ndarray
            Array of points at which to evaluate the velocity.

        Returns
        -------
        ndarray
            Array of velocities at each point.
        """

        # Assemble influence matrix
        inf_mat = np.zeros((points.shape[0], self._N_panels, 3))

        # Panels
        for i, panel in enumerate(self._mesh.panels):
            inf_mat[:,i] = panel.get_ring_influence(points)

        # Wake
        inf_mat += self._mesh.wake.get_influence_matrix(points=points, u_inf=self._u_inf, omega=self._omega, N_panels=self._N_panels)

        return np.einsum('ijk,j', inf_mat, self._mu)+self._v_inf[np.newaxis,:]


    def get_velocity_induced_by_body(self, points):
        """Determines the velocity at the given points off the body considering only the influence of the body (not the wake). Should not be used for points close to either.

        Parameters
        ----------
        points : ndarray
            Array of points at which to evaluate the velocity.

        Returns
        -------
        ndarray
            Array of velocities at each point.
        """

        # Assemble influence matrix
        inf_mat = np.zeros((points.shape[0], self._N_panels, 3))

        # Panels
        for i, panel in enumerate(self._mesh.panels):
            inf_mat[:,i] = panel.get_ring_influence(points)

        return np.einsum('ijk,j', inf_mat, self._mu)