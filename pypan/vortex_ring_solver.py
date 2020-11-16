"""Class for the first-order vortex ring (constant doublet) solver."""
import time

import numpy as np

from .solvers import Solver
from .pp_math import norm, vec_norm, vec_inner
from .helpers import OneLineProgress

class VortexRingSolver(Solver):
    """Vortex ring solver.

    Parameters
    ----------
    mesh : Mesh
        A mesh object.

    verbose : bool, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._verbose:
            print()
            prog = OneLineProgress(self._N_panels, msg="Determining panel influence matrix")

        # Create panel influence matrix; first index is the influenced panel, second is the influencing panel
        self._panel_influence_matrix = np.zeros((self._N_panels, self._N_panels, 3))
        for i, panel in enumerate(self._mesh.panels):
            self._panel_influence_matrix[:,i] = panel.get_ring_influence(self._cp)
            if self._verbose:
                prog.display()

        # Determine panel part of A matrix
        self._A_panels = np.einsum('ijk,ik->ij', self._panel_influence_matrix, self._n)

        # Determine projection matrix onto plane of each panel
        self._P_surf = np.repeat(np.identity(3)[np.newaxis,:,:], self._N_panels, axis=0)-np.matmul(self._n[:,:,np.newaxis], self._n[:,np.newaxis,:])


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
        self._b = -vec_inner(self._v_inf, self._n)


    def solve(self, **kwargs):
        """Solves the panel equations to determine the flow field around the mesh.

        Parameters
        ----------
        lifting : bool, optional
            Whether the Kutta condition is to be enforced. Defaults to False.

        verbose : bool, optional
        """
        start_time = time.time()

        # Get kwargs
        lifting = kwargs.get("lifting", False)
        self._verbose = kwargs.get("verbose", False)

        # Lifting
        if lifting:
            if self._N_edges==0:
                raise RuntimeError("Lifting case cannot be solved for a geometry with no Kutta edges.")

            # Create horseshoe vortex influence matrix; first index is the influenced panels (bordering the horseshoe vortex), second is the influencing panel
            if self._verbose:
                print()
                prog = OneLineProgress(self._mesh.N_edges, "Determining wake influence")
            self._vortex_influence_matrix = np.zeros((self._N_panels, self._N_panels, 3))
            for edge in self._mesh.kutta_edges:
                p_ind = edge.panel_indices
                V = edge.get_vortex_influence(self._cp, self._u_inf[np.newaxis,:])
                self._vortex_influence_matrix[:,p_ind[0]] = -V
                self._vortex_influence_matrix[:,p_ind[1]] = V
                if self._verbose:
                    prog.display()

            # Determine panel part of A matrix
            self._A_vortices = np.einsum('ijk,ik->ij', self._vortex_influence_matrix, self._n)

            if self._verbose: print("\nSolving lifting case...", end='', flush=True)

            # Specify A matrix
            A = self._A_panels+self._A_vortices

            # Specify b vector
            b = np.copy(self._b)

        # Nonlifting
        else:
            if self._verbose: print("\nSolving nonlifting case...", end='', flush=True)
            
            # Specify A matrix
            A = np.zeros((self._N_panels+1, self._N_panels))
            A[:-1] = self._A_panels
            A[-1,:] = 1.0

            # Specify b vector
            b = np.zeros(self._N_panels+1)
            b[:-1] = self._b

        # Solve system using least-squares approach
        self._gamma, res, rank, s_a = np.linalg.lstsq(A, b, rcond=None)
        end_time = time.time()
        if self._verbose:
            print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)
            try:
                print("    Maximum residual: {0}".format(np.max(res)))
            except:
                pass
            print("    Circulation sum: {0}".format(np.sum(self._gamma)))
            print("    Rank of A matrix: {0}".format(rank))
            print("    Max singular value of A: {0}".format(np.max(s_a)))
            print("    Min singular value of A: {0}".format(np.min(s_a)))

        if self._verbose: print("\nDetermining velocities, pressure coefficients, and forces...", end='', flush=True)
        start_time = time.time()

        # Determine velocities at each control point induced by panels
        self._v = self._v_inf[np.newaxis,:]+np.einsum('ijk,j', self._panel_influence_matrix, self._gamma)

        # Include vortex sheet principal value in the velocity
        self._grad_gamma = self._mesh.get_gradient(self._gamma)
        self._v -= 0.5*np.matmul(self._P_surf, self._grad_gamma[:,:,np.newaxis]).reshape(self._v.shape)

        # Determine velocities induced by horseshoe vortices
        if lifting:
            self._v += np.sum(self._vortex_influence_matrix*self._gamma[np.newaxis,:,np.newaxis], axis=1)

        # Determine coefficients of pressure
        self._V = vec_norm(self._v)
        self._C_P = 1.0-(self._V*self._V)/self._V_inf_2
        end_time = time.time()

        # Determine force acting on each panel
        self._dF = -(self._rho*self._V_inf_2*self._dA*self._C_P)[:,np.newaxis]*self._n

        # Sum force components starting with the smallest magnitudes (for numerical stability)
        self._F = np.zeros(3)
        x_sorted_ind = np.argsort(self._dF[:,0])
        self._F[0] = np.sum(self._dF[x_sorted_ind,0])
        y_sorted_ind = np.argsort(self._dF[:,1])
        self._F[1] = np.sum(self._dF[y_sorted_ind,1])
        z_sorted_ind = np.argsort(self._dF[:,2])
        self._F[2] = np.sum(self._dF[z_sorted_ind,2])
        if self._verbose: print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)
        return self._F