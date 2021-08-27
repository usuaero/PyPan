import time
import copy

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from pypan.solvers import Solver
from pypan.pp_math import norm, vec_norm, vec_inner, vec_cross
from pypan.helpers import OneLineProgress
from pypan.wake import StraightFixedWake, MarchingStreamlineWake, FullStreamlineWake, VelocityRelaxedWake


def get_panel_influences(args):
    """Calculates the influence of each panel on each point."""

    # Initialize storage
    panels, points = args
    inf_mat = np.zeros((len(points), len(panels), 3))
    
    # Loop through panels
    for i, panel in enumerate(panels):
        inf_mat[:,i,:] = panel.get_ring_influence(points)

    return inf_mat


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
        #N_processes = 8
        #with mp.Pool(processes=N_processes) as pool:
        #    N_per_process = self._N_panels//N_processes
        #    args = []
        #    for i in range(N_processes):
        #        if i<N_processes-1:
        #            panels = self._mesh.panels[i*N_per_process:(i+1)*N_per_process]
        #        else:
        #            panels = self._mesh.panels[i*N_per_process:]
        #        args.append((copy.deepcopy(panels), copy.deepcopy(self._mesh.cp)))

        #    res = pool.map(get_panel_influences, args)
        #    if self._verbose:
        #        prog.display()

        #self._panel_influence_matrix = np.concatenate(res, axis=1)
        #if self._verbose:
        #    prog.display()
        self._panel_influence_matrix = np.zeros((self._N_panels, self._N_panels, 3))
        for i, panel in enumerate(self._mesh.panels):
            self._panel_influence_matrix[:,i] = panel.get_ring_influence(self._mesh.cp)
            if self._verbose:
                prog.display()


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
        self._rho = kwargs["rho"]
        self._omega = np.array(kwargs.get("angular_rate", [0.0, 0.0, 0.0]))

        # Create part of b vector dependent upon v_inf and rotation
        v_rot = vec_cross(self._omega, self._mesh.cp)
        self._v_inf_and_rot = self._v_inf-v_rot
        self._b = -vec_inner(self._v_inf-v_rot, self._mesh.n)

        # Get solid body rotation
        self._omega = np.array(kwargs.get("angular_rate", [0.0, 0.0, 0.0]))

        # Finish Kutta edge search on mesh
        self._mesh.finalize_kutta_edge_search(self._u_inf)

        # Update wake
        self._mesh.wake.set_filament_direction(self._v_inf, self._omega)


    def solve(self, **kwargs):
        """Solves the panel equations to determine the flow field around the mesh.

        Parameters
        ----------
        method : str, optional
            Method for computing the least-squares solution to the system of equations. May be 'direct' or 'svd'. 'direct' solves the equation (A*)Ax=(A*)b using a standard linear algebra solver. 'svd' solves the equation Ax=b in a least-squares sense using the singular value decomposition. 'direct' is much faster but may be susceptible to numerical error due to a poorly conditioned system. 'svd' is more reliable at producing a stable solution. Defaults to 'direct'.

        wake_iterations : int, optional
            How many times the shape of the wake should be updated and the flow resolved. Only used if the mesh has been set with a "full_streamline" or "relaxed" wake. For "marching_streamline" wakes, the number of iterations is equal to the number of filament segments in the wake and this setting is ignored. Defaults to 2.

        export_wake_series : bool, optional
            Whether to export a vtk of the solver results after each wake iteration. Only used if the mesh has been set with an iterative wake. Defaults to False.

        wake_series_title : str, optional
            Gives a common file name and location for the wake series export files. Each file will be stored as "<wake_series_title>_<iteration_number>.vtk". May include a file path. Required if "export_wake_series" is True.

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
        dont_iterate_on_wake = not (isinstance(self._mesh.wake, VelocityRelaxedWake) or isinstance(self._mesh.wake, FullStreamlineWake) or isinstance(self._mesh.wake, MarchingStreamlineWake))

        # Non-iterative wake options
        if dont_iterate_on_wake:
            wake_iterations = 0
            export_wake_series = False

        # Iterative wake options
        else:

            # Number of iterations
            wake_iterations = kwargs.get("wake_iterations", 2)
            if isinstance(self._mesh.wake, MarchingStreamlineWake):
                wake_iterations = self._mesh.wake.N_segments_final

            # Wake series export
            export_wake_series = kwargs.get("export_wake_series", False)
            if export_wake_series:
                wake_series_title = kwargs.get("wake_series_title")
                if wake_series_title is None:
                    raise IOError("'wake_series_title' is required if 'export_wake_series' is true.")

        # Iterate on wake
        for i in range(wake_iterations+1):
            if self._verbose and not dont_iterate_on_wake:
                print("\nWake Iteration {0}/{1}".format(i, wake_iterations))
                print("========================")
            if self._verbose:
                print()
                start_time = time.time()
                print("    Solving singularity strengths (this may take a while)...", flush=True, end='')

            # Get wake influence matrix
            wake_influence_matrix = self._mesh.wake.get_influence_matrix(points=self._mesh.cp, u_inf=self._u_inf, omega=self._omega, N_panels=self._N_panels)

            # Specify A matrix
            A = np.zeros((self._N_panels+1,self._N_panels))
            A[:-1] = np.einsum('ijk,ik->ij', self._panel_influence_matrix, self._mesh.n)
            if not isinstance(wake_influence_matrix, float):
                A[:-1] += np.einsum('ijk,ik->ij', wake_influence_matrix, self._mesh.n)
            A[-1] = 1.0

            # Specify b vector
            b = np.zeros(self._N_panels+1)
            b[:-1] = self._b

            # Direct method
            if method=='direct':
                b = np.matmul(A.T, b[:,np.newaxis])
                A = np.matmul(A.T, A)
                self._mu = np.linalg.solve(A, b).flatten()

            # Singular value decomposition
            elif method == "svd":
                self._mu, res, rank, s_a = np.linalg.lstsq(A, b, rcond=None)

            # Clear up memory
            del A
            del b

            # Print computation results
            if self._verbose:
                print("Finished. Time: {0}".format(time.time()-start_time))
                print()
                print("    Solver Results:")
                print("        Sum of doublet strengths: {0}".format(np.sum(self._mu)))
                try:
                    print("        Maximum residual magnitude: {0}".format(np.max(np.abs(res))))
                    print("        Average residual magnitude: {0}".format(np.average(np.abs(res))))
                    print("        Median residual magnitude: {0}".format(np.median(np.abs(res))))
                    del res
                except:
                    pass

                if method=="svd":
                    print("        Rank of A matrix: {0}".format(rank))
                    print("        Max singular value of A: {0}".format(np.max(s_a)))
                    print("        Min singular value of A: {0}".format(np.min(s_a)))
                    del s_a

            if self._verbose:
                print()
                prog = OneLineProgress(4, msg="    Calculating derived quantities")

            # Determine velocities at each control point induced by panels
            self._v = self._v_inf_and_rot+np.einsum('ijk,j', self._panel_influence_matrix, self._mu)
            if self._verbose: prog.display()

            # Determine wake induced velocities
            self._v += np.sum(wake_influence_matrix*self._mu[np.newaxis,:,np.newaxis], axis=1)
            del wake_influence_matrix
            if self._verbose: prog.display()

            # Include doublet sheet principal value in the velocity
            self._v += -0.5*self._mesh.get_gradient(self._mu)
            if self._verbose: prog.display()

            # Determine coefficients of pressure
            V = vec_norm(self._v)
            self._C_P = 1.0-(V*V)/self._V_inf**2
            if self._verbose: prog.display()

            # export vtk
            if export_wake_series:
                self.export_vtk(wake_series_title+"_{0}.vtk".format(i+1))

            # Update wake
            if not dont_iterate_on_wake and i < wake_iterations: # Don't update the wake if this is the last iteration
                self._mesh.wake.update(self.get_velocity_induced_by_body, self._mu, self._v_inf, self._omega, self._verbose)

        # Determine force acting on each panel
        self._dF = -(0.5*self._rho*self._V_inf**2*self._mesh.dA*self._C_P)[:,np.newaxis]*self._mesh.n

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

        # Set solved flag
        self._solved = True

        return self._F, self._M

    
    def get_velocity_off_body(self, points):
        """Determines the velocity at the given points off the body. Considers the influence of both the body, wake, and freestream. Should not be used for points close to the body or wake.

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