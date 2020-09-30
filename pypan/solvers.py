"""Defines classes for solving potential flow scenarios."""

import time

import numpy as np

from .pp_math import vec_inner, vec_norm, norm

class Solver:
    """Base class for solvers."""

    def export_case_data(self, filename):
        """Writes the case data to the given file.

        Parameters
        ----------
        filename : str
            File location at which to store the case data.
        """

        # Setup data table
        item_types = [("cpx", "float"),
                      ("cpy", "float"),
                      ("cpz", "float"),
                      ("nx", "float"),
                      ("ny", "float"),
                      ("nz", "float"),
                      ("area", "float"),
                      ("u", "float"),
                      ("v", "float"),
                      ("w", "float"),
                      ("V", "float"),
                      ("C_P", "float"),
                      ("dFx", "float"),
                      ("dFy", "float"),
                      ("dFz", "float"),
                      ("circ", "float")]

        table_data = np.zeros(self._N_panels, dtype=item_types)

        # Geometry
        table_data[:]["cpx"] = self._cp[:,0]
        table_data[:]["cpy"] = self._cp[:,1]
        table_data[:]["cpz"] = self._cp[:,2]
        table_data[:]["nx"] = self._n[:,0]
        table_data[:]["ny"] = self._n[:,1]
        table_data[:]["nz"] = self._n[:,2]
        table_data[:]["area"] = self._dA

        # Velocities
        table_data[:]["u"] = self._v[:,0]
        table_data[:]["v"] = self._v[:,1]
        table_data[:]["w"] = self._v[:,2]
        table_data[:]["V"] = self._V
        table_data[:]["C_P"] = self._C_P

        # Circulation and forces
        table_data[:]["dFx"] = self._dF[:,0]
        table_data[:]["dFy"] = self._dF[:,1]
        table_data[:]["dFz"] = self._dF[:,2]
        table_data[:]["circ"] = self._gamma[:self._N_panels]

        # Define header and output
        header = "{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}{:<21}".format(
                 "Control (x)", "Control (y)", "Control (z)", "nx", "ny", "nz", "Area", "u", "v", "w", "V", "C_P", "dFx", "dFy",
                 "dFz", "circ")
        format_string = "%20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e"

        # Save
        np.savetxt(filename, table_data, fmt=format_string, header=header)



class VortexRingSolver(Solver):
    """Vortex ring solver.

    Parameters
    ----------
    mesh : Mesh
        A mesh object.

    verbose : bool, optional
    """

    def __init__(self, **kwargs):

        # Store mesh
        self._mesh = kwargs["mesh"]
        verbose = kwargs.get("verbose", False)

        # Gather control point locations and normals
        if verbose: print("\nParsing mesh...", end='', flush=True)
        self._N_panels = self._mesh.N
        self._N_edges = self._mesh.N_edges
        self._cp = np.copy(self._mesh.cp)
        self._n = np.copy(self._mesh.n)
        self._dA = np.copy(self._mesh.dA)

        # Gather edges
        self._N_edges = self._mesh.N_edges
        if self._N_edges != 0:
            self._edge_panel_ind = np.zeros((self._N_edges, 2))
            for i, edge in enumerate(self._mesh.kutta_edges):
                self._edge_panel_ind[i,:] = edge.panel_indices
        if verbose: print("Finished", flush=True)

        # Create panel influence matrix; first index is the influencing panel, second is the influenced panel
        if verbose: print("\nDetermining panel influence matrix...", end='', flush=True)
        self._influence_matrix = np.zeros((self._N_panels, self._N_panels, 3))
        for i, panel in enumerate(self._mesh.panels):
            self._influence_matrix[i,:] = panel.get_ring_influence(self._cp)

        # Determine panel part of A matrix
        self._A_panels = vec_inner(self._influence_matrix, self._n[np.newaxis,:])
        if verbose: print("Finished", flush=True)


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
        verbose = kwargs.get("verbose", False)

        # Lifting
        if lifting:
            if verbose: print("\nSolving lifting case...", end='', flush=True)
            pass

        # Nonlifting
        else:
            if verbose: print("\nSolving nonlifting case...", end='', flush=True)
            
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
        if verbose:
            print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)
            print("    Maximum residual: {0}".format(np.max(res)))
            print("    Circulation sum: {0}".format(np.sum(self._gamma)))

        # Determine velocities at each control point
        if verbose: print("\nDetermining velocities, pressure coefficients, and forces...", end='', flush=True)
        start_time = time.time()
        self._v = np.sum(self._influence_matrix*self._gamma[:,np.newaxis,np.newaxis], axis=0)
        self._V = vec_norm(self._v)

        # Determine coefficients of pressure
        self._C_P = 1.0-(self._V*self._V)/self._V_inf_2
        end_time = time.time()

        # Determine forces
        self._dF = self._rho*self._V_inf_2*(self._dA*self._C_P)[:,np.newaxis]*self._n
        self._F = np.sum(self._dF, axis=0).flatten()
        if verbose: print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)
        return self._F