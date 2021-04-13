import time
import stl
import warnings
import copy

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from pypan.pp_math import vec_cross, vec_inner, vec_norm, norm, inner
from pypan.helpers import OneLineProgress
from pypan.panels import Tri, Quad
from pypan.wake import Wake, StraightFixedWake, FullStreamlineWake, VelocityRelaxedWake, MarchingStreamlineWake
from pypan.kutta_edges import KuttaEdge


class Mesh:
    """A class for defining collections of panels.

    Parameters
    ----------
    name : str
        Name of the mesh.

    mesh_file : str
        File path to the mesh file. Please note that PyPan assumes the panel normals all point outward. Failure to meet this condition can produce erroneous results. Can be "STL" or "VTK".

        ASCII or binary STL files may be used.

        Currently PyPan can import a VTK *unstructured mesh*. The panels should be given as POLYGONS. PyPan can accept no other format currently. Within a VTK file, the normal vector, area, and centroid may also be given under CELL_DATA. In all cases LOOKUP_TABLE should be default (PyPan is not currently able to parse non-default lookup tables).

    adjacency_file : str, optional
        Name of the panel adjacency mapping file for this mesh. Must be previously generated using Mesh.export_panel_adjacency_mapping(). Defaults to None, in which case the panel adjacency mapping will be determined using a brute force approach. Also, if the file cannot be found as specified, it will be ignored and the panel adjacency mapping will be determined using the brute force approach.

    CG : list, optional
        Location of the center of gravity for the mesh. This is the location about which moments are computed. Defaults to [0.0, 0.0, 0.0]. This is relative to the coordinate system of the mesh.

    gradient_fit_type : str, optional
        The type of basis functions to use for least-squares estimation of gradients. May be 'linear' or 'quad'. Defaults to 'quad' (recommended).
    """

    def __init__(self, **kwargs):

        # Load kwargs
        self.name = kwargs["name"]
        mesh_file = kwargs["mesh_file"]
        self._verbose = kwargs.get("verbose", False)
        self.CG = np.array(kwargs.get("CG", [0.0, 0.0, 0.0]))
        self._gradient_type = kwargs.get('gradient_fit_type', 'quad')

        # Load mesh
        if self._verbose:
            start_time = time.time()
            print("\nReading in mesh...", end='', flush=True)
        self._load_mesh(mesh_file)
        if self._verbose:
            end_time = time.time()
            print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)

        # Create panel vertex mapping
        # VTK does this inherently; STL has a faster way than the brute-force method
        if self._vertex_mapping_needed:
            self._determine_panel_vertex_mapping()

        # Determine panel adjacency mapping
        self._determine_panel_adjacency_mapping(**kwargs)

        # Calculate moment arms
        self.r_CG = self.cp-self.CG[np.newaxis,:]

        # Set up dummy wake
        self.wake = Wake(kutta_edges=[])

        # Display mesh information
        if self._verbose:
            print("\nMesh Parameters:")
            print("    # panels: {0}".format(self.N))
            print("    # vertices: {0}".format(self._vertices.shape[0]))

    
    def _load_mesh(self, mesh_file):
        # Loads the mesh from the input file

        self._vertex_mapping_needed = True

        # STL
        if "stl" in mesh_file or "STL" in mesh_file:
            self._load_stl(mesh_file)
            self._vertex_mapping_needed = False

        # VTK
        elif "vtk" in mesh_file or "VTK" in mesh_file:
            self._load_vtk(mesh_file)
            self._vertex_mapping_needed = False

        # Unrecognized type
        else:
            raise IOError("{0} is not a supported mesh type for PyPan.".format(mesh_file_type))


    def _load_stl(self, stl_file):
        # Loads mesh from an stl file

        # Load stl file
        raw_mesh = stl.mesh.Mesh.from_file(stl_file)

        # Initialize storage
        N = raw_mesh.v0.shape[0]
        self.N = N
        self.panels = []
        bad_facets = []

        # Loop through panels and initialize objects
        for i in range(N):

            # Check for finite area
            if norm(raw_mesh.normals[i]) == 0.0:
                self.N -= 1
                warnings.warn("Panel {0} has zero area. Skipping...".format(i))
                bad_facets.append(i)
                continue

            # Initialize
            panel = Tri(v0=raw_mesh.v0[i],
                        v1=raw_mesh.v1[i],
                        v2=raw_mesh.v2[i])

            self.panels.append(panel)

        self.panels = np.array(self.panels)

        # Store panel information
        self.cp = np.zeros((self.N, 3))
        self.n = np.zeros((self.N, 3))
        self.dA = np.zeros(self.N)
        for i in range(self.N):
            self.n[i], self.dA[i], self.cp[i] = self.panels[i].get_info()

        # Get vertex list
        good_facets = [i for i in range(N) if i not in bad_facets]
        raw_vertices = np.concatenate((raw_mesh.v0[good_facets], raw_mesh.v1[good_facets], raw_mesh.v2[good_facets]))
        self._vertices, inverse_indices = np.unique(raw_vertices, return_inverse=True, axis=0)
        self._panel_vertex_indices = []
        for i in range(self.N):
            self._panel_vertex_indices.append([3, *inverse_indices[i::self.N]])


    def _load_vtk(self, vtk_file):
        # Loads mesh from vtk file

        # Get data from file
        mesh_data = pv.read(vtk_file)

        # Get vertices
        self._vertices = np.copy(mesh_data.points)

        # Initialize panels
        self.panels = []
        self.N = mesh_data.n_faces
        self._panel_vertex_indices = []
        curr_ind = 0
        cell_info = mesh_data.faces
        self._poly_list_size = len(cell_info)
        self.cp = np.zeros((self.N, 3))
        self.n = np.zeros((self.N, 3))
        self.dA = np.zeros(self.N)
        for i in range(self.N):

            # Determine number of edges and vertex indices
            n = cell_info[curr_ind]
            vertex_ind = cell_info[curr_ind+1:curr_ind+1+n]
            self._panel_vertex_indices.append([n, *list(vertex_ind)])
            vertices = self._vertices[vertex_ind]

            # Initialize panel object
            if n==3:
                panel_obj = Tri(v0=vertices[0],
                                v1=vertices[1],
                                v2=vertices[2])
            elif n==4:
                panel_obj = Quad(v0=vertices[0],
                                 v1=vertices[1],
                                 v2=vertices[2],
                                 v3=vertices[3])

            # Check for zero area
            self.n[i], self.dA[i], self.cp[i] = panel_obj.get_info()
            if abs(self.dA[i])<1e-10:
                raise IOError("Panel {0} in the mesh has zero area.".format(i))
            
            # Store
            self.panels.append(panel_obj)

            # Update index
            curr_ind += n+1


    def _rescale_3D_axes(self, ax):
        # Rescales 3D axes to dt

        # Get current limits
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        z_lims = ax.get_zlim()

        # Determine ranges
        x_diff = x_lims[1]-x_lims[0]
        y_diff = y_lims[1]-y_lims[0]
        z_diff = z_lims[1]-z_lims[0]
        
        # Determine max range
        max_diff = max(max(x_diff, y_diff), z_diff)

        # Determine center of each axis
        x_avg = 0.5*(x_lims[0]+x_lims[1])
        y_avg = 0.5*(y_lims[0]+y_lims[1])
        z_avg = 0.5*(z_lims[0]+z_lims[1])

        # Set new limits
        ax.set_xlim3d(x_avg-max_diff, x_avg+max_diff)
        ax.set_ylim3d(y_avg-max_diff, y_avg+max_diff)
        ax.set_zlim3d(z_avg-max_diff, z_avg+max_diff)


    def _check_for_vertex(self, vertex, v_list):
        # Checks for the vertex in the list; if there, the index is returned

        # Loop through list
        for i, v in enumerate(v_list):
            if np.allclose(v, vertex, atol=1e-8):
                return i
        
        return -1


    def _determine_panel_vertex_mapping(self):
        # Creates a list of all unique vertices and maps each panel to those vertices

        if self._verbose:
            print()
            prog = OneLineProgress(self.N, msg="Determining panel->vertex mapping")

        # Collect vertices and panel vertex indices
        self._vertices = []
        self._panel_vertex_indices = [] # First index is the number of vertices, the rest are the vertex indices
        self._poly_list_size = 0

        # Loop through panels
        i = 0 # Index of last added vertex
        for panel in self.panels:

            # Initialize panel info
            if isinstance(panel, Tri):
                panel_info = [3]
                self._poly_list_size += 4
            elif isinstance(panel, Quad):
                panel_info = [4]
                self._poly_list_size += 5

            # Check if vertices are in the list
            for vertex in panel.vertices:
                ind = self._check_for_vertex(vertex, self._vertices)

                # Not in list
                if ind == -1:
                    self._vertices.append(vertex)
                    panel_info.append(i)
                    i += 1

                # In list
                else:
                    panel_info.append(ind)

            # Store panel info
            self._panel_vertex_indices.append(panel_info)
            if self._verbose:
                prog.display()

        self._vertices = np.array(self._vertices) # Cannot do this for _panel_vertex_indices because the length of each list element is not necessarily the same


    def _determine_panel_adjacency_mapping(self, **kwargs):
        # Stores a list of the indices to each adjacent panel for each panel
        
        not_determined = True

        # Check for adjacency file
        adjacency_file = kwargs.get("adjacency_file", None)
        if adjacency_file is not None:
            
            # Try to find file
            try:
                with open(adjacency_file, 'r') as adj_handle:

                    # Get lines
                    lines = adj_handle.readlines()
                    lines = lines[1:] # Skip header

                    # Check number of panels
                    if len(lines)%2 != 0:
                        raise IOError("Data error in {0}. Should have two lines for each panel!".format(adjacency_file))
                    if len(lines)//2 != self.N:
                        raise IOError("Data error in {0}. Mesh has {0} panels. File describes mapping for {2} panels.".format(adjacency_file, self.N, len(lines)//2))

                    # Loop through lines to store mapping
                    for i, line in enumerate(lines):
                        info = line.split()
                        panel_ind = i//2

                        # Check the panel index is correct
                        if panel_ind != int(info[0]):
                            raise IOError("Input mismatch at line {0} of {1}. Panel index should be {2}; got {3}.".format(i, adjacency_file, panel_ind, int(info[0])))

                        # Store
                        if i%2==0:
                            self.panels[panel_ind].abutting_panels = [int(ind) for ind in info[1:]]
                        else:
                            self.panels[panel_ind].touching_panels = [int(ind) for ind in info[1:]]

                not_determined = False

            except OSError:
                warnings.warn("Adjacency file not found as specified. Reverting to brute force determination.")

        # Brute force approach
        if not_determined:

            if self._verbose:
                print()
                prog = OneLineProgress(self.N, msg="Determining panel adjacency mapping")

            # Loop through possible combinations
            for i, panel_i in enumerate(self.panels):

                for j in range(i+1, self.N):
                    panel_j = self.panels[j]

                    # Determine if we're touching and/or abutting
                    num_shared = 0
                    for i_vert in self._panel_vertex_indices[i][1:]:

                        # Check for shared vertex
                        if i_vert in self._panel_vertex_indices[j][1:]:
                            num_shared += 1
                            if num_shared==2:
                                break # Don't need to keep going
                            
                    # Touching panels (at least one shared vertex)
                    if num_shared>0 and j not in panel_i.touching_panels:
                        panel_i.touching_panels.append(j)
                        panel_j.touching_panels.append(i)

                    # Abutting panels (two shared vertices)
                    if num_shared==2 and j not in panel_i.abutting_panels:
                        panel_i.abutting_panels.append(j)
                        panel_j.abutting_panels.append(i)

                if self._verbose:
                    prog.display()


    def _initialize_kutta_search(self, **kwargs):
        # Sets up the Kutta edge search; does everything not dependent on the freestream vector; relies on an adjacency mapping already being created

        if self._verbose:
            print()
            prog = OneLineProgress(self.N, msg="Locating potential Kutta edges")

        # Get parameters
        theta_K = np.radians(kwargs.get("kutta_angle", 90.0))
        self._check_freestream = kwargs.get("check_freestream", True)

        # Initialize edge storage
        self._kutta_edges = []

        # Look for adjacent panels where the angle between their normals is greater than the Kutta angle
        self._potential_kutta_panels = []

        # Get panel angles
        with np.errstate(invalid='ignore'):
            theta = np.abs(np.arccos(np.einsum('ik,jk->ij', self.n, self.n)))

        # Determine which panels have an angle greater than the Kutta angle
        angle_greater = (theta>theta_K).astype(int)
        i_panels = np.argwhere(np.sum(angle_greater, axis=1).flatten()).flatten()

        # Loop through possible combinations
        for i in i_panels:
            panel_i = self.panels[i]

            # Check abutting panels for Kutta angle
            for j in panel_i.abutting_panels:

                # Don't repeat
                if j <= i:
                    continue

                # Check angle
                if angle_greater[i,j]:
                    self._potential_kutta_panels.append([i,j])

            if self._verbose:
                prog.display()


    def finalize_kutta_edge_search(self, u_inf):
        """Determines where the Kutta condition should exist based on previously located adjacent panels and the freestream velocity.

        Parameters
        ----------
        u_inf : ndarray
            Freestream velocity vector (direction of the oncoming flow).
        """

        if len(self._potential_kutta_panels)>0:

            if self._verbose:
                print()
                prog = OneLineProgress(len(self._potential_kutta_panels), msg="Finalizing Kutta edge locations")

            # Loop through previously determined possibilities
            for i,j in self._potential_kutta_panels:

                # Get panel objects
                panel_i = self.panels[i]
                panel_j = self.panels[j]

                # Check freestream condition
                if not self._check_freestream or inner(self.n[i], u_inf)>0 or inner(self.n[i], u_inf)>0:
                
                    # Get edge vertices
                    v0 = None
                    for ii, vi in enumerate(panel_i.vertices):
                        for vj in panel_j.vertices:

                            # Check distance
                            d = norm(vi-vj)
                            if d<1e-10:

                                # Store first
                                if v0 is None:
                                    v0 = vi
                                    ii0 = ii

                                # Initialize edge object; vertices are stored in the same order as the first panel
                                else:
                                    if ii-ii0 == 1: # Order is important for definition of circulation
                                        self._kutta_edges.append(KuttaEdge(v0, vi, [i, j]))
                                    else:
                                        self._kutta_edges.append(KuttaEdge(vi, v0, [i, j]))
                                    break

                if self._verbose:
                    prog.display()

            # Store number of edges
            self.N_edges = len(self._kutta_edges)

        else:
            self.N_edges = 0

        if self._verbose:
            print("    Found {0} Kutta edges.".format(self.N_edges))

        if self._verbose:
            print()
            prog = OneLineProgress(self.N, msg="Locating panels for gradient calculation")

        # Store touching and abutting panels not across Kutta edge
        for i, panel in enumerate(self.panels):

            # Loop through panels touching this one
            for j in panel.touching_panels:

                # Check for kutta edge
                for kutta_edge in self._kutta_edges:
                    pi = kutta_edge.panel_indices
                    if (pi[0]==i and pi[1]==j) or (pi[0]==j and pi[1]==i):
                        break

                else:
                    panel.touching_panels_not_across_kutta_edge.append(j)

                    # Check if the panel is abutting
                    if j in panel.abutting_panels:
                        panel.abutting_panels_not_across_kutta_edge.append(j)

            if self._verbose:
                prog.display()

        # Store second abutting panels not across Kutta edge
        # Note we're not tracking the progress of this loop. It's super fast.
        for i, panel in enumerate(self.panels):
            for j in panel.abutting_panels_not_across_kutta_edge:

                # This panel obviously counts
                panel.second_abutting_panels_not_across_kutta_edge.append(j)

                # Get second panels
                for k in self.panels[j].abutting_panels_not_across_kutta_edge:
                    if k not in panel.second_abutting_panels_not_across_kutta_edge and k!=i:
                        panel.second_abutting_panels_not_across_kutta_edge.append(k)

        # Set up least-squares matrices
        self._set_up_lst_sq()

        # Initialize wake
        if self.N_edges>0:
            if self._wake_type == "fixed":
                self.wake = StraightFixedWake(kutta_edges=self._kutta_edges, **self._wake_kwargs)
            elif self._wake_type == "full_streamline":
                self.wake = FullStreamlineWake(kutta_edges=self._kutta_edges, **self._wake_kwargs)
            elif self._wake_type == "relaxed":
                self.wake = VelocityRelaxedWake(kutta_edges=self._kutta_edges, **self._wake_kwargs)
            elif self._wake_type == "marching_streamline":
                self.wake = MarchingStreamlineWake(kutta_edges=self._kutta_edges, **self._wake_kwargs)
            else:
                raise IOError("{0} is not a valid wake type.".format(wake_type))


    def _set_up_lst_sq(self):
        # Determines the A matrix to least-squares estimation of the gradient. Must be called after kutta edges are determined.

        if self._verbose:
            print()
            prog = OneLineProgress(self.N, msg="Calculating least-squares matrices")

        # Initialize
        self.A_lsq = []

        # Loop through panels
        for i, panel in enumerate(self.panels):

            # Determine which neighbors to use
            if self._gradient_type=='quad':
                neighbors = panel.second_abutting_panels_not_across_kutta_edge
            else:
                neighbors = panel.touching_panels_not_across_kutta_edge

            # Get centroids of neighboring panels in local panel coordinates
            dp = np.einsum('ij,kj->ki', panel.A_t, self.cp[neighbors]-self.cp[i][np.newaxis,:])

            # Get basis functions
            dx = dp[:,0][:,np.newaxis]
            dy = dp[:,1][:,np.newaxis]
            
            # Assemble A matrix
            if self._gradient_type=='quad':
                A = np.concatenate((dx**2, dy**2, dx*dy, dx, dy), axis=1)
            else:
                A = dp

            # Store
            self.A_lsq.append(A)

            if self._verbose:
                prog.display()


    def set_wake(self, **kwargs):
        """Sets up a wake for this mesh.

        A fixed wake consists of straight, semi-infinite vortex filaments attached to the Kutta edges. The direction is specified by "fixed_direction_type".
        
        An iterative wake consists of segmented semi-infinite vortex filaments. These initially be set in the direction of the local freestream vector resulting from the freestream velocity and rotation.

        Parameters
        ----------
        type : str, optional
            May be "fixed", "full_streamline", "relaxed", or "marching_streamline". Defaults to "fixed".

        kutta_angle : float, optional
            The angle threshold in degrees for determining which edges are Kutta edges. Defaults to 90.0.

        check_freestream : bool, optional
            Whether to include freestream information in the Kutta edge search. If True, a Kutta edge will only be specified between panels which satisfy the Kutta angle and and which have at least one of their normals at less than 90.0 to the freestream velocity (pointing towards the body). Defaults to True.

        fixed_direction_type : str, optional
            May be "custom", "freestream", or "freestream_and_rotation". Defaults to "freestream_and_rotation".

        custom_dir : list or ndarray, optional
            Direction of the vortex filaments. Only used if "fixed_direction_type" is "custom", and then it is required.

        N_segments : int, optional
            Number of segments to use for each filament. Defaults to 20. If type is "marching_streamline", this number determines number of wake iterations specified for the solver. Not required for type "fixed".

        segment_length : float, optional
            Length of each discrete filament segment. Defaults to 1.0. Not required for type "fixed".

        end_segment_infinite : bool, optional
            Whether the final segment of the filament should be treated as infinite. Only used if type is "full_streamline" or "relaxed". Defaults to False. Not required for type "fixed".

        corrector_iterations : int, optional
            How many times to correct the streamline (velocity) prediction for each segment within a streamline wake (not "relaxed" or "fixed"). Defaults to 1.

        K : float
            Time stepping factor for shifting the filament vertices based on the local induced velocity and distance from the trailing edge. Only required for type "relaxed".
        """

        # Find possible Kutta edges
        self._initialize_kutta_search(**kwargs)

        # Get type
        self._wake_type = kwargs.get("type", "fixed")
        if self._wake_type is None:
            raise IOError("Kwarg 'type' is required for set_iterative_wake().")
        self._wake_kwargs = copy.deepcopy(kwargs)

        # A note to the developer: the actual wake object is initialized at the end of finalize_kutta_search(); really all that's done here is storage.


    def plot(self, **kwargs):
        """Plots the mesh in 3D. Python's plotting library is very slow for large datasets and has a poor UI. We recommend using export_vtk() instead and importing into a dedicated renderer, such as Paraview.

        Parameters
        ----------
        panels : bool, optional
            Whether to display panels. Defaults to True.

        centroids : bool, optional
            Whether to display centroids. Defaults to False.

        kutta_edges : bool, optional
            Whether to display the edges at which the Kutta condition will be enforced. Defaults to True.
        """

        # Set up plot
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.gca(projection='3d')
        
        # Plot vertices
        if kwargs.get("panels", True):
            for i, panel in enumerate(self.panels):
                n_vert = panel.vertices.shape[1]
                ind = [x%n_vert for x in range(n_vert+1)]
                ax.plot(panel.vertices[ind,0], panel.vertices[ind,1], panel.vertices[ind,2], 'k-', label='Panel' if i==0 else '')
        
        ## Plot adjacency
        #ind = 0
        #neighbors = self.panels[ind].touching_panels_not_across_kutta_edge
        #ax.plot(self.panels[ind].v_c[0], self.panels[ind].v_c[1], self.panels[ind].v_c[2], 'r.')
        #for i in neighbors:
        #    ax.plot(self.panels[i].v_c[0], self.panels[i].v_c[1], self.panels[i].v_c[2], 'g.')
        
        # Plot centroids
        if kwargs.get("centroids", False):
            for i, panel in enumerate(self.panels):
                ax.plot(self.cp[i][0], self.cp[i][1], self.cp[i][2], 'r.', label='Centroid' if i==0 else '')

        # Plot Kutta edges
        if kwargs.get("kutta_edges", True):
            for i, edge in enumerate(self._kutta_edges):
                ax.plot(edge.vertices[:,0], edge.vertices[:,1], edge.vertices[:,2], 'b', label='Kutta Edge' if i==0 else '')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        self._rescale_3D_axes(ax)
        plt.legend()
        plt.show()


    def export_vtk(self, filename):
        """Exports the mesh to a VTK file. Please note this exports the mesh only. There is an export_vtk() method within the Solver class which will export the mesh along with the flow data.

        Parameters
        ----------
        filename : str
            Name of the file to write the results to. Must have '.vtk' extension.
        """

        # Check extension
        if '.vtk' not in filename:
            raise IOError("Filename for VTK export must contain .vtk extension.")

        # Open file
        with open(filename, 'w') as export_handle:
            
            # Write header
            print("# vtk DataFile Version 3.0", file=export_handle)
            print("PyPan mesh file. Generated by PyPan, USU AeroLab (c) 2020.", file=export_handle)
            print("ASCII", file=export_handle)

            # Write dataset
            print("DATASET POLYDATA", file=export_handle)

            # Write vertices
            vertices, panel_indices = self.get_vtk_data()
            print("POINTS {0} float".format(len(vertices)), file=export_handle)
            for vertex in vertices:
                print("{0:<20.12}{1:<20.12}{2:<20.12}".format(*vertex), file=export_handle)

            # Determine polygon list size
            size = 0
            for pi in panel_indices:
                size += len(pi)

            # Write panel polygons
            print("POLYGONS {0} {1}".format(self.N, size), file=export_handle)
            for panel in panel_indices:
                print(" ".join([str(i) for i in panel]), file=export_handle)

            # Write Kutta edges

            # Write panel data
            print("CELL_DATA {0}".format(self.N), file=export_handle)

            # Area
            print("SCALARS panel_area float 1", file=export_handle)
            print("LOOKUP_TABLE default", file=export_handle)
            for dA in self.dA:
                print("{0:<20.12}".format(dA), file=export_handle)

            # Centroids
            print("VECTORS panel_centroids float", file=export_handle)
            for cp in self.cp:
                print("{0:<20.12} {1:<20.12} {2:<20.12}".format(cp[0], cp[1], cp[2]), file=export_handle)

            # Normals
            print("NORMALS panel_normals float", file=export_handle)
            for n in self.n:
                print("{0:<20.12} {1:<20.12} {2:<20.12}".format(n[0], n[1], n[2]), file=export_handle)

        if self._verbose:
            print()
            print("Mesh successfully written to '{0}'.".format(filename))


    def get_vtk_data(self):
        """Returns a list of vertices and a list of indices referencing each panel to its vertices in the first list.
        """
        return self._vertices, self._panel_vertex_indices


    def get_gradient(self, phi):
        """Returns a least-squares estimate of the gradient of phi **in the plane of the panel** at each panel centroid assuming a quadratic model for phi in the plane of the panel. Phi should be given as the value of a scalar function at each panel centroid, in the correct order.

        Parameters
        ----------
        phi : ndarray
            Value of the scalar field at each panel centroid.

        Returns
        -------
        grad_phi : ndarray
            The gradient of phi at each panel centroid wrt the Cartesian axes.
        """

        # Initialize
        grad_phi = np.zeros((self.N, 3))

        # Loop through panels
        for i, panel in enumerate(self.panels):

            # Determine which neighbors to use
            if self._gradient_type=='quad':
                neighbors = panel.second_abutting_panels_not_across_kutta_edge
            else:
                neighbors = panel.touching_panels_not_across_kutta_edge

            # Get delta phi
            b = phi[neighbors]-phi[i]

            # Solve
            A = self.A_lsq[i]
            c = np.linalg.solve(np.einsum('ij,ik', A, A), np.einsum('ij,i', A, b))

            # Transform back to global coords
            if self._gradient_type=='quad':
                grad_phi[i] = np.einsum('ij,i', panel.A_t, np.array([c[3], c[4], 0.0]))
            else:
                grad_phi[i] = np.einsum('ij,i', panel.A_t, np.array([c[0], c[1], 0.0]))

        return grad_phi


    def export_panel_adjacency_mapping(self, filename):
        """Writes the panel adjacency mapping to the specified file. This mapping can then be read into PyPan when initializing the mesh on subsequent runs, speeding up initialization times.

        Parameters
        ----------
        filename : str
            Name of the file to write the panel adjacency mapping to. Should be type ".pam".

        """

        # Check file extension
        if ".pam" not in filename:
            raise IOError("Filename for writing a panel adjacency mapping must be of type '.pam'.")

        # Open file
        with open(filename, 'w') as file_handle:

            # Write header
            print("### Panel adjacency mapping for {0}".format(self.name), file=file_handle)

            # Loop through panels to write to file
            for i, panel in enumerate(self.panels):

                # Write abutting panels
                print(str(i)+" "+(" ".join(["{}"]*len(panel.abutting_panels))).format(*panel.abutting_panels), file=file_handle)

                # Write touching panels
                print(str(i)+" "+(" ".join(["{}"]*len(panel.touching_panels))).format(*panel.touching_panels), file=file_handle)