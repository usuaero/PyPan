import copy

import numpy as np

from abc import abstractmethod
from pypan.pp_math import vec_cross, vec_inner, vec_norm, norm, cross

class Wake:
    """A base class for wake models in PyPan. This class can be used as a dummy class for there being no wake.

    Parameters
    ----------
    kutta_edges : list of KuttaEdge
        List of Kutta edges which define this wake.
    """

    def __init__(self, **kwargs):

        # Store Kutta edges
        self._kutta_edges = kwargs["kutta_edges"]
        self._N_edges = len(self._kutta_edges)

        # Create unique list of vertices
        self._arrange_kutta_vertices()


    def _arrange_kutta_vertices(self):
        # Determines a unique list of the vertices defining all Kutta edges for the wake and the panels associated with each vertex

        if self._N_edges>0:

            # Get array of all vertices
            vertices = np.zeros((2*self._N_edges,3))
            for i, edge in enumerate(self._kutta_edges):

                # Store vertices
                vertices[2*i] = edge.vertices[0]
                vertices[2*i+1] = edge.vertices[1]

            # Determine unique vertices
            unique_vertices, inverse_indices = np.unique(vertices, return_inverse=True, axis=0)

            # Initialize filaments
            inbound_panels = []
            outbound_panels = []
            for i, vertex in enumerate(unique_vertices):

                # Determine associated panels
                ip = []
                op = []
                for j, ind in enumerate(inverse_indices):
                    if ind==i:
                        if j%2==0: # Inbound node for these panels
                            ip = copy.copy(self._kutta_edges[j//2].panel_indices)
                        else: # Outbound node
                            op = copy.copy(self._kutta_edges[j//2].panel_indices)

                # Store panels
                inbound_panels.append(ip)
                outbound_panels.append(op)

        return unique_vertices, inbound_panels, outbound_panels


    def get_influence_matrix(self, **kwargs):
        """Create wake influence matrix; first index is the influenced panels (bordering the horseshoe vortex), second is the influencing panel, third is the velocity component.

        Parameters
        ----------
        points : ndarray
            Array of points at which to calculate the influence.

        u_inf : ndarray
            Freestream direction vector.
        
        omega : ndarray
            Body-fixed rotation rates.

        Returns
        -------
        ndarray
            Trailing vortex influences.
        """

        return 0.0


    def get_vtk_data(self, **kwargs):
        """Returns a list of vertices and line indices describing this wake.
        
        Parameters
        ----------
        length : float, optional
            Length each fixed vortex filament should be. Defaults to 5.0.
        """

        return [], []

    
    def set_filament_direction(self, v_inf, omega):
        """Sets the initial direction of the vortex filaments."""
        pass


class NonIterativeWake(Wake):
    """Defines a non-iterative wake consisting of straight, semi-infinite vortex filaments.

    Parameters
    ----------
    kutta_edges : list of KuttaEdge
        List of Kutta edges which define this wake.

    type : str
        May be "custom", "freestream", "freestream_constrained", "freestream_and_rotation", or "freestream_and_rotation_constrained". Defaults to "freestream".

    dir : list or ndarray, optional
        Direction of the vortex filaments. Required for type "custom".

    normal_dir : list or ndarray, optional
        Normal direction of the plane in which the direction of the vortex filaments should be constrained. Required for type "freestream_constrainted" or "freestream_and_rotation_constrained".
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store type
        self.is_iterative = False
        self._type = kwargs.get("type", "freestream")

        # Initialize filaments
        self.filaments = []
        vertices, inbound_panels, outbound_panels = self._arrange_kutta_vertices()
        for vertex, ip, op in zip(vertices, inbound_panels, outbound_panels):
            self.filaments.append(FixedVortexFilament(vertex, ip, op))

        # Store number of filaments
        self.N = len(self.filaments)

        # Get direction for custom wake
        if self._type=="custom":
            try:
                self.dir = np.array(kwargs.get("dir"))
                self.dir /= norm(self.dir)
                for filament in self.filaments:
                    filament.set_dir(self.dir)
            except:
                raise IOError("'dir' is required for wake type 'fixed'.")

        # Get normal direction for constrained wake
        if "constrained" in self._type:
            try:
                self._n = np.array(kwargs.get("normal_dir"))
                self._n /= norm(self._n)
            except:
                raise IOError("'normal_dir' is required for wake type {0}.".format(self._type))
            
            # Create projection matrix
            self._P = np.eye(3)-np.matmul(self._n[:,np.newaxis], self._n[np.newaxis,:])


    def set_filament_direction(self, v_inf, omega):
        """Updates the direction of the vortex filaments based on the velocity params.

        Parameters
        ----------
        v_inf : ndarray
            Freestream velocity vector.

        omega : ndarray
            Angular rate vector.
        """

        # Freestream direction
        if self._type=="freestream":
            u = v_inf/norm(v_inf)
            for filament in self.filaments:
                filament.set_dir(u)

        # Freestream constrained
        elif self._type=="freestream_constrained":
            u = np.einsum('ij,j', self._P, v_inf)
            u /= norm(u)
            for filament in self.filaments:
                filament.set_dir(u)

        # Freestream with rotation
        elif self._type=="freestream_and_rotation":
            for filament in self.filaments:
                u = v_inf-cross(omega, filament.p0)
                u /= norm(u)
                filament.set_dir(u)

        # Freestream with rotation constrained
        elif self._type=="freestream_and_rotation_constrained":
            for filament in self.filaments:
                u = v_inf-cross(omega, filament.p0)
                u = np.einsum('ij,j', self._P, u)
                u /= norm(u)
                filament.set_dir(u)


    def get_influence_matrix(self, **kwargs):
        """Create wake influence matrix; first index is the influenced panels
        (bordering the horseshoe vortex), second is the influencing panel, third is the 
        velocity component.

        Parameters
        ----------
        points : ndarray
            Array of points at which to calculate the influence.

        N_panels : int
            Number of panels in the mesh to which this wake belongs.

        u_inf : ndarray
            Freestream direction vector.
        
        omega : ndarray
            Body-fixed rotation rates.

        Returns
        -------
        ndarray
            Trailing vortex influences.
        """

        # Get kwargs
        points = kwargs.get("points")

        # Initialize storage
        N = len(points)
        vortex_influence_matrix = np.zeros((N, kwargs["N_panels"], 3))

        # Get influence of edges
        for edge in self._kutta_edges:

            # Get indices of panels defining the edge
            p_ind = edge.panel_indices

            # Get infulence
            V = edge.get_vortex_influence(points, kwargs.get("u_inf")[np.newaxis,:])

            # Store
            vortex_influence_matrix[:,p_ind[0]] = -V
            vortex_influence_matrix[:,p_ind[1]] = V

        # Get influence of filaments
        for filament in self.filaments:

            # Get influence
            V = filament.get_influence(points)

            # Add for outbound panels
            outbound_panels = filament.outbound_panels
            if len(outbound_panels)>0:
                vortex_influence_matrix[:,filament.outbound_panels[0]] -= V
                vortex_influence_matrix[:,filament.outbound_panels[1]] += V

            # Add for inbound panels
            inbound_panels = filament.inbound_panels
            if len(inbound_panels)>0:
                vortex_influence_matrix[:,filament.inbound_panels[0]] += V
                vortex_influence_matrix[:,filament.inbound_panels[1]] -= V
        
        return vortex_influence_matrix


    def get_vtk_data(self, **kwargs):
        """Returns a list of vertices and line indices describing this wake.
        
        Parameters
        ----------
        length : float, optional
            Length each vortex filament should be. Defaults to 5.0.
        """

        # Get kwargs
        l = kwargs.get("length", 5.0)

        # Initialize storage
        vertices = []
        line_vertex_indices = []

        # Loop through filaments
        i = 0
        for filament in self.filaments:

            # Add vertices
            vertices.append(filament.p0)
            vertices.append(filament.p0+l*filament.dir)

            # Add indices
            line_vertex_indices.append([2, i, i+1])

            # Increment index
            i += 2

        return vertices, line_vertex_indices, self.N


class FixedVortexFilament:
    """Defines a straight, semi-infinite vortex filament.

    Parameters
    ----------
    origin : ndarray
        Origin point of the filament.

    inbound_panels : list
        Indices of the two panels for which this is an inbound filament.

    outbound_panels : list
        Indices of the two panels for which this is an outbound filament.
    """

    def __init__(self, origin, inbound_panels, outbound_panels):

        # Store info
        self.p0 = origin
        self.inbound_panels = inbound_panels
        self.outbound_panels = outbound_panels
        self.N = 1


    def set_dir(self, direction):
        """Sets the direction of the filament.

        Parameters
        ----------
        direction : ndarray
            Direction of the vortex filament.
        """

        # Store direction
        self.dir = direction


    def get_influence(self, points):
        """Determines the influence of this vortex filament on the specified points.

        Parameters
        ----------
        points : ndarray
            Points at which to calculate the influence.
        """

        # Determine displacement vector magnitudes
        r = points-self.p0[np.newaxis,:]
        r_mag = vec_norm(r)

        # Calculate influence
        return 0.25/np.pi*vec_cross(self.dir, r)/(r_mag*(r_mag-vec_inner(self.dir, r)))[:,np.newaxis]


class IterativeWake(Wake):
    """Defines an iterative wake consisting of segmented semi-infinite vortex filaments. Will initially be set in the direction of the local freestream vector resulting from the freestream velocity and rotation.

    Parameters
    ----------
    kutta_edges : list of KuttaEdge
        List of Kutta edges which define this wake.

    N_segments : int, optional
        Number of segments to use for each filament. Defaults to 20.

    segment_length : float, optional
        Length of each discrete filament segment. Defaults to 1.0.

    end_segment_infinite : bool, optional
        Whether the final segment of the filament should be treated as infinite. Defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store type
        self.is_iterative = True

        # Initialize filaments
        self.filaments = []
        vertices, inbound_panels, outbound_panels = self._arrange_kutta_vertices()
        for vertex, ip, op in zip(vertices, inbound_panels, outbound_panels):
            self.filaments.append(StreamlineVortexFilament(vertex, ip, op, **kwargs))

        # Store number of filaments
        self.N = len(self.filaments)


    def set_filament_direction(self, v_inf, omega):
        """Updates the initial direction of the vortex filaments based on the velocity params.

        Parameters
        ----------
        v_inf : ndarray
            Freestream velocity vector.

        omega : ndarray
            Angular rate vector.
        """

        # Freestream with rotation
        for filament in self.filaments:
            u = v_inf-cross(omega, filament.p0)
            u /= norm(u)
            filament.set_dir(u)


    def get_vtk_data(self, **kwargs):
        """Returns a list of vertices and line indices describing this wake.
        
        Parameters
        ----------
        length : float, optional
            Length of the final filament segment, if set as infinite.
        """

        # Get kwargs
        l = kwargs.get("length", 5.0)

        # Initialize storage
        vertices = []
        line_vertex_indices = []

        # Loop through filaments
        i = 0
        for filament in self.filaments:

            # Add vertices
            for j, vertex in enumerate(filament.points):
                vertices.append(vertex)

                # Add indices
                if j!=len(filament.points)-1:
                    line_vertex_indices.append([2, i+j, i+j+1])

            # Treat infinite end segment
            if filament.end_inf:
                u = vertices[-1]-vertices[-2]
                u /= norm(u)
                vertices[-1] = vertices[-2]+u*l

            # Increment index
            i += filament.points.shape[0]

        return vertices, line_vertex_indices, self.N*self.filaments[0].N


class StreamlineVortexFilament:
    """Defines a semi-infinite vortex filament which follows a streamline.

    Parameters
    ----------
    origin : ndarray
        Origin point of the filament.

    inbound_panels : list
        Indices of the two panels for which this is an inbound filament.

    outbound_panels : list
        Indices of the two panels for which this is an outbound filament.

    N_segments : int, optional
        Number of segments to use for the filament. Defaults to 20.

    segment_length : float, optional
        Length of each discrete filament segment. Defaults to 1.0.

    end_segment_infinite : bool, optional
        Whether the final segment of the filament should be treated as infinite. Defaults to False.
    """

    def __init__(self, origin, inbound_panels, outbound_panels, **kwargs):

        # Store info
        self.p0 = origin
        self.inbound_panels = inbound_panels
        self.outbound_panels = outbound_panels
        self.N = kwargs.get("N_segments", 20)
        self.l = kwargs.get("segment_length", 1.0)
        self.end_inf = kwargs.get("end_segment_infinite", False)


    def set_dir(self, direction):
        """Sets the initial direction of the filament.

        Parameters
        ----------
        direction : ndarray
            Direction of the vortex filament.
        """

        # Store direction
        self.dir = direction

        # Initialize points
        self.points = self.p0[np.newaxis,:]+self.l*np.linspace(0.0, self.N*self.l, self.N+1)[:,np.newaxis]*self.dir[np.newaxis,:]


    def get_influence(self, points):
        """Determines the influence of this vortex filament on the specified points.

        Parameters
        ----------
        points : ndarray
            Points at which to calculate the influence.
        """

        ## Determine displacement vectors
        #r0 = points-self.vertices[0,:]
        #r1 = points-self.vertices[1,:]

        ## Determine displacement vector magnitudes
        #r0_mag = vec_norm(r0)
        #r1_mag = vec_norm(r1)

        ## Calculate influence of bound segment
        #return 0.25*((r0_mag+r1_mag)/(np.pi*r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1))))[:,np.newaxis]*vec_cross(r0, r1)

        # Determine displacement vector magnitudes
        r = points-self.p0[np.newaxis,:]
        r_mag = vec_norm(r)

        # Calculate influence
        return 0.25*vec_cross(self.dir, r)/(np.pi*r_mag*(r_mag-vec_inner(self.dir, r)))[:,np.newaxis]


class KuttaEdge:
    """A class for defining an edge segment at which the Kutta condition is applied.

    Parameters
    ----------
    v0 : ndarray
        Start vertex.

    v1 : ndarray
        End vertex.

    panel_indices : list
        Indices (within the mesh) of the panels neighboring this edge.
    """

    def __init__(self, v0, v1, panel_indices):

        # Store
        self.vertices = np.zeros((2, 3))
        self.vertices[0] = v0
        self.vertices[1] = v1
        self.panel_indices = panel_indices


    def __str__(self):
        s = "E "+" ".join(["{:<20}"]*6)+" {} {}"
        s = s.format(self.vertices[0,0],
                     self.vertices[0,1],
                     self.vertices[0,2],
                     self.vertices[1,0],
                     self.vertices[1,1],
                     self.vertices[1,2],
                     self.panel_indices[0],
                     self.panel_indices[1])
        return s


    def get_vortex_influence(self, points, u_inf):
        """Determines the velocity vector induced by this edge at arbitrary
        points, assuming a horseshoe vortex is shed from this edge.

        Parameters
        ----------
        points : ndarray
            An array of points where the first index is the point index and 
            the second index is the coordinate.

        u_inf : ndarray
            Freestream direction vectors. Same shape as points.

        Returns
        -------
        ndarray
            The velocity vector induced at each point.
        """

        # Determine displacement vectors
        r0 = points-self.vertices[0,:]
        r1 = points-self.vertices[1,:]

        # Determine displacement vector magnitudes
        r0_mag = vec_norm(r0)
        r1_mag = vec_norm(r1)

        # Calculate influence of bound segment
        return 0.25*((r0_mag+r1_mag)/(np.pi*r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1))))[:,np.newaxis]*vec_cross(r0, r1)