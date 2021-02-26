import numpy as np

from abc import abstractmethod
from pypan.pp_math import vec_cross, vec_inner, vec_norm

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


    def get_influence_matrix(self, **kwargs):
        """Create wake influence matrix; first index is the influenced panels
        (bordering the horseshoe vortex), second is the influencing panel, third is the 
        velocity component.

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

        points = kwargs.get("points")
        N = len(points)
        return np.zeros((N, N, 3))


class FixedWake(Wake):
    """Defines a fixed wake consisting of straight, semi-infinite vortex
    filaments.

    Parameters
    ----------
    kutta_edges : list of KuttaEdge
        List of Kutta edges which define this wake.

    type : str
        May be "fixed", "freestream", "freestream_constrained", "freestream_and_rotation",
        or "freestream_and_rotation_constrained".

    dir : list or ndarray, optional
        Direction of the vortex filaments. Required for type "fixed".

    normal_dir : list or ndarray, optional
        Normal direction of the plane in which the direction of the vortex filaments should
        be constrained. Required for type "freestream_constrainted" or 
        "freestream_and_rotation_constrained".
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store type
        self._type = kwargs.get("type")
        if self._type is None:
            raise IOError("Wake type must be specified.")

        # Get direction for fixed wake
        if self._type=="fixed":
            try:
                self._dir = np.array(kwargs.get("dir"))
            except:
                raise IOError("'dir' is required for wake type 'fixed'.")

        # Get normal direction for constrained wake
        if "constrained" in self._type:
            try:
                self._n = np.array(kwargs.get("normal_dir"))
            except:
                raise IOError("'normal_dir' is required for wake type {0}.".format(self._type))


    def get_influence_matrix(self, **kwargs):
        """Create wake influence matrix; first index is the influenced panels
        (bordering the horseshoe vortex), second is the influencing panel, third is the 
        velocity component.

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

        # Get kwargs
        points = kwargs.get("points")

        # Initialize storage
        N = len(points)

        vortex_influence_matrix = np.zeros((N, N, 3))
        for edge in self._kutta_edges:

            # Get indices of panels defining the edge
            p_ind = edge.panel_indices

            # Get infulence
            V = edge.get_vortex_influence(points, kwargs.get("u_inf")[np.newaxis,:])

            # Store
            vortex_influence_matrix[:,p_ind[0]] = -V
            vortex_influence_matrix[:,p_ind[1]] = V
        
        return vortex_influence_matrix


class FixedFilament:
    """Defines a straight, semi-infinite vortex filament.

    Parameters
    ----------
    origin : ndarray
        Origin point of the filament.
    """

    def __init__(self, origin):

        # Store origin point
        self._p0 = origin


    def set_dir(self, dir):
        """Sets the direction of the filament.

        Parameters
        ----------
        dir : ndarray
            Direction of the vortex filament.
        """

        # Store direction
        self._dir = dir


    def get_influence(self, points):
        """Determines the influence of this vortex filament on the specified points.

        Parameters
        ----------
        points : ndarray
            Points at which to calculate the influence.
        """
        pass


class KuttaEdge:
    """A class for defining an edge segment at which the Kutta condition is applied. These are used
    for the horseshoe vortex type wake.

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
        r0_mag = vec_norm(r0)[:,np.newaxis]
        r1_mag = vec_norm(r1)[:,np.newaxis]

        # Calculate influence of bound segment
        v_01 = ((r0_mag+r1_mag)*vec_cross(r0, r1))/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1)[:,np.newaxis]))

        # Calculate influence of trailing segments
        v_0_inf = vec_cross(u_inf, r0)/(r0_mag*(r0_mag-vec_inner(u_inf, r0)[:,np.newaxis]))
        v_1_inf = vec_cross(u_inf, r1)/(r1_mag*(r1_mag-vec_inner(u_inf, r1)[:,np.newaxis]))

        return 0.25/np.pi*(-v_0_inf+v_01+v_1_inf)