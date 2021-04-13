"""Defines classes for handling panels."""

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod

from pypan.pp_math import vec_norm, norm, vec_inner, inner, vec_cross, cross


class Panel:
    """A base class defining a panel for potential flow simulation."""

    def __init__(self, **kwargs):

        # Initialize some storage
        self.touching_panels = [] # Panels which share at least one vertex with this panel
        self.abutting_panels = [] # Panels which share two vertices with this panel
        self.touching_panels_not_across_kutta_edge = [] # Panels which share at least one vertex with this panel where those two vertices do not define a Kutta edge
        self.abutting_panels_not_across_kutta_edge = [] # Panels which share two vertices with this panel where those two vertices do not define a Kutta edge
        self.second_abutting_panels_not_across_kutta_edge = [] # Panels which share two vertices with this panel or its abutting panels where those two vertices do not define a Kutta edge


    def get_info(self):
        """Returns the panel normal, area, and centroid.

        Returns
        -------
        ndarray
            Normal vector.
        
        float
            Panel area.

        ndarray
            Panel centroid.
        """

        return self._calc_normal(), self._calc_area(), self._calc_centroid()


    def _calc_normal(self):
        # Calculates the panel unit normal vector
        # Assumes the panel is planar
        d1 = self.vertices[1]-self.vertices[0]
        d2 = self.vertices[2]-self.vertices[1]
        N = cross(d1, d2)
        return N/norm(N)


    def _calc_centroid(self):
        # Calculates the centroid of the panel
        return 1.0/self.N*np.sum(self.vertices, axis=0).flatten()


    def get_ring_influence(self, points):
        """Determines the velocity vector induced by this panel at arbitrary
        points, assuming a vortex ring (0th order) model and a unit positive
        vortex strength.

        Parameters
        ----------
        points : ndarray
            An array of points where the first index is the point index and 
            the second index is the coordinate.

        Returns
        -------
        ndarray
            The velocity vector induced at each point.
        """

        # Determine displacement vectors
        r = points[np.newaxis,:,:]-self.vertices[:,np.newaxis,:]
        r_mag = vec_norm(r)

        # Calculate influence
        v = np.zeros_like(points)
        with np.errstate(divide='ignore'):
            for i in range(self.N):
                d = (r_mag[i-1]*r_mag[i]*(r_mag[i-1]*r_mag[i]+vec_inner(r[i-1], r[i])))
                n = (r_mag[i-1]+r_mag[i])/d
                n = np.nan_to_num(n, copy=False)
                v += vec_cross(r[i-1], r[i])*n[:,np.newaxis]

        return 0.25/np.pi*v


    @abstractmethod
    def _calc_area(self):
        pass


    @abstractmethod
    def mirror(self):
        pass


class Quad(Panel):
    """A quadrilateral panel."""

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((4,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")
        self.vertices[3] = kwargs.get("v3")
        self.midpoints = 0.5*(self.vertices+np.roll(self.vertices, 1, axis=0))

        self.N = 4

        super().__init__(**kwargs)

        # Set up local coordinate transformation
        n = self._calc_normal()
        self.A_t = np.zeros((3,3))
        self.A_t[0] = self.midpoints[1]-self.midpoints[0]
        self.A_t[0] /= norm(self.A_t[0])
        self.A_t[1] = cross(n, self.A_t[0])
        self.A_t[2] = n


    def _calc_area(self):
        # Calculates the panel area from the two constituent triangles
        A = 0.5*norm(cross(self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0]))
        return A + 0.5*norm(cross(self.vertices[2]-self.vertices[0], self.vertices[3]-self.vertices[0]))


    def _calc_normal(self):
        # Calculates the normal based off of the edge midpoints
        n = cross(self.midpoints[1]-self.midpoints[0], self.midpoints[2]-self.midpoints[1])
        return n/norm(n)


class Tri(Panel):
    """A triangular panel."""

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((3,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")

        self.N = 3

        super().__init__(**kwargs)

        # Set up local coordinate transformation
        n,_,_ = self.get_info()
        self.A_t = np.zeros((3,3))
        self.A_t[0] = self.vertices[1]-self.vertices[0]
        self.A_t[0] /= norm(self.A_t[0])
        self.A_t[1] = cross(n, self.A_t[0])
        self.A_t[2] = n


    def _calc_area(self):
        # Calculates the panel area
        return 0.5*norm(cross(self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0]))