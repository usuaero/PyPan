"""Defines classes for handling basic geometric elements."""

import stl
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod

from .pp_math import vec_norm, norm, vec_inner, inner, vec_cross, cross
from .helpers import OneLineProgress


class Panel:
    """A base class defining a panel for potential flow simulation."""

    def __init__(self, **kwargs):

        # Get normal vector
        self.n = kwargs.get("n")
        if self.n is None:
            self._calc_normal()
        else:
            try:
                self.n = self.n/norm(self.n)
            except RuntimeWarning:
                self._calc_normal()

        # Determine area
        self.A = kwargs.get("A", None)
        if self.A is None:
            self._calc_area()

        # Determine centroid
        self.v_c = kwargs.get("v_c", None)
        if self.v_c is None:
            self._calc_centroid()

        # Determine max side length
        self.d_max = kwargs.get("d_max", None)
        if self.d_max is None:
            self.d_max = np.max(vec_norm(self.vertices-np.roll(self.vertices, 1, axis=0)))

        self.adjacent_panels = []


    def _calc_normal(self):
        # Calculates the panel unit normal vector
        # Assumes the panel is planar
        d1 = self.vertices[1]-self.vertices[0]
        d2 = self.vertices[2]-self.vertices[1]
        N = cross(d1, d2)
        self.n = N/norm(N)


    @abstractmethod
    def _calc_centroid(self):
        pass


    @abstractmethod
    def _calc_area(self):
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

        super().__init__(**kwargs)


class Tri(Panel):
    """A triangular panel."""

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((3,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")

        super().__init__(**kwargs)


    def _calc_area(self):
        # Calculates the panel area

        # Get vector components
        nx, ny, nz = self.n
        x0, y0, z0 = self.vertices[0]
        x1, y1, z1 = self.vertices[1]
        x2, y2, z2 = self.vertices[2]

        # Get the area using Stoke's theorem
        dA0 = 0.5*ny*(z1+z0)*(x1-x0)\
             +0.5*nz*(x1+x0)*(y1-y0)\
             +0.5*nx*(y1+y0)*(z1-z0)
        dA1 = 0.5*ny*(z2+z1)*(x2-x1)\
             +0.5*nz*(x2+x1)*(y2-y1)\
             +0.5*nx*(y2+y1)*(z2-z1)
        dA2 = 0.5*ny*(z0+z2)*(x0-x2)\
             +0.5*nz*(x0+x2)*(y0-y2)\
             +0.5*nx*(y0+y2)*(z0-z2)

        self.A = dA0+dA1+dA2

    
    def _calc_centroid(self):
        # Calculates the location of the panel centroid

        # Construct transformation matrix
        T = np.zeros((3,3))
        T[0] = self.vertices[1]-self.vertices[0]
        T[0] /= norm(T[0])
        T[1] = cross(self.n, T[0])
        T[2] = np.copy(self.n)
        
        # Transform vertices
        v0_p = np.einsum('ij,j', T, self.vertices[0])
        v1_p = np.einsum('ij,j', T, self.vertices[1])
        v2_p = np.einsum('ij,j', T, self.vertices[2])

        # Get transformed coordinates of centroid
        x_c_p = 1.0/(6.0*self.A)*(
            (v0_p[0]+v1_p[0])*(v0_p[0]*v1_p[1]-v1_p[0]*v0_p[1])+
            (v1_p[0]+v2_p[0])*(v1_p[0]*v2_p[1]-v2_p[0]*v1_p[1])+
            (v2_p[0]+v0_p[0])*(v2_p[0]*v0_p[1]-v0_p[0]*v2_p[1]))
        y_c_p = 1.0/(6.0*self.A)*(
            (v0_p[1]+v1_p[1])*(v0_p[0]*v1_p[1]-v1_p[0]*v0_p[1])+
            (v1_p[1]+v2_p[1])*(v1_p[0]*v2_p[1]-v2_p[0]*v1_p[1])+
            (v2_p[1]+v0_p[1])*(v2_p[0]*v0_p[1]-v0_p[0]*v2_p[1]))

        # Transform back to standard coordinates
        self.v_c = np.einsum('ji,j', T, np.array([x_c_p, y_c_p, v0_p[2]]))


    def __str__(self):
        s = "P "+" ".join(["{:<20}"]*17)
        s = s.format(self.vertices[0,0],
                     self.vertices[0,1],
                     self.vertices[0,2],
                     self.vertices[1,0],
                     self.vertices[1,1],
                     self.vertices[1,2],
                     self.vertices[2,0],
                     self.vertices[2,1],
                     self.vertices[2,2],
                     self.n[0],
                     self.n[1],
                     self.n[2],
                     self.v_c[0],
                     self.v_c[1],
                     self.v_c[2],
                     self.A,
                     self.d_max)
        return s


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
        r0 = points-self.vertices[0,:]
        r1 = points-self.vertices[1,:]
        r2 = points-self.vertices[2,:]

        # Determine displacement vector magnitudes
        r0_mag = vec_norm(r0)[:,np.newaxis]
        r1_mag = vec_norm(r1)[:,np.newaxis]
        r2_mag = vec_norm(r2)[:,np.newaxis]

        # Calculate influence
        v_01 = ((r0_mag+r1_mag)*vec_cross(r0, r1))/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1)[:,np.newaxis]))
        v_12 = ((r1_mag+r2_mag)*vec_cross(r1, r2))/(r1_mag*r2_mag*(r1_mag*r2_mag+vec_inner(r1, r2)[:,np.newaxis]))
        v_20 = ((r2_mag+r0_mag)*vec_cross(r2, r0))/(r2_mag*r0_mag*(r2_mag*r0_mag+vec_inner(r2, r0)[:,np.newaxis]))

        return 0.25/np.pi*(v_01+v_12+v_20)


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
        r0_mag = vec_norm(r0)[:,np.newaxis]
        r1_mag = vec_norm(r1)[:,np.newaxis]

        # Calculate influence of bound segment
        v_01 = ((r0_mag+r1_mag)*vec_cross(r0, r1))/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1)[:,np.newaxis]))

        # Calculate influence of trailing segments
        v_0_inf = vec_cross(u_inf, r0)/(r0_mag*(r0_mag-vec_inner(u_inf, r0)[:,np.newaxis]))
        v_1_inf = vec_cross(u_inf, r1)/(r1_mag*(r1_mag-vec_inner(u_inf, r1)[:,np.newaxis]))

        return 0.25/np.pi*(-v_0_inf+v_01+v_1_inf)