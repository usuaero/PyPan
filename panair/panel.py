import copy

import numpy as np

from pypan.pp_math import dist, cross, norm, inner
from panair.exceptions import MachInclinedError


class BasePanel:
    """A class containing methods common to both panels and subpanels."""


    def _calc_geom_props(self):
        # Calculates various geometric properties of this panel

        # Store number of vertices
        self.N = self.vertices.shape[0]

        # Determine edge midpoints and center point (these all lie in the average plane of the panel)
        self.center = (1.0/self.N)*np.sum(self.vertices, axis=0).flatten()

        # Get radius and diameter
        self._calc_radius_and_diameter()


    def _calc_radius_and_diameter(self):
        # Calculates the radius and diameter of the panel

        self.radius = 0.0
        self.diameter = 0.0
        for i, vertex0 in enumerate(self.vertices):

            # Check disance from center
            d = dist(self.center, vertex0)
            if d > self.radius:
                self.radius = d

            # Check distance from other vertices
            for vertex1 in self.vertices[i+1:]:
                d = dist(vertex0, vertex1)
                if d > self.diameter:
                    self.diameter = d


    def calc_local_coords(self, **kwargs):
        """Calculates panel local coords (dependent on flow properties).

        Parameters
        ----------
        M : float
            Freestream Mach number.
        """

        # Get kwargs
        M = kwargs["M"]
        c_0 = kwargs["c_0"]
        C_0 = kwargs["C_0"]
        B_0 = kwargs["B_0"]
        s = kwargs["s"]
        B = kwargs["B"]

        # Calculate tangent vector compressible norms (if applicable)
        if hasattr(self, "t"):
            self.t_comp_norm = np.zeros(3)
            for i, t in enumerate(self.t):
                self.t_comp_norm[i] = inner(t, t)-M**2*inner(c_0, t)**2

        # Calculate conormal vector
        self.n_co = self.n-M**2*inner(c_0, self.n)*c_0

        # Check inclination
        self.n_co = np.einsum('ij,j', B_0, self.n)
        self._incl = inner(self.n, self.n_co)
        if abs(self._incl)<1e-10:
            raise MachInclinedError
        self._r = np.sign(self._incl)

        # Get panel coordinate directions
        v_0 = cross(self.n, c_0)
        v_0 /= norm(v_0)
        u_0 = cross(v_0, self.n)
        u_0 /= norm(u_0)

        # Calculate transformation matrix
        # It should be that det(A) = B**2 (see Epton & Magnus pp. E.3-16)
        self._A = np.zeros((3,3))
        denom = abs(self._incl)**-0.5
        self._A[0,:] = denom*np.einsum('ij,j', C_0, u_0)
        self._A[1,:] = self._r*s/B*np.einsum('ij,j', C_0, v_0)
        self._A[2,:] = B*denom*self.n

        # Calculate area Jacobian
        self._J = 1.0/B*denom


class Panel(BasePanel):
    """A class for defining the panels used in PAN AIR.

    Parameters
    ----------
    v0 : list
        First corner vertex.

    v1 : list
        Second corner vertex.

    v2 : list
        Third corner vertex.

    v3 : list, optional
        Fourth corner vertex. May be omitted for triangular panel.

    tol : float, optional
        Tolerance for determining if two points are collapsed onto each other.
        Defaults to 1e-10.

    projected : bool, optional
        Whether this panel has been projected to the average plane. Defaults to False.
    """

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((4,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")
        self.vertices[3] = kwargs.get("v3", self.vertices[2]) # Will get removed by _check_collapsed_vertices()

        # Determine if this is a projected panel
        self._projected = kwargs.get("projected", False)

        # Check for collapsed points
        self._check_collapsed_vertices(kwargs.get("tol", 1e-8))

        # Calculate midpoints
        self.midpoints = 0.5*(self.vertices+np.roll(self.vertices, 1, axis=0))

        # Calculate normal vector; this is simpler than the method used in PAN AIR, which is able to handle
        # the case where the midpoints and center point do not lie in a flat plane [Epton & Magnus section D.2]
        self.n = cross(self.midpoints[1]-self.midpoints[0], self.midpoints[2]-self.midpoints[1])
        self.n /= norm(self.n)

        # Other calculations
        self._calc_geom_props()
        self._calc_skewness()

        # Setup projected panel
        if not self._projected:
            self._initialize_projected_panel()

        # Initialize subpanels
        self.subpanels = []
        for i in range(self.N):

            # Outer subpanel
            self.subpanels.append(Subpanel(v0=self.midpoints[i-1], v1=self.vertices[i], v2=self.midpoints[i], projected=self._projected))

            # Inner subpanel
            self.subpanels.append(Subpanel(v0=self.midpoints[i], v1=self.center, v2=self.midpoints[i-1], projected=self._projected))

        # Initialize half panels (only if the panel is not already triangular)
        if self.N==4:
            self.half_panels = []
            for i in range(self.N):
                self.half_panels.append(Subpanel(v0=self.vertices[i-2], v1=self.vertices[i-1], v2=self.vertices[i]))
        else:
            self.half_panels = False


    def _check_collapsed_vertices(self, tol):
        # Determines if any of the vertices in this panel are collapsed (i.e. triangular panel)

        # Loop through vertices
        collapsed = None
        for i in range(4):
            
            # Check distance
            d = dist(self.vertices[i-1], self.vertices[i])
            if d<tol:
                collapsed = i
                break

        # Delete point
        if collapsed is not None:
            ind = [i for i in range(4) if i != collapsed]
            self.vertices = self.vertices[ind]


    def _initialize_projected_panel(self):
        # Calculates the properties of this panel projected into the average plane
        # The normal vector and conormal vector will be the same

        # For a 3-sided panel, it is already projected
        if self.N == 3:
            self.projected_panel = copy.copy(self)

        else:
            # Calculate projection matrix
            P = np.eye(3)-np.einsum('i,j->ij', self.n, self.n)

            # Project vertices into average plane set at origin
            vertices_p = np.einsum('ij,kj->ki', P, self.vertices)

            # Determine offset from origin
            offset = self.midpoints[0]-np.einsum('ij,j', P, self.midpoints[0])

            # Apply offset to projected points
            vertices_p += offset[np.newaxis,:]

            # Initialize new panel
            self.projected_panel = Panel(v0=vertices_p[0],
                                         v1=vertices_p[1],
                                         v2=vertices_p[2],
                                         v3=vertices_p[3],
                                         projected=True)

    
    def _calc_skewness(self):
        # Calculates the skewness parameters for this panel (if not triangular)

        # Get skewness parameters for 4-sided panel
        if self.N==4:
            self.C_skew = np.zeros((2,4))
            denom = inner(cross(self.midpoints[3]-self.center, self.midpoints[0]-self.center), self.n)
            self.C_skew[0,0] = inner(cross(self.vertices[0]-self.midpoints[3], self.midpoints[0]-self.center), self.n)/denom
            self.C_skew[1,0] = inner(cross(self.midpoints[3]-self.center, self.vertices[0]-self.center), self.n)/denom
            self.C_skew[0,1] = self.C_skew[0,0]
            self.C_skew[1,1] = -self.C_skew[1,0]
            self.C_skew[0,2] = -self.C_skew[0,0]
            self.C_skew[1,2] = -self.C_skew[1,0]
            self.C_skew[0,3] = -self.C_skew[0,0]
            self.C_skew[1,3] = self.C_skew[1,0]


    def mirror(self, plane):
        """Returns a mirror of this panel.

        Parameters
        ----------
        plane : str
            Plane across which to mirror this panel. May be 'xy' or 'xz'.
        
        Returns
        -------
        PANAIRPanel
            A mirrored panel of this panel.
        """
        
        # Copy vertices
        refl_vert = np.copy(self.vertices)

        # xy mirror
        if plane=='xy':
            refl_vert[:,2] *= -1.0

        # xz mirror
        elif plane=='xz':
            refl_vert[:,1] *= -1.0

        # Create new panel with reordered vertices
        if refl_vert.shape[0]==4:
            return Panel(v0=refl_vert[3],
                         v1=refl_vert[2],
                         v2=refl_vert[1],
                         v3=refl_vert[0])
        else:
            return Panel(v0=refl_vert[2],
                         v1=refl_vert[1],
                         v2=refl_vert[0])


    def calc_local_coords(self, **kwargs):
        """Calculates the local coordinate system transform."""

        # Get kwargs
        B_0 = kwargs['B_0']
        C_0 = kwargs['C_0']
        c_0 = kwargs['c_0']
        s = kwargs['s']
        B = kwargs['B']
        M = kwargs['M']

        super().calc_local_coords(**kwargs)

        # Calculate properties for subpanels
        for subpanel in self.subpanels:
            subpanel.calc_local_coords(**kwargs)

        # Calculate properties for half panels
        if self.half_panels:
            for half_panel in self.half_panels:
                half_panel.calc_local_coords(**kwargs)


class Subpanel(BasePanel):
    """Defines a subpanel to a panel.

    Parameters
    ----------
    v0 : list
        First corner vertex.

    v1 : list
        Second corner vertex.

    v2 : list
        Third corner vertex.
        
    projected : bool, optional
        Whether this panel has been projected to the average plane. Defaults to False.
    """

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((3,3))
        self.vertices[0] = kwargs["v0"]
        self.vertices[1] = kwargs["v1"]
        self.vertices[2] = kwargs["v2"]
        self._projected = kwargs.get("projected", False)

        # Calculate area and normal vector
        n = cross(self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[1])
        N = norm(n)
        self.A = 0.5*N
        self.n = n/N

        # Check for zero area in projected panel
        if self.A<1e-10 and self._projected:
            self.null_panel = True
        else:
            self.null_panel = False

            # Calculate edge tangents
            self.t = np.roll(self.vertices, 1, axis=0)-self.vertices
            self.t /= np.linalg.norm(self.t, axis=1, keepdims=True)

            self._calc_geom_props()