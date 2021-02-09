import numpy as np

from pypan.pp_math import dist, cross, norm, inner


class Panel:
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
    """

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((4,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")
        self.vertices[3] = kwargs.get("v3", self.vertices[2]) # Will get removed by _check_collapsed()

        # Check for collapsed points
        self._check_collapsed(kwargs.get("tol", 1e-8))

        # Store number of vertices
        self.N = self.vertices.shape[0]

        # Determine edge midpoints and center point (these all lie in the average plane of the panel)
        self.midpoints = 0.5*(self.vertices+np.roll(self.vertices, 1, axis=0))
        self.center = (1.0/self.N)*np.sum(self.vertices, axis=0).flatten()

        # Calculate normal vector; this is simpler than the method used in PAN AIR, which is able to handle
        # the case where the midpoints and center point do not lie in a flat plane [E.&M. D.2]
        self.n = cross(self.midpoints[1]-self.midpoints[0], self.midpoints[2]-self.midpoints[1])
        self.n /= norm(self.n)

        # Initialize subpanels
        self.subpanels = []
        for i in range(self.N):

            # Outer subpanel
            self.subpanels.append(Subpanel(v0=self.midpoints[i-1], v1=self.vertices[i], v2=self.midpoints[i]))

            # Inner subpanel
            self.subpanels.append(Subpanel(v0=self.midpoints[i], v1=self.center, v2=self.midpoints[i-1]))

        # Calculate projected panel information
        self._calc_projected_panel()

        # Create half panels
        if self.N==4:
            self.half_panels = []
            for i in range(self.N):
                self.half_panels.append(Subpanel(v0=self.vertices[i-2], v1=self.vertices[i-1], v2=self.vertices[i]))
        else:
            self.half_panels = False


    def _check_collapsed(self, tol):
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


    def _calc_projected_panel(self):
        # Calculates the properties of this panel projected into the average plane

        # Calculate projection matrix
        P = np.eye(3)-np.einsum('i,j->ij', self.n, self.n)

        # Project vertices
        self.vertices_p = np.einsum('ij,kj->ki', P, self.vertices)

        # Calculate edge tangents
        self.t_p = np.roll(self.vertices, 1, axis=0)-self.vertices
        self.t_p /= np.linalg.norm(self.t_p, axis=1, keepdims=True)


    def calc_local_coords(self, **kwargs):
        """Calculates the local coordinate system transform."""

        # Get kwargs
        B_0 = kwargs['B_0']
        C_0 = kwargs['C_0']
        c_0 = kwargs['c_0']
        s = kwargs['s']
        B = kwargs['B']
        M = kwargs['M']

        # Calculate panel inclination
        self.n_co = np.einsum('ij,j', B_0, self.n)
        self._incl = inner(self.n, self.n_co)
        if abs(self._incl)<1e-10:
            raise MachInclinedError
        self._r = np.sign(self._incl)

        # Calculate projected tangent vector compressible norms
        self.t_p_comp_norm = np.zeros(self.N)
        for i, t in enumerate(self.t_p):
            self.t_p_comp_norm[i] = inner(t, t)-M**2*inner(c_0, t)**2

        # Get panel coordinate directions
        v_0 = cross(self.n, c_0)
        u_0 = cross(v_0, self.n)

        # Calculate transformation matrix
        self._A = np.zeros((3,3))
        d = abs(self._incl)**-0.5
        self._A[0,:] = d*np.einsum('ij,j', C_0, u_0)
        self._A[1,:] = self._r*s/B*np.einsum('ij,j', C_0, v_0)
        self._A[1,:] = B*d*self.n

        # Calculate properties for subpanels
        for subpanel in self.subpanels:
            subpanel.calc_local_coords(**kwargs)

        # Calculate properties for half panels
        if self.half_panels:
            for half_panel in self.half_panels:
                half_panel.calc_local_coords(**kwargs)


class Subpanel:
    """Defines a subpanel to a panel.

    Parameters
    ----------
    v0 : list
        First corner vertex.

    v1 : list
        Second corner vertex.

    v2 : list
        Third corner vertex.
    """

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((3,3))
        self.vertices[0] = kwargs["v0"]
        self.vertices[1] = kwargs["v1"]
        self.vertices[2] = kwargs["v2"]

        # Calculate area
        n = cross(self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[1])
        self.A = 0.5*norm(n)

        # Calculate panel normal
        self.n = 0.5*n/self.A

        # Calculate edge tangents
        self.t = np.roll(self.vertices, 1, axis=0)-self.vertices
        self.t /= np.linalg.norm(self.t, axis=1, keepdims=True)


    def calc_local_coords(self, **kwargs):
        """Calculates subpanel local coords (dependent on flow properties).

        Parameters
        ----------
        M : float
            Freestream Mach number.
        """

        # Get kwargs
        M = kwargs["M"]
        c_0 = kwargs["c_0"]
        s = kwargs["s"]
        B = kwargs["B"]

        # Calculate tangent vector compressible norms
        self.t_comp_norm = np.zeros(3)
        for i, t in enumerate(self.t):
            self.t_comp_norm[i] = inner(t, t)-M**2*inner(c_0, t)**2

        # Calculate conormal vector
        self.n_co = self.n-M**2*inner(c_0, self.n)*c_0

        # Check inclination
        self._incl = inner(self.n, self.n_co)
        if abs(self._incl)<1e-10:
            raise MachInclinedError


class MachInclinedError(Exception):
    """An exception thrown when a panel is Mach inclined."""

    def __init__(self):
        super().__init__("This panel is Mach inclined.")