"""A set of classes which can mimic PAN AIR for a limited set of cases."""

import time

import math as m
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pypan.pp_math import dist


class PANAIRPanel:
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
        self._check_collapsed(kwargs.get("tol", 1e-10))

        # Determine edge midpoints
        self.midpoints = 0.5*(self.vertices+np.roll(self.vertices, 1, axis=0))


    def _check_collapsed(self, tol):
        # Determines if any of the vertices in this panel are collapsed (i.e. triangular panel)

        # Loop through vertices
        collapsed = False
        for i in range(4):
            
            # Check distance
            d = dist(self.vertices[i-1], self.vertices[i])
            if d<tol:
                collapsed = i
                break

        # Delete point
        if collapsed:
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
            return PANAIRPanel(v0=refl_vert[3],
                               v1=refl_vert[2],
                               v2=refl_vert[1],
                               v3=refl_vert[0])
        else:
            return PANAIRPanel(v0=refl_vert[2],
                               v1=refl_vert[1],
                               v2=refl_vert[0])


class PANAIRNetwork:
    """A class for defining a PAN AIR network. A network may be defined from input file
    lines or arrays of panel objects and vertices.

    Parameters
    ----------
    name : str
        Name of this network.

    lines : list, optional
        Lines from the input file defining this network.

    panels : ndarray, optional
        Array of PANAIRPanel objects defining this network.

    vertices : ndarray, optional
        Array of vertices defining this network.

    kn : float

    kt : float
    """

    def __init__(self, **kwargs):

        # Get kwargs
        self.kn = kwargs.get("kn")
        self.kt = kwargs.get("kt")
        self.name = kwargs.get("name")

        # Parse input
        lines = kwargs.get("lines", False)
        if not lines:
            self._parse_from_panels(kwargs["panels"], kwargs["vertices"])
        else:
            self._parse_from_input_file(lines)


    def _parse_from_input_file(self, lines):
        # Parses the lines given to create the network

        # Get shape
        shape = lines[1].split()
        self.n_rows = int(float(shape[0]))-1
        self.n_cols = int(float(shape[1]))-1

        # Determine number of panels and vertices
        self.N = int(self.n_rows*self.n_cols)
        self.N_vert = int((self.n_rows+1)*(self.n_cols+1))

        # Get vertices
        self.vertices = []
        for j, line in enumerate(lines[2:]):
            N_coords = len(line)/10
            N_vert = int(N_coords/3)
            for j in range(N_vert):
                vertex = [float(line[int(j*30):int(j*30+10)]),
                          float(line[int(j*30+10):int(j*30+20)]),
                          float(line[int(j*30+20):int(j*30+30)])]
                self.vertices.append(vertex)
        
        # Convert to numpy array
        self.vertices = np.array(self.vertices)

        # Turn grid of vertices into panels
        self.panels = np.empty((self.n_rows, self.n_cols), dtype=PANAIRPanel)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                
                # Vertices are stored going down the columns first (huh, who would've thought with FORTRAN)
                # Order of the panel vertices determines panel orientation
                self.panels[i,j] = PANAIRPanel(v0=self.vertices[j*(self.n_rows+1)+i],
                                               v1=self.vertices[(j+1)*(self.n_rows+1)+i],
                                               v2=self.vertices[(j+1)*(self.n_rows+1)+i+1],
                                               v3=self.vertices[j*(self.n_rows+1)+i+1])


    def _parse_from_panels(self, panels, vertices):
        # Stores the information for the network based on arrays of panels and vertices

        # Determine shape
        self.n_rows, self.n_cols = panels.shape
        self.N = int(self.n_rows*self.n_cols)
        self.N_vert = int((self.n_rows+1)*(self.n_cols+1))

        # Store
        self.panels = panels
        self.vertices = vertices


    def mirror(self, plane):
        """Creates a mirrored copy of this network about the given plane

        Parameters
        ----------
        plane : str
            May be 'xy' or 'xz'.
        """

        # Create new array of mirrored panels
        panels = np.empty((self.n_rows, self.n_cols), dtype=PANAIRPanel)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                panels[i,j] = self.panels[i,j].mirror(plane)

        # Mirror vertices
        vertices = np.copy(self.vertices)
        if plane=='xy':
            vertices[:,2] *= -1.0
        else:
            vertices[:,1] *= -1.0

        # Create new network
        return PANAIRNetwork(name=self.name+"_{0}_mirror".format(plane), panels=panels, vertices=vertices, kn=self.kn, kt=self.kt)


class PANAIRMesh:
    """Class containing all the mesh information for PAN AIR.

    Parameters
    ----------
    input_file : str
        Path to a PAN AIR input file.
    """

    def __init__(self, **kwargs):
        
        # Read in mesh
        self._load_mesh(kwargs.get("input_file"))


    def _load_mesh(self, panair_file):
        # Reads in the structured mesh from a PAN AIR input file

        # Initialize storage
        self._networks = []

        # Open file
        with open(panair_file, 'r') as input_handle:

            # Read in lines
            lines = input_handle.readlines()
            i = -1

            # Loop through lines
            for i, line in enumerate(lines):

                # Get symmetry
                if "$SYMMETRIC" in lines[i]:
                    planes = lines[i+1].split()
                    plane_toggles = lines[i+2].split()

                    # Set symmetry options
                    for plane, plane_toggle in zip(planes, plane_toggles):

                        # XY symmetry
                        if "xy" in plane:
                            xy_sym = bool(float(plane_toggle))

                        # XZ symmetry
                        if "xz" in plane:
                            xz_sym = bool(float(plane_toggle))

                # Get panel parameters
                if "=kn" in line:
                    kn = int(float(lines[i+1].split()[0]))
                elif "=kt" in line:
                    kt = int(float(lines[i+1].split()[0]))

                # Parse network
                elif "=nm" in line and "nn" in line:

#                    # Check for wake
#                    if "wake" in line:
#                        wake = True
#                    else:
#                        wake = False

                    # Determine number of rows and columns of panels in this network
                    info = lines[i+1].split()
                    n_rows = int(float(info[0]))
                    n_cols = int(float(info[1]))

                    # Initialize network object
                    if n_rows%2 != 0:
                        self._networks.append(PANAIRNetwork(name=line.split()[-1], lines=lines[i:i+int((n_rows//2+1)*n_cols)+2], kn=kn, kt=kt))
                    else:
                        self._networks.append(PANAIRNetwork(name=line.split()[-1], lines=lines[i:i+int(n_rows//2*n_cols)+2], kn=kn, kt=kt))

                # End mesh parsing
                elif "$FLOW-FIELD" in line:
                    break

        # Determine total number of panels
        self.N = (n_rows-1)*(n_cols-1)
        self.N *= xy_sym*2
        self.N *= int(xz_sym*2)

        # Apply xz symmetry
        if xz_sym:
            
            # Mirror panels
            for i in range(len(self._networks)):
                self._networks.append(self._networks[i].mirror('xz'))

        # Apply xy symmetry (will often be skipped)
        if xy_sym:
            
            # Mirror panels
            for i in range(len(self._networks)):
                self._networks.append(self._networks[i].mirror('xy'))


    def plot(self):
        """Plots the PAN AIR mesh in 3D.

        Parameters
        ----------
        """

        # Arbitrary colors for each type of panel
        colors = {
            11 : "#0000FF",
            18 : "#FF0000",
            20 : "#00FF00",
            5 : "#AA00AA",
            1 : "#000000"
        }

        # Set up plot
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.gca(projection='3d')
        
        # Plot vertices
        for network in self._networks:
            for i in range(network.n_rows):
                for j in range(network.n_cols):
                    panel = network.panels[i,j]
                    ax.plot(panel.vertices[:,0], panel.vertices[:,1], panel.vertices[:,2], '-', color=colors[network.kt], linewidth=0.2)

                    #for midpoint in panel.midpoints:
                    #    ax.plot(midpoint[0], midpoint[1], midpoint[2], '.', linewidth=0.4)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        self._rescale_3D_axes(ax)
        plt.show()


    def _rescale_3D_axes(self, ax):
        # Rescales 3D axes to have same scale

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


class PANAIRMain:
    """Main class for mimicing PAN AIR functionality.

    Parameters
    ----------
    input_file : str
        Path to a PAN AIR input file.
    """

    def __init__(self, **kwargs):
        
        # Load mesh
        start_time = time.time()
        print("\nReading in mesh...", end='', flush=True)
        self.mesh = PANAIRMesh(**kwargs)
        end_time = time.time()
        print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)

        # Load in case parameters
        start_time = time.time()
        print("\nReading in case parameters...", end='', flush=True)
        self._load_params(**kwargs)
        end_time = time.time()
        print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)


    def _load_params(self, **kwargs):
        # Loads the case parameters from the input file

        # Open file
        input_file = kwargs.get("input_file")
        with open(input_file, 'r') as input_handle:
            lines = input_handle.readlines()

        # Loop through case information
        for i, line in enumerate(lines):

            # Mach number
            if "=amach" in line:
                self.M = float(lines[i+1])

            # Angles of attack
            elif "=alpc" in line:
                self.alpha_c = float(lines[i+1])
            elif "=alpha(0)" in line:
                self.alpha = float(lines[i+1])

            # Sideslip angles
            elif "=betc" in line:
                self.beta_c = float(lines[i+1])
            elif "=beta(0)" in line:
                self.beta = float(lines[i+1])

            # Reference parameters
            elif "=sref" in line:
                ref_info = lines[i+1].split()
                self.S_ref = float(ref_info[0])
                self.b_ref = float(ref_info[1])
                self.c_ref = float(ref_info[2])
                self.d_ref = float(ref_info[3])

            # Exit
            elif "$POINTS" in line:
                break


    def execute_case(self, verbose=False):
        """Executes the case as specified in the input file.

        Parameters
        ----------
        verbose : bool, optional
        """
        
        # Calculate transforms
        self._calc_transforms()


    def _calc_transforms(self):
        # Calculates the various transforms between coordinate systems

        # Determine s and B
        quant = 1.0-self.M**2
        self._s = quant/abs(quant)
        self._B = m.sqrt(self._s*quant)

        # Determine compressibility transformation matrix
        S_a = m.sin(self.alpha_c)
        C_a = m.cos(self.alpha_c)
        S_B = m.sin(self.beta_c)
        C_B = m.cos(self.beta_c)
        self._gamma_c = np.array([[C_a*C_B, -S_B, S_a*C_B],
                                  [C_a*S_B, C_B, S_a*S_B],
                                  [-S_a, 0.0, C_a]])


    def plot_mesh(self):
        """Plots the input mesh."""
        self.mesh.plot()