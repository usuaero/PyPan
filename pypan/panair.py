"""A set of classes which can mimic PAN AIR for a limited set of cases."""

import time

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


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

    v3 : list
        Fourth corner vertex.
    """

    def __init__(self, **kwargs):

        # Store vertices
        self.vertices = np.zeros((4,3))
        self.vertices[0] = kwargs.get("v0")
        self.vertices[1] = kwargs.get("v1")
        self.vertices[2] = kwargs.get("v2")
        self.vertices[3] = kwargs.get("v3")


    def mirror(self):
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
        pass


class PANAIRNetwork:
    """A class for defining a PAN AIR network.

    Parameters
    ----------
    lines : list
        Lines from the input file defining this network.

    xy_sym : bool, optional
        Whether this network is to be mirrored across the xy plane.

    xz_sym : bool, optional
        Whether this network is to be mirrored across the xz plane.

    kn : float

    kt : float
    """

    def __init__(self, lines, **kwargs):

        # Get kwargs
        self._xy_sym = kwargs.get("xy_sym", False)
        self._xz_sym = kwargs.get("xz_sym", False)
        self._kn = kwargs.get("kn")
        self._kt = kwargs.get("kt")

        # Parse input
        self._parse(lines)


    def _parse(self, lines):
        # Parses the lines given to create the network

        # Get shape
        shape = lines[1].split()
        self._n_rows = int(float(shape[0]))
        self._n_cols = int(float(shape[1]))

        # Determine number of panels and vertices
        self.N = int(self._n_rows*self._n_cols)
        self.N_vert = int((self._n_rows+1)*(self._n_cols+1))

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


    def _mirror(self, plane):
        # Creates a mirrored copy of this network about the given plane
        pass


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

                    self._networks.append(PANAIRNetwork(lines[i:i+int(n_rows//2*n_cols)+2]))

                # End mesh parsing
                elif "$FLOW-FIELD" in line:
                    break

        # Determine total number of panels
        self.N = (n_rows-1)*(n_cols-1)
        self.N *= xy_sym*2
        self.N *= int(xz_sym*2)

        ## Apply xz symmetry
        #if xz_sym:
        #    
        #    # Run through vertices already there
        #    N_vert_orig = len(vertices)
        #    for i in range(N_vert_orig):
        #        vertex = vertices[i]
        #        
        #        # Check we won't just be duplicating a point
        #        if abs(vertex[1])>1e-10:
        #            vertices.append([vertex[0], -vertex[1], vertex[2]])

        ## Apply xy symmetry (will often be skipped)
        #if xy_sym:
        #    
        #    # Run through vertices already there
        #    N_vert_orig = len(vertices)
        #    for i in range(N_vert_orig):
        #        vertex = vertices[i]
        #        
        #        # Check we won't just be duplicating a point
        #        if abs(vertex[2])>1e-10:
        #            vertices.append([vertex[0], vertex[1], -vertex[2]])


    def plot(self):
        """Plots the PAN AIR mesh in 3D.

        Parameters
        ----------
        """

        # Set up plot
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.gca(projection='3d')
        
        # Plot vertices
        for network in self._networks:
            for vertex in network.vertices:
                ax.plot(vertex[0], vertex[1], vertex[2], 'k.')

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


    def execute_case(self):
        pass