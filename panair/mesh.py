import copy

import matplotlib.pyplot as plt
import multiprocessing as mp

from mpl_toolkits.mplot3d import Axes3D
from panair.network import Network


class Mesh:
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

                    # Determine number of rows and columns of panels in this network
                    info = lines[i+1].split()
                    n_rows = int(float(info[0]))
                    n_cols = int(float(info[1]))

                    # Initialize network object
                    if n_rows%2 != 0:
                        self._networks.append(Network(name=line.split()[-1], lines=lines[i:i+int((n_rows//2+1)*n_cols)+2], kn=kn, kt=kt))
                    else:
                        self._networks.append(Network(name=line.split()[-1], lines=lines[i:i+int(n_rows//2*n_cols)+2], kn=kn, kt=kt))

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
        fig = plt.figure(figsize=plt.figaspect(1.0)*2.0)
        ax = fig.gca(projection='3d')
        
        # Plot vertices
        for network in self._networks:
            for i in range(network.n_rows):
                for j in range(network.n_cols):
                    panel = network.panels[i,j].projected_panel
                    ax.plot(panel.vertices[:,0], panel.vertices[:,1], panel.vertices[:,2], '-', color=colors[network.kt], linewidth=0.2)

                    #for midpoint in panel.midpoints:
                    #    ax.plot(midpoint[0], midpoint[1], midpoint[2], '.', linewidth=0.4)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        self._rescale_3D_axes(ax)
        plt.show()


    def _rescale_3D_axes(self, ax):
        # Rescales 3D plot axes to have same scale

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


    def calc_local_coords(self, **kwargs):
        """Sets up local coordinate systems for each panel."""

        for network in self._networks:
            network.calc_local_coords(**kwargs)