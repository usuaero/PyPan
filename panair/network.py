import numpy as np

from panair.panel import Panel, MachInclinedError


class Network:
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

    type_code : float
        Number code for the type of network this is.
    """

    def __init__(self, **kwargs):

        # Get kwargs
        self.type_code = kwargs.get("type_code")
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
        self.panels = np.empty((self.n_rows, self.n_cols), dtype=Panel)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                
                # Vertices are stored going down the columns first (huh, who would've thought with FORTRAN)
                # Order of the panel vertices determines panel orientation
                self.panels[i,j] = Panel(v0=self.vertices[j*(self.n_rows+1)+i],
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
        panels = np.empty((self.n_rows, self.n_cols), dtype=Panel)
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
        return Network(name=self.name+"_{0}_mirror".format(plane), panels=panels, vertices=vertices, type_code=self.type_code)


    def calc_local_coords(self, **kwargs):
        """Sets up the local coordinate system transform for the panels in this network."""

        # Loop through panels
        try:
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    self.panels[i,j].calc_local_coords(**kwargs)

        # Handle Mach inclined error
        except MachInclinedError:
            raise RuntimeError("Panel ({0},{1}) (or a subpanel or half panel thereof) in network {2} is Mach inclined.".format(i, j, self.name))