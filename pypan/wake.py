import copy

import numpy as np

from abc import abstractmethod
from pypan.pp_math import vec_cross, vec_inner, vec_norm, norm, cross
from pypan.helpers import OneLineProgress

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
        self.filaments = []
        self.N = 0
        self.N_segments = 0


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

        return [], [], 0

    
    def set_filament_direction(self, v_inf, omega):
        """Sets the initial direction of the vortex filaments."""
        pass


class StraightFixedWake(Wake):
    """Defines a non-iterative wake consisting of straight, semi-infinite vortex filaments.

    Parameters
    ----------
    kutta_edges : list of KuttaEdge
        List of Kutta edges which define this wake.

    fixed_direction_type : str, optional
        May be "custom", "freestream", or "freestream_and_rotation". Defaults to "freestream_and_rotation".

    custom_dir : list or ndarray, optional
        Direction of the vortex filaments. Only used if "fixed_direction_type" is "custom", and then it is required.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store type
        self._type = kwargs.get("fixed_direction_type", "freestream_and_rotation")

        # Initialize filaments
        self._vertices, self.inbound_panels, self.outbound_panels = self._arrange_kutta_vertices()

        # Store number of filaments and segments
        self.N = len(self._vertices)
        self.N_segments = 1

        # Initialize filament directions
        self.filament_dirs = np.zeros((self.N, 3))

        # Get direction for custom wake
        if self._type=="custom":
            try:
                u = np.array(kwargs.get("custom_dir"))
                u /= norm(u)
                self.filament_dirs[:] = u
            except:
                raise IOError("'custom_dir' is required for non-iterative wake type 'custom'.")


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
            self.filament_dirs[:] = u

        # Freestream with rotation
        elif self._type=="freestream_and_rotation":
            self.filament_dirs = v_inf[np.newaxis,:]-vec_cross(omega, self._vertices)
            self.filament_dirs /= vec_norm(self.filament_dirs)[:,np.newaxis]


    def get_influence_matrix(self, **kwargs):
        """Create wake influence matrix; first index is the influenced panels, second is the influencing panel, third is the velocity component.

        Parameters
        ----------
        points : ndarray
            Array of points at which to calculate the influence.

        N_panels : int
            Number of panels in the mesh to which this wake belongs.

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
            V = edge.get_vortex_influence(points)

            # Store
            vortex_influence_matrix[:,p_ind[0]] = -V
            vortex_influence_matrix[:,p_ind[1]] = V

        # Determine displacement vector magnitudes
        r = points[:,np.newaxis,:]-self._vertices[np.newaxis,:,:]
        r_mag = vec_norm(r)

        # Calculate influences
        V = 0.25/np.pi*vec_cross(self.filament_dirs[np.newaxis,:,:], r)/(r_mag*(r_mag-vec_inner(self.filament_dirs[np.newaxis,:,:], r)))[:,:,np.newaxis]
        for i in range(self.N):

            # Add for outbound panels
            outbound_panels = self.outbound_panels[i]
            if len(outbound_panels)>0:
                vortex_influence_matrix[:,outbound_panels[0],:] -= V[:,i,:]
                vortex_influence_matrix[:,outbound_panels[1],:] += V[:,i,:]

            # Add for inbound panels
            inbound_panels = self.inbound_panels[i]
            if len(inbound_panels)>0:
                vortex_influence_matrix[:,inbound_panels[0],:] += V[:,i,:]
                vortex_influence_matrix[:,inbound_panels[1],:] -= V[:,i,:]
        
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
        for j in range(self.N):

            # Add vertices
            vertices.append(self._vertices[j])
            vertices.append(self._vertices[j]+l*self.filament_dirs[j])

            # Add indices
            line_vertex_indices.append([2, i, i+1])

            # Increment index
            i += 2

        return vertices, line_vertex_indices, self.N


class SegmentedWake(Wake):
    """Defines a wake consisting of segmented semi-infinite vortex filaments. Vortex filaments will initially be set in the direction of the local freestream vector resulting from the freestream velocity and rotation.

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

        # Get kwargs
        self.l = kwargs.get('segment_length', 1.0)
        self.N_segments = kwargs.get('N_segments', 20)
        self._end_infinite = kwargs.get("end_segment_infinite", False)

        # Initialize filaments
        vertices, self.inbound_panels, self.outbound_panels = self._arrange_kutta_vertices()

        # Initialize filament points
        self.N = vertices.shape[0]
        self._vertices = np.zeros((self.N, self.N_segments+1, 3))
        self._vertices[:,0,:] = vertices


    def set_filament_direction(self, v_inf, omega):
        """Updates the initial direction of the vortex filaments based on the velocity params.

        Parameters
        ----------
        v_inf : ndarray
            Freestream velocity vector.

        omega : ndarray
            Angular rate vector.
        """

        # Determine directions
        origins = self._vertices[:,0,:]
        self._filament_dirs = v_inf[np.newaxis,:]-vec_cross(omega, origins)
        self._filament_dirs /= vec_norm(self._filament_dirs)[:,np.newaxis]

        # Set vertices
        self._vertices = origins[:,np.newaxis,:]+np.linspace(0.0, self.N_segments*self.l, self.N_segments+1)[np.newaxis,:,np.newaxis]*self._filament_dirs[:,np.newaxis,:]


    def get_vtk_data(self, **kwargs):
        """Returns a list of vertices and line indices describing this wake.
        
        Parameters
        ----------
        length : float, optional
            Length of the final filament segment, if set as infinite. Defaults to 20 times the filament segment length.
        """

        # Get kwargs
        l = kwargs.get("length", 20.0*self.l)

        # Initialize storage
        vertices = []
        line_vertex_indices = []

        # Loop through filaments
        i = 0
        for j in range(self.N):

            # Add vertices
            for k in range(self.N_segments+1):
                vertices.append(self._vertices[j,k])

                # Add indices
                if k != self.N_segments:
                    line_vertex_indices.append([2, i+k, i+k+1])

            # Treat infinite end segment
            if self._end_infinite:
                u = vertices[-1]-vertices[-2]
                u /= norm(u)
                vertices[-1] = vertices[-2]+u*l

            # Increment index
            i += self.N_segments+1

        return vertices, line_vertex_indices, self.N*self.N_segments


    def get_influence_matrix(self, **kwargs):
        """Create wake influence matrix; first index is the influenced points, second is the influencing panel, third is the velocity component.

        Parameters
        ----------
        points : ndarray
            Array of points at which to calculate the influence.

        N_panels : int
            Number of panels in the mesh to which this wake belongs.

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
            V = edge.get_vortex_influence(points)

            # Store
            vortex_influence_matrix[:,p_ind[0]] = -V
            vortex_influence_matrix[:,p_ind[1]] = V

        # Get influence of filaments
        V = self._get_filament_influences(points)
        for i in range(self.N):

            # Add for outbound panels
            outbound_panels = self.outbound_panels[i]
            if len(outbound_panels)>0:
                vortex_influence_matrix[:,outbound_panels[0]] -= V[:,i]
                vortex_influence_matrix[:,outbound_panels[1]] += V[:,i]

            # Add for inbound panels
            inbound_panels = self.inbound_panels[i]
            if len(inbound_panels)>0:
                vortex_influence_matrix[:,inbound_panels[0]] += V[:,i]
                vortex_influence_matrix[:,inbound_panels[1]] -= V[:,i]
        
        return vortex_influence_matrix


    def _get_filament_influences(self, points):
        # Determines the unit vortex influence from the wake filaments on the given points

        # Determine displacement vectors: first index is point, second is filament, third is segment, fourth is vector component
        if self._end_infinite:
            r0 = points[:,np.newaxis,np.newaxis,:]-self._vertices[np.newaxis,:,:-2,:] # Don't add the last segment at this point
            r1 = points[:,np.newaxis,np.newaxis,:]-self._vertices[np.newaxis,:,1:-1,:]
        else:
            r0 = points[:,np.newaxis,np.newaxis,:]-self._vertices[np.newaxis,:,:-1,:]
            r1 = points[:,np.newaxis,np.newaxis,:]-self._vertices[np.newaxis,:,1:,:]

        # Determine displacement vector magnitudes
        r0_mag = vec_norm(r0)
        r1_mag = vec_norm(r1)

        # Calculate influence of each segment
        inf = np.sum(((r0_mag+r1_mag)/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1))))[:,:,:,np.newaxis]*vec_cross(r0, r1), axis=2)

        # Add influence of last segment, if needed
        if self._end_infinite:

            # Determine displacement vector magnitudes
            r = r1[:,:,-1,:]
            r_mag = vec_norm(r)
            u = self._vertices[:,-1,:]-self._vertices[:,-2,:]
            u /= vec_norm(u)[:,np.newaxis]

            # Calculate influence
            inf += vec_cross(u[np.newaxis,:,:], r)/(r_mag*(r_mag-vec_inner(u[np.newaxis,:,:], r)))[:,:,np.newaxis]

        return 0.25/np.pi*np.nan_to_num(inf)


class FullStreamlineWake(SegmentedWake):
    """Defines a segmented wake which is updated to trace out entire streamlines beginning at the Kutta edges on each iteration.

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

    corrector_iterations : int, optional
        How many times to correct the streamline (velocity) prediction for each segment. Defaults to 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get kwargs
        self._corrector_iterations = kwargs.get('corrector_iterations', 1)


    def update(self, velocity_from_body, mu, v_inf, omega, verbose):
        """Updates the shape of the wake based on solved flow results.

        Parameters
        ----------
        velocity_from_body : callable
            Function which will return the velocity induced by the body at a given set of points.

        mu : ndarray
            Vector of doublet strengths.

        v_inf : ndarray
            Freestream velocity vector.

        omega : ndarray
            Angular rate vector.

        verbose : bool
        """

        if verbose:
            print()
            prog = OneLineProgress(self.N_segments+1, msg="    Updating wake shape")

        # Initialize storage
        new_locs = np.zeros((self.N, self.N_segments, 3))

        # Get starting locations (offset slightly from origin to avoid singularities)
        curr_loc = self._vertices[:,0,:]+self._filament_dirs*0.01
        
        if verbose: prog.display()

        # Loop through filament segments (the first vertex never changes)
        next_loc = np.zeros((self.N, 3))
        for i in range(1,self.N_segments+1):

            # Determine velocities at current point
            v0 = velocity_from_body(curr_loc)+v_inf[np.newaxis,:]-vec_cross(omega, curr_loc)
            v0 += self._get_velocity_from_other_filaments_and_edges(curr_loc, mu)

            # Guess of next location
            next_loc = curr_loc+self.l*v0/vec_norm(v0)[:,np.newaxis]

            # Iteratively correct
            for j in range(self._corrector_iterations):

                # Velocities at next location
                v1 = velocity_from_body(next_loc)+v_inf[np.newaxis,:]
                v1 += self._get_velocity_from_other_filaments_and_edges(next_loc, mu)

                # Correct location
                v_avg = 0.5*(v0+v1)
                next_loc = curr_loc+self.l*v_avg/vec_norm(v_avg)[:,np.newaxis]

            # Store
            new_locs[:,i-1,:] = np.copy(next_loc)

            # Move downstream
            curr_loc = np.copy(next_loc)

            if verbose: prog.display()

        # Store the new locations
        self._vertices[:,1:,:] = new_locs


    def _get_velocity_from_other_filaments_and_edges(self, points, mu):
        # Determines the velocity at each point (assumed to be one on each filament in order) induced by all other filaments and Kutta edges

        # Initialize storage
        v_ind = np.zeros((self.N, 3))

        # Get filament influences
        with np.errstate(divide='ignore', invalid='ignore'):
            V = self._get_filament_influences(points) # On the first segment of the first iteration, this will throw warnings because the initial point is on the filament; these can safely be ignored

        # Loop through filaments
        for i in range(self.N):

            # Get indices of points not belonging to this filament
            ind = [j for j in range(self.N) if j!=i]

            # Add for outbound panels
            outbound_panels = self.outbound_panels[i]
            if len(outbound_panels)>0:
                v_ind[ind] -= V[ind,i]*mu[outbound_panels[0]]
                v_ind[ind] += V[ind,i]*mu[outbound_panels[1]]

            # Add for inbound panels
            inbound_panels = self.inbound_panels[i]
            if len(inbound_panels)>0:
                v_ind[ind] += V[ind,i]*mu[inbound_panels[0]]
                v_ind[ind] -= V[ind,i]*mu[inbound_panels[1]]

        # Get influence of edges
        for edge in self._kutta_edges:

            # Get indices of panels defining the edge
            p_ind = edge.panel_indices

            # Get infulence
            v = edge.get_vortex_influence(points)

            # Store
            v_ind += -v*mu[p_ind[0]]
            v_ind += v*mu[p_ind[1]]

        return v_ind


class VelocityRelaxedWake(SegmentedWake):
    """Defines a segmented wake which is updated by shifting the segment vertices by the induced velocity on each iteration.

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

    K : float
        Time stepping factor for shifting the filament vertices based on the local induced velocity and distance from the trailing edge.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get kwargs
        self._K = kwargs["K"]


    def update(self, velocity_from_body, mu, v_inf, omega, verbose):
        """Updates the shape of the wake based on solved flow results.

        Parameters
        ----------
        velocity_from_body : callable
            Function which will return the velocity induced by the body at a given set of points.

        mu : ndarray
            Vector of doublet strengths.

        v_inf : ndarray
            Freestream velocity vector.

        omega : ndarray
            Angular rate vector.

        verbose : bool
        """

        if verbose:
            print()
            prog = OneLineProgress(4, msg="    Updating wake shape")

        # Reorder vertices for computation
        points = self._vertices[:,1:,:].reshape((self.N*(self.N_segments), 3))

        # Get velocity from body and rotation
        v_ind = velocity_from_body(points)-vec_cross(omega, points)
        if verbose: prog.display()

        # Get velocity from wake elements
        v_ind += self._get_velocity_from_filaments_and_edges(points, mu)
        if verbose: prog.display()

        # Calculate time-stepping parameter
        U = norm(v_inf)
        u = v_inf/U
        dl = self._vertices[:,1:,:]-self._vertices[:,0,:][:,np.newaxis,:]
        d = vec_inner(dl, u[np.newaxis,:])
        dt = self._K*d/U
        if verbose: prog.display()

        # Shift vertices
        self._vertices[:,1:,:] += dt[:,:,np.newaxis]*v_ind.reshape((self.N, self.N_segments, 3))
        if verbose: prog.display()


    def _get_velocity_from_filaments_and_edges(self, points, mu):
        # Determines the velocity at the given points induced by all filaments and Kutta edges

        # Initialize storage
        v_ind = np.zeros_like(points)

        # Get filament influences
        with np.errstate(divide='ignore', invalid='ignore'):
            V = self._get_filament_influences(points)

        # Loop through filaments
        for i in range(self.N):

            # Add for outbound panels
            outbound_panels = self.outbound_panels[i]
            if len(outbound_panels)>0:
                v_ind[:] -= V[:,i]*mu[outbound_panels[0]]
                v_ind[:] += V[:,i]*mu[outbound_panels[1]]

            # Add for inbound panels
            inbound_panels = self.inbound_panels[i]
            if len(inbound_panels)>0:
                v_ind[:] += V[:,i]*mu[inbound_panels[0]]
                v_ind[:] -= V[:,i]*mu[inbound_panels[1]]

        # Get influence of edges
        for edge in self._kutta_edges:

            # Get indices of panels defining the edge
            p_ind = edge.panel_indices

            # Get infulence
            v = edge.get_vortex_influence(points)

            # Store
            v_ind += -v*mu[p_ind[0]]
            v_ind += v*mu[p_ind[1]]

        return v_ind


class MarchingStreamlineWake(SegmentedWake):
    """Defines a segmented wake which is updated by adding a filament segment in the direction of the local velocity at each iteration.

    Parameters
    ----------
    kutta_edges : list of KuttaEdge
        List of Kutta edges which define this wake.

    N_segments : int, optional
        Number of segments to use for each filament. Must be the same as the number of wake iterations for the solver. Defaults to 20.

    segment_length : float, optional
        Length of each discrete filament segment. Defaults to 1.0.

    end_segment_infinite : bool, optional
        Whether the final segment of the filament should be treated as infinite. Defaults to False.

    corrector_iterations : int, optional
        How many times to correct the streamline (velocity) prediction for each segment. Defaults to 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get kwargs
        self._corrector_iterations = kwargs.get("corrector_iterations", 1)

        # Set segment counters
        self.N_segments_final = copy.copy(self.N_segments) # How many segments this wake should have when done iterating. It starts with zero.


    def set_filament_direction(self, v_inf, omega):
        """Resets the counter for determining how far along the wake has been solved. Does not update filament vertices (yeah it's a misnomer, but hey, consistency).

        Parameters
        ----------
        v_inf : ndarray
            Freestream velocity vector.

        omega : ndarray
            Angular rate vector.
        """

        # Get filament starting directions (required for offsetting the initial point to avoid infinite velocities)
        origins = self._vertices[:,0,:]
        self._filament_dirs = v_inf[np.newaxis,:]-vec_cross(omega, origins)
        self._filament_dirs /= vec_norm(self._filament_dirs)[:,np.newaxis]

        # Reset number of segments which have been set
        self.N_segments = 0


    def get_vtk_data(self, **kwargs):
        """Returns a list of vertices and line indices describing this wake.
        
        Parameters
        ----------
        length : float, optional
            Length of the final filament segment, if set as infinite. Defaults to 20 times the filament segment length.
        """

        # Get kwargs
        l = kwargs.get("length", 20.0*self.l)

        # Initialize storage
        vertices = []
        line_vertex_indices = []

        # Loop through filaments
        i = 0
        for j in range(self.N):

            # Add vertices
            for k in range(self.N_segments+1):
                vertices.append(self._vertices[j,k])

                # Add indices
                if k != self.N_segments:
                    line_vertex_indices.append([2, i+k, i+k+1])

            # Increment index
            i += self.N_segments+1

        return vertices, line_vertex_indices, self.N*self.N_segments


    def get_influence_matrix(self, **kwargs):
        """Create wake influence matrix; first index is the influenced points, second is the influencing panel, third is the velocity component.

        Parameters
        ----------
        points : ndarray
            Array of points at which to calculate the influence.

        N_panels : int
            Number of panels in the mesh to which this wake belongs.

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
            V = edge.get_vortex_influence(points)

            # Store
            vortex_influence_matrix[:,p_ind[0]] = -V
            vortex_influence_matrix[:,p_ind[1]] = V

        # Get influence of filaments
        if self.N_segments > 0:
            V = self._get_filament_influences(points)
            for i in range(self.N):

                # Add for outbound panels
                outbound_panels = self.outbound_panels[i]
                if len(outbound_panels)>0:
                    vortex_influence_matrix[:,outbound_panels[0]] -= V[:,i]
                    vortex_influence_matrix[:,outbound_panels[1]] += V[:,i]

                # Add for inbound panels
                inbound_panels = self.inbound_panels[i]
                if len(inbound_panels)>0:
                    vortex_influence_matrix[:,inbound_panels[0]] += V[:,i]
                    vortex_influence_matrix[:,inbound_panels[1]] -= V[:,i]
        
        return vortex_influence_matrix


    def _get_filament_influences(self, points):
        # Determines the unit vortex influence from the wake filaments on the given points

        # Determine displacement vectors: first index is point, second is filament, third is segment, fourth is vector component
        r0 = points[:,np.newaxis,np.newaxis,:]-self._vertices[np.newaxis,:,:self.N_segments,:]
        r1 = points[:,np.newaxis,np.newaxis,:]-self._vertices[np.newaxis,:,1:self.N_segments+1,:]

        # Determine displacement vector magnitudes
        r0_mag = vec_norm(r0)
        r1_mag = vec_norm(r1)

        # Calculate influence of each segment
        inf = np.sum(((r0_mag+r1_mag)/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1))))[:,:,:,np.newaxis]*vec_cross(r0, r1), axis=2)

        return 0.25/np.pi*np.nan_to_num(inf)



    def update(self, velocity_from_body, mu, v_inf, omega, verbose):
        """Updates the shape of the wake based on solved flow results.

        Parameters
        ----------
        velocity_from_body : callable
            Function which will return the velocity induced by the body at a given set of points.

        mu : ndarray
            Vector of doublet strengths.

        v_inf : ndarray
            Freestream velocity vector.

        omega : ndarray
            Angular rate vector.

        verbose : bool
        """

        # Update number of segments
        self.N_segments += 1

        if verbose:
            print()
            prog = OneLineProgress(self.N_segments+1, msg="    Updating wake shape with {0} segments".format(self.N_segments))

        # Initialize storage
        new_locs = np.zeros((self.N, self.N_segments, 3))

        # Get starting locations (offset slightly from origin to avoid singularities)
        curr_loc = self._vertices[:,0,:]+self._filament_dirs*0.01
        
        if verbose: prog.display()

        # Loop through filament segments (the first vertex never changes)
        next_loc = np.zeros((self.N, 3))
        for i in range(1,self.N_segments+1):

            # Determine velocities at current point
            v0 = velocity_from_body(curr_loc)+v_inf[np.newaxis,:]-vec_cross(omega, curr_loc)
            v0 += self._get_velocity_from_other_filaments_and_edges(curr_loc, mu)

            # Guess of next location
            next_loc = curr_loc+self.l*v0/vec_norm(v0)[:,np.newaxis]

            # Iteratively correct
            for j in range(self._corrector_iterations):

                # Velocities at next location
                v1 = velocity_from_body(next_loc)+v_inf[np.newaxis,:]
                v1 += self._get_velocity_from_other_filaments_and_edges(next_loc, mu)

                # Correct location
                v_avg = 0.5*(v0+v1)
                next_loc = curr_loc+self.l*v_avg/vec_norm(v_avg)[:,np.newaxis]

            # Store
            new_locs[:,i-1,:] = np.copy(next_loc)

            # Move downstream
            curr_loc = np.copy(next_loc)

            if verbose: prog.display()

        # Store the new locations
        self._vertices[:,1:self.N_segments+1,:] = new_locs


    def _get_velocity_from_other_filaments_and_edges(self, points, mu):
        # Determines the velocity at each point (assumed to be one on each filament in order) induced by all other filaments and Kutta edges

        # Initialize storage
        v_ind = np.zeros((self.N, 3))

        # Get filament influences
        with np.errstate(divide='ignore', invalid='ignore'):
            V = self._get_filament_influences(points) # On the first segment of the first iteration, this will throw warnings because the initial point is on the filament; these can safely be ignored

        # Loop through filaments
        for i in range(self.N):

            # Get indices of points not belonging to this filament
            ind = [j for j in range(self.N) if j!=i]

            # Add for outbound panels
            outbound_panels = self.outbound_panels[i]
            if len(outbound_panels)>0:
                v_ind[ind] -= V[ind,i]*mu[outbound_panels[0]]
                v_ind[ind] += V[ind,i]*mu[outbound_panels[1]]

            # Add for inbound panels
            inbound_panels = self.inbound_panels[i]
            if len(inbound_panels)>0:
                v_ind[ind] += V[ind,i]*mu[inbound_panels[0]]
                v_ind[ind] -= V[ind,i]*mu[inbound_panels[1]]

        # Get influence of edges
        for edge in self._kutta_edges:

            # Get indices of panels defining the edge
            p_ind = edge.panel_indices

            # Get infulence
            v = edge.get_vortex_influence(points)

            # Store
            v_ind += -v*mu[p_ind[0]]
            v_ind += v*mu[p_ind[1]]

        return v_ind