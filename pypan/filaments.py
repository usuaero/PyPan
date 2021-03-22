import numpy as np

from pypan.pp_math import vec_norm, vec_cross, vec_inner, norm

class StraightVortexFilament:
    """Defines a straight, semi-infinite vortex filament.

    Parameters
    ----------
    origin : ndarray
        Origin point of the filament.

    inbound_panels : list
        Indices of the two panels for which this is an inbound filament.

    outbound_panels : list
        Indices of the two panels for which this is an outbound filament.
    """

    def __init__(self, origin, inbound_panels, outbound_panels):

        # Store info
        self.p0 = origin
        self.inbound_panels = inbound_panels
        self.outbound_panels = outbound_panels
        self.N = 1


    def set_dir(self, direction):
        """Sets the direction of the filament.

        Parameters
        ----------
        direction : ndarray
            Direction of the vortex filament.
        """

        # Store direction
        self.dir = direction


    def get_influence(self, points):
        """Determines the influence of this vortex filament on the specified points.

        Parameters
        ----------
        points : ndarray
            Points at which to calculate the influence.
        """

        # Determine displacement vector magnitudes
        r = points-self.p0[np.newaxis,:]
        r_mag = vec_norm(r)

        # Calculate influence
        return 0.25/np.pi*vec_cross(self.dir, r)/(r_mag*(r_mag-vec_inner(self.dir, r)))[:,np.newaxis]


class SegmentedVortexFilament:
    """Defines a (potentially) semi-infinite vortex filament made up of discrete segments.

    Parameters
    ----------
    origin : ndarray
        Origin point of the filament.

    inbound_panels : list
        Indices of the two panels for which this is an inbound filament.

    outbound_panels : list
        Indices of the two panels for which this is an outbound filament.

    N_segments : int, optional
        Number of segments to use for the filament. Defaults to 20.

    segment_length : float, optional
        Length of each discrete filament segment. Defaults to 1.0.

    end_segment_infinite : bool, optional
        Whether the final segment of the filament should be treated as infinite. Defaults to False.
    """

    def __init__(self, origin, inbound_panels, outbound_panels, **kwargs):

        # Store info
        self.p0 = origin
        self.inbound_panels = inbound_panels
        self.outbound_panels = outbound_panels
        self.N = kwargs.get("N_segments", 20)
        self.l = kwargs.get("segment_length", 1.0)
        self.end_inf = kwargs.get("end_segment_infinite", False)


    def initialize_points(self, direction):
        """Initializes the points making up this filament. Includes the origin point.

        Parameters
        ----------
        direction : ndarray
            Direction of the vortex filament.
        """

        # Store direction
        self.dir = direction

        # Initialize points
        self.points = self.p0[np.newaxis,:]+self.l*np.linspace(0.0, self.N*self.l, self.N+1)[:,np.newaxis]*self.dir[np.newaxis,:]


    def get_influence(self, points):
        """Determines the influence of this vortex filament on the specified points.

        Parameters
        ----------
        points : ndarray
            Points at which to calculate the influence.
        """

        # Determine displacement vectors
        if self.end_inf:
            r0 = points[:,np.newaxis,:]-self.points[np.newaxis,:-2,:]
            r1 = points[:,np.newaxis,:]-self.points[np.newaxis,1:-1,:]
        else:
            r0 = points[:,np.newaxis,:]-self.points[np.newaxis,:-1,:]
            r1 = points[:,np.newaxis,:]-self.points[np.newaxis,1:,:]

        # Determine displacement vector magnitudes
        r0_mag = vec_norm(r0)
        r1_mag = vec_norm(r1)

        # Calculate influence of each segment
        inf = np.sum(((r0_mag+r1_mag)/(r0_mag*r1_mag*(r0_mag*r1_mag+vec_inner(r0, r1))))[:,:,np.newaxis]*vec_cross(r0, r1), axis=1)

        # Add influence of last segment, if needed
        if self.end_inf:

            # Determine displacement vector magnitudes
            r = r1[:,-1,:]
            r_mag = vec_norm(r)
            u = self.points[-1]-self.points[-2]
            u /= norm(u)

            # Calculate influence
            inf += vec_cross(u, r)/(r_mag*(r_mag-vec_inner(u, r)))[:,np.newaxis]

        return 0.25/np.pi*inf