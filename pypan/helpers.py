"""Helper functions for PyPan."""

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime as dt
from datetime import timedelta as td

from pypan.pp_math import dist


class OneLineProgress():
    """Displays a progress bar in one line.
    Written by Zach Montgomery.
    """
    
    def __init__(self, total, msg='', show_etr=True):
        self.total = total
        self.msg = msg
        self.count = 0
        self.show_etr = show_etr
        self.start = dt.now()
        self.roll_timer = dt.now()
        self.roll_count = -1
        self.roll_delta = 0.2
        self.run_time = 0.0
        self.display()
    
    def increment(self):
        self.count += 1
    
    def decrement(self):
        self.count -= 1
    
    def __str__(self):
        pass
    
    def __len__(self):
        l = len(str(self))
        self.decrement()
        return l
    
    def Set(self, count):
        self.count = count
    
    def display(self):
        rolling = '-\\|/'
        roll_delta = (dt.now()-self.roll_timer).total_seconds()
        
        p2s = False
        if roll_delta >= self.roll_delta or self.roll_count == -1:
            p2s = True
            self.roll_timer = dt.now()
            self.roll_count += 1
            if self.roll_count >= len(rolling):
                self.roll_count = 0
        
        perc = self.count/self.total*100.
        self.increment()
        
        if not p2s and perc < 100.: return
        
        s = '\r' + ' '*(len(self.msg)+50) + '\r'
        s += self.msg + ' '*4
        
        # j = 0
        for i in range(10):
            if perc >= i*10:
                j = i
        
        if perc < 100.:
            s += u'\u039e'*j + rolling[self.roll_count] + '-'*(9-j)
        else:
            s += u'\u039e'*10
        
        # for i in range(1,11):
            # if i*10 <= perc:
                # s += u'\u039e'
            # else:
                # s += '-'
        s += ' '*4 + '{:7.3f}%'.format(perc)
        if not self.show_etr:
            if perc >= 100.: s += '\n'
            print(s, end='')
            return
        
        if perc <= 0:
            etr = '-:--:--.------'
            s += ' '*4 + 'ETR = {}'.format(etr)
        elif perc >= 100.:
            self.run_time = dt.now()-self.start
            s += ' '*4 + 'Run Time {}'.format(self.run_time) + '\n'
        else:
            time = (dt.now()-self.start).total_seconds()
            etr = td(seconds=time / perc * 100. - time)
            s += ' '*4 + 'ETR = {}'.format(etr)
        print(s, end='')
        return


def compare_mirror(mesh1, mesh2, mirror_plane):
    """Checks whether the two meshes are mirrors of each other.

    Parameters
    ----------
    mesh1, mesh2 : pypan.Mesh
        Mesh objects which may or may not be mirrors of each other.

    mirror_plane : int
        Index of the normal component to the mirror plane (yz = 0, xz = 1, xy = 2).
    """

    # Check number of vertices
    if mesh1.N_vert != mesh2.N_vert:
        print("Meshes do not have the same number of vertices.")
    else:
        print("Both meshes have {0} vertices.".format(mesh1.N_vert))

    # Check number of panels
    if mesh1.N != mesh2.N:
        print("Meshes do not have the same number of panels.")
    else:
        print("Both meshes have {0} panels.".format(mesh1.N))

    # Try to mirror vertices naively
    mirrored_verts = np.copy(mesh2.vertices)
    mirrored_verts[:,mirror_plane] *= -1.0

    # Check locations
    non_matches = 0
    for i in range(mesh1.N_vert):

        # Calculate distance
        d = dist(mesh1.vertices[i], mirrored_verts[i,:])
        
        if d > 1e-12:
            non_matches += 1

    print("Assuming same order, {0} vertices do not match.".format(non_matches))

    # Check locations in reverse order
    non_matches = 0
    for i in range(mesh1.N_vert):

        # Calculate distance
        d = dist(mesh1.vertices[i], mirrored_verts[mesh2.N_vert-i-1,:])
        
        if d > 1e-12:
            non_matches += 1

    print("Assuming opposite order, {0} vertices do not match.".format(non_matches))

    # Set up plot of mirrored vertices
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.gca(projection='3d')
    
    # Plot vertices
    for i, vertex in enumerate(mesh1.vertices):
        ax.plot(vertex[0], vertex[1], vertex[2], 'b.', markersize=4)
    for i, vertex in enumerate(mirrored_verts):
        ax.plot(vertex[0], vertex[1], vertex[2], '.', color='orange', markersize=2)

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    mesh1._rescale_3D_axes(ax)
    plt.show()