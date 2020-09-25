import pypan as pp
import numpy as np

if __name__=="__main__":

    # Load mesh
    my_mesh = pp.Mesh(mesh_file="dev/swept_wing_10.stl", mesh_file_type="STL", verbose=True)
    my_mesh.plot(centroids=True)