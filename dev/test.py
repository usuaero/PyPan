import copy
import pypan as pp
import numpy as np

if __name__=="__main__":

    # Load mesh
    #mesh_file = "dev/21.ppmsh"
    #mesh_file = "dev/swept_wing_21_tapered.stl"
    #mesh_file = "dev/swept_wing_21.stl"
    #mesh_file = "dev/swept_wing_51.stl"
    #mesh_file = "dev/1250_polygon_sphere_100mm.STL"

    #my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="STL", kutta_angle=90.0, verbose=True)

    # Parse STL
    mesh_file = "dev/swept_wing_21_rounded.stl"
    my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="STL", kutta_angle=90.0, verbose=True)
    #my_mesh.plot(centroids=False)
    my_mesh.export_vtk("dev/swept_wing_21.vtk")
    mapping1 = copy.copy(my_mesh._panel_vertex_indices)

    # Load exported VTK and get Kutta edges
    mesh_file = "dev/swept_wing_21.vtk"
    my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="VTK", kutta_angle=90.0, verbose=True)
    #my_mesh.plot(centroids=False)
    mapping2 = copy.copy(my_mesh._panel_vertex_indices)

    # Compare mappings
    for ind1, ind2 in zip(mapping1, mapping2):
        for i1, i2 in zip(ind1, ind2):
            if i1!=i2:
                print("Mismatch!")
                print(i1)
                print(i2)

    ## Initialize solver
    #my_solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)

    ## Set condition
    #my_solver.set_condition(V_inf=[0.0, 0.0, -100.0], rho=0.0023769)

    ## Solve
    #F = my_solver.solve(verbose=True, lifting=False)
    #print("F: ", F)
    #print("Max C_P: ", np.max(my_solver._C_P))
    #print("Min C_P: ", np.min(my_solver._C_P))

    ## Export VTK
    #my_solver.export_vtk("sphere_1250.vtk")