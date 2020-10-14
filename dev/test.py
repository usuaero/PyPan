import copy
import pypan as pp
import numpy as np

if __name__=="__main__":

    # Load mesh
    #mesh_file = "dev/meshes/swept_wing_21_tapered.stl"
    #mesh_file = "dev/meshes/swept_wing_21.stl"
    #mesh_file = "dev/meshes/swept_wing_51.stl"
    mesh_file = "dev/meshes/1250_polygon_sphere_100mm.STL"
    #mesh_file = "dev/meshes/Dodecahedron.stl"
    #mesh_file = "dev/meshes/5000_polygon_sphere_100mm.STL"
    #mesh_file = "dev/meshes/20000_polygon_sphere_100mm.STL"
    #mesh_file = "dev/meshes/1250_sphere.vtk"
    #mesh_file = "dev/meshes/swept_wing_21_rounded.stl"
    #mesh_file = "dev/meshes/swept_wing_21.vtk"
    #mesh_file = "dev/meshes/F16_Original_withFins.vtk"

    my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="STL", kutta_angle=90.0, verbose=True)
    #my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="VTK", kutta_angle=90.0, verbose=True)

    #my_mesh.export_vtk("dev/meshes/5000_sphere.vtk")
    #my_mesh.plot(centroids=False)

    # Initialize solver
    my_solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)

    # Set condition
    my_solver.set_condition(V_inf=[0.0, 0.0, -100.0], rho=0.0023769)

    # Solve
    F = my_solver.solve(verbose=True, lifting=False)
    print("F: ", F)
    print("Max C_P: ", np.max(my_solver._C_P))
    print("Min C_P: ", np.min(my_solver._C_P))

    # Export VTK
    my_solver.export_vtk("dev/results/1250_sphere.vtk")