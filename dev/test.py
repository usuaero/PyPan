import pypan as pp
import numpy as np

if __name__=="__main__":

    # Load mesh
    mesh_file = "dev/swept_wing_21_tapered.stl"
    #mesh_file = "dev/swept_wing_21_rounded.stl"
    #mesh_file = "dev/swept_wing_21.stl"
    #mesh_file = "dev/swept_wing_51.stl"
    my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="STL", kutta_angle=90.0, verbose=True)
    my_mesh.plot(centroids=False, panels=True)
    my_mesh.export("21.ppmsh")

    # Initialize solver
    my_solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)

    # Set condition
    my_solver.set_condition(V_inf=[-100.0, 0.0, 0.0], rho=0.0023769)

    # Solve
    F = my_solver.solve(verbose=True)
    print(F)
    my_solver.export_case_data("case_data_21.txt")