import copy
import time
import pypan as pp
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    # Load mesh
    #mesh_file = "dev/meshes/swept_wing_21_tapered.stl"
    #mesh_file = "dev/meshes/swept_wing_21.stl"
    #mesh_file = "dev/meshes/swept_wing_51.stl"
    #mesh_file = "dev/meshes/1250_polygon_sphere_100mm.STL"
    #mesh_file = "dev/meshes/Dodecahedron.stl"
    #mesh_file = "dev/meshes/5000_polygon_sphere_100mm.STL"
    #mesh_file = "dev/meshes/20000_polygon_sphere_100mm.STL"
    #mesh_file = "dev/meshes/1250_sphere.vtk"
    mesh_file = "dev/meshes/swept_wing_21_rounded.stl"
    #mesh_file = "dev/meshes/swept_wing_21.vtk"
    #mesh_file = "dev/meshes/F16_Original_withFins.vtk"
    start_time = time.time()

    my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="STL", kutta_angle=90.0, verbose=True)
    #my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="VTK", kutta_angle=90.0, verbose=True)

    #my_mesh.export_vtk("dev/meshes/5000_sphere.vtk")
    #my_mesh.plot(centroids=False)

    # Initialize solver
    my_solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)

    # Set condition
    my_solver.set_condition(V_inf=[-100.0, 0.0, -15.0], rho=0.0023769)

    # Solve
    F = my_solver.solve(verbose=True, lifting=True)
    print("F: ", F)
    print("Max C_P: ", np.max(my_solver._C_P))
    print("Min C_P: ", np.min(my_solver._C_P))

    # Export VTK
    my_solver.export_vtk(mesh_file.replace("meshes", "results").replace(".STL", ".vtk").replace(".stl", ".vtk"))
    end_time = time.time()
    print("\nTotal run time: {0} s".format(end_time-start_time))

    # Run alpha sweep
    alphas, F, F_w = my_solver.alpha_sweep(V_inf=100, alpha_lims=[-20.0, 20.0], N_alpha=21, rho=0.0023769, results_dir="dev/results/swept_wing_21_alpha_sweep/")

    # Plot
    non_dim = 1.0/(0.5*0.0023769*100.0**2*8.0)
    plt.figure()
    plt.plot(alphas, F_w[:,2]*non_dim)
    plt.xlabel("Alpha [deg]")
    plt.ylabel("CL")
    plt.show()

    plt.figure()
    plt.plot(F_w[:,2]*non_dim, F_w[:,0]*non_dim)
    plt.xlabel("CL")
    plt.ylabel("CD")
    plt.show()