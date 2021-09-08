import time
import os

import pypan as pp
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    # Load mesh
    #mesh_file = "dev/meshes/F16_surfaceMesh_single.vtk"
    mesh_file = "dev/meshes/swept_wing_low_grid.vtk"

    # Start timer
    start_time = time.time()
    pam_file = mesh_file.replace(".stl", ".pam").replace(".vtk", ".pam")
    name = mesh_file.replace("dev/meshes/", "").replace(".stl", "").replace(".vtk", "")

    # Load mesh
    my_mesh = pp.Mesh(name=name, mesh_file=mesh_file, adjacency_file=pam_file, verbose=True, multi_file='multi' in mesh_file)

    # Export vtk if we need to
    vtk_file = mesh_file.replace(".stl", ".vtk")
    if not os.path.isfile(vtk_file):
        my_mesh.export_vtk(vtk_file)

    # Export adjacency mapping if we need to
    if not os.path.isfile(pam_file):
        my_mesh.export_panel_adjacency_mapping(pam_file)

    # Set wake
    my_mesh.set_wake(type='fixed')

    # Initialize solver
    my_solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)

    # Set condition
    alpha = np.radians(5.0)
    V = 100.0
    V_inf = np.array([-V*np.cos(alpha), 0.0, -V*np.sin(alpha)])
    rho = 0.0023769
    my_solver.set_condition(V_inf=V_inf, rho=rho, angular_rate=[0.0, 0.0, 0.0])
    u_inf = V_inf/V
    
    # Plot
    #my_mesh.plot(panels=False, vertices=True, kutta_edges=True)

    # Solve
    results_file = mesh_file.replace("meshes", "results").replace("stl", "vtk").replace("tri", "vtk")
    F, M = my_solver.solve(verbose=True, method='gauss-seidel', gs_convergence=1e-4)

    # Print results
    print()
    print("F: ", F)
    print("M: ", M)
    print("Max C_P: ", np.max(my_solver._C_P))
    print("Min C_P: ", np.min(my_solver._C_P))

    # Calculate lift and drag
    D = np.inner(F, u_inf)
    L = np.linalg.norm(F-D*u_inf)
    Sw = 20.0
    print("L: ", L)
    print("D: ", D)
    print("L/D: ", L/D)
    print("CL: ", L/(0.5*rho*V**2*Sw))
    print("CD: ", D/(0.5*rho*V**2*Sw))

    # Export results as VTK
    my_solver.export_vtk(results_file)

    print()
    print("Total execution time: {0} s".format(time.time()-start_time))
