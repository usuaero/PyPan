import time
import os

import pypan as pp
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    # Load mesh
    #mesh_file = "dev/meshes/F16_surfaceMesh_single.stl"
    mesh_file = "dev/meshes/F16_surfaceMesh_multi.stl"

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
    alpha = np.radians(20.0)
    V = 100.0
    V_inf = np.array([V*np.cos(alpha), 0.0, V*np.sin(alpha)])
    my_solver.set_condition(V_inf=[0.0, 0.0, 100.0], rho=0.0023769, angular_rate=[0.0, 0.0, 0.0])

    # Plot
    my_mesh.plot()

    # Solve
    results_file = mesh_file.replace("meshes", "results").replace("stl", "vtk").replace("tri", "vtk")
    F, M = my_solver.solve(verbose=True, wake_iterations=0)#, export_wake_series=True, wake_series_title=results_file.replace(".vtk", "_series"), method='direct')
    print()
    print("F: ", F)
    print("M: ", M)
    print("Max C_P: ", np.max(my_solver._C_P))
    print("Min C_P: ", np.min(my_solver._C_P))

    # Export results as VTK
    my_solver.export_vtk(results_file)

    # Export potential
    my_solver.export_potential('dev/results/potential.vtk', verbose=True, res=[10, 10, 10], buffers=[1.0, 1.0, 1.0])

    print()
    print("Total execution time: {0} s".format(time.time()-start_time))