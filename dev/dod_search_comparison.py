import time
import os

import pypan as pp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat


if __name__=="__main__":

    # Parameters
    mesh_files = ["dev/meshes/swept_wing_low_grid.vtk",
                  "dev/meshes/swept_wing_high_grid.vtk",
                  "dev/meshes/F-22.vtk",
                  "dev/meshes/DPW-W1.tri",
                  "dev/meshes/HSCT.tri",
                  "dev/meshes/supersonic_wing_body.vtk",]

    mesh_files = ["dev/meshes/supersonic_wing_body_ultra_low_res.stl",
                  "dev/meshes/supersonic_wing_body_very_low_res.stl",
                  "dev/meshes/supersonic_wing_body_low_res.stl",
                  "dev/meshes/supersonic_wing_body_med_res.stl",]
                  #"dev/meshes/supersonic_wing_body_hi_res.stl",
                  #"dev/meshes/supersonic_wing_body_very_hi_res.stl",
                  #"dev/meshes/supersonic_wing_body_ultra_hi_res.stl",]

    mach_nums = [1.2, 1.6, 2.2, 3.0]
    colors = ["#000000", "#333333", "#555555", "#888888", "#AAAAAA", "#DDDDDD"]
    N_vert = []
    recursive_times = np.zeros((len(mesh_files), len(mach_nums)))
    brute_force_times = np.zeros((len(mesh_files), len(mach_nums)))

    for i, mesh_file in enumerate(mesh_files):

        print()
        print("----------------------------------")
        print(mesh_file)
        print("----------------------------------")

        # Start timer
        start_time = time.time()
        name = mesh_file.replace("dev/meshes/", "").replace(".vtk", "").replace(".tri", "").replace(".stl", "")
        pam_file = "dev/meshes/"+name+".pam"

        # Load mesh
        my_mesh = pp.Mesh(name=name, mesh_file=mesh_file, adjacency_file=pam_file, verbose=True)
        N_vert.append(my_mesh.N_vert)

        # Export vtk if we need to
        vtk_file = "dev/meshes/"+name+".vtk"
        if not os.path.isfile(vtk_file):
            my_mesh.export_vtk(vtk_file)

        # Export adjacency mapping if we need to
        if not os.path.isfile(pam_file):
            my_mesh.export_panel_adjacency_mapping(pam_file)

        # Set wake
        my_mesh.set_wake(type='full_streamline')

        # Loop through Mach numbers
        for j, M in enumerate(mach_nums):

            # Initialize solver
            my_solver = pp.SupersonicSolver(mesh=my_mesh, verbose=True)

            # Set condition
            my_solver.set_condition(M=M)

            # Store time results
            recursive_times[i,j] = my_solver._recursive_time
            brute_force_times[i,j] = my_solver._brute_force_time

    # Write times to files
    recursive_times.tofile('dev/results/recursive_times.txt', sep=',')
    brute_force_times.tofile('dev/results/brute_force_times.txt', sep=',')

    N_vert = np.array(N_vert)

    # Calculate regressions
    for j, M in enumerate(mach_nums):
        print()
        print("Mach number: {0}".format(M))
        
        # Brute force
        reg = stat.linregress(np.log10(N_vert), np.log10(brute_force_times[:,j]))
        print("    Brute force is O(N^{0})".format(reg.slope))
        
        # Recursive
        reg = stat.linregress(np.log10(N_vert), np.log10(recursive_times[:,j]))
        print("    Recursive is O(N^{0})".format(reg.slope))

    # Plot
    plt.figure(figsize=(8.0,8.0))
    for j, M in enumerate(mach_nums):
        plt.plot(N_vert, recursive_times[:,j], '^-', label='Recur. M={0}'.format(M), color=colors[j])
        plt.plot(N_vert, brute_force_times[:,j], 's--', label='BF M={0}'.format(M), color=colors[j])
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Search Execution Time [s]')
    plt.savefig('dev/results/search_comparison.pdf')