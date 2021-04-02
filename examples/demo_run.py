import pypan as pp

if __name__=="__main__":

    # Load mesh
    my_mesh = pp.Mesh(name="demo_wing", mesh_file="dev/meshes/swept_wing_low_grid.vtk", verbose=True)

    # Set wake
    my_mesh.set_wake(type="full_streamline", N_segments=40, segment_length=0.5)

    # Initialize solver
    my_solver = pp.VortexRingSolver(mesh=my_mesh)

    # Solve
    my_solver.set_condition(V_inf=[-100.0, 0.0, -10.0], rho=0.0023769, angular_rate=[1.0, 0.0, 0.0])
    F, M = my_solver.solve(verbose=True, wake_iterations=4, export_wake_series=True, wake_series_title='demo')

    print()
    print("Force vector: {0}".format(F))
    print("Moment vector: {0}".format(M))

    # Export VTK
    my_solver.export_vtk("results.vtk")