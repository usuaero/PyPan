import pypan as pp

if __name__=="__main__":

    # Load mesh
    my_mesh = pp.Mesh(name="demo_wing", mesh_file="mesh.vtk", mesh_file_type="VTK", kutta_angle=90.0, verbose=True)
    my_mesh.set_iterative_wake(segment_length=0.5)

    # Initialize solver
    my_solver = pp.VortexRingSolver(mesh=my_mesh)

    # Solve
    my_solver.set_condition(V_inf=[-100.0, 0.0, -10.0], rho=0.0023769)
    F, M = my_solver.solve(verbose=True)

    print()
    print("Force vector: {0}".format(F))
    print("Moment vector: {0}".format(M))

    # Export VTK
    my_solver.export_vtk("results.vtk")