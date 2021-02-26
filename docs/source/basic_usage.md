# Basic Usage

A simple, introductory case of using PyPan is shown below.

```python
import pypan as pp

if __name__=="__main__":

    # Load mesh
    mesh_file = "dev/meshes/swept_wing_vtk.vtk"
    my_mesh = pp.Mesh(mesh_file=mesh_file, mesh_file_type="VTK", verbose=True, kutta_angle=90.0)

    # Set wake
    my_mesh.set_wake(iterative=False, type="freestream")

    # Plot mesh
    my_mesh.plot()

    # Initialize solver
    my_solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)

    # Set condition
    my_solver.set_condition(V_inf=[-100.0, 0.0, -10.0], rho=0.0023769)

    # Solve
    F, M = my_solver.solve(verbose=True)
    print("F: ", F)
    print("M: ", M)
    print("Max C_P: ", np.max(my_solver._C_P))
    print("Min C_P: ", np.min(my_solver._C_P))

    # Export results as VTK
    my_solver.export_vtk(mesh_file.replace("meshes", "results").replace("stl", "vtk"))
```
