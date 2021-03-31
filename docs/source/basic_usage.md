# Basic Usage

A simple, introductory case of using PyPan is shown below.

```python
import pypan as pp

if __name__=="__main__":

    # Load mesh from vtk file
    mesh_file = "swept_wing.vtk"
    my_mesh = pp.Mesh(name="test_mesh", mesh_file=mesh_file, mesh_file_type="VTK", verbose=True, kutta_angle=90.0)

    # Set fixed wake in the direction of the freestream
    my_mesh.set_fixed_wake(type="freestream")

    # Initialize vortex ring solver using the mesh already created
    my_solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)

    # Set condition
    my_solver.set_condition(V_inf=[-100.0, 0.0, -10.0], rho=0.0023769)

    # Solve for the forces and moments acting on the body
    F, M = my_solver.solve(verbose=True)
    print("F: ", F)
    print("M: ", M)

    # Export results as VTK
    my_solver.export_vtk("case_results.vtk")
```
