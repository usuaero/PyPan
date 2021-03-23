# Wake Models

As with all panel solvers, the accuracy of the flow results is dependent upon appropriately modeling the wake of the body. Various wake models are available in PyPan, as described below.

The inclusion of a wake in PyPan is dependent upon the mesh having Kutta edges. If the mesh has no Kutta edges, no wake elements will be generated. A dummy wake which contains no trailing vortex elements and does not enforce the Kutta condition is initialized when a mesh object is instantiated. The mesh may then be passed to the solver with this dummy wake in place (appropriate for nonlifting bodies) or another wake may be set. To set a fixed or iterative wake, ```Mesh.set_fixed_wake()``` or ```Mesh.set_iterative_wake()``` must be called, respectively. The wake should be set on the mesh before the mesh is passed to the solver and ```Solver.set_condition()``` is called as some wake types are dependent upon the flow condition and these are updated when ```Solver.set_condition()``` is called.

Below, the various types of wake models available in PyPan are described on a conceptual level. For how to implement a given wake model in the API, see [Meshes](mesh).

## Fixed Wake

This is the simplest wake model available in PyPan. It consists of a single, semi-infinite vortex filament extending from each vertex defining the Kutta edges on the body. The direction of the filaments is set and does not change throughout computation.

The direction of these trailing filaments may be set in the following ways:

| Direction Specification                | Description                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| "freestream"                           | Each vortex filament extends in the freestream direction.                    |
| "freestream_constrained"               | Same as "freestream", except the direction of the filaments is constrained to be in the plane defined by the user-specified normal vector. |
| "freestream_and_rotation"              | Each vortex filament extends in the direction of the local velocity vector determined from the summation of the freestream velocity and the local velocity due to solid-body rotation of the body. |
| "freestream_and_rotation_constrained"  | Same as "freestream_and_rotation", except the direction of the filaments is constrained to be in the plane defined by the user-specified normal vector. |
| "custom"                               | Each vortex filament extends in the same direction as specified by the user. |

## Iterative Wake

Iterative wake models allow for iteratively updating the shape of the wake filaments to follow streamlines of the flow. This is done over multiple iterations of solving the entire flow field. The number of iterations is set in the call to ```Solver.solve()```. Each filament is made up of a set of finite segments which change position with every iteration. In general, it is recommended that the number and length of the filament segments be specified such that the wake extends a significant distance behind the body. Setting ```"end_segment_infinite"``` can help with this, but should be used with judgement.

There are three iterative wake models available in PyPan. They are as follows:

### Full Streamline Integration

This wake model initializes the wake as straight, semi-infinite filaments extending in the direction of the freestream (including rotation). The flow about the body is then solved. The shape of each filament is then updated to follow a streamline of the current flow field, ignoring the influence of the filament on itself. This is determined by integrating backward from the origin of the wake filament using a simple predictor-corrector integrator. Each iteration of this method is slow, but few iterations are required to obtain reasonable shape calculations.

### Wake Relaxation

This method is similar to the second wake rollup model described in Katz and Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Chapter 15.1, 2001. Essentially, the wake filaments are initialized as with the full streamline integration method. The induced velocity at each filament segment endpoint is then calculated. Each endpoint is then moved by the induced velocity multiplied by a time parameter (specified by the user). Each iteration of this method is very fast. There is a tradeoff between the size of the time parameter and the accuracy of the wake shape calculation. A large time parameter will move the wake quickly, but accuracy is degraded. A small time parameter will have the opposite effect.
