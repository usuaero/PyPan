# Wake Models

As with all panel solvers, the accuracy of the flow results is dependent upon appropriately modeling the wake of the body. Various wake models are available in PyPan, as described below.

The inclusion of a wake in PyPan is dependent upon the mesh having Kutta edges. If the mesh has no Kutta edges, no wake elements will be generated. A dummy wake which contains no trailing vortex elements and does not enforce the Kutta condition is initialized when a mesh object is instantiated. The mesh may then be passed to the solver with this dummy wake in place (appropriate for nonlifting bodies) or another wake may be set. To set a fixed or iterative wake, ```Mesh.set_fixed_wake()``` or ```Mesh.set_iterative_wake()``` must be called, respectively. The wake should be set on the mesh before the mesh is passed to the solver and ```Solver.set_condition()``` is called as some wake types are dependent upon the flow condition and these are updated when ```Solver.set_condition()``` is called.

Below, the various types of wake models available in PyPan are described on a conceptual level. For how to implement a given wake model in the API, see [Meshes](mesh).

## Fixed Wake

This is the simplest wake model available in PyPan. It consists of a single, semi-infinite vortex filament extending from each vertex defining the Kutta edges on the body. The direction of the filaments is set and does not change throughout computation.

The direction of these trailing filaments may be set in the following ways:

### "freestream"

Each vortex filament extends opposite the freestream direction.

### "freestream_constrained"

Same as "freestream", except the direction of the filaments is constrained to be in the plane defined by the user-specified normal vector.

### "freestream_and_rotation"

Each vortex filament extends in the direction of the local velocity vector determined from the summation of the freestream velocity and the local velocity due to solid-body rotation of the body.

### "freestream_and_rotation_constrained"

Same as "freestream_and_rotation", except the direction of the filaments is constrained to be in the plane defined by the user-specified normal vector.

### "custom"

Each vortex filament extends in the same direction as specified by the user.

## Iterative Wake

This wake model allows for updating the shape of the wake iteratively to have the wake filaments follow the streamlines of the flow. This is done over multiple iterations of solving the potential flow about the body. Each filament is made up of a set of finite segments which change position with every iteration. In general, it is recommended that the number and length of the filament segments be specified such that the wake extends a significant distance behind the body. Setting ```"end_segment_infinite"``` can help with this, but should be used with judgement.
