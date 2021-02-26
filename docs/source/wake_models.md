# Wake Models

As with all panel solvers, the accuracy of the flow results is dependent upon appropriately modeling the wake of the body. Various wake models are available in PyPan, as described below.

The inclusion of a wake in PyPan is dependent upon the mesh having Kutta edges. If the mesh has no Kutta edges, no wake elements will be generated.

**CURRENTLY "FIXED" IS THE ONLY AVAILABLE WAKE MODEL**

## Fixed Wake

This is the simplest wake model available in PyPan. It consists of a single, semi-infinite vortex filament extending from each vertex defining the Kutta edges on the body. The direction of the filaments is set and does not change throughout computation. The direction of these trailing filaments may be set in the following ways:

### "fixed"

Each vortex filament extends in the same direction as specified by the user.

### "freestream"

Each vortex filament extends opposite the freestream direction.

### "freestream_constrained"

Same as "freestream", except the direction of the filaments is constrained to be in the plane defined by the user-specified normal vector.

### "freestream_and_rotation"

Each vortex filament extends in the direction of the local velocity vector determined from the summation of the freestream velocity and the local velocity due to solid-body rotation of the body.

### "freestream_and_rotation_constrained"

Same as "freestream_and_rotation", except the direction of the filaments is constrained to be in the plane defined by the user-specified normal vector.

## Iterative Wake

This wake model allows for updating the shape of the wake iteratively to enforce zero force acting on the wake filaments.
