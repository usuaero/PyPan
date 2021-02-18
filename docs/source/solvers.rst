Solvers
=======

There are various solvers written into PyPan. Each is an instance of the Solver base class, which has the following available methods:

.. automodule:: pypan.solvers
.. autoclass:: Solver
   :members:

Vortex Ring (Constant Doublet) Solver
-------------------------------------

Uses vortex ring panels to solve for zero normal flow at the centroid of each panel. This is an incompressible method. Both lifting and nonlifting flow can be modeled by specifying a Kutta angle for the mesh.

.. automodule:: pypan
.. autoclass:: VortexRingSolver
   :members:

Newton's Method Solver
----------------------

Uses Newton's method (a local inclination method) for solving inviscid, hypersonic flow about a body. May use either the original or modified method.

.. automodule:: pypan
.. autoclass:: NewtonsSolver
   :members: