# Helpful Hints

## MachUpX Mesh Generation
MachUpX can be used to create reasonable meshes for PyPan. The following advice should be followed:

* If NACA airfoils are being used, be sure to specify ```"NACA_closed_te" : True``` in the geometry dict for each NACA airfoil.
* For each wing under ```"CAD_options"```, be sure to specify ```"round_wing_tip"``` and ```"n_rounding_sections"```.
