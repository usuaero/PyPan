import json

import numpy as np
import machupX as MX
import pypan as pp
import subprocess as sp
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

if __name__=="__main__":
    
    # Specify input
    input_dict = {
        "solver" : {
            "type" : "nonlinear"
        },
        "units" : "English",
        "scene" : {
            "atmosphere" : {
            }
        }
    }

    # Specify airplane
    airplane_dict = {
        "weight" : 50.0,
        "units" : "English",
        "airfoils" : {
            "NACA_0010" : {
                "geometry" : {
                    "NACA" : "0010",
                    "NACA_closed_te" : True
                }
            }
        },
        "plot_lacs" : False,
        "wings" : {
            "main_wing" : {
                "ID" : 1,
                "side" : "both",
                "is_main" : True,
                "airfoil" : [[0.0, "NACA_0010"],
                             [1.0, "NACA_0010"]],
                "semispan" : 6.0,
                "dihedral" : [[0.0, 0.0],
                              [1.0, 0.0]],
                "chord" : [[0.0, 1.0],
                           [1.0, 1.0]],
                "sweep" : [[0.0, 45.0],
                           [1.0, 45.0]],
                "grid" : {
                    "N" : 40
                },
                "CAD_options" :{
                    "round_wing_tip" : True,
                    "round_wing_root" : False,
                    "n_rounding_sections" : 20
                }
            }
        }
    }

    # Specify state
    state = {
        "velocity" : [100.0, 0.0, 10]
    }

    # Load scene for stl
    scene = MX.Scene(input_dict)
    scene.add_aircraft("plane", airplane_dict, state=state)
    FM = scene.solve_forces(non_dimensional=False, verbose=True)["plane"]["total"]
    print(json.dumps(FM, indent=4))

    # Export stl
    stl_file = "dev/meshes/swept_wing_stl.stl"
    scene.export_stl(filename=stl_file, section_resolution=51)

    # Load scene for vtk
    scene = MX.Scene(input_dict)
    airplane_dict["wings"]["main_wing"]["grid"]["N"] = 55
    scene.add_aircraft("plane", airplane_dict, state=state)
    FM = scene.solve_forces(non_dimensional=False, verbose=True)["plane"]["total"]
    print(json.dumps(FM, indent=4))

    # Export vtk
    vtk_file = "dev/meshes/swept_wing_vtk.vtk"
    scene.export_vtk(filename=vtk_file, section_resolution=71)

    # Run PyPan with vtk mesh
    print()
    print("VTK Mesh")
    mesh = pp.Mesh(mesh_file=vtk_file, mesh_file_type="VTK", kutta_angle=90.0, verbose=True)
    solver = pp.VortexRingSolver(mesh=mesh, verbose=True)
    solver.set_condition(V_inf=[-100.0, 0.0, -10.0], rho=0.0023769)
    F, M = solver.solve(lifting=True, verbose=True)
    print()
    print("Force Vector: {0}".format(F))
    print("Moment Vector: {0}".format(M))
    print("Max Pressure Coef: {0}".format(np.max(solver._C_P)))
    print("Min Pressure Coef: {0}".format(np.min(solver._C_P)))
    solver.export_vtk(vtk_file.replace("meshes", "results"))

    # Run PyPan with stl mesh
    print()
    print("STL Mesh")
    mesh = pp.Mesh(mesh_file=stl_file, mesh_file_type="STL", kutta_angle=90.0, verbose=True)
    #mesh.plot()
    solver = pp.VortexRingSolver(mesh=mesh, verbose=True)
    solver.set_condition(V_inf=[-100.0, 0.0, -10.0], rho=0.0023769)
    F, M = solver.solve(lifting=True, verbose=True)
    print()
    print("Force Vector: {0}".format(F))
    print("Moment Vector: {0}".format(M))
    print("Max Pressure Coef: {0}".format(np.max(solver._C_P)))
    print("Min Pressure Coef: {0}".format(np.min(solver._C_P)))
    solver.export_vtk(stl_file.replace("meshes", "results").replace(".stl", ".vtk"))