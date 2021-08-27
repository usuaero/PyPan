import machupX as MX
import pypan as pp
import json
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh

if __name__=="__main__":
    
    # Specify input
    input_dict = {
        "solver" : {
            "type" : "nonlinear"
        },
        "units" : "English",
        "scene" : {
        }
    }

    # Specify airplane
    airplane_dict = {
        "weight" : 50.0,
        "units" : "English",
        "airfoils" : {
            "NACA_0010" : {
                #"CLa" : 6.808347851,
                "geometry" : {
                    "NACA" : "0002"
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
                "chord" : [[0.0, 1.0],
                           [1.0, 1.0]],
                "sweep" : [[0.0, 45.0],
                           [1.0, 45.0]],
                "grid" : {
                    "N" : 30
                },
                "CAD_options" :{
                    "round_wing_tip" : True,
                    "round_wing_root" : False,
                    "n_rounding_sections" : 30
                }
            }
        }
    }

    # Specify state
    state = {
        "position" : [0.0, 0.0, 0.0],
        "velocity" : [100.0, 0.0, 10],
        "orientation" : [0.0, 0.0, 0.0]
    }

    # Load scene
    scene = MX.Scene(input_dict)
    scene.add_aircraft("plane", airplane_dict, state=state)
    FM = scene.solve_forces(non_dimensional=False, verbose=True)["plane"]["total"]
    print(json.dumps(FM, indent=4))

    # Export vtk
    vtk_file = "dev/meshes/mux_compare.vtk"
    scene.export_vtk(filename=vtk_file, section_resolution=81)

    # Generate mesh
    my_mesh = pp.Mesh(name='mux_compare', mesh_file=vtk_file, verbose=True)
    #my_mesh.export_vtk("swept_wing_pp.vtk")
    my_mesh.set_wake(type='fixed', kutta_angle=90.0)

    # Initialize solver
    solver = pp.VortexRingSolver(mesh=my_mesh, verbose=True)
    solver.set_condition(V_inf=[-100.0, 0.0, -10.0], rho=0.0023769)

    # Solve
    F, M = solver.solve(lifting=True, verbose=True)
    print("Force vector: {0}".format(F))
    print("Moment vector: {0}".format(M))

    # Export results
    solver.export_vtk(vtk_file.replace("meshes", "results"))

    # Comparison
    print()
    print("{0:<5}{1:<25}{2:<25}{3:<25}".format("", "PyPan", "MachUpX", "% Error"))
    print("".join(["-"]*80))
    print("{0:<5}{1:<25.8f}{2:<25.8f}{3:<25.8f}".format("Fx", F[0], FM['Fx'], 100.0*abs((F[0]-FM['Fx'])/F[0])))
    print("{0:<5}{1:<25.8f}{2:<25.8f}{3:<25.8f}".format("Fy", F[1], FM['Fy'], 100.0*abs((F[1]-FM['Fy'])/F[1])))
    print("{0:<5}{1:<25.8f}{2:<25.8f}{3:<25.8f}".format("Fz", F[2], FM['Fz'], 100.0*abs((F[2]-FM['Fz'])/F[2])))
    print("{0:<5}{1:<25.8f}{2:<25.8f}{3:<25.8f}".format("Mx", M[0], FM['Mx'], 100.0*abs((M[0]-FM['Mx'])/M[0])))
    print("{0:<5}{1:<25.8f}{2:<25.8f}{3:<25.8f}".format("My", M[1], FM['My'], 100.0*abs((M[1]-FM['My'])/M[1])))
    print("{0:<5}{1:<25.8f}{2:<25.8f}{3:<25.8f}".format("Mz", M[2], FM['Mz'], 100.0*abs((M[2]-FM['Mz'])/M[2])))
