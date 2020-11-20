# This script is for me to test the functionality of whatever I'm working on at the moment.
import machupX as MX
import pypan as pp
import json
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d

if __name__=="__main__":

    # Specify airplane
    airplane_dict = {
        "weight" : 50.0,
        "units" : "English",
        "airfoils" : {
            "NACA_0010" : {
                "geometry" : {
                    "NACA" : "0012",
                    "NACA_closed_te" : True
                }
            }
        },
        "wings" : {
            "main_wing" : {
                "ID" : 1,
                "side" : "both",
                "is_main" : True,
                "airfoil" : "NACA_0010",
                "semispan" : 1.5,
                "chord" : [[0.0, 5.0],
                           [0.05, 4.9],
                           [0.1, 4.5],
                           [0.15, 3.0],
                           [0.2, 1.5],
                           [1.0, 0.25]],
                "sweep" : [[0.0, 0.0],
                           [0.2, 80.0],
                           [0.2, 55.0],
                           [1.0, 55.0]],
                "grid" : {
                    "N" : 40,
                    "wing_ID" : 1,
                    "reid_corrections" : True,
                },
                "CAD_options" :{
                    "round_stl_tip" : True,
                    "round_stl_root" : False,
                    "n_rounding_sections" : 10,
                    "cluster_points" : [0.2]
                }
            }
        }
    }

    # Load scene
    state = {
        "velocity" : 100.0
    }
    scene = MX.Scene()
    scene.add_aircraft("plane", airplane_dict, state=state)
    stl_file = "dev/meshes/cool_body_7000.stl"
    scene.export_stl(filename=stl_file, section_resolution=41)