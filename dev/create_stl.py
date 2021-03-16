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
                    "NACA" : "0010",
                    "NACA_closed_te" : True
                }
            },
            "NACA_2410" : {
                "geometry" : {
                    "NACA" : "2410",
                    "NACA_closed_te" : True
                }
            }
        },
        "wings" : {
            "main_wing" : {
                "ID" : 1,
                "side" : "both",
                "is_main" : True,
                "semispan" : 3.0,
                "airfoil" : "NACA_2410",
                "chord" : [[0.0, 2.0],
                           [1.0, 1.0]],
                "sweep" : [[0.0, 45.0],
                           [1.0, 45.0]],
                "grid" : {
                    "N" : 30,
                },
                "CAD_options" :{
                    "round_wing_tip" : True,
                    "round_wing_root" : False,
                    "n_rounding_sections" : 20
                }
            },
            "h_stab" : {
                "ID" : 2,
                "side" : "both",
                "is_main" : False,
                "connect_to" : {
                    "ID" : 1,
                    "location" : "root",
                    "dx" : -4.0,
                },
                "semispan" : 1.0,
                "chord" : [[0.0, 0.8],
                           [1.0, 0.5]],
                "sweep" : [[0.0, 25.0],
                           [1.0, 25.0]],
                "grid" : {
                    "N" : 20,
                },
                "CAD_options" :{
                    "round_wing_tip" : True,
                    "round_wing_root" : False,
                    "n_rounding_sections" : 20
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
    stl_file = "dev/meshes/swept_wing_and_tail.stl"
    scene.export_stl(filename=stl_file, section_resolution=41)