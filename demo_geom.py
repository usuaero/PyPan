import machupX as mx

if __name__=="__main__":

    # MachUpX input
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
        "controls" : {
            "aileron" : {
                "is_symmetric" : False
            }
        },
        "wings" : {
            "main_wing" : {
                "ID" : 1,
                "side" : "both",
                "is_main" : True,
                "semispan" : 3.0,
                "airfoil" : "NACA_0010",
                "chord" : [[0.0, 2.0],
                           [1.0, 1.0]],
                "sweep" : [[0.0, 45.0],
                           [1.0, 45.0]],
                "dihedral" : [[0.0, 0.0],
                              [0.1, 20.0],
                              [1.0, 20.0]],
                "grid" : {
                    "N" : 40,
                },
                "control_surface" : {
                    "root_span" : 0.2,
                    "tip_span" : 0.8,
                    "chord_fraction" : 0.3,
                    "control_mixing" : {
                        "aileron" : 1.0
                    }
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
    control_state = {
        "aileron" : 0.0
    }
    scene = mx.Scene()
    scene.add_aircraft("plane", airplane_dict, state=state, control_state=control_state)
    vtk_file = "mesh.vtk"
    scene.export_vtk(filename=vtk_file, section_resolution=41)