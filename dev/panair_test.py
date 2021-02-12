# For testing the PAN AIR capabilities.

from panair.main import Main

if __name__=="__main__":

    # Declare input file
    #input_file = "dev/meshes/axiebody.INP"
    input_file = "dev/meshes/wingbody.INP"

    # Load case
    my_case = Main(input_file=input_file, verbose=True)
    my_case.plot_mesh()
    my_case.execute_case(verbose=True)