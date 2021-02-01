# For testing the PAN AIR capabilities.

from pypan.panair import PANAIRMain

if __name__=="__main__":

    # Declare input file
    #input_file = "dev/meshes/axiebody.INP"
    input_file = "dev/meshes/wingbody.INP"

    # Load case
    my_case = PANAIRMain(input_file=input_file)
    my_case.execute_case()