# PAN AIR Capabilities
PyPan is being developed as a modern successor to PAN AIR, the legacy compressible panel code developed by NASA and Boeing. As such, a first step in the development of PyPan was to mimic limited PAN AIR behavior so as to understand the inner workings of a panel code. Not all the capabilities of PAN AIR are available in PyPan; however, it is capable of the following:

* Cool stuff.
* I'll flesh this out once it's actually capable of anything...

There are some minor capabilities written into PyPan which were not originally part of PAN AIR. These are

* Mesh plotting.

## PAN AIR Usage
PAN AIR is another module (```panair```) installed alongside PyPan. Since PAN AIR is a FORTRAN code, it is meant to be run from a single input file. As such, the script interface for PAN AIR in PyPan is limited. The code below shows this full interface:

```python
# PAN AIR 

from panair import Main

if __name__=="__main__":

    # Declare input file
    input_file = "dev/meshes/wingbody.INP"

    # Load case
    my_case = Main(input_file=input_file, verbose=True)

    # Plot mesh
    my_case.plot_mesh()

    # Execute
    my_case.execute_case(verbose=True)
```
It is not possible to modify the input parameters from the Python API. For creating PAN AIR input files, you are directed to the original PAN AIR documentation, freely available online from NASA.