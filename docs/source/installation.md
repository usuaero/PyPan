# Installation

## Getting Python

If you do not have Python installed on your machine, it can be downloaded from a number of locations. We use [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/). Please be sure you have Python 3.6 or later.

## Getting the Source Code

PyPan is available on [Github](https://github.com/usuaero/PyPan).

You can either download the source as a ZIP file and extract the contents, or clone the PyPan repository using Git. If your system does not already have a version of Git installed, you will not be able to use this second option unless you first download and install Git. If you are unsure, you can check by typing `git --version` into a command prompt.

### Cloning the Github repository (recommended)

1. From the command prompt navigate to the directory where PyPan will be installed. Note: git will automatically create a folder within this directory called PyPan. Keep this in mind if you do not want multiple nested folders called PyPan.
2. Execute

    $ git clone https://github.com/usuaero/PyPan

Cloning from the repository allows you to most easily download and install periodic updates (because we will always be updating PyPan!). This can be done using the following command

    $ git pull

### Downloading source as a ZIP file (less recommended)

1. Open a web browser and navigate to [https://github.com/usuaero/PyPan](https://github.com/usuaero/PyPan)
2. Make sure the Branch is set to `Master`
3. Click the `Clone or download` button
4. Select `Download ZIP`
5. Extract the downloaded ZIP file to a local directory on your machine

## Installing

Once you have the source code downloaded, navigate to the root (PyPan/) directory and execute

    $ pip install .

Any time you update the source code (e.g. after executing a git pull), PyPan will need to be reinstalled by executing the above command.