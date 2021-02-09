"""Mimics PAN AIR for a limited set of cases."""

import time

import math as m
import numpy as np

from pypan.pp_math import dist
from panair.mesh import Mesh


class Main:
    """Main class for mimicing PAN AIR functionality.

    Parameters
    ----------
    input_file : str
        Path to a PAN AIR input file.

    verbose : bool, optional
    """

    def __init__(self, **kwargs):

        # Get kwargs
        verbose = kwargs.get("verbose", False)

        # Load in case parameters
        if verbose:
            start_time = time.time()
            print("\nReading in case parameters...", end='', flush=True)
        self._load_params(**kwargs)
        if verbose:
            end_time = time.time()
            print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)
        
        # Load mesh
        if verbose:
            start_time = time.time()
            print("\nReading in mesh...", end='', flush=True)
        self.mesh = Mesh(**kwargs)
        if verbose:
            end_time = time.time()
            print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)


    def _load_params(self, **kwargs):
        # Loads the case parameters from the input file

        # Open file
        input_file = kwargs.get("input_file")
        with open(input_file, 'r') as input_handle:
            lines = input_handle.readlines()

        # Loop through case information
        for i, line in enumerate(lines):

            # Mach number
            if "=amach" in line:
                self.M = float(lines[i+1])

            # Angles of attack
            elif "=alpc" in line:
                self.alpha_c = float(lines[i+1])
            elif "=alpha(0)" in line:
                self.alpha = float(lines[i+1])

            # Sideslip angles
            elif "=betc" in line:
                self.beta_c = float(lines[i+1])
            elif "=beta(0)" in line:
                self.beta = float(lines[i+1])

            # Reference parameters
            elif "=sref" in line:
                ref_info = lines[i+1].split()
                self.S_ref = float(ref_info[0])
                self.b_ref = float(ref_info[1])
                self.c_ref = float(ref_info[2])
                self.d_ref = float(ref_info[3])

            # Exit
            elif "$POINTS" in line:
                break

        # Check for transonic case
        if abs(self.M-1.0)<1e-10:
            raise IOError("Transonic cases are not allowed in PAN AIR. Got M={0}.".format(self.M))


    def execute_case(self, verbose=False):
        """Executes the case as specified in the input file.

        Parameters
        ----------
        verbose : bool, optional
        """
        
        # Calculate transforms
        if verbose:
            start_time = time.time()
            print("\nCalculating panel coordinate transforms...", end='', flush=True)
        self._calc_transforms()
        if verbose:
            end_time = time.time()
            print("Finished. Time: {0} s.".format(end_time-start_time), flush=True)


    def _calc_transforms(self):
        # Calculates the various transforms between coordinate systems

        # Determine s and B
        quant = 1.0-self.M**2
        self._s = quant/abs(quant)
        self._B = m.sqrt(self._s*quant)

        # Determine compressibility transformation matrix
        S_a = m.sin(self.alpha_c)
        C_a = m.cos(self.alpha_c)
        S_B = m.sin(self.beta_c)
        C_B = m.cos(self.beta_c)
        self._gamma_c = np.array([[C_a*C_B, -S_B, S_a*C_B],
                                  [C_a*S_B, C_B, S_a*S_B],
                                  [-S_a, 0.0, C_a]])

        # Store compressibility direction
        self._c_0 = self._gamma_c[0]

        # Calculate metric matrices
        self._C_0 = self._s*self._B*np.eye(3)+(1.0-self._s*self._B)*np.einsum('i,j->ij', self._c_0, self._c_0)
        self._B_0 = np.eye(3)+(self._s*self._B-1.0)*np.einsum('i,j->ij', self._c_0, self._c_0)

        # Set up local coordinate systems
        self.mesh.calc_local_coords(c_0=self._c_0, B_0=self._B_0, C_0=self._C_0, s=self._s, B=self._B, M=self.M)


    def plot_mesh(self):
        """Plots the input mesh."""
        self.mesh.plot()