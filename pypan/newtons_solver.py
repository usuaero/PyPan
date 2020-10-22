"""Classes for Newton's method and the modified Newton's method for inviscid hypersonic flow."""

import numpy as np

from .solvers import Solver
from .pp_math import norm, vec_inner

class NewtonsSolver(Solver):
    """Solves the aerodynamics using Newton's method."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set maximum pressure coef
        self._C_P_max = 2.0


    def set_condition(self, **kwargs):
        """Sets the atmospheric conditions for the computation.

        V_inf : list
            Freestream velocity vector.
        """

        # Get freestream params
        self._v_inf = np.array(kwargs["V_inf"])
        self._V_inf = norm(self._v_inf)
        self._u_inf = self._v_inf/self._V_inf


    def solve(self, **kwargs):
        """Solves the panel equations to determine the flow field around the mesh.

        Parameters
        ----------
        verbose : bool, optional
        """

        # Determine panel sines
        S_theta = vec_inner(self._u_inf, self._n)

        # Calculate pressure coefficient
        self._C_P = np.where(S_theta>=0.0, self._C_P_max*S_theta**2, 0.0)

        return None


class ModifiedNewtonsSolver(NewtonsSolver):
    """Solves the aerodynamics using the modified Newton's method."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def set_condition(self, **kwargs):
        """Sets the atmospheric conditions for the computation.

        V_inf : list
            Freestream velocity vector.
        
        a_inf : float
            Freestream speed of sound.

        spec_heat_ratio : float
            Ratio of specific heats.
        """

        # Get freestream params
        self._v_inf = np.array(kwargs["V_inf"])
        self._V_inf = norm(self._v_inf)
        self._u_inf = self._v_inf/self._V_inf
        self._a_inf = kwargs.get("a_inf")
        self._M_inf = self._V_inf/self._a_inf
        self._g = kwargs.get("spec_heat_ratio")

        # Determine maximum pressure coefficient
        A = ((self._g+1.0)**2*self._M_inf**2/(4.0*self._g*self._M_inf**2-2.0*(self._g-1.0)))**(self._g/(self._g-1.0))
        B = (1.0-self._g+2.0*self._g*self._M_inf**2)/(self._g+1.0)
        self._C_P_max = 2.0/(self._g*self._M_inf**2)*(A*B-1.0)