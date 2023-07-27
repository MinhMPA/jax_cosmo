import jax.numpy as np
from jax.experimental.ode import odeint
from jax.tree_util import register_pytree_node_class

import jax_cosmo.constants as const
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a

__all__ = ["Cosmology"]


@register_pytree_node_class
class Cosmology:
    def __init__(self, Omega_m, Omega_c, Omega_b, h, n_s, sigma8, S8, Omega_k, w0, wa, gamma0=None, gamma1=None):
        """
        Cosmology object, stores primary and derived cosmological parameters.

        Parameters:
        -----------
        Omega_m, float
          Total matter density fraction.
        Omega_c, float
          Cold dark matter density fraction.
        Omega_b, float
          Baryonic matter density fraction.
        h, float
          Hubble constant divided by 100 km/s/Mpc; unitless.
        n_s, float
          Primordial scalar perturbation spectral index.
        sigma8, float
          Variance of matter density perturbations at an 8 Mpc/h scale
        S8, float
          sigma8*(Omega_m/0.3)**0.5
        Omega_k, float
          Curvature density fraction.
        w0, float
          First order term of dark energy equation
        wa, float
          Second order term of dark energy equation of state
        gamma0: float
          Index of the growth rate (optional), does not scale with z
        gamma1: float
          Index of the growth rate (optional), scales as z^2/(1+z)

        Notes:
        ------

        If `gamma` is specified, the emprical characterisation of growth in
        terms of  dlnD/dlna = \omega^\gamma will be used to define growth throughout.
        Otherwise the linear growth factor and growth rate will be solved by ODE.

        """
        # Store primary parameters
        self._Omega_m = Omega_m
        self._Omega_c = Omega_c
        self._Omega_b = Omega_b
        self._h = h
        self._n_s = n_s
        self._sigma8 = sigma8
        self._S8 = S8
        self._Omega_k = Omega_k
        self._w0 = w0
        self._wa = wa

        self._flags = {}

        # Secondary optional parameters
        self._gamma0 = gamma0
        self._gamma1 = gamma1
        self._flags["gamma_growth"] = gamma0 is not None

        # Create a workspace where functions can store some precomputed
        # results
        self._workspace = {}

    def __str__(self):
        return (
            "Cosmological parameters: \n"
            + "    h:        "
            + str(self.h)
            + " \n"
            + "    Omega_m:  "
            + str(self.Omega_m)
            + " \n"
            + "    Omega_c:  "
            + str(self.Omega_c)
            + " \n"
            + "    Omega_b:  "
            + str(self.Omega_b)
            + " \n"
            + "    Omega_k:  "
            + str(self.Omega_k)
            + " \n"
            + "    w0:       "
            + str(self.w0)
            + " \n"
            + "    wa:       "
            + str(self.wa)
            + " \n"
            + "    n:        "
            + str(self.n_s)
            + " \n"
            + "    sigma8:   "
            + str(self.sigma8)
            + " \n"
            + "    S8:   "
            + str(self.S8)
        )

    def __repr__(self):
        return self.__str__()

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        params = (
            self._Omega_m,
            self._Omega_c,
            self._Omega_b,
            self._h,
            self._n_s,
            self._sigma8,
            self._S8,
            self._Omega_k,
            self._w0,
            self._wa,
        )

        if self._flags["gamma_growth"]:
            params += (self._gamma0,self._gamma1)

        return (
            params,
            self._flags,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Retrieve base parameters
        Omega_m, Omega_c, Omega_b, h, n_s, sigma8, S8, Omega_k, w0, wa = children[:10]
        children = list(children[10:])
        children.reverse()

        # We extract the remaining parameters in reverse order from how they
        # were inserted
        if aux_data["gamma_growth"]:
            gamma0, gamma1 = children[-2:]
        else:
            gamma0 = None
            gamma1 = None

        return cls(
            Omega_m=Omega_m,
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            n_s=n_s,
            sigma8=sigma8,
            S8=S8,
            Omega_k=Omega_k,
            w0=w0,
            wa=wa,
            gamma0=gamma0,
            gamma1=gamma1,
        )

    # Cosmological parameters, base and derived
    @property
    def Omega(self):
        return 1.0 - self._Omega_k

    @property
    def Omega_m(self):
        return self._Omega_m

    @property
    def Omega_c(self):
        return self._Omega_m - self._Omega_b

    @property
    def Omega_b(self):
        return self._Omega_b

    @property
    def Omega_de(self):
        return self.Omega - self.Omega_m

    @property
    def Omega_k(self):
        return self._Omega_k

    @property
    def k(self):
        return -np.sign(self._Omega_k).astype(np.int8)

    @property
    def sqrtk(self):
        return np.sqrt(np.abs(self._Omega_k))

    @property
    def h(self):
        return self._h

    @property
    def w0(self):
        return self._w0

    @property
    def wa(self):
        return self._wa

    @property
    def n_s(self):
        return self._n_s

    @property
    def sigma8(self):
        return self._sigma8

    @property
    def S8(self):
        return self._sigma8*(self._Omega_m/0.30)**0.5

    @property
    def gamma0(self):
        return self._gamma0
    @property
    def gamma1(self):
        return self._gamma1
