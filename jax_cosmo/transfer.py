# This module contains various transfer functions from the literatu
import jax.numpy as np

import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const

__all__ = ["Eisenstein_Hu"]


def Eisenstein_Hu(cosmo, k, type="eisenhu_osc"):
    """Computes the Eisenstein & Hu matter transfer function.

    Parameters
    ----------
    cosmo: Background
      Background cosmology

    k: array_like
      Wave number in h Mpc^{-1}

    type: str, optional
      Type of transfer function. Either 'eisenhu' or 'eisenhu_osc'
      (def: 'eisenhu_osc')

    Returns
    -------
    T: array_like
      Value of the transfer function at the requested wave number

    Notes
    -----
    The Eisenstein & Hu transfer functions are computed using the fitting
    formulae of :cite:`1998:EisensteinHu`

    """
    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - k_eq   : scale of the particle horizon at equality epoch
    #            - z_eq   : redshift of equality epoch
    #            - R_eq   : ratio of the baryon to photon momentum density
    #                       at z_eq
    #            - z_d    : redshift of drag epoch
    #            - R_d    : ratio of the baryon to photon momentum density
    #                       at z_d
    #            - sh_d   : sound horizon at drag epoch
    #            - k_silk : Silk damping scale
    T_2_7_sqr = (const.tcmb / 2.7) ** 2
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2
    fb = cosmo.Omega_b / cosmo.Omega_m
    fc = (cosmo.Omega_m - cosmo.Omega_b) / cosmo.Omega_m

    k_eq = 7.46e-2 * w_m / T_2_7_sqr / cosmo.h  # Eq. (3) [h/Mpc]
    z_eq = 2.50e4 * w_m / (T_2_7_sqr) ** 2  # Eq. (2)

    # z drag from Eq. (4)
    b1 = 0.313 * np.power(w_m, -0.419) * (1.0 + 0.607 * np.power(w_m, 0.674))
    b2 = 0.238 * np.power(w_m, 0.223)
    z_d = (
        1291.0
        * np.power(w_m, 0.251)
        / (1.0 + 0.659 * np.power(w_m, 0.828))
        * (1.0 + b1 * np.power(w_b, b2))
    )

    # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
    R_d = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_d)
    # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
    R_eq = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_eq)
    # Sound horizon at drag epoch in h^-1 Mpc from arXiv:2106.00428v2 Eq. (10)
    sh_d = sh_d_AAN(cosmo)
    # Eq. (7) but in [hMpc^{-1}]
    k_silk = (
        1.6
        * np.power(w_b, 0.52)
        * np.power(w_m, 0.73)
        * (1.0 + np.power(10.4 * w_m, -0.95))
        / cosmo.h
    )
    #############################################

    alpha_gamma = (
        1.0
        - 0.328 * np.log(431.0 * w_m) * w_b / w_m
        + 0.38 * np.log(22.3 * w_m) * (cosmo.Omega_b / cosmo.Omega_m) ** 2
    )
    gamma_eff = (
        cosmo.Omega_m
        * cosmo.h
        * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * sh_d) ** 4))
    )

    if type == "eisenhu":
        q = k * np.power(const.tcmb / 2.7, 2) / gamma_eff

        # EH98 (29) #
        L = np.log(2.0 * np.exp(1.0) + 1.8 * q)
        C = 14.2 + 731.0 / (1.0 + 62.5 * q)
        res = L / (L + C * q * q)

    elif type == "eisenhu_osc":
        # Cold dark matter transfer function

        # EH98 (11, 12)
        a1 = np.power(46.9 * w_m, 0.670) * (1.0 + np.power(32.1 * w_m, -0.532))
        a2 = np.power(12.0 * w_m, 0.424) * (1.0 + np.power(45.0 * w_m, -0.582))
        alpha_c = np.power(a1, -fb) * np.power(a2, -(fb**3))
        b1 = 0.944 / (1.0 + np.power(458.0 * w_m, -0.708))
        b2 = np.power(0.395 * w_m, -0.0266)
        beta_c = 1.0 + b1 * (np.power(fc, b2) - 1.0)
        beta_c = 1.0 / beta_c

        # EH98 (19). [k] = h/Mpc
        def T_tilde(k1, alpha, beta):
            # EH98 (10); [q] = 1 BUT [k] = h/Mpc
            q = k1 / (13.41 * k_eq)
            L = np.log(np.exp(1.0) + 1.8 * beta * q)
            C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * np.power(q, 1.08))
            T0 = L / (L + C * q * q)
            return T0

        # EH98 (17, 18)
        f = 1.0 / (1.0 + (k * sh_d / 5.4) ** 4)
        Tc = f * T_tilde(k, 1.0, beta_c) + (1.0 - f) * T_tilde(k, alpha_c, beta_c)

        # Baryon transfer function
        # EH98 (19, 14, 21)
        y = (1.0 + z_eq) / (1.0 + z_d)
        x = np.sqrt(1.0 + y)
        G_EH98 = y * (-6.0 * x + (2.0 + 3.0 * y) * np.log((x + 1.0) / (x - 1.0)))
        alpha_b = 2.07 * k_eq * sh_d * np.power(1.0 + R_d, -0.75) * G_EH98

        beta_node = 8.41 * np.power(w_m, 0.435)
        tilde_s = sh_d / np.power(1.0 + (beta_node / (k * sh_d)) ** 3, 1.0 / 3.0)

        beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * w_m) ** 2 + 1.0)

        # [tilde_s] = Mpc/h
        Tb = (
            T_tilde(k, 1.0, 1.0) / (1.0 + (k * sh_d / 5.2) ** 2)
            + alpha_b
            / (1.0 + (beta_b / (k * sh_d)) ** 3)
            * np.exp(-np.power(k / k_silk, 1.4))
        ) * np.sinc(k * tilde_s / np.pi)

        # Total transfer function
        res = fb * Tb + fc * Tc
    else:
        raise NotImplementedError
    return res


def sh_d_Eisenstein_Hu(cosmo):
    """Computes the Eisenstein & Hu matter transfer function.

    Parameters
    ----------
    cosmo: Background
      Background cosmology

    Returns
    -------
    T: float
      Value of the transfer function at the requested wave number

    """
    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - k_eq   : scale of the particle horizon at equality epoch
    #            - z_eq   : redshift of equality epoch
    #            - R_eq   : ratio of the baryon to photon momentum density
    #                       at z_eq
    #            - z_d    : redshift of drag epoch
    #            - R_d    : ratio of the baryon to photon momentum density
    #                       at z_d
    #            - sh_d   : sound horizon at drag epoch
    #            - k_silk : Silk damping scale
    T_2_7_sqr = (const.tcmb / 2.7) ** 2
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2

    k_eq = 7.46e-2 * w_m / T_2_7_sqr / cosmo.h  # Eq. (3) [h/Mpc]
    z_eq = 2.50e4 * w_m / (T_2_7_sqr) ** 2  # Eq. (2)

    # z drag from Eq. (4)
    b1 = 0.313 * np.power(w_m, -0.419) * (1.0 + 0.607 * np.power(w_m, 0.674))
    b2 = 0.238 * np.power(w_m, 0.223)
    z_d = (
        1291.0
        * np.power(w_m, 0.251)
        / (1.0 + 0.659 * np.power(w_m, 0.828))
        * (1.0 + b1 * np.power(w_b, b2))
    )

    # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
    R_d = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_d)
    # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
    R_eq = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_eq)
    # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
    sh_d = (
        2.0
        / (3.0 * k_eq)
        * np.sqrt(6.0 / R_eq)
        * np.log((np.sqrt(1.0 + R_d) + np.sqrt(R_eq + R_d)) / (1.0 + np.sqrt(R_eq)))
    )
    return sh_d

def sh_d_EH_AAN(cosmo):
    """Computes the Eisenstein & Hu matter transfer function.

    Parameters
    ----------
    cosmo: Background
      Background cosmology

    Returns
    -------
    T: float
      Value of the transfer function at the requested wave number

    """
    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - k_eq   : scale of the particle horizon at equality epoch
    #            - z_eq   : redshift of equality epoch
    #            - R_eq   : ratio of the baryon to photon momentum density
    #                       at z_eq
    #            - z_d    : redshift of drag epoch
    #            - R_d    : ratio of the baryon to photon momentum density
    #                       at z_d
    #            - sh_d   : sound horizon at drag epoch
    #            - k_silk : Silk damping scale
    T_2_7_sqr = (const.tcmb / 2.7) ** 2
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2

    k_eq = 7.46e-2 * w_m / T_2_7_sqr / cosmo.h  # Eq. (3) [h/Mpc]
    z_eq = 2.50e4 * w_m / (T_2_7_sqr) ** 2  # Eq. (2)

    # z drag from arXiv:2106.00428 Eq. (A2)
    z_d = (
        (1+428.169*w_b**0.256459*w_m**0.616388+925.56*w_m**0.751615)/
        w_m**0.714129
    )

    # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
    R_d = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_d)
    # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
    R_eq = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_eq)
    # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
    sh_d = (
        2.0
        / (3.0 * k_eq)
        * np.sqrt(6.0 / R_eq)
        * np.log((np.sqrt(1.0 + R_d) + np.sqrt(R_eq + R_d)) / (1.0 + np.sqrt(R_eq)))
    )
    return sh_d

def sh_d_AAN_no_neutrino(cosmo):
    """Computes sound horizon at drag epoch from Aispuru et al. 2021 (arXiv:2106.00428v2).

    Parameters
    ----------
    cosmo: Background
      Background cosmology

    Returns
    -------
    sh_d: float
      Sound horizon at the baryon drag epoch, in h^{-1}Mpc

    """
    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - sh_d   : sound horizon at drag epoch
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2

    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473
    # r_s from arXiv:2106.00428v2 Eq. (8)
    sh_d = (
        (a1*w_b**a2
         +a3*w_b**a4*w_m**a5
         +a6*w_m**a7)**-1
         -a8/(w_m**a9)
    )*cosmo.h
    return sh_d

def sh_d_AAN(cosmo):
    """Computes sound horizon at drag epoch from Aispuru et al. 2021 (arXiv:2106.00428v2).

    Parameters
    ----------
    cosmo: Background
      Background cosmology

    Returns
    -------
    sh_d: float
      Sound horizon at the baryon drag epoch, in h^{-1}Mpc

    """
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2

    #Defaults to m_nu = 0.06 eV if m_nu is not specified in the given cosmology.
    m_nu = .06 #eV
    try:
        m_nu = cosmo.m_nu
    except:
        pass

    w_nu = .017*m_nu #Assumes m_nu is in units of [eV]

    a1 = 0.0034917
    a2 = -19.972694
    a3 = 0.000336186
    a4 = 0.0000305
    a5 = 0.22752
    a6 = 0.00003142567
    a7 = 0.5453798
    a8 = 374.14994
    a9 = 4.022356899
    # r_s from arXiv:2106.00428v2 Eq. (10)
    sh_d = (
        a1*np.exp(a2*(a3+w_nu)**2)
        /(a4*w_b**a5 + a6*w_m**a7 + a8*(w_b*w_m)**a9)
    )*cosmo.h
    return sh_d

def z_d_Eisenstein_Hu(cosmo):
    """Computes the Eisenstein & Hu matter transfer function.

    Parameters
    ----------
    cosmo: Background
      Background cosmology

    Returns
    -------
    T: float
      Value of the transfer function at the requested wave number

    """
    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - k_eq   : scale of the particle horizon at equality epoch
    #            - z_eq   : redshift of equality epoch
    #            - R_eq   : ratio of the baryon to photon momentum density
    #                       at z_eq
    #            - z_d    : redshift of drag epoch
    #            - R_d    : ratio of the baryon to photon momentum density
    #                       at z_d
    #            - sh_d   : sound horizon at drag epoch
    #            - k_silk : Silk damping scale
    T_2_7_sqr = (const.tcmb / 2.7) ** 2
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2

    k_eq = 7.46e-2 * w_m / T_2_7_sqr / cosmo.h  # Eq. (3) [h/Mpc]
    z_eq = 2.50e4 * w_m / (T_2_7_sqr) ** 2  # Eq. (2)

    # z drag from Eq. (4)
    b1 = 0.313 * np.power(w_m, -0.419) * (1.0 + 0.607 * np.power(w_m, 0.674))
    b2 = 0.238 * np.power(w_m, 0.223)
    z_d = (
        1291.0
        * np.power(w_m, 0.251)
        / (1.0 + 0.659 * np.power(w_m, 0.828))
        * (1.0 + b1 * np.power(w_b, b2))
    )
    return k_eq, z_eq, z_d