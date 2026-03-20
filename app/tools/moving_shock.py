# -*- coding: utf-8 -*-

import numpy as np

#
# Goal: compute post-shock state knowing the pre-shock state and the shock Mach number.
#
# To do that one must use the Rankine-Hugoniot relations.
# In the following drawing, we use the same notation as in wikipedia page `Moving shock`
#
# In the laboratory frame:
#
#   post-choc             pre-choc
#
#                u1+W
#                --->
#     rho2        |         rho1
#     u2          |         u1
#     P2          |         P1
#
# In the frame moving with the shock:
#
#   post-choc             pre-choc
#
#                 0
#                ---
#     rho2        |         rho1
#     uy=W+u1-u2  |         ux = W
#     P2          |         P1
#
#     <------            <-------
#
# in the frame moving with shock, the is stationary.
#
# The shock Mach number is defined as M=W/c2 where c2 is the speed of sound in pre-shock state.
#
# references:
#
# - Moving shock (take care that state 1 and 2 are inverted compared to
#   Marc Buffat's convention):
#   https://en.wikipedia.org/wiki/Moving_shock
# - Rankine-Hugoniot (in French):
#   https://perso.univ-lyon1.fr/marc.buffat/COURS/BOOK_MECAFLU_HTML/AERO/chap4.html#relations-a-travers-un-choc-droit
# - Piston-generated expansion wave:
#   https://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node201.html
#
# Input:
# - shock Mach number Mx is known,
# - pre-shock (rho2, u2, P2) state is known
#
# Output:
# - post-shock state
# - bonus: pressure inside bubble at mechanical and thermal equilibrium with pre-shock gas
#
#
# This script is useful to cross-check initial condition for the multi-bubble
# test case presented in reference:
#
# Triangular metric-based mesh adaptation for compressible multi-material flows in semi-Lagrangian coordinates
# Stéphane Del Pino and Isabelle Marmajou
# Journal of Computational Physics, Volume 478, 1 April 2023, 111975
# https://doi.org/10.1016/j.jcp.2023.111975
# See in section 4.5

def sound_speed(gamma, P, rho):
    """Compute speed of sound.

    Parameters
    ----------
    gamma : real
            specific heat ratio
    P : real
        pressure
    rho : real
          density
    """
    return np.sqrt(gamma * P / rho)

def compute_p2(p1, gamma, Mx):
    """Compute pressure in post-shock region.

    Parameters
    ----------
    p1 : real
         pressure in pre-shock region
    gamma : real
            specific heat ratio
    Mx : real
         Mach number in the pre-shock region (stationary shock)
    """
    return p1 * (1 + 2 * gamma / (gamma+1) * (Mx**2 - 1))

def compute_rho2(rho1, gamma, Mx):
    """Compute density in post-shock region.

    Parameters
    ----------
    rho1 : real
         density in pre-shock region
    gamma : real
            specific heat ratio
    Mx : real
         Mach number in the pre-shock region (stationary shock)
    """
    return rho1 / (1.0 - 2.0 / (gamma+1) * (1.0 - 1.0 / Mx**2))

#def compute_u1(gamma, u2, M1):
#    return u2 * (gamma+1) * M1**2 / ( (gamma-1) * M1**2 + 2)

def compute_u1_over_u2(gamma, Mx):
    """Compute velocity ratio (pre over post-shock region velocities) in stationary frame.

    Parameters
    ----------
    gamma : real
            specific heat ratio
    Mx : real
         Mach number in the pre-shock region (stationary shock)
    """
    return ((gamma-1)*Mx**2+2)/((gamma+1)*Mx**2)

def compute_speed_of_sound(a, gamma, Mx):
    """Compute speed of sound in post-shock region.

    Parameters
    ----------
    a : real
        speed of sound in pre-shock region
    gamma : real
            specific heat ratio
    Mx : real
         Mach number in the pre-shock region (stationary shock)
    """
    return a * np.sqrt(1 + 2*(gamma-1)/(gamma+1)**2 * (gamma*Mx**2 - 1.0/Mx**2 - (gamma-1)) )

def compute_post_shock_state(rho1, u1, P1, gamma, Mx):
    """Compute post-shock state knowing the pre-shock state.

    Parameters
    ----------
    rho1 : real
           pre-shock state density
    u1 : real
           pre-shock state velocity
    P1 : real
           pre-shock state pressure
    gamma : real
            specific heat ratio
    Mx : real
         Mach number in the pre-shock region (stationary shock)
    """
    a1 = np.sqrt(gamma * P1 / rho1)
    W = Mx * a1

    P2 = compute_p2(P1, gamma, Mx)
    rho2 = compute_rho2(rho1, gamma, Mx)
    a2 = compute_speed_of_sound(a1, gamma, Mx)

    u2 = W * (1 - compute_u1_over_u2(gamma, Mx))

    print("pre-shock state is rho1={}, u1={}, P1={}".format(rho1, u1, P1))
    print("shock speed W={}".format(W))
    print("post-shock state is rho2={}, u2={}, P2={}".format(rho2, u2, P2))

def compute_bubble_density(rho1, P1, gamma1, gamma2, cv1, cv2):
    """Compute bubble density assuming thermal and mechanical equilibrium with surrounding gas.

    Ideal gas law: P = rho * Rs * T
    where P is pressure, Rs=R/M is specific gas constant, M is the molar mass and T temperature

    At mechanical and thermal equilibrium P1 = P2 and T1 = T2 thus rho1 * Rs1 = rho2 * Rs2

    Parameters
    ----------
    rho1 : real
           surrounding gas density
    P1 : real
           surrounding gas pressure
    gamma1 : real
             surrounding gas specific heat ratio
    gamma2 : real
             bubble gas specific heat ratio
    """
    # perfect gas constant
    R = 8.314

    # specific gas constant
    Rs1 = (gamma1-1)*cv1
    Rs2 = (gamma2-1)*cv2

    rho2 = rho1 * Rs1 / Rs2
    print("bubble density: {}".format(rho2))

###############################################################################
if __name__ == "__main__":

    # pre-shock is air at rest, so in the lab frame:
    # air density and pressure are taken from
    # Finite-volume WENO scheme for viscous compressible multicomponent flows
    # Coralic and Colonius, Journal of Computational Physics
    # Volume 274, 1 October 2014, Pages 95-121
    # https://doi.org/10.1016/j.jcp.2014.06.003
    rho1 = 1.204
    u1 = 0.0
    P1 = 101325

    # shock wave Mach number
    Mx = 1.22

    # specific heat at constant volume and specific heat ratio (air)
    cv = 0.72
    gamma = 1.4

    # specific heat at constant volume and specific heat ratio (helium)
    #cv2 = 3.11
    #gamma2 = 1.67

    # specific heat at constant volume and specific heat ratio (helium + 28% air)
    cv2 = 2.44
    gamma2 = 1.648

    # specific heat at constant volume and specific heat ratio (R22 refrigerant)
    #cv2 = 0.365
    #gamma2 = 1.249

    compute_post_shock_state(rho1, u1, P1, gamma, Mx)
    compute_bubble_density(rho1, P1, gamma, gamma2, cv, cv2)
