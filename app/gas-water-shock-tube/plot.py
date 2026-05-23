#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import configparser
#import StiffenedGas as sg

def do_plot(ini_filename1, ini_filename2, shock_tube_name):

    config1 = configparser.ConfigParser()
    config1.read(ini_filename1)

    tEnd = config1.getfloat('run', 'tEnd', fallback=0.2)
    level_min = config1.getint('amr', 'level_min', fallback=0)
    level_max = config1.getint('amr', 'level_max', fallback=0)

    # water
    gamma0 = config1.getfloat('material0', 'gamma', fallback=4.4)
    pinf0 = config1.getfloat('material0', 'pinf', fallback=6e3)

    #
    gamma1 = config1.getfloat('material1', 'gamma', fallback=1.4)
    pinf1 = config1.getfloat('material1', 'pinf', fallback=0)

    prefix1 = config1.get('two_fluid_shock_tube', 'name', fallback='')

    config2 = configparser.ConfigParser()
    config2.read(ini_filename2)

    prefix2 = config2.get('two_fluid_shock_tube', 'name', fallback='')

    #eos = sg.StiffenedGas(gamma0, gamma1, pinf0, pinf1)
    #compute_pressure = np.vectorize(eos.mixture_pressure)

    # load numerical solution
    pos1 = np.load(prefix1+'_positions.npy')
    rho1_0 = np.load(prefix1+'_alpha_rho_0.npy')
    rho1_1 = np.load(prefix1+'_alpha_rho_1.npy')
    rho1 = rho1_0+rho1_1
    etot1 = np.load(prefix1+'_e_tot.npy')
    alpha_0_1 = np.load(prefix1+'_alpha_0.npy')
    rho_vx1 = np.load(prefix1+'_rho_vx.npy')
    level1 = np.load(prefix1+'_level.npy')
    eint1 = etot1 / rho1 - 0.5 * rho_vx1**2 / rho1**2
    #P1 = compute_pressure(rho1, eint1, alpha_0_1)
    P1 = np.load(prefix1+'_pressure.npy')
    c1 = np.load(prefix1+'_speed_of_sound.npy')

    pos2 = np.load(prefix2+'_positions.npy')
    rho2_0 = np.load(prefix2+'_alpha_rho_0.npy')
    rho2_1 = np.load(prefix2+'_alpha_rho_1.npy')
    rho2 = rho2_0+rho2_1
    etot2 = np.load(prefix2+'_e_tot.npy')
    alpha_0_2 = np.load(prefix2+'_alpha_0.npy')
    rho_vx2 = np.load(prefix2+'_rho_vx.npy')
    level2 = np.load(prefix2+'_level.npy')
    eint2 = etot2 / rho2 - 0.5 * rho_vx2**2 / rho2**2
    #P2 = compute_pressure(rho2, eint2, alpha_0_2)
    P2 = np.load(prefix2+'_pressure.npy')
    c2 = np.load(prefix2+'_speed_of_sound.npy')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(11,13))

    ax1.plot(pos1, alpha_0_1, 'r-', label='alpha_0')
    ax2.plot(pos1, rho1, 'g-', label='rho')
    ax3.plot(pos1, P1, 'b-', label='pressure')
    #ax3.plot(pos1, c1, 'b-', label='speed of sound')
    #ax3.semilogy(pos1, P1, 'b-', label='pressure')
    ax4.plot(pos1, level1, 'k-', label='AMR levels')
    #ax4.plot(pos1, rho_vx1, 'k-', label='rho_u')

    ax1.plot(pos2, alpha_0_2, 'r--', label='alpha_0 thinc')
    ax2.plot(pos2, rho2, 'g--', label='rho thinc')
    ax3.plot(pos2, P2, 'b--', label='pressure thinc')
    #ax3.plot(pos2, c2, 'b--', label='speed of sound thinc')
    #ax3.semilogy(pos2, P2, 'b--', label='pressure thinc')
    ax4.plot(pos2, level2, 'k--', label='AMR levels thinc')
    #ax4.plot(pos2, rho_vx2, 'k--', label='rho_u thinc')

    tsize=14
    ax1.tick_params(labelsize=tsize)
    ax2.tick_params(labelsize=tsize)
    ax3.tick_params(labelsize=tsize)
    ax4.tick_params(labelsize=tsize)

    lsize=15
    ax1.legend(prop={'size':lsize})
    ax2.legend(prop={'size':lsize})
    ax3.legend(prop={'size':lsize})
    ax4.legend(prop={'size':lsize})
    plt.suptitle('{} at tEnd={}\nAMR levels {} to {}'.format(shock_tube_name+' shock tube', tEnd, level_min, level_max), fontsize=28)
    plt.show()

###############################################################################
if __name__ == "__main__":

    shock_tube_name="Gas-Water"
    parser = argparse.ArgumentParser(description='Display '+shock_tube_name+' plots.')
    parser.add_argument('--ini1', type=str, default='test_gas_water_shock_tube_2d.ini', help='ini parameter file 1')
    parser.add_argument('--ini2', type=str, default='test_gas_water_shock_tube_2d_thinc.ini', help='ini parameter file 2')
    args = parser.parse_args()

    do_plot(args.ini1, args.ini2, shock_tube_name)
