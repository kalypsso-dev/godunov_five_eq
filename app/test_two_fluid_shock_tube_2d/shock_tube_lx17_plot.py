#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import configparser
import subprocess

def shock_tube_plot(ini_filename):

    config = configparser.ConfigParser()
    config.read(ini_filename)

    tEnd=config.getfloat('run','tEnd', fallback=0.2)
    prefix=config.get('output','outputPrefix',fallback='shock_tube')

    if (config['run']['dimension'] == '2'):
        # doing 2d
        dim=2
    else:
        print("Error: only 2D supported here")
        return

    alphaL = config.getfloat('region0','alpha0',fallback=1.0)
    alpha_rhoL = config.get('region0','alpha_rho', fallback="1.0,0.0")
    rhoL = alphaL*float(alpha_rhoL.split(',')[0])+(1.0-alphaL)*float(alpha_rhoL.split(',')[1])
    pL = config.getfloat('region0', 'p', fallback=1.0)
    uL = config.getfloat('region0', 'u', fallback=0.0)

    alphaR = config.getfloat('region1','alpha0',fallback=0.0)
    alpha_rhoR = config.get('region1', 'alpha_rho', fallback="0.0,1.0")
    rhoR = alphaR*float(alpha_rhoR.split(',')[0])+(1.0-alphaR)*float(alpha_rhoR.split(',')[1])
    pR = config.getfloat('region1', 'p', fallback=0.1)
    uR = config.getfloat('region1', 'u', fallback=0.0)

    print('shock_tube args : dim={} tEnd={}'.format(dim,tEnd))
    print('shock_tube left state: rhoL={} pL={} uL={}'.format(rhoL,pL,uL))
    print('shock_tube right state: rhoR={} pR={} uR={}'.format(rhoR,pR,uR))

    # load numerical solution
    #shock_tube_num=np.loadtxt('shock_tube_numerical_solution.csv', skiprows=1, delimiter=',',usecols=(1,2,6))
    #shock_tube_num_x = shock_tube_num[:,2]
    #shock_tube_num_rho = shock_tube_num[:,1]
    #shock_tube_num_level = shock_tube_num[:,0]

    shock_tube_num_x = np.load(prefix+'_positions.npy')
    shock_tube_num_rho = np.load(prefix+'_rho_mix.npy')
    shock_tube_num_p = np.load(prefix+'_thermal_pressure.npy')
    shock_tube_num_rho_vx = np.load(prefix+'_rho_vx.npy')
    shock_tube_num_vx = shock_tube_num_rho_vx/shock_tube_num_rho
    shock_tube_num_level = np.load(prefix+'_level.npy')

    time_integrator = config.get('amr','time_integrator',fallback='unknown')
    slope_type = config.get('hydro','slope_type',fallback=-1)
    order = -1
    if time_integrator=='RK1':
        if slope_type==0:
            order = 1
    elif time_integrator=='RK2_SSP' or time_integrator=='HANCOCK':
        order = 2

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(8,16))
    #ax1 = plt.subplot(2,1,1)
    #ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax1.plot(shock_tube_num_x, shock_tube_num_rho, 'xb-', label='rho')
    ax2.plot(shock_tube_num_x, shock_tube_num_vx, 'xb-', label='velocity')
    ax3.plot(shock_tube_num_x, shock_tube_num_p, 'xb-', label='pressure')
    ax4.plot(shock_tube_num_x, shock_tube_num_level, 'xb-', label='AMR levels')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.suptitle('Shock-tube:\n LX17 with Mie-Gruneisen JWL at tEnd={}\n Godunov (5 equations) MUSCL-{} order {}'.format(tEnd, time_integrator, order), fontsize=20)
    plt.show()

###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot results of a shock-tube problem.')
    parser.add_argument('--ini', type=str, default='test_two_fluid_shock_tube_2d_lx17_jwl.ini', help='kalypsso ini parameter file')
    args = parser.parse_args()

    shock_tube_plot(args.ini)
