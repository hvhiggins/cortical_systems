# Michael Schwemmer, modified by Brent Doiron
# Translated from MATLAB to python by Holden Higgins
# Hodgkin-Huxley Model
# ====================
#
# simulates the response of the squid axon to a pulse of appiled current.
#
# parameters as in 
#   Hoppensteadt and Peskin
#   Modeling and Simulation in Medicine and the Life Sciences
#
# to run, type the following at the matlab prompt: 
#    
#    HH_pulse(vstart,tstart,textra,iapp,twidth)

#    vstart is the initial voltage.
#    tstart is the start time of the pulse.
#    textra is the simulation time after the pulse is over. 
#    iapp is the amplitude of the pulse. 
#    twidth is the width of the pulse.
#
# voltages in mV, current in uA/cm^2, conductance in mS/cm^2, time is msec
#

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from scipy.integrate import solve_ivp

###############################
# Global Variables:
# membrance capacitance  uF/cm^2
C = 1.0 
# Max Na conductance  mS/cm^2
gNabar = 120
# Max K conductance  mS/cm^2
gKbar = 36
# Max leakage conductance  mS/cm^2
gLbar = 0.3
# Na Reversal Potential  mV
ENa = 45
# K Reversal Potential  mV
EK = -82
# Leakage Reversal Potential  mV
EL = -59

def alpham(v):
    theta = (v+45)/10
    a = np.ones(np.shape(theta))
    a[theta != 0] = 1.0*theta/(1-exp(-theta))
    return a

def betam(v):
    b = 4.0*exp(-(v+70)/18)
    return b

def alphah(v):
    a = 0.07*exp(-(v+70)/20)
    return a

def betah(v):
    b = 1.0/(1+exp(-(v+40)/10))
    return b

def alphan(v):
    theta = (v+60)/10
    if (theta == 0):
        a = 0.1
    else:
        a = 0.1*theta/(1-exp(-theta))
    return a

def betan(v):
    b = 0.125*exp(-(v+70)/80)
    return b


def sim_HH_pulse(
    vstart,
    tstart,
    textra,
    iapp,
    twidth,
):
    #######################################################
    # Initial Values for m, n, and h are set to rest state

    v = vstart

    m = alpham(v)/(alpham(v) + betam(v))
    n = alphan(v)/(alphan(v) + betan(v))
    h = alphah(v)/(alphah(v) + betah(v))


    ###################################
    # Solve the equations using ode45

    # Initial Conditions 
    s0 = np.array([m, n, h, v])

    tmax = tstart+twidth+textra 

    def fxn(t,s):

        ds = [[],[],[],[]]#np.zeros((4,1))
        ds[0] = alpham(s[3])*(1-s[0])-betam(s[3])*s[0]
        ds[1] = alphan(s[3])*(1-s[1])-betan(s[3])*s[1]
        ds[2] = alphah(s[3])*(1-s[2])-betah(s[3])*s[2]
        gNa = gNabar*(s[0]**3)*s[2]
        gK = gKbar*(s[1]**4)
        
        if ((tstart<t) and (t<tstart+twidth)):
            ix = iapp 
        else:
            ix = 0

        ds[3] = -(1/C)*(gNa*(s[3]-ENa) + gK*(s[3]-EK) + gLbar*(s[3]-EL)) + ix/C

        return ds

    timespan = [0, tmax]

    soln = solve_ivp(fxn,timespan,s0)
    if soln.status != 0:
        print(soln.message)
        return
    T = soln.t
    S = soln.y

    dt = 0.01
    t_iapp=[0, tstart-dt, tstart, tstart+twidth, tstart+twidth+dt, tmax]
    c_iapp=[0, 0, iapp, iapp, 0, 0]      

    return T, S, t_iapp, c_iapp

def HH_pulse(
    vstart, # initial voltage (mV)
    tstart, # start time of pulse (ms)
    textra, # sim time after pulse (ms)
    iapp, # amplitude of pulse (mA)
    twidth, # width of pulse (ms)
    fname = None):
    # Wrapper function to maintain function signature from original code
    T, S, t_iapp, c_iapp = sim_HH_pulse(vstart, tstart, textra, iapp, twidth)
    print(np.max(S[3]))
    # Plotting the pulse
    fig, ax = plt.subplots(3)
    ax[0].plot(T,S[3])
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Voltage')
    ax[1].plot(t_iapp, c_iapp, color='r', linewidth=4)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('iapp')
    ax[2].plot(T,S[0], label = "m") 
    ax[2].plot(T,S[1], label = "n") 
    ax[2].plot(T,S[2], label = "h") 
    ax[2].legend()
    ax[2].set_xlabel('time')
    if fname:
        return fig.savefig(fname)
    else:
        plt.show()

if __name__ == "__main__":
    HH_pulse(
        vstart = -69.9, 
        tstart = 10,
        textra = 10,
        iapp = 20,
        twidth = 5)
            