# Code originally written in MATLAB by Brent Doiron
# translated to python by Holden Higgins on 2-3-2023
#
# Compares Numnerical simulations of the first passage time of an LIF to the 
# (quasi)analytic estimate of the mean first passge time computed from the 
# Backwards-Kolmogorov equation 
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import math
from time import time

mu = 0.9
V_T = 1
V_R = 0

Nmesh = 2000 #mesh size for the BVP

V0_low = -4*(V_T-V_R) #the lower bound for V0 appromimating -infty

# the sigma sweep 

sigmalow = 0.05
sigmahigh = 1
Nsigma = 10

# other functions
def fV(V: ArrayLike) -> ArrayLike:
    return mu-V #for LIF

def theory(sigmaline):
    # Theory ############
    T_theory =  np.ones(sigmaline.size)
    for k, sigma in enumerate(sigmaline):

        ###### Theory functions for bvp solver
        def T1bvp(V: ArrayLike, y: ArrayLike) -> ArrayLike:
            """ first moment
            V: shape (m,)
            y: shape (2,m)
            returns dydx: shape (2,m)
            """
            dydV=[y[1], (-2/sigma**2)*(fV(V)*y[1]+1)]
            return dydV

        def T1bc(ya: ArrayLike, yb: ArrayLike):
            return [ya[1], yb[0]]

        def T1guess(V: ArrayLike):
            """
            V: shape (m,)
            returns guess: shape (2,m)"""
            guess_V = (V_T-V)/mu
            guess_y = np.zeros(guess_V.size)
            guess = np.array([guess_V, guess_y])
            return guess

        V0mesh = np.linspace(V0_low,V_T,Nmesh) #domain of the BVP

        T1 = solve_bvp(T1bvp, T1bc, V0mesh, T1guess(V0mesh), max_nodes = 5000) #solve the BVP subject to the BCs

        min_idx=np.argmin(np.abs(T1.x-V_R)) # Find the index for x = V_R
        T_theory[k]=T1.y[0,min_idx] #save the MFP for a neuron that starts at V_R
    return T_theory

def simulation(sigmaline):
    T_sim =  np.ones(sigmaline.size)

    for k, sigma in enumerate(sigmaline):
        start = time()
        R = 4000 #number of passages
        dt = 0.005
        sqrtdt = math.sqrt(dt)
        T = np.zeros((R,1))

        for i in range(0,R): #loop over realizations
            stop = 0
            V = V_R
            j = 0 # loop counter

            while (stop == 0):
                dV = dt*fV(V) + sigma*sqrtdt*np.random.normal() #memebrane integration
                V += dV

                if (V >= V_T or j>5000):  #spike
                    stop = 1  #exit loop

                j = j+1
                
            T[i]=j*dt
        
        T_sim[k]=np.mean(T)
        print(f"ran sim sigma = {sigma:.3f} in {time()-start}s")
    return T_sim

def main(
    run_theory = False,
    run_simulation = False,
):
    sigmaline = np.linspace(sigmalow,sigmahigh,Nsigma)
    
    T_sim = T_theory = np.zeros(sigmaline.size)
    if run_theory:
        T_theory = theory(sigmaline)
    # Sims ###########
    if run_simulation:
        T_sim = simulation(sigmaline)
    #plot stuff
    plt.plot(sigmaline, 1./T_theory, label = 'Theory') 
    plt.plot(sigmaline, 1./T_sim, label = 'Simulations')
    plt.xlabel('σ')
    plt.ylabel('r (arb units)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__=="__main__":
    main(run_theory=True)