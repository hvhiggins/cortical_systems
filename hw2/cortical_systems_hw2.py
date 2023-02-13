import numpy as np
import matplotlib.pyplot as plt
import math
import modified_mfp
import HH_pulse
from time import time
import itertools

def q1a():
    # (1)(a)
    # model params
    dt = .05 # s
    duration = 1000 # s
    sigma_vals = np.linspace(.00, 3, 500)
    sim_times = np.arange(0, duration, dt)

    V_R = 0
    V_T = 1
    mu = .9

    # collected data 
    CVs = np.zeros(sigma_vals.shape)
    firing_rates = np.zeros(sigma_vals.shape)

    # simulation
    for trial in range(1):
        for idx_s, sigma in enumerate(sigma_vals):
            T_k = []
            last_spike = 0
            V = V_R
            noise = np.random.normal(0, sigma, size=sim_times.size)
            noise_plus_mu = noise + mu
            for idx_t, time in enumerate(sim_times):
                dV = -V + noise_plus_mu[idx_t]
                V += dV * dt
                if V >= V_T:
                    V = V_R
                    T_k.append(time-last_spike)
                    last_spike = time
            firing_rate = len(T_k)/duration
            firing_rates[idx_s] = firing_rate
            if firing_rate>0:
                CVs[idx_s] = firing_rate * np.std(T_k)
        plt.plot(sigma_vals, CVs, label = f"Trial #{trial}")
    plt.grid()
    plt.xlabel("σ")
    plt.ylabel("CV")
    plt.title("CV vs σ")
    plt.legend()
    plt.savefig('q1a.png')

def q1b():
    # (1)(b)
    # model params
    dt = .005 # s
    duration = 5000 # s
    sigma_vals = np.linspace(.2, 2, 20)
    sim_times = np.arange(0, duration, dt)

    V_R = 0
    V_T = 1
    mu = .2

    # collected data 
    CVs = np.zeros(sigma_vals.shape)
    firing_rates = np.zeros(sigma_vals.shape)

    # simulation
    for trial in range(5):
        for idx_s, sigma in enumerate(sigma_vals):
            T_k = []
            last_spike = 0
            V = V_R
            Vs = []
            noise = np.random.normal(0, sigma, size=sim_times.size)
            dV = (noise + mu) * dt
            for idx_t, time in enumerate(sim_times):
                V += dV[idx_t]
                if V >= V_T:
                    V = V_R
                    T_k.append(time-last_spike)
                    last_spike = time
            firing_rate = 1/np.mean(T_k)
            firing_rates[idx_s] = firing_rate
            CVs[idx_s] = firing_rate * np.std(T_k)
        plt.plot(sigma_vals, CVs, label = f"Trial #{trial}")
    plt.grid()
    plt.xlabel("σ")
    plt.ylabel("CV")
    plt.title("CV vs σ")
    plt.legend()
    plt.savefig('q1b.png')

def q1c():
    modified_mfp.run_modified_mfp(
        run_theory = True,
        run_simulation = True,
        output_fname = "q1c.png",
    )


def helper_2ac(iapp_vals):
    min_twidths = np.zeros(iapp_vals.shape)
    
    resolution = 10
    tw_initial = 20
    for idx, iapp in enumerate(iapp_vals):
        twidth = tw_initial

        # Use binary search to find min valid twidth to within 2^(-resolution)
        for res in range(1,resolution+1):
            T, S, t_iapp, c_iapp = HH_pulse.sim_HH_pulse(
                vstart=-69.9, tstart=5, textra=10, iapp=iapp, twidth=twidth)
            soln_voltages = S[3]
            pulse = np.max(soln_voltages) > 0
            step_size= tw_initial/(2**res)
            if pulse:
                step_size = -step_size
            twidth+=step_size

        if idx % 10 == 0:
            print((iapp, twidth))

        min_twidths[idx] = twidth
    return min_twidths

def q2a():
    start = time()
    iapp_vals = np.arange(1, 25, .1)
    
    min_twidths = helper_2ac(iapp_vals)

    print(f"did q2a in {time() - start}s")
    plt.plot(iapp_vals, min_twidths)
    plt.ylim(0,10)
    plt.ylabel("min delta t for spike(ms)")
    plt.xlabel("current applied (mA)")
    plt.grid()
    plt.savefig("q2a.png")

def q2b():
    HH_pulse.HH_pulse(
        vstart = -69.9, 
        tstart = 5,
        textra = 20,
        iapp = -10,
        twidth = 2,
        fname = "q2b.png")

def q2c():
    start = time()
    iapp_vals = np.arange(-25, -1, .1)
    
    min_twidths = helper_2ac(iapp_vals)

    print(f"did q2c in {time() - start}s")
    plt.plot(iapp_vals, min_twidths)
    plt.ylim(0,10)
    plt.ylabel("min delta t for spike(ms)")
    plt.xlabel("current applied (mA)")
    plt.grid()
    plt.savefig("q2c.png")

def helper_q3ab(I_0, beta_n, res, output_fname):
    
    # Class 1 excitability regime
    # model params
    C = 2 #mF/cm^2
    g_L = 2
    g_K = 20
    g_Na = 20
    E_K = -100 # mV
    E_Na = 50 # mV
    E_L = -70 # mV
    beta_m = -1.2
    gamma_m = 18
    gamma_n = 10
    phi = .15
    # steady-state activation curves
    m_inf = lambda V : .5 * (1 + np.tanh((V-beta_m)/gamma_m))
    n_inf = lambda V : .5 * (1 + np.tanh((V-beta_n)/gamma_n))
    tau_n = lambda V : 1/(phi*math.cosh((V-beta_n)/gamma_n))

    dt = .005 # ms
    t_vals = np.arange(0, 1000, dt)

    I_1_vals = np.linspace(.04, 2 , res)
    f_vals = np.linspace(.04, .17, res)
    param_space = [(I_1_idx, I_1, f_idx, f) 
                    for I_1_idx, I_1 in enumerate(I_1_vals)
                    for f_idx, f in enumerate(f_vals)]
    makes_spike = np.full((I_1_vals.size, f_vals.size), False)


    start1 = time()
    for I_1_idx, I_1, f_idx, f in param_space:
        I_vals = I_0 + I_1 * np.sin(2*math.pi*f*t_vals)
        V = -53
        n = n_inf(V)

        start2 = time()
        for t_idx, t in enumerate(t_vals):
            dn = (dt/tau_n(V)) * (n_inf(V) - n)
            n += dn

            Na_term = g_Na * m_inf(V) * (V-E_Na)
            K_term = g_K * n * (V-E_K)
            L_term = g_L * (V-E_L)

            internal_term = -(Na_term + K_term + L_term)
            dV = (dt/C) * (I_vals[t_idx] + internal_term)
            V += dV
            if t > 200 and V > 0:
                makes_spike[I_1_idx, f_idx] = True
                break

        print(f"calced {(I_1, f, V>0)} in {time()-start2}s")

    print(f'swept param space in {time()-start1}s')
    im = plt.imshow(makes_spike)
    cb = plt.colorbar(im, ticks = [0, 1])
    cb.set_ticklabels(['No spikes', 'Spikes'])
    plt.yticks(
        ticks = range(len(I_1_vals)), 
        labels = [f'{x:.3f}' for x in I_1_vals])
    plt.xticks(
        ticks = range(len(I_1_vals)),
        labels = [f'{x:.3f}' for x in f_vals],
        rotation = "vertical")
    plt.ylabel("I_1")
    plt.xlabel("f")
   
    plt.savefig(output_fname)


def q3a():
    # Class I excitability regime
    helper_q3ab(
        I_0 = 36.70,
        beta_n = 0,
        res = 20,
        output_fname = "q3a.png",
    )

def q3b():
    # Class II excitability regime
    helper_q3ab(
        I_0 = 45.75,
        beta_n = -13,
        res = 20,
        output_fname = "q3b.png",
    )

def q3c():
    # model params
    C = 2 #mF/cm^2
    g_L = 2
    g_K = 20
    g_Na = 20
    E_K = -100 # mV
    E_Na = 50 # mV
    E_L = -70 # mV
    beta_m = -1.2
    gamma_m = 18
    gamma_n = 10
    phi = .15
    
    dt = .005 # ms
    t_vals = np.arange(0, 3000, dt)
    
    regime_classes = [
        ('I', 36.70, 0),
        ('II', 45.75, -13),
    ]

    # see notes, dV/dt = f(V,n), dn/dt = g(V,n)
    # helpers to keep derivatives clean
    n_term = lambda V : (V - beta_n) / gamma_n
    m_term = lambda V : (V - beta_m) / gamma_m
    dNa_dV = lambda V, n :(g_Na/2) * (1 + np.tanh(m_term(V))
                            + ((V - E_Na)/gamma_m 
                            * np.cosh(m_term(V))**(-2)))
    dK_dV = lambda V, n : g_K * n
    dL_dV = lambda V, n : g_L
    
    # partial derivatives of the system 
    df_dn = lambda V, n : (1/C) *g_K * (V - E_K)
    dg_dV = lambda V, n : (phi/(2*gamma_n)) * (np.cosh(n_term(V))**(-1)
                            + (np.sinh(n_term(V)) * (1-n-np.tanh(n_term(V)))))
    dg_dn = lambda V, n : -phi * np.cosh(n_term(V))
    df_dV = lambda V, n : (1/C) * ( -dNa_dV(V, n) - dK_dV(V,n) - dL_dV(V, n))

    fig, ax = plt.subplots(2,1)
    for class_id, I_0, beta_n in regime_classes:
        # steady-state activation curves
        m_inf = lambda V : .5 * (1 + np.tanh((V-beta_m)/gamma_m))
        n_inf = lambda V : .5 * (1 + np.tanh((V-beta_n)/gamma_n))
        tau_n = lambda V : 1/(phi*math.cosh((V-beta_n)/gamma_n))

        I_vals = np.full(t_vals.shape, I_0)
        V = -40
        n = 0
        V_vals = np.zeros(t_vals.shape)
        n_vals = np.zeros(t_vals.shape)
        for t_idx, t in enumerate(t_vals):
            dn = (dt/tau_n(V)) * (n_inf(V) - n)
            n += dn

            Na_term = g_Na * m_inf(V) * (V-E_Na)
            K_term = g_K * n * (V-E_K)
            L_term = g_L * (V-E_L)

            internal_term = -(Na_term + K_term + L_term)
            dV = (dt/C) * (I_vals[t_idx] + internal_term)
            V += dV

            V_vals[t_idx] = V
            n_vals[t_idx] = n

        # Question output
        ax[0].plot(t_vals, V_vals, label = f'V for class {class_id}')
        ax[1].plot(t_vals, n_vals, label = f'n for class {class_id}')
       
        print(f"Class {class_id} steady state (V,n): {(V,n)}")
        jac = np.array([
            [df_dV(V, n), df_dn(V, n)], 
            [dg_dV(V, n), dg_dn(V, n)]])
        print(f"Jacobian: \n{jac}")
        eigs, _ = np.linalg.eig(jac)
        print(f"Has eigeinvalues {eigs}\n")
    
    ax[0].set_ylabel("voltage (mV)")
    ax[1].set_ylabel("n (arb units)")
    for i in (0,1):
        ax[i].set_xlabel("time (s)")
        ax[i].grid()
        ax[i].legend()
    fig.savefig("q3c.png")




if __name__ == "__main__":
    # Uncomment these to regenerate figures
    # q1a()
    # q1b()
    q1c()
    # q2a()
    # q2b()
    # q2c()
    # q3a()
    # q3b()
    # q3c()
    pass

