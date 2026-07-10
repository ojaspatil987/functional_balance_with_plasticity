import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

N_exc = 400
N_inh = 100
v_rest = 0 * mV
tau = 30 * ms

start_scope()
defaultclock.dt = 0.01*ms

tau_minus = 10*ms
tau_plus = 7*ms
tau_x = 15*ms
A_LTD = 1e-4
A_LTP = 1e-4 / ms
w_max = 5
v_bar_plus_theta = 5*mV

# theta minus and theta plus
theta_minus_val = -20*mV
theta_plus_val = 7.5*mV

# parameters for the visual stimulus driven firing rate
s_b = 2*kHz          # baseline feedforward firing rate
mu_exc = 0.2        # tuning strength, excitatory neurons
mu_inh = 0.02         # tuning strength, inhibitory neurons 
J_ffw = 1*mV          # size of each feedforward PSP

# the orientation of the stimulus bar, in radians (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180 degrees)
stim_theta = 22.5 * (np.pi/180)   

eqs = '''
dv/dt = (-v)/tau : volt
dv_bar/dt = (v - v_bar) / tau_minus   : volt
dv_bar_plus/dt = (v - v_bar_plus) / tau_plus : volt
theta_minus : volt (constant)
theta_plus  : volt (constant)
theta_star  : 1 (constant)
'''
# theta star is the preferred orientation of the neuron

exc = NeuronGroup(N_exc, eqs, threshold='v > 70*mV', reset='v = v_rest', method='exact')
inh = NeuronGroup(N_inh, eqs, threshold='v > 70*mV', reset='v = v_rest', method='exact')

exc.theta_minus = theta_minus_val
exc.theta_plus = theta_plus_val
inh.theta_minus = theta_minus_val
inh.theta_plus = theta_plus_val

exc.v = v_rest + rand(N_exc) * 10 * mV
inh.v = v_rest + rand(N_inh) * 10 * mV

# assigning neurons their preferred stimulus orientation
exc.theta_star = np.random.uniform(0, np.pi, N_exc)
inh.theta_star = np.random.uniform(0, np.pi, N_inh)

# recurrent synapses
exc_eqn = '''
w : 1
dx_bar_i/dt = -x_bar_i / tau_x : 1 (clock-driven)
dw_LTP/dt = A_LTP * x_bar_i * int(v_post > v_bar_plus_theta) * clip(v_post - theta_plus_post, 0*mV, 1e9*mV) * clip(v_bar_plus_post - theta_minus_post, 0*mV, 1e9*mV) / mV**2 : 1 (clock-driven)
'''
exc_on_pre = '''
v_post += 35*mV
x_bar_i += 1
w = clip(w - A_LTD * clip(v_bar_post - theta_minus_post, 0*mV, 1e9*mV) / mV, 0, w_max)
'''
exc_to_exc = Synapses(exc, exc, model=exc_eqn, on_pre=exc_on_pre, method='euler')
exc_to_exc.connect(p=0.5)
exc_to_exc.w = 1
exc_to_exc.x_bar_i = 0

inh_model = '''
w : 1
'''
inh_on_pre = '''
v_post -= 15*mV
w = clip(w - A_LTD * clip(v_bar_post - theta_minus_post, 0*mV, 1e9*mV) / mV, 0, w_max)
'''
inh_to_exc = Synapses(inh, exc, model=inh_model, on_pre=inh_on_pre, method='euler')
inh_to_exc.connect(p=0.5)
inh_to_exc.w = 1

# actual stimulus, modelled on a poisson spike train
rate_exc = s_b * (1 + mu_exc * np.cos(2 * (stim_theta - exc.theta_star[:])))
rate_inh = s_b * (1 + mu_inh * np.cos(2 * (stim_theta - inh.theta_star[:])))

ff_exc = PoissonGroup(N_exc, rates=rate_exc)
ff_inh = PoissonGroup(N_inh, rates=rate_inh)

ff_exc_syn = Synapses(ff_exc, exc, on_pre='v_post += J_ffw')
ff_exc_syn.connect(j='i')   # channel i drives neuron i only -- a private feedforward line per neuron

ff_inh_syn = Synapses(ff_inh, inh, on_pre='v_post += J_ffw')
ff_inh_syn.connect(j='i')

# plots
state_monitor_exc = StateMonitor(exc, 'v', record=True)
state_monitor_inh = StateMonitor(inh, 'v', record=True)
spike_monitor_exc = SpikeMonitor(exc)
spike_monitor_inh = SpikeMonitor(inh)

run(50*ms)

plt.plot(spike_monitor_exc.t / ms, spike_monitor_exc.i, '|', color='tab:blue', label='Excitatory Neurons')
plt.plot(spike_monitor_inh.t / ms, spike_monitor_inh.i + N_exc, '|', color='tab:red', label='Inhibitory Neurons')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title(f'Spike Raster Plot (stimulus orientation = {np.degrees(stim_theta):.1f} deg)')
plt.legend()
plt.show()