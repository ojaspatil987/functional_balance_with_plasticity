import numpy as np 
import matplotlib.pyplot as plt 
from brian2 import *
N_exc = 400
N_inh = 100
v_rest = 0 * mV
tau = 30 * ms
I_app_val = 1 * nA
R = 100 * Mohm
start_scope()
defaultclock.dt = 0.01*ms
tau_minus = 20*ms
tau_plus = 5*ms
tau_x = 15*ms
A_LTD = 1e-4
A_LTP = 1e-4 / ms
w_max = 5
v_bar_plus_theta = 5*mV
eqs = '''
dv/dt = (-v + I_app*R)/(tau) : volt
dv_bar/dt = (v - v_bar) / tau_minus   : volt
dv_bar_plus/dt = (v - v_bar_plus) / tau_plus : volt
theta_minus : volt (constant)
theta_plus  : volt (constant)
I_app : ampere
'''
exc = NeuronGroup(N_exc, eqs, threshold='v > 70*mV', reset='v = v_rest', method='exact')
inh = NeuronGroup(N_inh, eqs, threshold='v > 70*mV', reset='v = v_rest', method='exact')
exc.I_app = I_app_val
inh.I_app = I_app_val
exc.v = v_rest + rand(N_exc) * 10 * mV
inh.v = v_rest + rand(N_inh) * 10 * mV

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
state_monitor_exc = StateMonitor(exc, 'v', record=True)
state_monitor_inh = StateMonitor(inh, 'v', record=True)
spike_monitor_exc = SpikeMonitor(exc)
spike_monitor_inh = SpikeMonitor(inh)
run(50*ms)
plt.plot(spike_monitor_exc.t / ms, spike_monitor_exc.i, '|', color='tab:blue', label='Excitatory Neurons')
plt.plot(spike_monitor_inh.t / ms, spike_monitor_inh.i + N_exc, '|', color='tab:red', label='Inhibitory Neurons')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster Plot')
plt.legend()
plt.savefig('/mnt/user-data/outputs/raster.png', dpi=100)
print("done")
