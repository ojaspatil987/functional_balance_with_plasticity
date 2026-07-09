import numpy as np 
import matplotlib.pyplot as plt 
from brian2 import *

N_exc = 400
N_inh = 100

# parameters 
v_rest = 0 * mV
tau = 30 * ms

I_app = 0.5 * mA  # Applied current
R = 0.5 * ohm  # Membrane resistance

start_scope()
defaultclock.dt = 0.01*ms

eqs='''
dv/dt = (v - v_rest)/tau + I_app*R/(1*ms) : volt
'''

exc = NeuronGroup(N_exc, eqs, threshold='v > 70*mV', reset='v = v_rest', method='exact')
inh = NeuronGroup(N_inh, eqs, threshold='v > 70*mV', reset='v = v_rest', method='exact')

exc.v = v_rest + rand(N_exc) * 10 * mV
inh.v = v_rest + rand(N_inh) * 10 * mV

# exc_to_exc = Synapses(exc, exc, on_pre='v_post += 35*mV')
# exc_to_inh = Synapses(exc, inh, on_pre='v_post += 15*mV')

taum = 10*ms
taupre = 20*ms
taupost = taupre
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax


exc_to_exc = Synapses(exc, exc,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )

exc_to_inh = Synapses(exc, inh,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )

exc_to_exc.connect(p=1.0)
exc_to_inh.connect(p=1.0)

inh_to_inh = Synapses(inh, inh, on_pre='v_post -= 25*mV')
inh_to_exc = Synapses(inh, exc, on_pre='v_post -= 15*mV')
inh_to_inh.connect(p=1.0)
inh_to_exc.connect(p=1.0)

state_monitor_exc = StateMonitor(exc, 'v', record=True)
state_monitor_inh = StateMonitor(inh, 'v', record=True)
spike_monitor_exc = SpikeMonitor(exc)
spike_monitor_inh = SpikeMonitor(inh)

run(50*ms)

# plt.plot(state_monitor_exc.t / ms, state_monitor_exc.v[0] / mV, label='Excitatory Neuron 0')
# plt.plot(state_monitor_inh.t / ms, state_monitor_inh.v[0] /mV, label='Inhibitory Neuron 0')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane potential (mV)')

plt.plot(spike_monitor_exc.t / ms, spike_monitor_exc.i, '|', color='tab:blue', label='Excitatory Neurons')
plt.plot(spike_monitor_inh.t / ms, spike_monitor_inh.i + N_exc, '|', color='tab:red', label='Inhibitory Neurons')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster Plot')
plt.legend()
plt.show()  