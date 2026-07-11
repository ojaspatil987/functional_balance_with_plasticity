import numpy as np
from brian2 import *
num_neurons = 500
f = .8 # fraction of excitatory neurons
ne = int(f*num_neurons) # number of exc neurons
ni = num_neurons - ne   # number of inh neurons

tau = 20*ms
u_th = 20*mV 
u_reset = 0*mV
t_ref = 0*ms

tau_minus = 10*ms
tau_plus  = 7*ms
tau_x     = 15*ms

vth_m = -20. # (mV)
vth_p = 7.5 # (mV)

# connection probability
eps_ee = .3 # exc to exc
eps_ei = .3 # exc to inh
eps_ie = 1. # inh to exc
eps_ii = 1. # inh to inh

# EPSP (mV)
J = .5 
g = 8. # inhibition dominance ratio (IPSP = -g EPSP)

# parameters for the visual stimulus driven firing rate
s_b = 2*kHz          # baseline feedforward firing rate
mu_exc = 0.2        # tuning strength, excitatory neurons
mu_inh = 0.02         # tuning strength, inhibitory neurons 
J_ffw = 1*mV          # size of each feedforward PSP


stim_theta = 22.5 * (np.pi/180)