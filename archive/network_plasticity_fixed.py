from brian2 import *
start_scope()
defaultclock.dt = 0.01*ms

# ============================================================
# Network size
# ============================================================
N_exc = 400
N_inh = 100

# ============================================================
# Neuron parameters
# ============================================================
v_rest = 0*mV
tau = 30*ms
R = 100*Mohm
I_app_val = 1*nA

# ============================================================
# Plasticity parameters (Clopath et al. 2010, Eqs 1-3)
# ============================================================
tau_minus = 20*ms     # time constant for v_bar   (u_bar_minus,  Eq. in image)
tau_plus = 5*ms        # time constant for v_bar_plus (u_bar_plus)
tau_x = 15*ms          # presynaptic trace time constant
tau_homeo = 1000*ms    # ~1s averaging window for A_LTD homeostasis
v_ref = 12*mV          # reference depolarization (u_ref) for A_LTD(u_bar_bar)
A_LTD = 1.5e-4         # depression amplitude
A_LTP = 1.5e-2         # potentiation amplitude
w_max = 5              # hard upper bound on w
theta_minus_val = 10*mV
theta_plus_val = 15*mV

# ============================================================
# Neuron equations
# v_bar       = u_bar_minus(t): slow low-pass trace, drives LTD
# v_bar_plus  = u_bar_plus(t):  fast low-pass trace, drives LTP
# v_homeo     = u_bar_bar(t):   very slow trace, drives A_LTD homeostasis
# x_trace     = x_bar_i(t):     presynaptic spike trace, drives LTP
# ============================================================
eqs = '''
dv/dt = (-v + I_app*R)/tau : volt
dv_bar/dt = (v - v_bar)/tau_minus : volt
dv_bar_plus/dt = (v - v_bar_plus)/tau_plus : volt
dv_homeo/dt = (v - v_rest - v_homeo)/tau_homeo : volt
dx_trace/dt = -x_trace/tau_x : 1
theta_minus : volt (constant)
theta_plus  : volt (constant)
I_app : ampere
'''

exc = NeuronGroup(N_exc, eqs, threshold='v > 70*mV',
                   reset='v = v_rest; x_trace += 1', method='euler')
inh = NeuronGroup(N_inh, eqs, threshold='v > 70*mV',
                   reset='v = v_rest; x_trace += 1', method='euler')

for grp in (exc, inh):
    grp.v = v_rest + rand(len(grp)) * 10*mV
    grp.I_app = I_app_val
    grp.theta_minus = theta_minus_val
    grp.theta_plus = theta_plus_val

# ============================================================
# Excitatory plasticity (Clopath rule, Eqs 1-3) -- ONLY applied
# to synapses with an excitatory presynaptic neuron, per the paper.
#
#   LTD (Eq. 1), triggered on presynaptic spike:
#     w -= A_LTD(v_homeo) * [v_bar - theta_minus]_+
#   LTP (Eq. 2), triggered on postsynaptic spike (event-based
#     approximation of the continuous rule -- see note below):
#     w += A_LTP * x_trace_pre * [v_bar_plus - theta_minus]_+
# ============================================================
exc_model = 'w : 1'

exc_on_pre = '''
v_post += w*mV
A_LTD_u = A_LTD * (v_homeo_post/v_ref)**2
w = clip(w - A_LTD_u * clip(v_bar_post - theta_minus_post, 0*mV, 1e9*mV)/mV, 0, w_max)
'''

exc_on_post = '''
w_plus = A_LTP * x_trace_pre * clip(v_bar_plus_post - theta_minus_post, 0*mV, 1e9*mV)/mV
w = clip(w + w_plus, 0, w_max)
'''

exc_to_exc = Synapses(exc, exc, model=exc_model, on_pre=exc_on_pre,
                       on_post=exc_on_post, method='euler')
exc_to_exc.connect(p=0.5)
exc_to_exc.w = 1

exc_to_inh = Synapses(exc, inh, model=exc_model, on_pre=exc_on_pre,
                       on_post=exc_on_post, method='euler')
exc_to_inh.connect(p=0.5)
exc_to_inh.w = 1

# ============================================================
# Inhibitory synapses: fixed weight, no plasticity.
# The 2010 paper does not define a plasticity rule for
# inhibitory presynaptic connections -- only excitatory ones.
# ============================================================
inh_to_inh = Synapses(inh, inh, on_pre='v_post -= 25*mV')
inh_to_inh.connect(p=0.5)

inh_to_exc = Synapses(inh, exc, on_pre='v_post -= 15*mV')
inh_to_exc.connect(p=0.5)

# ============================================================
# Monitors
# ============================================================
state_monitor_exc = StateMonitor(exc, 'v', record=True)
state_monitor_inh = StateMonitor(inh, 'v', record=True)
spike_monitor_exc = SpikeMonitor(exc)
spike_monitor_inh = SpikeMonitor(inh)
weight_monitor = StateMonitor(exc_to_exc, 'w', record=range(20))

run(50*ms, report='text')

print("exc spikes:", spike_monitor_exc.count[:].sum())
print("inh spikes:", spike_monitor_inh.count[:].sum())
print("sample exc_to_exc weights after run:", exc_to_exc.w[:10])

plt.figure(figsize=(10, 5))
plt.plot(spike_monitor_exc.t/ms, spike_monitor_exc.i, '|', color='tab:blue', label='Excitatory')
plt.plot(spike_monitor_inh.t/ms, spike_monitor_inh.i + N_exc, '|', color='tab:red', label='Inhibitory')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster Plot')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/raster_clopath.png', dpi=100)

plt.figure(figsize=(10, 4))
for k in range(20):
    plt.plot(weight_monitor.t/ms, weight_monitor.w[k])
plt.xlabel('Time (ms)')
plt.ylabel('w (exc_to_exc)')
plt.title('Sample synaptic weight evolution (Clopath rule)')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/weights_clopath.png', dpi=100)
print("done")