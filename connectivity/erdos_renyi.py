from brian2 import *
import sys
# sys.path.append('/content')
# from params import *

defaultclock.dt = 0.1*ms

# ---------------------------------------------------------
# Neuron model -- now extended with plasticity-support variables
# ---------------------------------------------------------
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

# Plasticity parameters
w_min = 0.
w_max = 2.
w_max_inh = -5.
A_ltd = 14.e-5
A_ltp = 8.e-5

R = 1*Mohm

eqs = '''
du/dt = (-u + I_app*R)/tau         : volt
du_lp1/dt = (u - u_lp1)/tau_minus  : volt   # low-pass, LTD side (their v_bar)
du_lp2/dt = (u - u_lp2)/tau_plus   : volt   # low-pass, LTP side (their v_bar_plus)
theta_star  : 1 (constant)
theta_minus : volt (constant)
theta_plus  : volt (constant)
I_app       : ampere
po          : radian
'''

neurons = NeuronGroup(num_neurons, eqs, threshold='u>u_th', reset='u=u_reset',
                       refractory=t_ref, method='euler')
neurons.u = 0*mV
neurons.I_app = 0*amp
neurons.theta_minus = vth_m*mV     # <-- the fix: actually assign these
neurons.theta_plus  = vth_p*mV     # <-- otherwise plasticity thresholds are 0
neurons.po[:ne] = np.arange(0, np.pi, np.pi/ne)
neurons.po[ne:] = np.arange(0, np.pi, np.pi/ni)

exc = neurons[:ne]
inh = neurons[ne:]

# ---------------------------------------------------------
# Plastic synapse model (ported from their exc_eqn / exc_on_pre)
# ---------------------------------------------------------
J_exc = J*mV
J_inh = -g*J_exc

exc_plastic_model = '''
dw/dt = A_ltp * x_bar * clip(u_post - theta_plus_post, 0*mV, 1e9*mV) * clip(u_lp2_post - theta_minus_post, 0*mV, 1e9*mV) / (mV * ms) : volt (clock-driven)
dx_bar/dt = -x_bar/tau_x : 1 (clock-driven)
'''

exc_on_pre = '''
u_post += w
x_bar += 1
w = clip(w/mV - A_ltd * clip(u_lp1_post/mV - theta_minus_post/mV, 0, 1e9), w_min, w_max) * mV
'''

S_EE = Synapses(exc, exc, model=exc_plastic_model, on_pre=exc_on_pre, method='euler')
S_EE.connect(condition='i!=j', p=eps_ee)
S_EE.w = J_exc
S_EE.x_bar = 0

S_EI = Synapses(exc, inh, model=exc_plastic_model, on_pre=exc_on_pre, method='euler')
S_EI.connect(p=eps_ei)
S_EI.w = J_exc
S_EI.x_bar = 0

# ---------------------------------------------------------
# I -> E: LTD only, no LTP term (matches their inh_model, which
# has no dw_ltp -- inhibitory plasticity in their file is LTD-only)
# ---------------------------------------------------------
inh_plastic_model = 'w : volt'
inh_on_pre = '''
u_post += w
w = clip(w/mV - A_ltd * clip(u_lp1_post/mV - theta_minus_post/mV, 0, 1e9), w_max_inh, 0) * mV
'''

S_IE = Synapses(inh, exc, model=inh_plastic_model, on_pre=inh_on_pre, method='euler')
S_IE.connect(p=eps_ie)
S_IE.w = J_inh

# I -> I stays fully static (never plastic in the paper)
S_II = Synapses(inh, inh, model='w : volt', on_pre='u_post += w')
S_II.connect(condition='i!=j', p=eps_ii)
S_II.w = J_inh

# ---------------------------------------------------------
# Feedforward pathway
# ---------------------------------------------------------


# parameters for the visual stimulus driven firing rate
s_b = 2*kHz          # baseline feedforward firing rate
mu_exc = 0.2        # tuning strength, excitatory neurons
mu_inh = 0.02         # tuning strength, inhibitory neurons 
J_ffw = 1*mV          # size of each feedforward PSP


stim_theta = 22.5 * (np.pi/180)

# assigning neurons their preferred stimulus orientation
exc.theta_star = np.random.uniform(0, np.pi, ne)
inh.theta_star = np.random.uniform(0, np.pi, ni)

# actual stimulus, modelled on a poisson spike train
rate_exc = s_b * (1 + mu_exc * np.cos(2 * (stim_theta - exc.theta_star[:])))
rate_inh = s_b * (1 + mu_inh * np.cos(2 * (stim_theta - inh.theta_star[:])))

ff_exc = PoissonGroup(ne, rates=rate_exc)
ff_inh = PoissonGroup(ni, rates=rate_inh)

ff_exc_syn = Synapses(ff_exc, exc, on_pre='u_post += J_ffw')
ff_exc_syn.connect(j='i')   # channel i drives neuron i only -- a private feedforward line per neuron

ff_inh_syn = Synapses(ff_inh, inh, on_pre='u_post += J_ffw')
ff_inh_syn.connect(j='i')

# ---------------------------------------------------------
# Plotting Initial Connectivity
# ---------------------------------------------------------
import matplotlib.pyplot as plt

# Save initial E->E weights for comparison
initial_EE_weights = np.array(S_EE.w / mV)

# Construct the full 500x500 connectivity matrix (unitless, normalized by mV)
W_init = np.zeros((num_neurons, num_neurons))
W_init[S_EE.i, S_EE.j] = S_EE.w / mV
W_init[S_EI.i, S_EI.j + ne] = S_EI.w / mV
W_init[S_IE.i + ne, S_IE.j] = S_IE.w / mV
W_init[S_II.i + ne, S_II.j + ne] = S_II.w / mV

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(W_init, cmap='RdBu_r', aspect='equal', interpolation='none', vmin=-4.0, vmax=4.0)
plt.colorbar(im, label='Synaptic Weight (mV)')
ax.set_title('Erdos-Renyi Network Connectivity Matrix (Initial Weights)')
ax.set_xlabel('Post-synaptic Neuron Index')
ax.set_ylabel('Pre-synaptic Neuron Index')

ax.axvline(x=ne - 0.5, color='black', linestyle='--', alpha=0.5)
ax.axhline(y=ne - 0.5, color='black', linestyle='--', alpha=0.5)

ax.text(ne / 2, ne / 2, 'E -> E', color='black', fontsize=12, ha='center', va='center')
ax.text(ne + ni / 2, ne / 2, 'E -> I', color='black', fontsize=12, ha='center', va='center')
ax.text(ne / 2, ne + ni / 2, 'I -> E', color='black', fontsize=12, ha='center', va='center')
ax.text(ne + ni / 2, ne + ni / 2, 'I -> I', color='black', fontsize=12, ha='center', va='center')

plt.tight_layout()
init_conn_path = 'connectivity_matrix_initial_erdos_renyi.png'
plt.savefig(init_conn_path, dpi=150)
print(f"Initial connectivity plot saved to {init_conn_path}")

# Keep old name for backward compatibility
plt.savefig('connectivity_matrix_erdos_renyi.png', dpi=150)

# ---------------------------------------------------------
# Running Simulation
# ---------------------------------------------------------
# Set up monitors
spikemon = SpikeMonitor(neurons)

# Run simulation
print("Running simulation for 200 ms...")
run(200*ms, report='text')

# ---------------------------------------------------------
# Plotting Spike Raster
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
exc_spikes = spikemon.i < ne
inh_spikes = spikemon.i >= ne

plt.plot(spikemon.t/ms, spikemon.i, '|', color='gray', alpha=0.3, label='All Neurons')
plt.plot(spikemon.t[exc_spikes]/ms, spikemon.i[exc_spikes], '|', color='tab:blue', label='Excitatory')
plt.plot(spikemon.t[inh_spikes]/ms, spikemon.i[inh_spikes], '|', color='tab:red', label='Inhibitory')

plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title(f'Erdos-Renyi Spike Raster Plot (Stimulus Orientation = {np.degrees(stim_theta):.1f}°)')
plt.legend()
plt.tight_layout()

raster_output_path = 'raster_plot_erdos_renyi.png'
plt.savefig(raster_output_path, dpi=150)
print(f"Raster plot saved to {raster_output_path}")

# ---------------------------------------------------------
# Plotting Final Connectivity (After Plasticity)
# ---------------------------------------------------------
W_final = np.zeros((num_neurons, num_neurons))
W_final[S_EE.i, S_EE.j] = S_EE.w / mV
W_final[S_EI.i, S_EI.j + ne] = S_EI.w / mV
W_final[S_IE.i + ne, S_IE.j] = S_IE.w / mV
W_final[S_II.i + ne, S_II.j + ne] = S_II.w / mV

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(W_final, cmap='RdBu_r', aspect='equal', interpolation='none', vmin=-4.0, vmax=4.0)
plt.colorbar(im, label='Synaptic Weight (mV)')
ax.set_title('Erdos-Renyi Network Connectivity Matrix (Final Weights after 200 ms)')
ax.set_xlabel('Post-synaptic Neuron Index')
ax.set_ylabel('Pre-synaptic Neuron Index')

ax.axvline(x=ne - 0.5, color='black', linestyle='--', alpha=0.5)
ax.axhline(y=ne - 0.5, color='black', linestyle='--', alpha=0.5)

ax.text(ne / 2, ne / 2, 'E -> E', color='black', fontsize=12, ha='center', va='center')
ax.text(ne + ni / 2, ne / 2, 'E -> I', color='black', fontsize=12, ha='center', va='center')
ax.text(ne / 2, ne + ni / 2, 'I -> E', color='black', fontsize=12, ha='center', va='center')
ax.text(ne + ni / 2, ne + ni / 2, 'I -> I', color='black', fontsize=12, ha='center', va='center')

plt.tight_layout()
final_conn_path = 'connectivity_matrix_final_erdos_renyi.png'
plt.savefig(final_conn_path, dpi=150)
print(f"Final connectivity plot saved to {final_conn_path}")

# ---------------------------------------------------------
# Plotting Weight Distribution (Initial vs Final for E->E)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
final_EE_weights = np.array(S_EE.w / mV)

plt.hist(initial_EE_weights, bins=30, alpha=0.5, label='Initial', color='tab:gray')
plt.hist(final_EE_weights, bins=30, alpha=0.7, label='Final (After Plasticity)', color='tab:blue')
plt.xlabel('Synaptic Weight (mV)')
plt.ylabel('Count')
plt.title('Erdos-Renyi E -> E Synaptic Weight Distribution (Initial vs Final)')
plt.legend()
plt.tight_layout()

dist_output_path = 'weight_distribution_erdos_renyi.png'
plt.savefig(dist_output_path, dpi=150)
print(f"Weight distribution plot saved to {dist_output_path}")

