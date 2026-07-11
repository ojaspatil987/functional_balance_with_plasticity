from brian2 import *
import sys
# sys.path.append('/content')
# from params import *

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
w : volt
dx_bar/dt = -x_bar/tau_x : 1 (clock-driven)
dw_ltp/dt = A_ltp * x_bar
              * clip(u_post - theta_plus_post, 0*mV, 1e9*mV)
              * clip(u_lp2_post - theta_minus_post, 0*mV, 1e9*mV)
              / mV**2 / ms                                    : 1 (clock-driven)
'''

exc_on_pre = '''
u_post += w
x_bar += 1
w = clip(w/mV - A_ltd * clip(u_lp1_post/mV - theta_minus_post/mV, 0, 1e9),
         w_min, w_max) * mV
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
w = clip(w/mV - A_ltd * clip(u_lp1_post/mV - theta_minus_post/mV, 0, 1e9),
         w_max_inh, 0) * mV
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

ff_exc_syn = Synapses(ff_exc, exc, on_pre='v_post += J_ffw')
ff_exc_syn.connect(j='i')   # channel i drives neuron i only -- a private feedforward line per neuron

ff_inh_syn = Synapses(ff_inh, inh, on_pre='v_post += J_ffw')
ff_inh_syn.connect(j='i')

# ---------------------------------------------------------
# Plotting Connectivity
# ---------------------------------------------------------
import matplotlib.pyplot as plt

# Construct the full 500x500 connectivity matrix (unitless, normalized by mV)
W = np.zeros((num_neurons, num_neurons))
W[S_EE.i, S_EE.j] = S_EE.w / mV
W[S_EI.i, S_EI.j + ne] = S_EI.w / mV
W[S_IE.i + ne, S_IE.j] = S_IE.w / mV
W[S_II.i + ne, S_II.j + ne] = S_II.w / mV

fig, ax = plt.subplots(figsize=(8, 7))

# We use a symmetric limits diverging colormap so positive is blue/red and negative is the opposite, or coolwarm
im = ax.imshow(W, cmap='RdBu_r', aspect='equal', interpolation='none', vmin=-4.0, vmax=4.0)
cbar = plt.colorbar(im, label='Synaptic Weight (mV)')
ax.set_title('Network Connectivity Matrix (Initial Weights)')
ax.set_xlabel('Post-synaptic Neuron Index')
ax.set_ylabel('Pre-synaptic Neuron Index')

# Draw lines to separate Excitatory (0 to 399) and Inhibitory (400 to 499) populations
ax.axvline(x=ne - 0.5, color='black', linestyle='--', alpha=0.5)
ax.axhline(y=ne - 0.5, color='black', linestyle='--', alpha=0.5)

# Label the regions
ax.text(ne / 2, ne / 2, 'E -> E', color='black', fontsize=12, ha='center', va='center')
ax.text(ne + ni / 2, ne / 2, 'E -> I', color='black', fontsize=12, ha='center', va='center')
ax.text(ne / 2, ne + ni / 2, 'I -> E', color='black', fontsize=12, ha='center', va='center')
ax.text(ne + ni / 2, ne + ni / 2, 'I -> I', color='black', fontsize=12, ha='center', va='center')

plt.tight_layout()
output_path = 'connectivity_matrix.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")