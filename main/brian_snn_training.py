from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os

start_scope()

# ============================================================
# Parameters (Table 1 of the paper / params.py)
# ============================================================
num_neurons = 500
f = 0.8
ne = int(f * num_neurons)
ni = num_neurons - ne

tau = 20 * ms
u_th = 20 * mV
u_reset = 0 * mV
t_ref = 0 * ms

tau_minus = 10 * ms   # tau- : LTD low-pass filter
tau_plus  = 7 * ms    # tau+ : LTP low-pass filter
tau_x     = 15 * ms   # presynaptic trace decay

vth_m = -20.0   # theta- (mV), stored as plain number
vth_p = 7.5     # theta+ (mV), stored as plain number

eps_ee = 0.3
eps_ei = 0.3
eps_ie = 1.0
eps_ii = 1.0

J = 0.5          # EPSP (mV)
g = 8.0          # inhibition dominance ratio

# plasticity learning rates -- straight from params.py, plain floats
# (the whole ported model works in "divided-by-mV" numeric space, so these
#  stay unitless multipliers, matching the existing codebase convention)
A_ltd = 14.e-5
A_ltp = 8.e-5
w_min = 0.0        # lower bound, exc (numeric, representing mV)
w_max = 2.0        # upper bound, exc
w_max_inh = -5.0   # lower bound, inh (upper bound is 0)

# feedforward / stimulus params
s_b = 2 * kHz
mu_exc = 0.2
mu_inh = 0.02
J_ffw = 1 * mV

# ============================================================
# Training schedule params (Fig 1 / Fig 6 protocol)
# ============================================================
block_no = 40          # number of batches
stim_no = 20            # orientations per batch
trial_time = 100 * ms  # presentation time per orientation

# ============================================================
# Data-saving protocol setup
# ============================================================
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# ============================================================
# Neuron model
# ============================================================
eqs = '''
du/dt = -u/tau                     : volt
du_lp1/dt = (u - u_lp1)/tau_minus  : volt   # low-pass, LTD side
du_lp2/dt = (u - u_lp2)/tau_plus   : volt   # low-pass, LTP side
theta_star  : 1 (constant)
theta_minus : volt (constant)
theta_plus  : volt (constant)
'''

neurons = NeuronGroup(num_neurons, eqs, threshold='u>u_th', reset='u=u_reset',
                       refractory=t_ref, method='euler')
neurons.u = 0 * mV
neurons.theta_minus = vth_m * mV
neurons.theta_plus  = vth_p * mV

exc = neurons[:ne]
inh = neurons[ne:]

# preferred orientations used for the feedforward tuning (theta_star)
exc.theta_star = np.random.uniform(0, np.pi, ne)
inh.theta_star = np.random.uniform(0, np.pi, ni)

J_exc = J * mV
J_inh = -g * J_exc

# ============================================================
# Plastic synapses
# LTP is now folded directly into w (clock-driven, continuous).
# LTD still happens at each presynaptic spike (event-driven), as in the paper.
# ============================================================
exc_plastic_model = '''
dx_bar/dt = -x_bar/tau_x : 1 (clock-driven)
dw/dt = A_ltp * x_bar * clip(u_post - theta_plus_post, 0*mV, 1e9*mV) * clip(u_lp2_post - theta_minus_post, 0*mV, 1e9*mV) / mV**2 / ms * mV : volt (clock-driven)
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

# I -> E: LTD only (matches the paper's simpler inhibitory rule)
inh_plastic_model = 'w : volt'
inh_on_pre = '''
u_post += w
w = clip(w/mV - A_ltd * clip(u_lp1_post/mV - theta_minus_post/mV, 0, 1e9), w_max_inh, 0) * mV
'''

S_IE = Synapses(inh, exc, model=inh_plastic_model, on_pre=inh_on_pre, method='euler')
S_IE.connect(p=eps_ie)
S_IE.w = J_inh

# I -> I: never plastic
S_II = Synapses(inh, inh, model='w : volt', on_pre='u_post += w')
S_II.connect(condition='i!=j', p=eps_ii)
S_II.w = J_inh

# ============================================================
# Build the training schedule (pure numpy -- no simulation yet)
# ============================================================
n_trials = block_no * stim_no  # 800

schedule = np.concatenate([
    np.random.permutation(np.linspace(0, np.pi, stim_no, endpoint=False))
    for _ in range(block_no)
])  # shape (800,) -- one orientation per 100 ms slot

# per-neuron, per-trial feedforward rate (Eq. 2 of the paper)
theta_star_exc = exc.theta_star[:]   # plain numbers, shape (ne,)
theta_star_inh = inh.theta_star[:]

rates_exc_trials = s_b * (1 + mu_exc * np.cos(2 * (schedule[:, None] - theta_star_exc[None, :])))
rates_inh_trials = s_b * (1 + mu_inh * np.cos(2 * (schedule[:, None] - theta_star_inh[None, :])))
# shapes: (800, ne) and (800, ni)

rates_exc_t = TimedArray(rates_exc_trials, dt=trial_time)
rates_inh_t = TimedArray(rates_inh_trials, dt=trial_time)

ff_exc = PoissonGroup(ne, rates='rates_exc_t(t, i)')
ff_inh = PoissonGroup(ni, rates='rates_inh_t(t, i)')

ff_exc_syn = Synapses(ff_exc, exc, on_pre='u_post += J_ffw')
ff_exc_syn.connect(j='i')

ff_inh_syn = Synapses(ff_inh, inh, on_pre='u_post += J_ffw')
ff_inh_syn.connect(j='i')

# ============================================================
# Data-saving protocol: static data, written once before the run
# ============================================================
# preferred orientations (theta_star) for every neuron -- these never
# change during training, so they're saved exactly once.
np.savez(os.path.join(data_dir, 'theta_star.npz'),
         theta_star_exc=theta_star_exc,
         theta_star_inh=theta_star_inh)

# connectivity (pre-index i, post-index j) for every synapse group.
# Brian2 keeps synapse ordering fixed once .connect() has been called,
# so every weight vector saved below lines up 1-to-1, element for
# element, with these (i, j) pairs. This is what lets a saved weight
# vector be turned back into a proper weight matrix later.
np.savez(os.path.join(data_dir, 'connectivity.npz'),
         ee_i=np.array(S_EE.i[:]), ee_j=np.array(S_EE.j[:]),
         ei_i=np.array(S_EI.i[:]), ei_j=np.array(S_EI.j[:]),
         ie_i=np.array(S_IE.i[:]), ie_j=np.array(S_IE.j[:]),
         ii_i=np.array(S_II.i[:]), ii_j=np.array(S_II.j[:]),
         ne=ne, ni=ni)

# full orientation schedule -- schedule[k] is the theta shown on trial
# k+1 (1-indexed), i.e. exactly the theta that produced
# data/weights_trial{k+1:04d}.npz below.
np.save(os.path.join(data_dir, 'schedule.npy'), schedule)

# ============================================================
# Weight snapshotting
#  - save_trial_weights: fires after EVERY single theta presentation
#    (every 100 ms trial), giving one file per learned orientation.
#  - snapshot: keeps the old once-per-batch bookkeeping used for the
#    convergence plot below.
# ============================================================
weight_snapshots = []           # E->E weights, one row per batch
weight_snapshots_ei = []        # E->I
weight_snapshots_ie = []        # I->E
batch_duration = stim_no * trial_time


@network_operation(dt=trial_time)
def save_trial_weights():
    # defaultclock.t is the time right after this trial's learning
    # step has been applied, so this captures the "post-theta" weights.
    trial_idx = int(round(defaultclock.t / trial_time))  # 1 .. n_trials
    if trial_idx < 1 or trial_idx > n_trials:
        return
    theta = schedule[trial_idx - 1]
    fname = os.path.join(data_dir, f'weights_trial{trial_idx:04d}.npz')
    np.savez_compressed(
        fname,
        trial=trial_idx,
        batch=(trial_idx - 1) // stim_no + 1,
        trial_in_batch=(trial_idx - 1) % stim_no + 1,
        theta=theta,
        theta_deg=np.degrees(theta),
        w_ee=np.array(S_EE.w[:] / mV, dtype=np.float32),
        w_ei=np.array(S_EI.w[:] / mV, dtype=np.float32),
        w_ie=np.array(S_IE.w[:] / mV, dtype=np.float32),
    )


@network_operation(dt=batch_duration)
def snapshot():
    weight_snapshots.append(np.array(S_EE.w[:] / mV))
    weight_snapshots_ei.append(np.array(S_EI.w[:] / mV))
    weight_snapshots_ie.append(np.array(S_IE.w[:] / mV))


# initial weights, for the "before learning" comparison
W_EE_initial = np.array(S_EE.w[:] / mV)

# ============================================================
# Run the whole thing in one call
# ============================================================
net = Network(collect())
total_time = n_trials * trial_time  # 800 * 100ms = 80 s
print(f"Running {block_no} batches x {stim_no} trials = {n_trials} trials "
      f"({total_time}) in a single run() call...")
print(f"Per-trial weight snapshots will be written to ./{data_dir}/ "
      f"as weights_trial0001.npz ... weights_trial{n_trials:04d}.npz")
net.run(total_time, report='text')

# ============================================================
# Convergence plot (Fig 6A equivalent)
# ============================================================
weight_snapshots = np.array(weight_snapshots)   # shape (block_no, n_synapses_EE)
mean_change_per_batch = np.mean(np.abs(np.diff(weight_snapshots, axis=0)), axis=1)

plt.figure(figsize=(7, 4))
plt.plot(mean_change_per_batch, marker='o')
plt.xlabel('Batch')
plt.ylabel('Mean |Δw| (E→E)')
plt.title('Convergence of E→E weights during training')
plt.tight_layout()
plt.savefig('weight_convergence.png', dpi=150)
print("Saved weight_convergence.png")

# quick sanity check: did anything actually potentiate?
print(f"Initial E->E mean weight: {W_EE_initial.mean():.4f} mV")
print(f"Final   E->E mean weight: {weight_snapshots[-1].mean():.4f} mV")
print(f"Fraction of E->E weights at/near w_max: "
      f"{np.mean(weight_snapshots[-1] > 0.9*w_max):.2%}")

# ============================================================
# Retrieving a saved snapshot later (e.g. in a brand-new Brian2
# session/script). See load_weights.py for a ready-made helper --
# in short:
#
#   from load_weights import load_trial
#   snap = load_trial(37)                  # 37th stimulated theta
#   snap['theta_deg']                      # orientation shown, in degrees
#   snap['w_ee'], snap['ee_i'], snap['ee_j']   # weight vector + its (i, j) pairs
#
#   S_EE.connect(i=snap['ee_i'], j=snap['ee_j'])
#   S_EE.w = snap['w_ee'] * mV
# ============================================================