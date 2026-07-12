"""
osi_analysis.py
================
Computes and plots the Orientation Selectivity Index (OSI) of the network
BEFORE training (uniform initial weights) and AFTER training (the final
learned weights saved by brian_snn_training.py), for:
    1. one example excitatory neuron    -- before
    2. one example inhibitory neuron    -- before
    3. the whole excitatory population  -- before (averaged)
    4. the whole inhibitory population  -- before (averaged)
    5-8. the same four, but AFTER training
That's 8 plots total, each saved as its own PNG file.

OSI is computed exactly as in the paper (Materials and Methods):
    OSI = 1 - circular variance = | sum_k r_k * exp(2i*theta_k) | / sum_k r_k
where r_k is the firing rate measured at stimulus orientation theta_k.

This script does NOT need to sit in the same file as brian_snn_training.py
or load_weights.py. It only needs the files brian_snn_training.py already
writes to disk:
    data/theta_star.npz          (each neuron's preferred orientation)
    data/connectivity.npz        (which neuron connects to which, as i/j pairs)
    data/weights_trial####.npz   (learned weights at any trial, incl. the last)
Connectivity is stored as plain index pairs, so it makes no difference
whether it came from an Erdos-Renyi random graph or a small-world graph --
this script just reads whatever numbers are on disk.
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os
from load_weights import load_trial, load_theta_star

# ============================================================
# 0. SETTINGS -- these are the only two lines you change between
#    an Erdos-Renyi run and a small-world run
# ============================================================
data_dir = 'data'      # folder produced by brian_snn_training.py for THIS connectivity
n_trials = 800         # block_no * stim_no used during that training run (last saved trial)

# ============================================================
# 1. Load what never changes during training: which neuron connects
#    to which, and every neuron's preferred orientation (theta_star)
# ============================================================
conn = np.load(os.path.join(data_dir, 'connectivity.npz'))
ee_i, ee_j = conn['ee_i'], conn['ee_j']
ei_i, ei_j = conn['ei_i'], conn['ei_j']
ie_i, ie_j = conn['ie_i'], conn['ie_j']
ii_i, ii_j = conn['ii_i'], conn['ii_j']
ne, ni = int(conn['ne']), int(conn['ni'])
num_neurons = ne + ni

theta_star_exc, theta_star_inh = load_theta_star(data_dir)

# initial (pre-training) weights are always the same fixed numbers in
# this model -- no need for a "trial 0" file, we just know the values:
J = 0.5          # mV, EPSP
g = 8.0          # inhibition dominance ratio
J_exc_initial = J          # mV -- for E->E and E->I
J_inh_initial = -g * J      # mV -- for I->E and I->I

final_snap = load_trial(n_trials, data_dir=data_dir)   # the fully trained weights

# ============================================================
# 2. Rebuild just enough of the network to READ OUT firing rates.
#    Weights are plain (non-plastic) here on purpose -- exactly like
#    the paper "freezes" weights when it measures tuning curves.
# ============================================================
tau = 20*ms
u_th = 20*mV
u_reset = 0*mV
t_ref = 0*ms

eqs = '''
du/dt = -u/tau : volt
theta_star : 1 (constant)
'''

neurons = NeuronGroup(num_neurons, eqs, threshold='u>u_th', reset='u=u_reset',
                       refractory=t_ref, method='euler')
neurons.u = 0*mV

exc = neurons[:ne]
inh = neurons[ne:]
exc.theta_star = theta_star_exc
inh.theta_star = theta_star_inh

S_EE = Synapses(exc, exc, model='w : volt', on_pre='u_post += w')
S_EE.connect(i=ee_i, j=ee_j)

S_EI = Synapses(exc, inh, model='w : volt', on_pre='u_post += w')
S_EI.connect(i=ei_i, j=ei_j)

S_IE = Synapses(inh, exc, model='w : volt', on_pre='u_post += w')
S_IE.connect(i=ie_i, j=ie_j)

S_II = Synapses(inh, inh, model='w : volt', on_pre='u_post += w')
S_II.connect(i=ii_i, j=ii_j)

# feedforward pathway (same structure as brian_snn_training.py)
s_b = 2*kHz
mu_exc = 0.2
mu_inh = 0.02
J_ffw = 1*mV

ff_exc = PoissonGroup(ne, rates=0*Hz)
ff_inh = PoissonGroup(ni, rates=0*Hz)
ff_exc_syn = Synapses(ff_exc, exc, on_pre='u_post += J_ffw')
ff_exc_syn.connect(j='i')
ff_inh_syn = Synapses(ff_inh, inh, on_pre='u_post += J_ffw')
ff_inh_syn.connect(j='i')

spikemon = SpikeMonitor(neurons)
net = Network(collect())

# ============================================================
# 3. Stimulus set for measuring tuning curves / OSI: 8 orientations,
#    same protocol the paper uses (0 to 157.5 deg in 22.5 deg steps)
# ============================================================
test_oris_deg = np.arange(0, 180, 22.5)     # degrees
test_oris_rad = np.radians(test_oris_deg)   # radians
stim_dur = 2*second
n_oris = len(test_oris_deg)

# ============================================================
# 4. Measure firing rates for BOTH phases (before / after training).
#    Same code runs twice in this loop -- only the weights differ.
# ============================================================
rates_before = np.zeros((num_neurons, n_oris))
rates_after = np.zeros((num_neurons, n_oris))

for phase in ['before', 'after']:

    if phase == 'before':
        S_EE.w = J_exc_initial * mV
        S_EI.w = J_exc_initial * mV
        S_IE.w = J_inh_initial * mV
        S_II.w = J_inh_initial * mV
    else:
        S_EE.w = final_snap['w_ee'] * mV
        S_EI.w = final_snap['w_ei'] * mV
        S_IE.w = final_snap['w_ie'] * mV
        S_II.w = J_inh_initial * mV   # I->I is never plastic, stays at its initial value

    net.store(phase)   # snapshot taken right after setting this phase's weights

    for k in range(n_oris):
        theta = test_oris_rad[k]
        rate_exc_k = s_b * (1 + mu_exc * np.cos(2 * (theta - theta_star_exc)))
        rate_inh_k = s_b * (1 + mu_inh * np.cos(2 * (theta - theta_star_inh)))
        ff_exc.rates = rate_exc_k
        ff_inh.rates = rate_inh_k

        spikes_before = spikemon.count[:].copy()
        net.run(stim_dur)
        spikes_after = spikemon.count[:].copy()
        rate_k = (spikes_after - spikes_before) / stim_dur

        if phase == 'before':
            rates_before[:, k] = rate_k
        else:
            rates_after[:, k] = rate_k

        net.restore(phase)   # reset membrane potential / spike counts for the next
                              # orientation -- weights are untouched, since the
                              # snapshot was taken AFTER they were set above

# ============================================================
# 5. OSI per neuron = 1 - circular variance (paper's Materials and Methods)
# ============================================================
osi_before = np.abs(np.sum(rates_before * np.exp(2j*test_oris_rad)[None, :], axis=1)) / \
             (np.sum(rates_before, axis=1) + 1e-12)
osi_after = np.abs(np.sum(rates_after * np.exp(2j*test_oris_rad)[None, :], axis=1)) / \
            (np.sum(rates_after, axis=1) + 1e-12)

print(f"Data from: {data_dir}")
print(f"OSI, example excitatory neuron  -- before: {osi_before[0]:.3f}   after: {osi_after[0]:.3f}")
print(f"OSI, example inhibitory neuron  -- before: {osi_before[ne]:.3f}   after: {osi_after[ne]:.3f}")
print(f"OSI, excitatory population avg -- before: {np.mean(osi_before[:ne]):.3f}   after: {np.mean(osi_after[:ne]):.3f}")
print(f"OSI, inhibitory population avg -- before: {np.mean(osi_before[ne:]):.3f}   after: {np.mean(osi_after[ne:]):.3f}")

# ============================================================
# 6. x-axis for plotting: difference between stimulus orientation and
#    each neuron's OWN preferred orientation, wrapped into (-180, 180]
#    degrees -- this is what makes tuning curves line up on "0 = tuned"
# ============================================================
theta_star_deg_exc = np.degrees(theta_star_exc)
theta_star_deg_inh = np.degrees(theta_star_inh)

dpo_exc = test_oris_deg[None, :] - theta_star_deg_exc[:, None]   # shape (ne, n_oris)
dpo_exc = ((dpo_exc + 180) % 360) - 180

dpo_inh = test_oris_deg[None, :] - theta_star_deg_inh[:, None]   # shape (ni, n_oris)
dpo_inh = ((dpo_inh + 180) % 360) - 180

bins = np.arange(-180, 181, 15)               # 15-degree-wide bins, -180 to 180
bin_centers = (bins[:-1] + bins[1:]) / 2

example_exc = 0     # index of the example excitatory neuron (within 0..ne-1)
example_inh = 0      # index of the example inhibitory neuron (within 0..ni-1, i.e. neuron ne+0)

# ============================================================
# 7. PLOT 1/8 -- single excitatory neuron, before training
# ============================================================
x = dpo_exc[example_exc, :]
y = rates_before[example_exc, :]
order = np.argsort(x)
plt.figure(figsize=(6, 4))
plt.plot(x[order], y[order], 'o-', color='tab:red')
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Excitatory neuron #{example_exc} -- BEFORE training\nOSI = {osi_before[example_exc]:.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_before_single_exc.png', dpi=150)
plt.close()

# ============================================================
# 8. PLOT 2/8 -- single inhibitory neuron, before training
# ============================================================
x = dpo_inh[example_inh, :]
y = rates_before[ne + example_inh, :]
order = np.argsort(x)
plt.figure(figsize=(6, 4))
plt.plot(x[order], y[order], 'o-', color='tab:blue')
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Inhibitory neuron #{example_inh} -- BEFORE training\nOSI = {osi_before[ne + example_inh]:.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_before_single_inh.png', dpi=150)
plt.close()

# ============================================================
# 9. PLOT 3/8 -- excitatory population, before training (binned average)
# ============================================================
x_flat = dpo_exc.flatten()
y_flat = rates_before[:ne, :].flatten()
bin_mean = np.full(len(bin_centers), np.nan)
bin_std = np.full(len(bin_centers), np.nan)
for b in range(len(bin_centers)):
    mask = (x_flat >= bins[b]) & (x_flat < bins[b + 1])
    if mask.any():
        bin_mean[b] = y_flat[mask].mean()
        bin_std[b] = y_flat[mask].std()

plt.figure(figsize=(6, 4))
plt.plot(bin_centers, bin_mean, '-', color='tab:red')
plt.fill_between(bin_centers, bin_mean - bin_std, bin_mean + bin_std, color='tab:red', alpha=0.2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Excitatory population -- BEFORE training\nAvg OSI = {np.mean(osi_before[:ne]):.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_before_population_exc.png', dpi=150)
plt.close()

# ============================================================
# 10. PLOT 4/8 -- inhibitory population, before training (binned average)
# ============================================================
x_flat = dpo_inh.flatten()
y_flat = rates_before[ne:, :].flatten()
bin_mean = np.full(len(bin_centers), np.nan)
bin_std = np.full(len(bin_centers), np.nan)
for b in range(len(bin_centers)):
    mask = (x_flat >= bins[b]) & (x_flat < bins[b + 1])
    if mask.any():
        bin_mean[b] = y_flat[mask].mean()
        bin_std[b] = y_flat[mask].std()

plt.figure(figsize=(6, 4))
plt.plot(bin_centers, bin_mean, '-', color='tab:blue')
plt.fill_between(bin_centers, bin_mean - bin_std, bin_mean + bin_std, color='tab:blue', alpha=0.2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Inhibitory population -- BEFORE training\nAvg OSI = {np.mean(osi_before[ne:]):.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_before_population_inh.png', dpi=150)
plt.close()

# ============================================================
# 11. PLOT 5/8 -- single excitatory neuron, after training
# ============================================================
x = dpo_exc[example_exc, :]
y = rates_after[example_exc, :]
order = np.argsort(x)
plt.figure(figsize=(6, 4))
plt.plot(x[order], y[order], 'o-', color='tab:red')
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Excitatory neuron #{example_exc} -- AFTER training\nOSI = {osi_after[example_exc]:.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_after_single_exc.png', dpi=150)
plt.close()

# ============================================================
# 12. PLOT 6/8 -- single inhibitory neuron, after training
# ============================================================
x = dpo_inh[example_inh, :]
y = rates_after[ne + example_inh, :]
order = np.argsort(x)
plt.figure(figsize=(6, 4))
plt.plot(x[order], y[order], 'o-', color='tab:blue')
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Inhibitory neuron #{example_inh} -- AFTER training\nOSI = {osi_after[ne + example_inh]:.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_after_single_inh.png', dpi=150)
plt.close()

# ============================================================
# 13. PLOT 7/8 -- excitatory population, after training (binned average)
# ============================================================
x_flat = dpo_exc.flatten()
y_flat = rates_after[:ne, :].flatten()
bin_mean = np.full(len(bin_centers), np.nan)
bin_std = np.full(len(bin_centers), np.nan)
for b in range(len(bin_centers)):
    mask = (x_flat >= bins[b]) & (x_flat < bins[b + 1])
    if mask.any():
        bin_mean[b] = y_flat[mask].mean()
        bin_std[b] = y_flat[mask].std()

plt.figure(figsize=(6, 4))
plt.plot(bin_centers, bin_mean, '-', color='tab:red')
plt.fill_between(bin_centers, bin_mean - bin_std, bin_mean + bin_std, color='tab:red', alpha=0.2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Excitatory population -- AFTER training\nAvg OSI = {np.mean(osi_after[:ne]):.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_after_population_exc.png', dpi=150)
plt.close()

# ============================================================
# 14. PLOT 8/8 -- inhibitory population, after training (binned average)
# ============================================================
x_flat = dpo_inh.flatten()
y_flat = rates_after[ne:, :].flatten()
bin_mean = np.full(len(bin_centers), np.nan)
bin_std = np.full(len(bin_centers), np.nan)
for b in range(len(bin_centers)):
    mask = (x_flat >= bins[b]) & (x_flat < bins[b + 1])
    if mask.any():
        bin_mean[b] = y_flat[mask].mean()
        bin_std[b] = y_flat[mask].std()

plt.figure(figsize=(6, 4))
plt.plot(bin_centers, bin_mean, '-', color='tab:blue')
plt.fill_between(bin_centers, bin_mean - bin_std, bin_mean + bin_std, color='tab:blue', alpha=0.2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Stim. orientation - preferred orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Inhibitory population -- AFTER training\nAvg OSI = {np.mean(osi_after[ne:]):.3f}')
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig('osi_after_population_inh.png', dpi=150)
plt.close()

print("Saved 8 plots: osi_{before,after}_{single_exc,single_inh,population_exc,population_inh}.png")
