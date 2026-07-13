import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# Ensure we can import load_weights from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_weights import load_trial, load_theta_star

# Set up data directories relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

# Seed the random number generators for reproducibility
np.random.seed(42)

def run_tuning_simulation():
    # 1. Load snapshots and orientations
    conn = np.load(os.path.join(data_dir, 'connectivity.npz'))
    ee_i, ee_j = conn['ee_i'], conn['ee_j']
    ei_i, ei_j = conn['ei_i'], conn['ei_j']
    ie_i, ie_j = conn['ie_i'], conn['ie_j']
    ii_i, ii_j = conn['ii_i'], conn['ii_j']
    ne, ni = int(conn['ne']), int(conn['ni'])
    num_neurons = ne + ni
    
    theta_star_exc, theta_star_inh = load_theta_star(data_dir)
    
    # Snapshot parameters
    J = 0.5          # mV, EPSP
    g = 8.0          # inhibition dominance ratio
    J_exc_initial = J
    J_inh_initial = -g * J
    
    snap_before = load_trial(1, data_dir=data_dir)
    snap_after = load_trial(799, data_dir=data_dir)
    
    # 2. Rebuild the frozen network model
    start_scope()
    defaultclock.dt = 0.1*ms
    
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
    
    # Stimulus feedforward Poisson pathway
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
    
    # Setup test orientations
    test_oris_deg = np.arange(0, 180, 22.5)     # degrees
    test_oris_rad = np.radians(test_oris_deg)   # radians
    stim_dur = 1.0 * second
    n_oris = len(test_oris_deg)
    
    rates_before = np.zeros((num_neurons, n_oris))
    rates_after = np.zeros((num_neurons, n_oris))
    
    # 3. Simulate both phases
    for phase in ['before', 'after']:
        print(f"  Measuring tuning curves for phase: {phase}...")
        if phase == 'before':
            S_EE.w = snap_before['w_ee'] * mV
            S_EI.w = snap_before['w_ei'] * mV
            S_IE.w = snap_before['w_ie'] * mV
            S_II.w = J_inh_initial * mV
        else:
            S_EE.w = snap_after['w_ee'] * mV
            S_EI.w = snap_after['w_ei'] * mV
            S_IE.w = snap_after['w_ie'] * mV
            S_II.w = J_inh_initial * mV
            
        net.store(phase)
        
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
                
            net.restore(phase)
            
    # Calculate OSI: OSI = | sum_k r_k * exp(2i*theta_k) | / sum_k r_k
    osi_before = np.abs(np.sum(rates_before * np.exp(2j * test_oris_rad)[None, :], axis=1)) / \
                 (np.sum(rates_before, axis=1) + 1e-12)
    osi_after = np.abs(np.sum(rates_after * np.exp(2j * test_oris_rad)[None, :], axis=1)) / \
                (np.sum(rates_after, axis=1) + 1e-12)
                
    return test_oris_deg, rates_before, rates_after, osi_before, osi_after, ne, ni, theta_star_exc, theta_star_inh

if __name__ == '__main__':
    print("Running firing rate and tuning curves simulation...")
    test_oris, rates_before, rates_after, osi_before, osi_after, ne, ni, theta_star_exc, theta_star_inh = run_tuning_simulation()
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True, sharey='row')
    
    # 10 random indices for Excitatory and Inhibitory groups to draw individual curves
    np.random.seed(123)
    sample_exc_idx = np.random.choice(ne, size=10, replace=False)
    sample_inh_idx = np.random.choice(ni, size=10, replace=False)
    sample_all_idx = np.random.choice(ne + ni, size=10, replace=False)
    
    # Combined theta_stars
    theta_star_all = np.concatenate([theta_star_exc, theta_star_inh])
    
    # Define row mapping: (Row Title, Indices slice/list, samples, color, label, osi_before, osi_after, theta_stars, rates_before, rates_after)
    row_mappings = [
        ("Entire Population", slice(None), sample_all_idx, 'tab:purple', 'Population', osi_before, osi_after, theta_star_all, rates_before, rates_after),
        ("Excitatory Population", slice(0, ne), sample_exc_idx, 'tab:blue', 'Excitatory', osi_before[:ne], osi_after[:ne], theta_star_exc, rates_before[:ne], rates_after[:ne]),
        ("Inhibitory Population", slice(ne, None), sample_inh_idx, 'tab:red', 'Inhibitory', osi_before[ne:], osi_after[ne:], theta_star_inh, rates_before[ne:], rates_after[ne:])
    ]
    
    for row, (name, pop_slice, samples, color, label, osi_b_sub, osi_a_sub, theta_star_sub, rates_b_sub, rates_a_sub) in enumerate(row_mappings):
        # Calculate OSI stats
        avg_osi_before = np.mean(osi_b_sub)
        avg_osi_after = np.mean(osi_a_sub)
        
        # We will pool all centered differences and rates to compute the binned population tuning curve
        x_pooled = []
        y_b_pooled = []
        y_a_pooled = []
        
        for i in range(len(rates_b_sub)):
            # Wrapped difference in [-90, 90] degrees
            x_i = ((test_oris - np.degrees(theta_star_sub[i]) + 90) % 180) - 90
            x_pooled.append(x_i)
            y_b_pooled.append(rates_b_sub[i])
            y_a_pooled.append(rates_a_sub[i])
            
        x_pooled = np.array(x_pooled).flatten()
        y_b_pooled = np.array(y_b_pooled).flatten()
        y_a_pooled = np.array(y_a_pooled).flatten()
        
        # Binning setup for mean population tuning curve
        bins = np.arange(-90, 91, 15)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_mean_b = np.zeros_like(bin_centers)
        bin_std_b = np.zeros_like(bin_centers)
        bin_mean_a = np.zeros_like(bin_centers)
        bin_std_a = np.zeros_like(bin_centers)
        
        for b in range(len(bin_centers)):
            mask = (x_pooled >= bins[b]) & (x_pooled < bins[b+1])
            if np.any(mask):
                bin_mean_b[b] = np.mean(y_b_pooled[mask])
                bin_std_b[b] = np.std(y_b_pooled[mask])
                bin_mean_a[b] = np.mean(y_a_pooled[mask])
                bin_std_a[b] = np.std(y_a_pooled[mask])
                
        # -------------------------------------------------------------
        # Column 0: Before Training
        # -------------------------------------------------------------
        # Plot individual samples (aligned so their theta_star is at 0)
        for idx in samples:
            x_i = ((test_oris - np.degrees(theta_star_sub[idx]) + 90) % 180) - 90
            y_i = rates_b_sub[idx]
            sort_idx = np.argsort(x_i)
            axes[row, 0].plot(x_i[sort_idx], y_i[sort_idx], color=color, alpha=0.15, linestyle='-', marker='o', markersize=2)
            
        # Plot average population tuning curve in thick line
        axes[row, 0].plot(bin_centers, bin_mean_b, color=color, linewidth=3.5, label=f'Avg Aligned {label} Rate')
        axes[row, 0].fill_between(bin_centers, bin_mean_b - bin_std_b, bin_mean_b + bin_std_b, color=color, alpha=0.15)
        axes[row, 0].set_ylabel('Firing Rate (Hz)')
        axes[row, 0].grid(True, linestyle=':', alpha=0.6)
        axes[row, 0].set_xlim(-90, 90)
        axes[row, 0].set_title(f'{name} BEFORE Training')
        
        # Overlay OSI in a text box
        text_box_b = f"Avg OSI = {avg_osi_before:.3f}"
        axes[row, 0].text(0.05, 0.95, text_box_b, transform=axes[row, 0].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        axes[row, 0].legend(loc='lower right')
        
        # -------------------------------------------------------------
        # Column 1: After Training
        # -------------------------------------------------------------
        # Plot individual samples (aligned so their theta_star is at 0)
        for idx in samples:
            x_i = ((test_oris - np.degrees(theta_star_sub[idx]) + 90) % 180) - 90
            y_i = rates_a_sub[idx]
            sort_idx = np.argsort(x_i)
            axes[row, 1].plot(x_i[sort_idx], y_i[sort_idx], color=color, alpha=0.15, linestyle='-', marker='o', markersize=2)
            
        # Plot average population tuning curve in thick line
        axes[row, 1].plot(bin_centers, bin_mean_a, color=color, linewidth=3.5, label=f'Avg Aligned {label} Rate')
        axes[row, 1].fill_between(bin_centers, bin_mean_a - bin_std_a, bin_mean_a + bin_std_a, color=color, alpha=0.15)
        axes[row, 1].grid(True, linestyle=':', alpha=0.6)
        axes[row, 1].set_xlim(-90, 90)
        axes[row, 1].set_title(f'{name} AFTER Training')
        
        # Overlay OSI in a text box
        text_box_a = f"Avg OSI = {avg_osi_after:.3f}"
        axes[row, 1].text(0.05, 0.95, text_box_a, transform=axes[row, 1].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        axes[row, 1].legend(loc='lower right')
        
    # Set x-label on bottom subplots
    axes[2, 0].set_xlabel('Stim. Orientation - Preferred Orientation (deg)')
    axes[2, 1].set_xlabel('Stim. Orientation - Preferred Orientation (deg)')
    
    # Make sure x-axis ticks are matching the visual steps
    for ax in axes.flat:
        ax.set_xticks(np.arange(-90, 91, 45))
        
    plt.tight_layout()
    plot_path = 'firing_rates_vs_theta.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Firing rate comparison plot saved to {plot_path}")
