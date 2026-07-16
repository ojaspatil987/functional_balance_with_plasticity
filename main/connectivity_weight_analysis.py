import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import load_weights from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_weights import load_trial, load_theta_star

# Set up data directories relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

def get_po_diff_deg(theta1, theta2):
    """
    Computes the wrapped circular difference in degrees between two orientation angles.
    Since orientations are periodic in [0, pi] (0 to 180 degrees), the maximum 
    difference is 90 degrees.
    """
    deg1 = np.degrees(theta1)
    deg2 = np.degrees(theta2)
    diff = np.abs(deg1 - deg2)
    return np.minimum(diff, 180.0 - diff)

def plot_binned_trend(ax, x, y, bins=15, color='darkblue'):
    """
    Computes and plots a running mean trend of the y values binned along x,
    along with a shaded area representing the first standard deviation about the mean.
    Uses pure numpy to avoid any external scipy dependencies.
    """
    if len(x) == 0:
        return
    bin_edges = np.linspace(0, 90, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = []
    bin_stds = []
    
    for i in range(bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if i == bins - 1:
            mask = mask | (x == bin_edges[i+1])
        
        if np.any(mask):
            bin_means.append(np.mean(y[mask]))
            bin_stds.append(np.std(y[mask]))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
            
    ax.plot(bin_centers, bin_means, color=color, linewidth=3.0, label='Mean Trend')
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, color=color, alpha=0.15, label='±1 SD')
    ax.legend(loc='upper right')

if __name__ == '__main__':
    # 1. Load snapshots and orientation data
    print("Loading network snapshots and orientation data...")
    snap_before = load_trial(1, data_dir=data_dir)
    snap_after = load_trial(799, data_dir=data_dir)
    theta_star_exc, theta_star_inh = load_theta_star(data_dir=data_dir)
    
    ne = int(snap_before['ne'])
    ni = int(snap_before['ni'])
    
    # 2. Compute orientation differences for each synapse group
    print("Computing orientation differences between presynaptic and postsynaptic neurons...")
    # E -> E
    po_diff_ee = get_po_diff_deg(theta_star_exc[snap_before['ee_i']], theta_star_exc[snap_before['ee_j']])
    w_ee_before = snap_before['w_ee']
    w_ee_after = snap_after['w_ee']
    
    # E -> I
    po_diff_ei = get_po_diff_deg(theta_star_exc[snap_before['ei_i']], theta_star_inh[snap_before['ei_j']])
    w_ei_before = snap_before['w_ei']
    w_ei_after = snap_after['w_ei']
    
    # I -> E
    po_diff_ie = get_po_diff_deg(theta_star_inh[snap_before['ie_i']], theta_star_exc[snap_before['ie_j']])
    w_ie_before = snap_before['w_ie']
    w_ie_after = snap_after['w_ie']
    
    # I -> I (Static, default weights J_inh = -4.0 mV)
    po_diff_ii = get_po_diff_deg(theta_star_inh[snap_before['ii_i']], theta_star_inh[snap_before['ii_j']])
    w_ii_before = np.ones_like(snap_before['ii_i']) * -4.0
    w_ii_after = np.ones_like(snap_after['ii_i']) * -4.0
    
    # Concatenate all groups for the "all synapses" plot
    po_diff_all = np.concatenate([po_diff_ee, po_diff_ei, po_diff_ie, po_diff_ii])
    w_all_before = np.concatenate([w_ee_before, w_ei_before, w_ie_before, w_ii_before])
    w_all_after = np.concatenate([w_ee_after, w_ei_after, w_ie_after, w_ii_after])
    
    # 3. Generate Plot 1: All Synapses Plot (Side-by-side comparison)
    print("Generating Plot 1: All Synapses...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Before Training
    # axes[0].scatter(po_diff_all, w_all_before, s=1, alpha=0.1, color='tab:gray', rasterized=True)
    plot_binned_trend(axes[0], po_diff_all, w_all_before, bins=15, color='tab:blue')
    axes[0].set_title('All Synaptic Weights BEFORE Training')
    axes[0].set_xlabel('Orientation Difference Δθ* (deg)')
    axes[0].set_ylabel('Synaptic Weight (mV)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].set_xlim(0, 90)
    
    # After Training
    # axes[1].scatter(po_diff_all, w_all_after, s=1, alpha=0.1, color='tab:gray', rasterized=True)
    plot_binned_trend(axes[1], po_diff_all, w_all_after, bins=15, color='tab:red')
    axes[1].set_title('All Synaptic Weights AFTER Training')
    axes[1].set_xlabel('Orientation Difference Δθ* (deg)')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].set_xlim(0, 90)
    
    plt.tight_layout()
    all_plot_path = 'sample_all_synapses.png'
    plt.savefig(all_plot_path, dpi=150)
    print(f"All synapses plot saved to {all_plot_path}")
    
    # 4. Generate Plot 2: Subgroups synapses Grid Plot (4 rows, 2 columns)
    print("Generating Plot 2: Subgroup Synapses (E->E, E->I, I->E, I->I)...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    
    # Setup for each row: (Row Name, diff data, weights before, weights after, ylimits, scatter color, trend color)
    subgroups = [
        ('E -> E', po_diff_ee, w_ee_before, w_ee_after, (0.0, 2.55), 'tab:blue', 'navy'),
        ('E -> I', po_diff_ei, w_ei_before, w_ei_after, (0.0, 2.55), 'tab:green', 'darkgreen'),
        ('I -> E', po_diff_ie, w_ie_before, w_ie_after, (-5.55, 0.05), 'tab:red', 'darkred'),
        ('I -> I (Static)', po_diff_ii, w_ii_before, w_ii_after, (-5.55, 0.05), 'tab:purple', 'indigo')
    ]
    
    for row, (name, diff, w_before, w_after, ylim, color_scatter, color_trend) in enumerate(subgroups):
        # Column 0: Before Training
        # axes[row, 0].scatter(diff, w_before, s=3, alpha=0.15, color=color_scatter, rasterized=True)
        plot_binned_trend(axes[row, 0], diff, w_before, bins=15, color=color_trend)
        axes[row, 0].set_title(f'{name} Weights BEFORE Training')
        axes[row, 0].set_ylabel('Synaptic Weight (mV)')
        axes[row, 0].grid(True, linestyle=':', alpha=0.6)
        axes[row, 0].set_xlim(0, 90)
        axes[row, 0].set_ylim(ylim)
        
        # Column 1: After Training
        # axes[row, 1].scatter(diff, w_after, s=3, alpha=0.15, color=color_scatter, rasterized=True)
        plot_binned_trend(axes[row, 1], diff, w_after, bins=15, color=color_trend)
        axes[row, 1].set_title(f'{name} Weights AFTER Training')
        axes[row, 1].grid(True, linestyle=':', alpha=0.6)
        axes[row, 1].set_xlim(0, 90)
        axes[row, 1].set_ylim(ylim)
        
    # Label the x-axis for the bottom row
    axes[3, 0].set_xlabel('Orientation Difference Δθ* (deg)')
    axes[3, 1].set_xlabel('Orientation Difference Δθ* (deg)')
    
    plt.tight_layout()
    subgroups_plot_path = 'subgroups_synapses.png'
    plt.savefig(subgroups_plot_path, dpi=150)
    print(f"Subgroup synapses plot saved to {subgroups_plot_path}")
