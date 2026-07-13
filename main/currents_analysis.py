import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# Ensure we can import load_weights from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_weights import load_trial, load_theta_star

# Set up directories relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

# Seed the random number generators for reproducibility
np.random.seed(42)

def run_simulation_for_trial(trial_idx, test_theta, stim_duration=200*ms):
    # Clear any previous Brian2 objects to prevent conflicts
    start_scope()
    defaultclock.dt = 0.1*ms
    
    # Load snapshot and preferred orientations
    snap = load_trial(trial_idx, data_dir=data_dir)
    theta_star_exc, theta_star_inh = load_theta_star(data_dir=data_dir)
    
    ne = int(snap['ne'])
    ni = int(snap['ni'])
    num_neurons = ne + ni
    
    # Physical parameters (Table 1 of paper)
    tau = 20*ms
    u_th = 20*mV
    u_reset = 0*mV
    t_ref = 0*ms
    
    tau_current = 5*ms  # time constant for current filtering
    
    # Neuron equations with Excitatory and Inhibitory current monitors
    eqs = '''
    du/dt = -u/tau                     : volt
    du_lp1/dt = (u - u_lp1)/tau_minus  : volt
    du_lp2/dt = (u - u_lp2)/tau_plus   : volt
    theta_star  : 1 (constant)
    theta_minus : volt (constant)
    theta_plus  : volt (constant)
    dI_exc/dt = -I_exc/tau_current     : volt
    dI_inh/dt = -I_inh/tau_current     : volt
    '''
    
    # Define globally used variables in equations (matching main code)
    tau_minus = 10*ms
    tau_plus = 7*ms
    vth_m = -20.0
    vth_p = 7.5
    
    neurons = NeuronGroup(num_neurons, eqs, threshold='u>u_th', reset='u=u_reset',
                          refractory=t_ref, method='euler')
    neurons.u = 0*mV
    neurons.theta_minus = vth_m*mV
    neurons.theta_plus  = vth_p*mV
    neurons.theta_star[:ne] = theta_star_exc
    neurons.theta_star[ne:] = theta_star_inh
    
    # Synapse models (weights are static during this analysis run)
    exc_model = 'w : volt'
    exc_on_pre = '''
    u_post += w
    I_exc_post += w
    '''
    
    inh_model = 'w : volt'
    inh_on_pre = '''
    u_post += w
    I_inh_post += w
    '''
    
    exc = neurons[:ne]
    inh = neurons[ne:]
    
    # Synapses creation and restoration
    S_EE = Synapses(exc, exc, model=exc_model, on_pre=exc_on_pre, method='euler')
    S_EE.connect(i=snap['ee_i'], j=snap['ee_j'])
    S_EE.w = snap['w_ee'] * mV
    
    S_EI = Synapses(exc, inh, model=exc_model, on_pre=exc_on_pre, method='euler')
    S_EI.connect(i=snap['ei_i'], j=snap['ei_j'])
    S_EI.w = snap['w_ei'] * mV
    
    S_IE = Synapses(inh, exc, model=inh_model, on_pre=inh_on_pre, method='euler')
    S_IE.connect(i=snap['ie_i'], j=snap['ie_j'])
    S_IE.w = snap['w_ie'] * mV
    
    # Inhibitory to inhibitory stays fully static (never plastic)
    S_II = Synapses(inh, inh, model='w : volt', on_pre='u_post += w')
    S_II.connect(i=snap['ii_i'], j=snap['ii_j'])
    J = 0.5
    g = 8.0
    J_exc = J * mV
    J_inh = -g * J_exc
    S_II.w = J_inh
    
    # Stimulus feedforward pathway
    s_b = 2 * kHz
    mu_exc = 0.2
    mu_inh = 0.02
    J_ffw = 1 * mV
    
    rate_exc = s_b * (1 + mu_exc * np.cos(2 * (test_theta - exc.theta_star[:])))
    rate_inh = s_b * (1 + mu_inh * np.cos(2 * (test_theta - inh.theta_star[:])))
    
    ff_exc = PoissonGroup(ne, rates=rate_exc)
    ff_inh = PoissonGroup(ni, rates=rate_inh)
    
    ff_exc_syn = Synapses(ff_exc, exc, on_pre='u_post += J_ffw; I_exc_post += J_ffw')
    ff_exc_syn.connect(j='i')
    
    ff_inh_syn = Synapses(ff_inh, inh, on_pre='u_post += J_ffw; I_exc_post += J_ffw')
    ff_inh_syn.connect(j='i')
    
    # Monitor to record I_exc and I_inh
    statemon = StateMonitor(neurons, ['I_exc', 'I_inh'], record=True)
    
    run(stim_duration)
    
    return statemon.t, statemon.I_exc, statemon.I_inh, ne

if __name__ == '__main__':
    # 1. Select a random visual stimulus orientation
    test_theta = np.random.uniform(0, np.pi)
    print(f"Using random visual stimulus theta: {np.degrees(test_theta):.2f}°")
    
    # 2. Run simulation before training (Trial 1)
    print("Running simulation before training (Trial 1)...")
    t_before, I_exc_before, I_inh_before, ne = run_simulation_for_trial(1, test_theta)
    
    # 3. Run simulation after training (Trial 799)
    print("Running simulation after training (Trial 799)...")
    t_after, I_exc_after, I_inh_after, _ = run_simulation_for_trial(799, test_theta)
    
    # 4. Generate Plot 1: Sample Excitatory Neuron
    sample_idx = np.random.randint(0, ne)
    print(f"Selected sample excitatory neuron index: {sample_idx}")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot before training
    axes[0].plot(t_before / ms, I_exc_before[sample_idx] / mV, label='Excitatory Current (recurrent + feedforward)', color='tab:blue')
    axes[0].plot(t_before / ms, I_inh_before[sample_idx] / mV, label='Inhibitory Current', color='tab:red')
    axes[0].plot(t_before / ms, (I_exc_before[sample_idx] + I_inh_before[sample_idx]) / mV, label='Net Current', color='black')
    axes[0].set_ylabel('Potential (mV)')
    axes[0].set_title(f'Sample Neuron {sample_idx} currents BEFORE training (theta = {np.degrees(test_theta):.1f}°)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot after training
    axes[1].plot(t_after / ms, I_exc_after[sample_idx] / mV, label='Excitatory Current (recurrent + feedforward)', color='tab:blue')
    axes[1].plot(t_after / ms, I_inh_after[sample_idx] / mV, label='Inhibitory Current', color='tab:red')
    axes[1].plot(t_after / ms, (I_exc_after[sample_idx] + I_inh_after[sample_idx]) / mV, label='Net Current', color='black')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Potential (mV)')
    axes[1].set_title(f'Sample Neuron {sample_idx} currents AFTER training (theta = {np.degrees(test_theta):.1f}°)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    sample_plot_path = 'sample_neuron_currents.png'
    plt.savefig(sample_plot_path, dpi=150)
    print(f"Sample neuron plot saved to {sample_plot_path}")
    
    # 5. Generate Plot 2: Average currents over the Excitatory population
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    avg_exc_before = np.mean(I_exc_before[:ne, :], axis=0)
    avg_inh_before = np.mean(I_inh_before[:ne, :], axis=0)
    
    avg_exc_after = np.mean(I_exc_after[:ne, :], axis=0)
    avg_inh_after = np.mean(I_inh_after[:ne, :], axis=0)
    
    # Plot before training
    axes[0].plot(t_before / ms, avg_exc_before / mV, label='Average Excitatory Current', color='tab:blue')
    axes[0].plot(t_before / ms, avg_inh_before / mV, label='Average Inhibitory Current', color='tab:red')
    axes[0].plot(t_before / ms, (avg_exc_before + avg_inh_before) / mV, label='Average Net Current', color='black')
    axes[0].set_ylabel('Potential (mV)')
    axes[0].set_title(f'Average Excitatory population currents BEFORE training (theta = {np.degrees(test_theta):.1f}°)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot after training
    axes[1].plot(t_after / ms, avg_exc_after / mV, label='Average Excitatory Current', color='tab:blue')
    axes[1].plot(t_after / ms, avg_inh_after / mV, label='Average Inhibitory Current', color='tab:red')
    axes[1].plot(t_after / ms, (avg_exc_after + avg_inh_after) / mV, label='Average Net Current', color='black')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Potential (mV)')
    axes[1].set_title(f'Average Excitatory population currents AFTER training (theta = {np.degrees(test_theta):.1f}°)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    avg_plot_path = 'average_currents.png'
    plt.savefig(avg_plot_path, dpi=150)
    print(f"Average currents plot saved to {avg_plot_path}")
