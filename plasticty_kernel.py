"""
Build the empirical plasticity kernel Δw(Δt) for each synapse type in the
full network model, using the exact model/on_pre code blocks from the
original script:

  - E->E and E->I both use `exc_plastic_model` / `exc_on_pre`
    (continuous LTP + event-driven LTD). Since theta_minus/theta_plus are
    identical for exc and inh neurons in the full model, E->E and E->I
    produce the SAME kernel -- postsynaptic population identity doesn't
    change the rule itself, only the connectivity it's embedded in.
  - I->E uses `inh_plastic_model` / `inh_on_pre` (LTD-only, no continuous
    LTP term, weight bounded in [w_max_inh, 0]).
  - I->I is non-plastic ('u_post += w' only) -- included for completeness,
    trivially flat.
"""
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# ---- shared model parameters (identical to the full network script) ----
tau = 20 * ms
u_th = 20 * mV
u_reset = 0 * mV
tau_minus = 10 * ms
tau_plus = 7 * ms
tau_x = 15 * ms

vth_m = -20.0   # theta- , mV  (the "as-is" value from the pasted script)
vth_p = 7.5     # theta+ , mV

A_ltd = 14.e-5
A_ltp = 8.e-5
w_min = 0.0
w_max = 2.0
w_max_inh = -5.0

force_w = 40 * mV  # forcing-synapse weight, guarantees post spikes exactly on schedule

eqs = '''
du/dt = -u/tau                     : volt
du_lp1/dt = (u - u_lp1)/tau_minus  : volt
du_lp2/dt = (u - u_lp2)/tau_plus   : volt
theta_minus : volt (constant)
theta_plus  : volt (constant)
'''

# ---- the three rule types, copied verbatim from the full network ----
exc_plastic_model = '''
dx_bar/dt = -x_bar/tau_x : 1 (clock-driven)
dw/dt = A_ltp * x_bar * clip(u_post - theta_plus_post, 0*mV, 1e9*mV) * clip(u_lp2_post - theta_minus_post, 0*mV, 1e9*mV) / mV**2 / ms * mV : volt (clock-driven)
'''
exc_on_pre = '''
u_post += w
x_bar += 1
w = clip(w/mV - A_ltd * clip(u_lp1_post/mV - theta_minus_post/mV, 0, 1e9), w_min, w_max) * mV
'''

inh_plastic_model = 'w : volt'
inh_on_pre = '''
u_post += w
w = clip(w/mV - A_ltd * clip(u_lp1_post/mV - theta_minus_post/mV, 0, 1e9), w_max_inh, 0) * mV
'''

nonplastic_model = 'w : volt'
nonplastic_on_pre = 'u_post += w'


def run_pair(dt_offset_ms, synapse_type):
    """dt_offset_ms = t_post - t_pre. Returns w_final - w0 (mV)."""
    start_scope()

    t_pre = 200 * ms
    t_post = t_pre + dt_offset_ms * ms
    t_end = max(t_pre, t_post) + 300 * ms

    pre_gen = SpikeGeneratorGroup(1, [0], [t_pre])
    post_force_gen = SpikeGeneratorGroup(1, [0], [t_post])

    post = NeuronGroup(1, eqs, threshold='u > u_th', reset='u = u_reset',
                        refractory=0 * ms, method='euler')
    post.u = 0 * mV
    post.theta_plus = vth_p * mV
    post.theta_minus = vth_m * mV

    force_syn = Synapses(post_force_gen, post, on_pre='u_post += force_w')
    force_syn.connect()

    if synapse_type in ('EE', 'EI'):
        model, on_pre, w0 = exc_plastic_model, exc_on_pre, 1.0 * mV
    elif synapse_type == 'IE':
        model, on_pre, w0 = inh_plastic_model, inh_on_pre, -1.0 * mV
    elif synapse_type == 'II':
        model, on_pre, w0 = nonplastic_model, nonplastic_on_pre, -1.0 * mV
    else:
        raise ValueError(synapse_type)

    S = Synapses(pre_gen, post, model=model, on_pre=on_pre, method='euler')
    S.connect()
    S.w = w0
    if synapse_type in ('EE', 'EI'):
        S.x_bar = 0

    net = Network(pre_gen, post_force_gen, post, force_syn, S)
    net.run(t_end)

    return float(S.w[0] / mV) - float(w0 / mV)


dts = np.arange(-60, 61, 2)  # ms, negative = post fires before pre

synapse_types = ['EE', 'EI', 'IE', 'II']
results = {}
for stype in synapse_types:
    dw = [run_pair(dt, stype) for dt in dts]
    results[stype] = dw
    print(f"{stype}: dw range [{min(dw):.6f}, {max(dw):.6f}] mV, peak at "
          f"Δt={dts[int(np.argmax(np.abs(dw)))]} ms")

plt.figure(figsize=(8, 5))
styles = {'EE': ('-', 'o'), 'EI': ('--', 's'), 'IE': ('-', '^'), 'II': ('-', 'x')}
for stype in synapse_types:
    ls, marker = styles[stype]
    plt.plot(dts, results[stype], ls, marker=marker, markersize=4,
              label=f'{stype[0]}\u2192{stype[1]}', alpha=0.85)
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.xlabel('Δt = t_post − t_pre (ms)')
plt.ylabel('Δw (mV)')
plt.title('Empirical plasticity kernel by synapse type (theta_minus = -20 mV, as-is)')
plt.legend()
plt.tight_layout()
plt.savefig('pair_kernel_by_type.png', dpi=150)
print("Saved pair_kernel_by_type.png")

# sanity check: EE and EI should be numerically identical
diff = np.max(np.abs(np.array(results['EE']) - np.array(results['EI'])))
print(f"Max |EE - EI| difference across all Δt: {diff:.10f} mV (expect ~0)")
