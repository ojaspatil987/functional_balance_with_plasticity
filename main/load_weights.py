"""
Helper for reloading the per-trial weight snapshots produced by
brian_snn_training.py's data-saving protocol.

Layout on disk (all under ./data/ by default):
    data/theta_star.npz            -- theta_star_exc, theta_star_inh (static)
    data/connectivity.npz          -- (i, j) pairs for EE/EI/IE/II (static)
    data/schedule.npy              -- theta shown on every trial, 1-indexed
    data/weights_trial0001.npz     -- weights right after trial 1's theta
    data/weights_trial0002.npz
    ...
    data/weights_trial0800.npz

Every weight vector (w_ee, w_ei, w_ie) is aligned element-for-element with
the corresponding (i, j) index arrays in connectivity.npz, so you can drop
them straight into a fresh Synapses object.
"""

import os
import numpy as np


def load_trial(trial_idx, data_dir='data'):
    """
    Load the weight snapshot taken right after the given trial number
    (1-indexed, i.e. trial_idx=37 -> weights after the 37th theta shown
    over the whole training run).

    Returns a dict with:
        theta, theta_deg   -- orientation shown on this trial
        batch, trial_in_batch
        w_ee, w_ei, w_ie   -- weight vectors (in mV, plain floats)
        ee_i, ee_j, ei_i, ei_j, ie_i, ie_j, ii_i, ii_j -- connectivity
        ne, ni
    """
    fname = os.path.join(data_dir, f'weights_trial{trial_idx:04d}.npz')
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"No snapshot for trial {trial_idx} at {fname}. "
            f"Check the trial number, or that training actually ran that far."
        )
    snap = dict(np.load(fname))
    conn = dict(np.load(os.path.join(data_dir, 'connectivity.npz')))
    snap.update(conn)
    return snap


def load_theta_star(data_dir='data'):
    """Load the (static) preferred orientations for every neuron."""
    d = np.load(os.path.join(data_dir, 'theta_star.npz'))
    return d['theta_star_exc'], d['theta_star_inh']


def load_schedule(data_dir='data'):
    """Load the full training schedule (theta shown at every trial)."""
    return np.load(os.path.join(data_dir, 'schedule.npy'))


def restore_into_synapses(S_EE, S_EI, S_IE, trial_idx, data_dir='data'):
    """
    Convenience function: given fresh (unconnected) Synapses objects
    S_EE, S_EI, S_IE built with the SAME model/on_pre code as in
    brian_snn_training.py, connect them with the saved connectivity and
    load in the weights from `trial_idx`. Requires brian2 to be
    imported (for the `mV` unit) in the calling script.

    Example
    -------
        from brian2 import mV
        from load_weights import restore_into_synapses
        restore_into_synapses(S_EE, S_EI, S_IE, trial_idx=37)
    """
    from brian2 import mV  # local import so this module works without brian2 too

    snap = load_trial(trial_idx, data_dir=data_dir)

    S_EE.connect(i=snap['ee_i'], j=snap['ee_j'])
    S_EE.w = snap['w_ee'] * mV

    S_EI.connect(i=snap['ei_i'], j=snap['ei_j'])
    S_EI.w = snap['w_ei'] * mV

    S_IE.connect(i=snap['ie_i'], j=snap['ie_j'])
    S_IE.w = snap['w_ie'] * mV

    return snap


if __name__ == '__main__':
    # quick smoke test / usage demo
    snap = load_trial(37)
    print(f"Trial 37: theta = {snap['theta_deg']:.2f} deg, "
          f"batch {snap['batch']}, trial {snap['trial_in_batch']} in batch")
    print(f"E->E weight vector shape: {snap['w_ee'].shape}")
    print(f"Mean E->E weight at trial 37: {snap['w_ee'].mean():.4f} mV")
