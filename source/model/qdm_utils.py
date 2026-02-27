##########################################################################
## Util functions used in Quantum Diffusion Model
##
## Author: Tran Quoc Hoan, Start date: 2024/09/27
##########################################################################


import numpy as np
import os
import math

import qutip as qt
from utils.distance_jax import *
from data.data_utils import *
from model.qdm_jax import *
from model.qae_utils import *
from data.mol_data import *

def generate_input_data(input_type, n_qubits, n_data, rseed):
  # Examine the input data
  if input_type == 'line':
    input_states = gen_line_data_with_jax(nstates = n_data, nqubits = n_qubits, seed = rseed)
  elif input_type == 'circle':
    input_states = gen_circle_data_with_jax(nstates = n_data, nqubits = n_qubits, seed = rseed)
  elif input_type == 'product':
    input_states = gen_Haar_product_states(nstates = n_data, nqubits = n_qubits, seed = rseed)
  elif input_type == 'diffusion':
    # random basic states then we will apply the projected ensenmble framework later
    input_states = gen_rand_basis_states(nstates = n_data, nqubits = n_qubits, seed = rseed)
  else:
    # Random input
    input_states = gen_Haar_states(nstates = n_data, nqubits = n_qubits, seed = rseed)
  return input_states

def generate_real_data(dat_name, n_qubits, n_real_data, rseed, g_range=[0.0, 1.0], n_atoms=None, n_rings=None):
  # Examine the input data
  if dat_name == 'cluster0':
    real_states = gen_cluster_0(nstates = n_real_data, nqubits = n_qubits, scale = 0.06, seed = rseed)
  elif dat_name == 'multi_cluster':
    real_states = generate_multi_clustered_states(n_qubits, seed=rseed, N=n_real_data, scale=0.05)
  elif dat_name == 'line':
    real_states = gen_line_data_with_jax(nstates = n_real_data, nqubits = n_qubits)
  elif dat_name == 'circle':
    real_states = gen_circle_data_with_jax(nstates = n_real_data, nqubits = n_qubits)
  elif dat_name == 'tfim':
    real_states = gen_tfim_ground_states_qt(N = n_real_data, g_range=g_range, n = n_qubits, seed = rseed, use_dmrg=False)
  elif dat_name == 'qm9':
    full_dataset = QDrugDataset(dat_name, n_qubits, load_from_cache=True, file_path='../datasets/mol/', n_atoms=n_atoms)
    dataset, _ = filter_dataset_by_properties(full_dataset, target_num_rings=n_rings, target_n_atoms=n_atoms, min_mol_weight=None, max_mol_weight=None)
    real_states = dataset[:n_real_data, :-1]  # Exclude the last column which is not part of the quantum state
    #print(f"Info dataset: {dataset.info[-5:]}")
  else:
    raise NameError(f"Data name {dat_name} used does not match cluster0 or line")
  return real_states


def generate_data(input_type, dat_name, n_qubits, n_train, n_test, rseed, g_range=[0.0, 1.0], n_atoms=None, n_rings=None):
  train_input_states = generate_input_data(input_type, n_qubits, n_train, rseed+27)
  test_input_states = generate_input_data(input_type, n_qubits, n_test, rseed+2728)

  # Generate real states
  n_full_data = 10 * max(n_train, n_test)
  real_states = generate_real_data(dat_name, n_qubits, n_full_data, rseed+72, g_range, n_atoms, n_rings)
  return real_states, train_input_states, test_input_states

def generate_real_diff(logger, diff_file, real_states, n_diff_steps, n_qubits, rseed, scramb, n_pj_qubits=1, type_evol='full', \
                       delta_t=1.0, J=(-1.0, 0.0, 0.0), b=(-0.8090, -0.9045, 0.0), W=1.0, rand_ancilla=1, noise_info=None):
  # Obtain the data from diffusion process
  # if the data is not loaded let create the data
  if os.path.isfile(diff_file) == True:
    logger.info(f'Load diffusion file {diff_file}')
    X_out = np.load(diff_file)
  else:
    diff_hs = None
    if 'Ising' in scramb:
      logger.info(f'Scrambing with Chaotic {scramb} and random ancilla={rand_ancilla} n_pj_qubits={n_pj_qubits}')
      gamma_phi = 0.0
      if noise_info is not None:
        noise_level = noise_info.get('level', 0.0)
        noise_type = noise_info.get('type', 0)
        if noise_type > 0 and noise_level > 0.0:
          if noise_type < 0.5:
            gamma_phi = - math.log(1 - 2 * noise_level) / delta_t
          else:
            gamma_phi = None
      scramble = ChaoticScramblingModel(n_qubits=n_qubits, n_ancilla=n_pj_qubits, type_ham=scramb, J=J, b=b, W=W, rand_ancilla=rand_ancilla, gamma_phi=gamma_phi)
    else:
      logger.info(f'Scrambing with RUC {scramb}')
      diff_hs = delta_t * 1e-3 * np.arange(1, n_diff_steps+1)**2
      p1, p2 = 0.0, 0.0
      if noise_info is not None:
        noise_level = noise_info.get('level', 0.0)
        noise_type = noise_info.get('type', 0)
        if noise_type == 1:
          p1 = noise_level
        elif noise_type == 2:
          p2 = noise_level
        elif noise_type == 3:
          p1 = noise_level
          p2 = noise_level
      scramble = ScramblingModel(n_qubits=n_qubits, T=n_diff_steps, p1=p1, p2=p2)

    X = jnp.asarray(real_states, dtype=jnp.complex64)
    n_full_data = X.shape[0]
    X_out = np.zeros((n_diff_steps+1, n_full_data, 2**n_qubits), dtype=np.complex64)
    X_out[0] = X

    prev_states = real_states
    for k in range(1, n_diff_steps+1):
      print(f'Diffusion Step~{k} to generate data')
      if diff_hs is None:
        if type_evol == 'full':
          X_out[k] = scramble.diffusion_step(k * delta_t, real_states)
        else:
          # Apply projected ensemble to previous states with delta_t
          prev_states = scramble.diffusion_step(delta_t, prev_states)
          X_out[k] = prev_states
      else:
        X_out[k] = scramble.diffusion_step(k, real_states, diff_hs[:k])
    
    np.save(diff_file, X_out)
  return X_out

def compute_moment_operator(X, k):
    """
    Computes the k-th moment operator for an ensemble X (shape (N, d), complex rows as state vectors).
    """
    N, d = X.shape
    if N == 0:
        raise ValueError("Ensemble must have at least one state.")
    p = 1.0 / N
    # Convert to list of column kets (d, 1)
    kets = [qt.Qobj(X[i, :, None]) for i in range(N)]
    # Dimensions for the k-fold space
    dims = [[d] * k, [d] * k]
    # Initialize zero operator
    rho_k = qt.Qobj(np.zeros((d**k, d**k), dtype=complex), dims=dims)
    for psi in kets:
        rho = psi * psi.dag()  # |psi><psi|
        rho_tensor = qt.tensor([rho] * k)  # (|psi><psi|)^⊗k
        rho_k += p * rho_tensor
    return rho_k

def metric_evolution(metric_file, X0, Xout, k, use_trace_distance=False):
    """
    Computes or loads the Wasserstein and normalized Hilbert-Schmidt distances evolution 
    between X0 and each time slice in Xout. Assumes X0 and Xout contain pure quantum states 
    as complex row vectors (shape (N, d) and (T, Nfull, d), dtype=complex).
    
    The normalized HS distance for moment k is Δ^{(k)} = ||ρ₀^{(k)} - ρ_t^{(k)}||₂ / ||ρ₀^{(k)}||₂.
    
    Args:
        metric_file (str): Path to the .npz file for caching metrics.
        X0 (np.ndarray): Initial ensemble, shape (N, d).
        Xout (np.ndarray): Time-evolved ensembles, shape (T, Nfull, d).
        k (int): Moment order.
    
    Returns:
        dict: Metrics with 'wass' (Wasserstein) and 'hs_delta' (normalized HS distances), each shape (T,).
    """
    if os.path.isfile(metric_file):
        loaded = np.load(metric_file)
        metrics = {key: loaded[key] for key in loaded}
    else:
        metrics = {}
        
        T, Nfull = Xout.shape[0], Xout.shape[1]
        N = X0.shape[0]
        
        if use_trace_distance:
          # Precompute for rho0
          rho0 = compute_moment_operator(X0, k)

        inner00 = np.einsum('nd,md->nm', X0.conj(), X0)
        gram00 = np.abs(inner00)**2
        norm0_sq = np.mean(gram00 ** k)
        
        wass = np.zeros(T)
        trace_dist = np.zeros(T)
        hs_delta = np.zeros(T)
        
        for t in range(T):
            if Nfull > N:
                idx = np.random.choice(Nfull, N, replace=False)
            else:
                idx = np.arange(Nfull)
            Xt = Xout[t, idx]
            
            # Wasserstein
            wass[t] = wass_distance_jax(X0, Xt)
            
            # HS components
            inner0t = np.einsum('nd,md->nm', X0.conj(), Xt)
            gram0t = np.abs(inner0t)**2
            innertt = np.einsum('nd,md->nm', Xt.conj(), Xt)
            gramtt = np.abs(innertt)**2
            
            inner = np.mean(gram0t ** k)
            normt_sq = np.mean(gramtt ** k)
            
            diff_sq = norm0_sq + normt_sq - 2 * inner
            delta = np.sqrt(diff_sq) / np.sqrt(norm0_sq)
            hs_delta[t] = delta

            if use_trace_distance:
              # Trace distance
              rhot = compute_moment_operator(Xt, k)
              diff = rho0 - rhot
              trace_dist[t] = diff.norm('tr') / 2.0
        
        metrics['Wass'] = wass
        metrics['HS_delta'] = hs_delta
        if use_trace_distance:
          metrics['Trace_dist'] = trace_dist
        np.savez(metric_file, **metrics)
    
    return metrics


def distance_evolution(dist_file, X0, Xout, eval_mode=False, plot=True):
  if os.path.isfile(dist_file) == True:
    dists = np.load(dist_file)
  else:
    T1, Nfull = Xout.shape[0], Xout.shape[1]
    #print(T1, Nfull)

    # Sample N points from the full data
    N = X0.shape[0]
    
    mmd = np.zeros(T1)
    wass = np.zeros(T1)
    vendi = np.zeros(T1)

    for t in range(T1):
      if eval_mode == False:
        idx = np.random.choice(Nfull, N, replace=False)
        Xt = Xout[t, idx]
      else:
        Xt = Xout[t, :]
      mmd[t] = natural_distance_jax(X0, Xt)
      wass[t] = wass_distance_jax(X0, Xt)
      vendi[t] = vendi_score(Xt)
      
      #wass[t] = wass_distance_ott(X0, Xt, epsilon=0.1)
      #print(f'Step {t}, MMD: {mmd[t]}, WASS: {wass[t]}')
    
    dists = np.vstack((mmd, wass, vendi))
    np.save(dist_file, dists)
  if plot:
    plot_dist(dist_file, dists)
  return dists

def plot_dist(dist_file, dists):
  # Plot dist
  lw = 3
  mkz = 8
  putils.setPlot(fontsize=30, labelsize=30, lw=lw)
  fig, axs = plt.subplots(1, 1, figsize=(12,8), squeeze=False)
  axs = axs.ravel()
  ax = axs[0]
  putils.set_axes_facecolor(axs)
  ax.set_title(os.path.basename(dist_file), fontsize=12)
  ax.plot(dists[0], 'o--', mfc='white', markersize=mkz, c=putils.RED_m, lw=lw,
          label=r'$\mathcal{D}_{\rm MMD}(\mathcal{S}_t,\mathcal{S}^\prime_0)$')
  ax.plot(dists[1], 'o--', mfc='white', markersize=mkz, c=putils.BLUE_m, lw=lw,
          label=r'$W(\mathcal{S}_t,\mathcal{S}^\prime_0)$')

  #ax.legend(fontsize=20, framealpha=0, ncol=2, columnspacing=0.4, loc='upper left', bbox_to_anchor=(-0.1, 1.35))
  ax.set_yscale('log')
  #ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
  putils.set_axes_tick1([ax], xlabel='$t$', ylabel='Dist.', legend=True, tick_minor=False, top_right_spine=True, w=3, tick_length_unit=5)
  fig_file = dist_file.replace('.npy', '')
  plt.tight_layout()
  for ftype in ['pdf']:
      plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
  plt.show()
  plt.clf()
  plt.close()

def plot_dist2(dist_metric_file):
  if os.path.isfile(dist_metric_file) == False:
    print(f'File {dist_metric_file} not found')
    return
  loaded = np.load(dist_metric_file)
  metrics = {key: loaded[key] for key in loaded}

  # Plot dist
  lw = 3
  mkz = 8
  putils.setPlot(fontsize=30, labelsize=30, lw=lw)
  fig, axs = plt.subplots(1, len(metrics), figsize=(20,8), squeeze=False)
  axs = axs.ravel()
  putils.set_axes_facecolor(axs)
  
  for i, keystr in enumerate(metrics.keys()):
    dists = metrics[keystr]
    ax = axs[i]
    #ax.set_title(keystr, fontsize=12)
    ax.plot(dists, '-', mfc='white', markersize=mkz, c=putils.modern[i], lw=lw, label=keystr)

  putils.set_axes_tick1(axs, xlabel='$k$', ylabel='Dist.', legend=True, tick_minor=False, top_right_spine=True, w=3, tick_length_unit=5)
  fig_file = dist_metric_file.replace('.npz', '')
  plt.tight_layout()
  for ftype in ['pdf']:
      plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
  plt.show()
  plt.clf()
  plt.close()

def plot_loss_hist(save_file, loss_hist_all):
  T = loss_hist_all.shape[0]
  M = max(1, math.ceil(T/10))

  putils.setPlot(fontsize=20, labelsize=20, lw=2)
  fsz = int(4*M)
  fig, axs = plt.subplots(M, 10, figsize=(40, fsz), sharex=True, sharey=True, squeeze=False)
  for i in range(T):
    ax = axs[i//10, i%10]
    ax.plot(loss_hist_all[i], lw=2, color=putils.BLUE_m)
    ax.tick_params(direction='in', length=6, width=2, top='on', right='on', labelsize=20)
    ax.set_title(f'$t={i}$')
    ax.set_yscale('log')

  fig.supxlabel(r'$\rm Iterations$', fontsize=30)
  fig.supylabel(r'$\mathcal{L}(t)$', fontsize=30)
  plt.tight_layout()
  for ftype in ['pdf']:
    plt.savefig('{}.{}'.format(f'{save_file}_LOSS', ftype), bbox_inches = 'tight', dpi=300)
  plt.show()
  plt.clf()
  plt.close()

def plot_single_loss_hist(save_file, loss_hist, ylog=False):
    putils.setPlot(fontsize=26, labelsize=26, lw=2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True, sharey=False, squeeze=False)
    ax = ax.ravel()[0]
    xs = np.arange(loss_hist.shape[0])
    ax.plot(xs, loss_hist, lw=2, color=putils.BLUE_m, alpha=0.7,  zorder=1)
    
    putils.set_axes_tick1([ax], xlabel=r'$\rm Epochs$', ylabel='Loss', legend=False, tick_minor=True, top_right_spine=True, w=3, tick_length_unit=5)
    ax.set_ylabel('Loss', fontsize=30)
    if ylog:
      ax.set_yscale('log')
    plt.tight_layout()
    for ftype in ['pdf']:
      plt.savefig('{}.{}'.format(f'{save_file}_LOSS', ftype), bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.clf()
    plt.close()

def plot_loss_hist_all(save_file, loss_hist_all, dist_hist_all, vendi_hist_all):
    putils.setPlot(fontsize=26, labelsize=26, lw=2)
    fig, axs = plt.subplots(3, 1, figsize=(24, 18), sharex=True, sharey=False, squeeze=False)
    axs = axs.ravel()
    ax, bx, cx = axs[0], axs[1], axs[2]
    T = loss_hist_all.shape[0]
    tmp = 0
    for i in range(T-1, -1, -1):
      xs = np.arange(loss_hist_all[i].shape[0])
      xs += tmp
      if max(loss_hist_all[i]) == 0.0:
        break 
      ax.plot(xs, loss_hist_all[i], lw=2, color=putils.BLUE_m, alpha=0.7,  zorder=1)
      bx.plot(xs, dist_hist_all[i], lw=2, color=putils.VERMILLION_m, alpha=0.7,  zorder=1)
      cx.plot(xs, vendi_hist_all[i], lw=2, color=putils.GREEN_m, alpha=0.7,  zorder=1)
      tmp += loss_hist_all[i].shape[0]
    
    putils.set_axes_tick1(axs, xlabel=r'$\rm Epochs$', ylabel='Loss', legend=False, tick_minor=True, top_right_spine=True, w=3, tick_length_unit=5)
    ax.set_ylabel('Total loss', fontsize=30)
    bx.set_ylabel('Distance loss', fontsize=30)
    cx.set_ylabel('Vendi loss', fontsize=30)
    plt.tight_layout()
    for ftype in ['pdf']:
      plt.savefig('{}.{}'.format(f'{save_file}_LOSS', ftype), bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.clf()
    plt.close()

def plot_diffusion_dir(save_file, real_dist_file, train_dist_file, test_dist_file):
  if os.path.isfile(real_dist_file) and os.path.isfile(train_dist_file) and os.path.isfile(test_dist_file):
    real_dists = np.load(real_dist_file)
    train_dists = np.load(train_dist_file)
    test_dists = np.load(test_dist_file)
    lw = 3
    mkz = 8
    putils.setPlot(fontsize=30, labelsize=30, lw=lw)
    fig, axs = plt.subplots(1, 1, figsize=(12,8), squeeze=False)
    axs = axs.ravel()
    ax = axs[0]
    putils.set_axes_facecolor(axs)
    ax.plot(real_dists[1], 'o--', mfc='white', markersize=mkz, c=putils.BLUE_m, lw=lw, label=r'Real')
    ax.plot(train_dists[1], 'o--', mfc='white', markersize=mkz, c=putils.RED_m, lw=lw, label=r'Train')
    ax.plot(test_dists[1], 'o--', mfc='white', markersize=mkz, c=putils.GREEN_m, lw=lw, label=r'Test')
        
    #ax.legend(fontsize=20, framealpha=0, ncol=2, columnspacing=0.4, loc='upper left', bbox_to_anchor=(-0.1, 1.35))
    ax.set_yscale('log')
    #ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
    putils.set_axes_tick1([ax], xlabel='$t$', ylabel=r'$W(\mathcal{S}_t,\mathcal{S}^\prime_0)$', legend=True, tick_minor=False, top_right_spine=True, w=3, tick_length_unit=5)
    plt.tight_layout()
    for ftype in ['png', 'pdf']:
        plt.savefig('{}_EVAL.{}'.format(save_file, ftype), bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.clf()
    plt.close()
  else:
    return


def eval_QDM(save_file, model, real_states, input_states, params_cul, plot_bloch=False, qae_model=None):
  """
  Eval the backward process
  Args:
    model: QDM model
    input_states: input state of the system
  """
  backward_data = model.backward_gen_states(input_states, params_cul)[:, :, :2**model.n_qubits]
  if qae_model is not None:
    # Decode the backward data
    theta_qae = qae_model.theta
    n_qubits_orig = qae_model.n_qubits
    qae_latent = qae_model.n_latent
    qae_layers = qae_model.n_layers
    restore_data = []
    print('Decoding the backward data with QAE model', backward_data.shape)
    for i in range(backward_data.shape[0]):
      restore_state = decode_vmap(backward_data[i], theta_qae, n_qubits_orig, qae_latent, qae_layers)
      restore_data.append(restore_state)
    backward_data = np.array(restore_data)
    print('Decoded backward data shape:', backward_data.shape)

  distance_evolution(f'{save_file}_DIST.npy', real_states, backward_data, eval_mode=True)

  if plot_bloch > 0:
    T = backward_data.shape[0]
    for t in range(T):
      Xt = backward_data[t]
      #plot_Bloch_sphere(f'{save_file}_t_{t}', Xt, f'Backward states step={t}')
      plot_low_2D(f'{save_file}_2D_t_{t}', Xt, f'Backward step={t}')