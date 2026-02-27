import numpy as np
import warnings

class ComplexWarning(Warning):
    """Re-create the old NumPy ComplexWarning class."""
    pass

np.ComplexWarning = ComplexWarning
warnings.simplefilter('ignore', ComplexWarning)

import os
from data.data_utils import *
from model.qdm_jax import *
from model.qdm_utils import *
from model.qae_utils import *

from utils.distance_jax import *
from utils.utils import vendi_score
import jax
from jax import numpy as jnp
from jax import random
import optax

def training_QDM_t(logger, model, t, inputs_T, params_cul, n_train, n_epochs, lr, mag, dist_type, indices, key, \
                   batch_size=50, round_epochs=1, vendi_lambda=0.0):
  """
  Training for the backward process at step t using minibatch learning
  Args:
    logger: logger to log the training process
    model: QDM model
    t: diffusion step
    inputs_T:      [full_batch, 2**n_qubits] complex64 JAX array
    params_cul: collection of PQC parameters for steps > t
    n_train: number of training samples
    n_epochs: number of training epochs
    lr: learning rate (float)
    mag: magnitude of initial parameters
    dist_type: type of distance to use, 'mmd' or 'wass'
    indices: [n_train] integer array of indices into `forward_states_diff[t]`
    key: a JAX PRNGKey
    batch_size: size of each minibatch (default: 32)
    round_epochs: how often (in epochs) to log progress
  Returns:
    params_t: [model.backward_n_params] JAX array of parameters for backward step t
    loss_hist: [n_epochs] JAX array of average loss values for each epoch
  """
  input_tplus, _ = model.prepare_input_t(inputs_T, params_cul, t, n_train)
  forward_states_diff = model.forward_states_diff
  real_data = forward_states_diff[t, indices, :]  # shape [n_train, 2**n_qubits], complex64

  # Initialize parameters
  subkey, next_key = jax.random.split(key)
  #params_t = jax.random.normal(subkey, (model.backward_n_params,), dtype=jnp.float32)
  params_t = jax.random.uniform(subkey, (model.backward_n_params,), minval=-mag, maxval=mag, dtype=jnp.float32)

  # Adam optimizer with the learning rate schedule
  optimizer = optax.adam(learning_rate=lr)
  opt_state = optimizer.init(params_t)

  # Define a loss function for a single minibatch
  def loss_fn(params_t_inner: jnp.ndarray, batch_data: jnp.ndarray, batch_input: jnp.ndarray, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
      subkey, next_key = jax.random.split(key)
      # shape [batch_size, 2**n_qubits], complex64
      output_t, _ = backward_output_pure(batch_input, params_t_inner, model.n_ancilla, model.n_qubits, model.backward_circuit_vmap, subkey)

      # Compute distance to batch_data
      if dist_type == 'mmd':
          loss = natural_distance_jax(output_t, batch_data)  # scalar
      else:
          # use OTT-based Wasserstein (JIT-capable)
          loss = wass_distance_ott(output_t, batch_data)  # scalar
      
      # Vendi score and loss
      vendi_loss, vendi_out = 0.0, 0.0
      if vendi_lambda > 0.0:
        vendi_train = vendi_score(batch_data)
        vendi_out = vendi_score(output_t)
        
        vendi_loss = jnp.square(vendi_out - vendi_train)

      total_loss = loss + vendi_lambda * vendi_loss
      return total_loss, (next_key, (loss, vendi_loss, vendi_out))

  # JIT-compile both the loss and its gradient
  loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
  loss_hist_list, dist_hist_list, vendi_hist_list = [], [], []

  # Calculate number of batches
  n_batches = n_train // batch_size + (1 if n_train % batch_size != 0 else 0)

  for epoch in range(n_epochs):
      # Shuffle indices for this epoch
      subkey, next_key = jax.random.split(next_key)
      shuffled_indices = jax.random.permutation(subkey, jnp.arange(n_train))

      epoch_loss, epoch_dist_loss, epoch_vendi_loss = 0.0, 0.0, 0.0
      for batch_idx in range(n_batches):
          # Get batch indices
          start_idx = batch_idx * batch_size
          end_idx = min((batch_idx + 1) * batch_size, n_train)
          batch_indices = shuffled_indices[start_idx:end_idx]

          # Extract batch data
          batch_real_data = real_data[batch_indices, :]  # shape [batch_size, 2**n_qubits]
          batch_input_tplus = input_tplus[batch_indices, :]  # shape [batch_size, 2**n_qubits]

          # Compute loss and gradients for the minibatch
          (loss_val, (next_key, (batch_dist_loss, batch_vendi_loss, batch_vendi_score))), grads = loss_and_grads(params_t, batch_real_data, batch_input_tplus, next_key)
          epoch_loss += float(loss_val) * (end_idx - start_idx) / n_train  # Weighted average
          epoch_dist_loss += float(batch_dist_loss) * (end_idx - start_idx) / n_train
          epoch_vendi_loss += float(batch_vendi_loss) * (end_idx - start_idx) / n_train
          
          # Update parameters
          updates, opt_state = optimizer.update(grads, opt_state, params_t)
          params_t = optax.apply_updates(params_t, updates)

      # Log progress
      loss_hist_list.append(epoch_loss)
      dist_hist_list.append(epoch_dist_loss)
      vendi_hist_list.append(epoch_vendi_loss)
      if epoch == 0 or ((epoch + 1) % round_epochs == 0):
          logger.info(f"Training backward-{t} epoch={epoch}/{n_epochs}, total loss={epoch_loss:.6f}, batch dist loss={epoch_dist_loss:.6f}, vendi loss={epoch_vendi_loss:.6f}")

  loss_history = jnp.stack(loss_hist_list)
  dist_history = jnp.stack(dist_hist_list)
  vendi_history = jnp.stack(vendi_hist_list)
  return params_t, loss_history, dist_history, vendi_history

def train_QDM(logger, model, save_file, diff_file, real_states, train_input_states, test_input_states,\
              n_outer_epochs, lr, rseed, plot_bloch, dist_type, scramb='random', n_pj_qubits=1, type_evol='full', delta_t=1.0,
              J=(-1.0, 0.0, 0.0), b=(-0.8090, -0.9045, 0.0), W=1.0, rand_ancilla=1, 
              batch_size=100, round_epochs=10, mag=1.0, vendi_lambda=0.0, qae_model=None, noise_info=None):
  """
  Create QDM model and train this model
  Arg:
    save_file: basename to save file
  """
  n_train = train_input_states.shape[0]
  n_test = test_input_states.shape[0]
  n_full_data = real_states.shape[0]
  n_qubits = model.n_qubits
  n_diff_steps = model.T
  
  key = jax.random.PRNGKey(rseed)
  idx = np.random.choice(n_full_data, n_test, replace=False)
  X0_org = real_states[idx]

  if qae_model is not None:
    n_qubits_orig = qae_model.n_qubits
    qae_latent = qae_model.n_latent
    qae_layers = qae_model.n_layers
    qae_epochs = qae_model.n_epochs
    qae_save_file = qae_model.save_file
    qae_lr = qae_model.lr

    qae_param_file = f'{qae_save_file}_QAE_PARAMS.npy'
    qae_loss_file = f'{qae_save_file}_QAE_LOSS.npy'

    qae_loss_hist = []
    if os.path.isfile(qae_param_file):
      logger.info(f'Load QAE params from {qae_param_file}')
      theta_qae = np.load(qae_param_file)
      qae_model.theta = theta_qae

      if os.path.isfile(qae_loss_file):
        qae_loss_hist = np.load(qae_loss_file)
    else:
      logger.info(f"Training QAE to compress from {n_qubits_orig} to {qae_latent} qubits")
      qae_key, key = random.split(key)
      theta_qae, qae_loss_hist = train_qae(logger, real_states, n_qubits_orig, qae_latent, qae_layers, qae_epochs, batch_size, qae_lr, qae_key)

      np.save(qae_param_file, theta_qae)
      np.save(qae_loss_file, qae_loss_hist)

    qae_model.theta = theta_qae
    plot_single_loss_hist(f'{qae_save_file}_QAE', qae_loss_hist, ylog=True)

    # Compress all states and update for latent space
    real_states = encode_vmap(real_states, theta_qae, n_qubits_orig, qae_latent, qae_layers)

    # Check that the train_input_states and test_input_states have qae_latent qubits
    assert train_input_states.shape[1] == 2**qae_latent, f"train_input_states has shape {train_input_states.shape}, expected 2**{qae_latent}"
    assert test_input_states.shape[1] == 2**qae_latent, f"test_input_states has shape {test_input_states.shape}, expected 2**{qae_latent}"

  if scramb == 'identical' or delta_t == 0.0:
    logger.info('No scrambing, use the same original data')
    X = jnp.asarray(real_states, dtype=jnp.complex64)
    X_out = np.tile(X[None, :, :], (n_diff_steps+1, 1, 1)).astype(np.complex64)
  else:
    X_out = generate_real_diff(logger, diff_file, real_states, n_diff_steps, n_qubits, rseed + 5678, scramb, \
                  n_pj_qubits=n_pj_qubits, type_evol=type_evol, delta_t=delta_t, J=J, b=b, W=W, rand_ancilla=rand_ancilla, noise_info=noise_info)

  X0 = real_states[idx]
  print(f'X0 shape={X0.shape}, Xout shape = {X_out.shape}')

  if plot_bloch > 0:
    #plot_Bloch_sphere(f'{save_file}_in', train_input_states, 'Input states')
    #plot_Bloch_sphere(f'{save_file}_real', X0, 'Real states')
    plot_low_2D(f'{save_file}_2D_real', X0, 'True states')
    # for debug
    if True:
      for t in range(n_diff_steps+1):
        #plot_Bloch_sphere(f'{save_file}_diff_{t}', X_out[t][idx], f'step={t}')
        plot_low_2D(f'{save_file}_2D_diff_{t}', X_out[t][idx], f'Forward step={t}')

  real_dist_file = f'{save_file}_DIST.npy'
  if qae_model is not None:
    # convert back X_out to original space
    X_out_restore = []
    for i in range(X_out.shape[0]):
      restore_state = decode_vmap(X_out[i], theta_qae, n_qubits_orig, qae_latent, qae_layers)
      X_out_restore.append(restore_state)
    X_out_restore = np.array(X_out_restore)
    distance_evolution(real_dist_file, X0_org, X_out_restore)
  else:
    distance_evolution(real_dist_file, X0, X_out)

  if n_outer_epochs == 0:
    return
  
  # Training phase
  indices = np.random.choice(n_full_data, size=n_train, replace=False)

  param_file = f'{save_file}_PARAMS.npy'
  loss_file  = f'{save_file}_LOSS.npy'
  dist_loss_file = f'{save_file}_DLOSS.npy'
  vendi_loss_file = f'{save_file}_VLOSS.npy'

  loss_hist_all = np.zeros((n_diff_steps, n_outer_epochs))
  dist_hist_all = np.zeros((n_diff_steps, n_outer_epochs))
  vendi_hist_all = np.zeros((n_diff_steps, n_outer_epochs))
  
  if os.path.isfile(param_file):
    logger.info(f'Load params from {param_file}')
    params_cul = np.load(param_file)
  else:
    model.set_forward_states_diff(X_out)
    params_cul = np.zeros((n_diff_steps, model.backward_n_params))
    for t in range(n_diff_steps-1, -1, -1):
      params, loss_hist, dist_hist, vendi_hist = training_QDM_t(logger, model, t, train_input_states, params_cul, n_train, n_outer_epochs, lr, mag, dist_type, indices, key, \
            batch_size=batch_size, round_epochs=round_epochs, vendi_lambda=vendi_lambda)
      params_cul[t] = params
      loss_hist_all[t] = loss_hist
      dist_hist_all[t] = dist_hist
      vendi_hist_all[t] = vendi_hist

      plot_loss_hist_all(save_file, loss_hist_all, dist_hist_all, vendi_hist_all)

    # save params for backward generation
    np.save(param_file, params_cul)

    # save loss history
    if np.max(loss_hist_all) > 0.0:
      np.save(loss_file, loss_hist_all)
    if np.max(dist_hist_all) > 0.0:
      np.save(dist_loss_file, dist_hist_all)
    if np.max(vendi_hist_all) > 0.0:
      np.save(vendi_loss_file, vendi_hist_all)

  if os.path.isfile(loss_file):
    loss_hist_all = np.load(loss_file)
  if os.path.isfile(dist_loss_file):
    dist_hist_all = np.load(dist_loss_file)
  if os.path.isfile(vendi_loss_file):
    vendi_hist_all = np.load(vendi_loss_file)

  plot_loss_hist_all(save_file, loss_hist_all, dist_hist_all, vendi_hist_all)

  # eval the training data
  # eval_QDM(f'{save_file}_train', model, X0, train_input_states, params_cul, plot_bloch)

  # eval the testing data
  eval_QDM(f'{save_file}_test', model, X0_org, test_input_states, params_cul, plot_bloch, qae_model=qae_model)

  # eval the diffusion
  # train_dist_file = f'{save_file}_train_DIST.npy'
  # test_dist_file = f'{save_file}_test_DIST.npy'
  
  # plot_diffusion_dir(save_file, real_dist_file, train_dist_file, test_dist_file)

