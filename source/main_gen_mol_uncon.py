##########################################################################
## Code to train the circuit-based quantum diffusion model
## to generate quantum data based on molecular datasets
## Author: Tran Quoc Hoan, Start date: 2025/05/27
##########################################################################
import numpy as np
import warnings

class ComplexWarning(Warning):
    """Re-create the old NumPy ComplexWarning class."""
    pass

np.ComplexWarning = ComplexWarning
warnings.simplefilter('ignore', ComplexWarning)

import os
import argparse

from model.qdm_utils import *
from model.train_utils import train_QDM
from model.qdm_jax import ChaoticScramblingModel, ScramblingModel, QDM
from model.qae_utils import *
from utils.loginit import get_module_logger


def check_state_norms(real_states: np.ndarray, tolerance: float = 1e-10) -> tuple:
    """Check if each row in real_states has a norm of 1, suitable for a quantum state.

    Args:
        real_states (np.ndarray): Array of shape (n_molecules, 2^n_qbits) containing feature vectors.
        tolerance (float): Tolerance for checking if norm is 1 (default: 1e-10).

    Returns:
        tuple: (is_valid, norms, normalized_states)
            - is_valid (list): Boolean list indicating if each row's norm is 1 (within tolerance).
            - norms (np.ndarray): Array of L2 norms for each row.
            - normalized_states (np.ndarray): Array with each row normalized to norm=1.
    """
    n_molecules = real_states.shape[0]
    norms = np.zeros(n_molecules)
    is_valid = []
    normalized_states = np.zeros_like(real_states)

    for i in range(n_molecules):
        try:
            vector = real_states[i]
            norm = np.linalg.norm(vector)
            norms[i] = norm
            is_valid.append(abs(norm - 1.0) < tolerance)
            
            # Normalize the vector to ensure norm=1
            # if norm == 0:
            #     logger.warning(f"Molecule {i} has zero norm; assigning uniform vector")
            #     normalized_states[i] = np.ones_like(vector) / np.sqrt(len(vector))
            # else:
            #     normalized_states[i] = vector / norm
            
            # logger.debug(f"Molecule {i} Norm={norm:.6f}, Valid={is_valid[-1]}")
        except Exception as e:
            logger.warning(f"Error processing molecule {i}: {str(e)}")
            is_valid.append(False)
            normalized_states[i] = np.zeros_like(vector)
    
    return is_valid, norms, normalized_states


if __name__ == "__main__":
  # Check for command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='../results/del_qdm_test1')
  parser.add_argument('--load_params', type=str, default='params_file')

  # For diffusion setting
  parser.add_argument('--scramb', type=str, default='identical', help='Type of scrambling: random, Ising_nearest, Ising_random_all')
  parser.add_argument('--type_evol', type=str, default='full')
  parser.add_argument('--n_pj_qubits', type=int, default=4, help='number of projective qubits')
  parser.add_argument('--delta_t', type=float, default=0.01, help='time interval for evolution')
  parser.add_argument('--rand_ancilla', type=int, default=1, help='random ancilla qubits, 1 for True, 0 for False')
  
  # For Ising_random_all
  parser.add_argument('--n_layers', type=int, default=10, help='number layers for backward circuit')
  parser.add_argument('--n_diff_steps', type=int, default=10, help='number steps for diffusion')
  parser.add_argument('--J', type=float, default=1.0, help='J coefficient for Ising_random_all')
  parser.add_argument('--bz', type=float, default=0.0, help='Bz coefficient for Ising_random_all')
  parser.add_argument('--W', type=float, default=1.0, help='W coefficient for Ising_random_all')

  # For data
  parser.add_argument('--dat_name', type=str, default='qm9', help='name of the data: qm9')
  parser.add_argument('--n_atoms', type=int, default=8, help='Number of atoms to filter molecules')
  parser.add_argument('--n_rings', type=int, default=1, help='Number of rings to filter molecules')
  
  parser.add_argument('--input_type', type=str, default='rand', help='type of the input')
  parser.add_argument('--n_qubits', type=int, default=7, help='Number of data qubits')
  parser.add_argument('--n_ancilla', type=int, default=4, help='Number of ancilla qubits')
  
  parser.add_argument('--n_train', type=int, default=100, help='Number of training data')
  parser.add_argument('--n_test', type=int, default=100, help='Number of test data')

  # For  training
  parser.add_argument('--n_outer_epochs', type=int, default=10, help='Number of outer training epoch')
  parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
  parser.add_argument('--round_epochs', type=int, default=10, help='Number of epochs to round')
  parser.add_argument('--dist_type', type=str, default='wass', help='Type of distance: wass or mmd')  

  parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
  parser.add_argument('--mag', type=float, default=0.001, help='Magnitude of initial parameters')
  parser.add_argument('--vendi_lambda', type=float, default=0.0, help='Vendi loss lambda')
  #parser.add_argument('--ep', type=float, default=0.01, help='Step size')

    # For QAE
  parser.add_argument('--use_qae', type=int, default=0, help='Use QAE or not, 1 for True, 0 for False')
  parser.add_argument('--qae_latent', type=int, default=2, help='Number of latent qubits for QAE')
  parser.add_argument('--qae_layers', type=int, default=5, help='Number of layers for QAE')
  parser.add_argument('--qae_epochs', type=int, default=1000, help='Number of training epochs for QAE')
  parser.add_argument('--qae_lr', type=float, default=0.001, help='Learning rate for QAE')

  # For gen circuit type
  parser.add_argument('--gen_circuit_type', type=str, default='ryzcz', help='type of generator circuit')

  # For system
  parser.add_argument('--rseed', type=int, default=0, help='Random seed')
  parser.add_argument('--bloch', type=int, default=0, help='Plot bloch')
  parser.add_argument('--threads', type=int, default=1, help='Number of threads')

  args = parser.parse_args()

  save_dir, n_layers, n_diff_steps = args.save_dir, args.n_layers, args.n_diff_steps
  n_train, n_test, n_outer_epochs = args.n_train, args.n_test, args.n_outer_epochs
  dat_name, input_type, n_qubits, n_ancilla, rseed = args.dat_name, args.input_type, args.n_qubits, args.n_ancilla, args.rseed
  plot_bloch, load_params = args.bloch, args.load_params
  scramb, type_evol, n_pj_qubits, delta_t, rand_ancilla = args.scramb, args.type_evol, args.n_pj_qubits, args.delta_t, args.rand_ancilla
  b, J, W = args.bz, args.J, args.W
  batch_size, round_epochs = args.batch_size, args.round_epochs
  gen_circuit_type,n_threads = args.gen_circuit_type, args.threads

  lr, mag, dist_type, vendi_lambda = args.lr, args.mag, args.dist_type, args.vendi_lambda
  use_qae, qae_latent, qae_layers, qae_epochs, qae_lr = args.use_qae, args.qae_latent, args.qae_layers, args.qae_epochs, args.qae_lr

  n_atoms, n_rings = args.n_atoms, args.n_rings

  # Create folder to save results
  log_dir = os.path.join(save_dir,'log' )
  res_dir = os.path.join(save_dir,'res')
  diff_dir = os.path.join(save_dir,'diff')

  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(res_dir, exist_ok=True)
  os.makedirs(diff_dir, exist_ok=True)

  scramb_str = f'{scramb}_{delta_t}'
  if 'Ising' in scramb:
    scramb_str = f'{scramb}_{n_pj_qubits}_{type_evol}_{delta_t}'
    if scramb == 'Ising_nearest':
      J = (-1.0, 0.0, 0.0)
      b = (-0.8090, -0.9045, 0.0)
      W = 1.0
    elif scramb == 'Ising_random_all':
      scramb_str = f'{scramb_str}_bz_{b}_W_{W}'

  if use_qae == 1:
    qae_str = f'qae_latent_{qae_latent}_lays_{qae_layers}_epoch_{qae_epochs}_lr_{qae_lr}'
    scramb_str = f'{scramb_str}_{qae_str}'
    qae_save_file = os.path.join(res_dir, f'{dat_name}_{qae_str}_qubits_{n_qubits}_dat_{n_train}_{n_test}_seed_{rseed}')

    qae_model = QAEModel(n_qubits=n_qubits, n_latent=qae_latent, n_layers=qae_layers, n_epochs=qae_epochs, lr=qae_lr, save_file=qae_save_file)
    n_input_qubits = qae_latent
  else:
    qae_model = None
    n_input_qubits = n_qubits

  basename = f'{dat_name}_atoms_{n_atoms}_rings_{n_rings}_{scramb_str}_{gen_circuit_type}_qubits_{n_qubits}_{n_ancilla}_steps_{n_diff_steps}_{dist_type}_lays_{n_layers}_in_{input_type}_dat_{n_train}_{n_test}_epoch_{n_outer_epochs}_{batch_size}_lr_{lr}_init_{mag}_vd_{vendi_lambda}_seed_{rseed}'
  
  log_filename = os.path.join(log_dir, f'{basename}.log')
  logger = get_module_logger(__name__, log_filename, level='info')
  
  logger.info(log_filename)
  logger.info(args)
  
  # set random seed
  np.random.seed(rseed)

  # Create data for inputs and training
  if dat_name == 'qm9':
    real_states = generate_real_data(dat_name, n_qubits, n_train + n_test, rseed+72, n_atoms=n_atoms, n_rings=n_rings)
    train_input_states = generate_input_data(input_type, n_input_qubits, n_train, rseed+27)
    test_input_states = generate_input_data(input_type, n_input_qubits, n_test, rseed+2728)
    # Check norms
    is_valid, norms, normalized_states = check_state_norms(real_states)
    valid_count = sum(is_valid)
    logger.info(f"Checked {len(real_states)} molecules: {valid_count} have norm=1 (within tolerance)")
  else:
    raise ValueError(f"Unknown dataset name: {dat_name}")
  
  if False:
    if input_type == 'diffusion':
      logger.info(f"Input_type={input_type}, Input states will be sampled from the forward diffusion process")
      input_states = np.concatenate((train_input_states, test_input_states), axis=0)
      diff_hs = None
      if 'Ising' in scramb:
        scramble = ChaoticScramblingModel(n_qubits=n_qubits, n_ancilla=n_pj_qubits, type_ham=scramb, J=J, b=b, W=W, rand_ancilla=rand_ancilla)
      else:
        diff_hs = delta_t * 1e-3 * np.arange(1, n_diff_steps+1)**2
        scramble = ScramblingModel(n_qubits=n_qubits, T=n_diff_steps)
      if diff_hs is None:
        if type_evol == 'full':
          input_states = scramble.diffusion_step(n_diff_steps * delta_t, input_states)
        else:
          prev_states = input_states
          for t in range(1, n_diff_steps+1):
            # Apply projected ensemble to previous states
            prev_states = scramble.diffusion_step(delta_t, prev_states)
          input_states = prev_states
      else:
        input_states = scramble.diffusion_step(n_diff_steps, input_states, diff_hs=diff_hs[:n_diff_steps])

      train_input_states = input_states[:n_train]
      test_input_states = input_states[n_train:(n_train+n_test)]

  print('Shapes', real_states.shape, train_input_states.shape)

  # Create data for inputs and training
  save_file = os.path.join(res_dir, f'{basename}')
  diffname = f'{dat_name}_atoms_{n_atoms}_rings_{n_rings}_{scramb_str}_qubits_{n_qubits}'
  if 'Ising' in scramb:
    diffname = f'{diffname}_{n_pj_qubits}'
  
  diffname = f'{diffname}_steps_{n_diff_steps}_dat_{len(real_states)}_seed_{rseed}'
  diff_file = os.path.join(diff_dir, f'{diffname}_DIFF.npy')

  model = QDM(n_qubits=n_input_qubits, n_ancilla=n_ancilla, T = n_diff_steps, n_layers=n_layers, backward_circuit_type=gen_circuit_type, rseed=rseed+2810)
  
  train_QDM(logger, model, save_file, diff_file, real_states, train_input_states, test_input_states,\
              n_outer_epochs, lr, rseed+1234, plot_bloch, dist_type, scramb=scramb, n_pj_qubits=n_pj_qubits, type_evol=type_evol, delta_t=delta_t, J=J, b=b, W=W, \
              rand_ancilla=rand_ancilla, batch_size=batch_size, round_epochs=round_epochs, mag=mag, vendi_lambda=vendi_lambda, qae_model=qae_model)
  
  
  logger.info('Training completed.')