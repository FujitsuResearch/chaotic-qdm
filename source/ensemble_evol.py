##########################################################################
## Code to see the ensemble evolution under chaotic Hamiltonian
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
from model.qae_utils import *
from utils.loginit import get_module_logger

if __name__ == "__main__":
  # Check for command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='../results/del_qdm_test2')
  # For diffusion setting
  parser.add_argument('--scramb', type=str, default='Ising_nearest', help='Type of scrambling: random, Ising_nearest, Ising_random_all')
  parser.add_argument('--type_evol', type=str, default='full', help='Type of evolution: full or trotter')
  parser.add_argument('--n_pj_qubits', type=int, default=4, help='number of projective qubits')
  parser.add_argument('--delta_t', type=float, default=0.01, help='time interval for evolution')
  parser.add_argument('--rand_ancilla', type=int, default=1, help='random basis for ancilla or not, 1 for True, 0 for False')
  parser.add_argument('--n_diff_steps', type=int, default=10, help='number steps for diffusion')
  
  # For Ising_random_all
  parser.add_argument('--J', type=float, default=1.0, help='J coefficient for Ising_random_all')
  parser.add_argument('--bz', type=float, default=0.0, help='Bz coefficient for Ising_random_all')
  parser.add_argument('--W', type=float, default=1.0, help='W coefficient for Ising_random_all')


  # For data
  parser.add_argument('--dat_name', type=str, default='multi_cluster', help='name of the data: cluster0, line, circle, tfim, multi_cluster')
  parser.add_argument('--n_qubits', type=int, default=4, help='Number of data qubits')
  parser.add_argument('--n_dat', type=int, default=100, help='Number of data instances')

  # For system
  parser.add_argument('--rseed', type=int, default=0, help='Random seed')

  # For moment order
  parser.add_argument('--k_moment', type=int, default=2, help='Moment order k for distance calculation')

  args = parser.parse_args()

  save_dir, n_diff_steps = args.save_dir, args.n_diff_steps
  dat_name, n_dat, n_qubits, rseed = args.dat_name, args.n_dat, args.n_qubits, args.rseed
  scramb, type_evol, n_pj_qubits, delta_t, rand_ancilla = args.scramb, args.type_evol, args.n_pj_qubits, args.delta_t, args.rand_ancilla
  b, J, W = args.bz, args.J, args.W

  k_moment = args.k_moment

  # Create folder to save results
  log_dir = os.path.join(save_dir,'log' )
  diff_dir = os.path.join(save_dir,'diff')
  dist_dir = os.path.join(save_dir,'dist')

  
  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(diff_dir, exist_ok=True)
  os.makedirs(dist_dir, exist_ok=True)


  scramb_str = f'{scramb}_{delta_t}'
  if 'Ising' in scramb:
    scramb_str = f'{scramb}_{n_pj_qubits}_{type_evol}_{delta_t}'
    if scramb == 'Ising_nearest':
      J = (-1.0, 0.0, 0.0)
      b = (-0.8090, -0.9045, 0.0)
      W = 1.0
    elif scramb == 'Ising_random_all':
      scramb_str = f'{scramb_str}_bz_{b}_W_{W}'

  n_input_qubits = n_qubits
  basename = f'{dat_name}_{scramb_str}_rdanc_{rand_ancilla}_qubits_{n_qubits}_steps_{n_diff_steps}_dat_{n_dat}_seed_{rseed}'
  
  log_filename = os.path.join(log_dir, f'{basename}.log')
  logger = get_module_logger(__name__, log_filename, level='info')
  
  logger.info(log_filename)
  logger.info(args)
  
  # set random seed
  np.random.seed(rseed)

  # Create data for inputs and training
  real_states = generate_real_data(dat_name, n_qubits, n_dat, rseed+72)
  diff_file = os.path.join(diff_dir, f'{basename}_DIFF.npy')

  # Generate diffusion file
  X_out = generate_real_diff(logger, diff_file, real_states, n_diff_steps, n_qubits, rseed + 5678, scramb, \
                  n_pj_qubits=n_pj_qubits, type_evol=type_evol, delta_t=delta_t, J=J, b=b, W=W, rand_ancilla=rand_ancilla)
  
  use_trace_distance = False
  if n_qubits*k_moment < 10:
    use_trace_distance = True
  
  # Compute the distance evolution with the real ensemble, the Haar random states ensemble and the Haar product states ensemble
  dist_with_target_file = os.path.join(dist_dir, f'{basename}_dist_k={k_moment}_target.npz')
  metric_evolution(dist_with_target_file, real_states, X_out, k=k_moment, use_trace_distance=use_trace_distance)
  plot_dist2(dist_with_target_file)
  logger.info(f'Saved distance evolution with target data to {dist_with_target_file}')

  dist_with_haar_file = os.path.join(dist_dir, f'{basename}_dist_k={k_moment}_haar.npz')
  haar_ensemble = gen_Haar_states(n_dat, n_qubits, rseed + 9102)
  metric_evolution(dist_with_haar_file, haar_ensemble, X_out, k=k_moment, use_trace_distance=use_trace_distance)
  plot_dist2(dist_with_haar_file)
  logger.info(f'Saved distance evolution with Haar data to {dist_with_haar_file}')

  dist_with_haar_product_file = os.path.join(dist_dir, f'{basename}_dist_k={k_moment}_haar_product.npz')  
  haar_product_ensemble = gen_Haar_product_states(n_dat, n_qubits, rseed + 12903)
  metric_evolution(dist_with_haar_product_file, haar_product_ensemble, X_out, k=k_moment, use_trace_distance=use_trace_distance)
  plot_dist2(dist_with_haar_product_file)
  logger.info(f'Saved distance evolution with Haar product data to {dist_with_haar_product_file}')

  logger.info('Completed.')