####################################################################
## Code for utility functions to implement quantum generative model
## Author: Tran Quoc Hoan, Start date: 2024/07/04
####################################################################

import qutip as qt
import numpy as np
import scipy as sc
from qutip import Bloch, Qobj
import matplotlib.pyplot as plt
import math
import utils.plot_utils as putils
import jax.numpy as jnp
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ket states
qubit0 = qt.basis(2, 0)
qubit1 = qt.basis(2, 1)

# density matrices
qubit0_mat = qubit0 * qubit0.dag()
qubit1_mat = qubit1 * qubit1.dag()

### Helper function to build the dissipative QNNs
def partial_trace_rem(obj, rem):
  # keep list
  rem.sort(reverse=True)
  keep = list(range(len(obj.dims[0])))
  for x in rem:
      keep.pop(x)
  res = obj

  # return partial trace:
  if len(keep) != len(obj.dims[0]):
      res = obj.ptrace(keep)
  return res

def partial_trace_keep(obj, keep):
  # return partial trace
  res = obj
  if len(keep) != len(obj.dims[0]):
    res = obj.ptrace(keep)
  return res

def swapped_op(obj, i, j):
  if i == j: return obj
  nqubits = len(obj.dims[0])
  permute = list(range(nqubits))
  permute[i], permute[j] = permute[j], permute[i]
  return obj.permute(permute)

def tensor_ID(N):
  # identity matrix
  res = qt.qeye(2**N)
  # dim list
  dims = [2 for i in range(N)]
  dims = [dims.copy(), dims.copy()]
  res.dims = dims
  return res

def tensor_qubit0(N):
  # make qubit matrix
  res = qt.fock(2**N).proj() # faster than fock_dm(2**N)
  # dim list
  dims = [2 for i in range(N)]
  dims = [dims.copy(), dims.copy()]
  res.dims = dims
  return res

def clone_unitaries(unitaries):
  new_unis = []
  for layer in unitaries:
    new_lay = []
    for uni in layer:
      new_lay.append(uni.copy())
    new_unis.append(new_lay)
  return new_unis

### Random generate unitaries, data and unitaries for DQNN

def random_qubit_unitary(nqubits):
  dim = 2 ** nqubits
  # Create unitary matrix
  res = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
  res = sc.linalg.orth(res)
  res = qt.Qobj(res)
  # Create dim list
  dims = [2 for i in range(nqubits)]
  dims = [dims.copy(), dims.copy()]
  res.dims = dims
  return res

def random_qubit_state(nqubits):
  dim = 2 ** nqubits
  # create normalized state
  res = np.random.nornaml(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
  res = (1/ sc.linalg.norm(res)) * res
  res = qt.Qobj(res)

  # create dim list
  dims1 = [2 for i in range(nqubits)]
  dims2 = [1 for i in range(nqubits)]
  dims = [dims1, dims2]
  res.dims = dims
  return res

def plot_Bloch_sphere(file_name, output_states, string_labels, fontsize=20, plot_type='vector'):
  density_matrices = []
  for state in output_states:
      state = np.array(state)
      if state.ndim != 2:
          state = np.outer(state, state.conjugate())
      density_matrices.append(Qobj(state))

  num_qubits = int(math.log(density_matrices[0].shape[0], 2))
  fig, axes = plt.subplots(ncols=num_qubits, figsize=(num_qubits * 5, 5), subplot_kw=dict(projection='3d'))

  if not isinstance(axes, np.ndarray):
      axes = [axes]
      for qubit in range(num_qubits):
          qubits_density_matrices = [partial_trace_keep(dm, qubit) for dm in density_matrices] if num_qubits > 1 else density_matrices
          axes[qubit].set_box_aspect((1, 1, 1))
          b = Bloch(fig=fig, axes=axes[qubit])
          b.point_color = [putils.RED_m]
          b.point_marker = ['o']
          b.point_size = [25]
          b.vector_color = [putils.RED_m]
          b.vector_alpha = [0.7]
          b.vector_width = 1
          
          b.add_states(qubits_density_matrices, kind=plot_type)
          b.render()
      fig.suptitle(f'{string_labels}', fontsize=fontsize)
      plt.savefig(file_name + '_bloch.png', dpi=150)
      plt.close(fig)
  
  # density = []
  # for state in density_matrices:
  #     density.append(state.data)
  # df = pd.DataFrame({'density_matrices': density})
  # df.to_csv(fileName + '_GenOutput.csv', index=False)

def plot_low_2D(file_name, rhos, string_labels, fontsize=20):
    # Vectorize density matrices for dimensionality reduction
    vectorized = []
    for rho in rhos:
        if hasattr(rho, 'full'):  # For QuTiP Qobj
            mat = rho.full()
        else:  # For raw JAX/NumPy array
            mat = jnp.asarray(rho) if 'jax' in str(type(rho)) else np.asarray(rho)
        vec = mat.flatten()  # Complex flatten (dim**2 elements)
        real_imag = np.hstack([np.real(vec), np.imag(vec)])  # Real vector (2 * dim**2)
        vectorized.append(real_imag)
    X = np.array(vectorized)  # Shape: (Ns, 2 * (2**n_qubits)**2)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    putils.setPlot(fontsize=20, labelsize=18, lw=2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Instead of scatter, use plot with markers to avoid issues with large datasets
    axs[0].plot(X_pca[:, 0], X_pca[:, 1], 'o', alpha=0.7, markersize=5, color=putils.BLUE_m, markeredgecolor='k', linewidth=0.2, rasterized=False)
    axs[0].set_title(f'PCA {string_labels}')
    #axs[0].set_xlabel('PC1')
    #axs[0].set_ylabel('PC2')
    
    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot 2D scatter for t-SNE
    axs[1].plot(X_tsne[:, 0], X_tsne[:, 1], 'o', alpha=0.7, markersize=5, color=putils.VERMILLION_m, markeredgecolor='k', linewidth=0.2, rasterized=False)
    axs[1].set_title(f't-SNE {string_labels}')
    #axs[1].set_xlabel('Dim 1')
    #axs[1].set_ylabel('Dim 2')
    putils.set_axes_tick1(axs, legend=False, w=2, tick_length_unit=4, tick_direction='in', tick_minor=True, alpha=0.8, top_right_spine=True, grid=True)
    
    for ftype in ['png', 'pdf', 'svg']:
        plt.savefig(f'{file_name}.{ftype}', bbox_inches='tight', dpi=150)
    plt.close()

def get_Ham_Ising_qt(L, J=1.0, bx=0.0, periodic=False):
  """
  Generate the Ising Hamiltonian for a 1D chain of L qubits.
  H_ising = - J\sum_{i} Z_i Z_{i+1} - bx\sum_{i} X_i
  Args:
      L (int): Number of qubits.
      J (float): Coupling strength.
      bx (float): Magnetic field strength in the x-direction.
      periodic (bool): Whether the chain is periodic, default is False (open boundary)
  Returns:
      H (Qobj): The Ising Hamiltonian.
  """

  # Build Hamiltonian
  I = qt.qeye(2)
  X = qt.sigmax()
  Z = qt.sigmaz()

  H = qt.tensor([I] * L) * 0
  for i in range(L):
      # ZZ term: -(1-g) Z_i Z_{i+1}
      j_next = (i + 1) % L if periodic else i + 1
      if periodic or j_next < L:
          ops = [I] * L
          ops[i] = Z
          ops[j_next] = Z
          H += -J * qt.tensor(ops)
      
      # X term: -g X_i
      ops = [I] * L
      ops[i] = X
      H += -bx * qt.tensor(ops)
  return H


def get_Ham_HEIS_qt(L, J=1.0, b=0.0, periodic=False):
  """
  Generate the nearest neighbor Heisenbert Hamiltonian for a 1D chain of L qubits
  H_Heis = - \sum_{i} (J_X*X_i*X_{i+1} + J_Y*Y_i*Y_{i+1} + J_Z*Z_i*Z_{i+1}) - b_x\sum_{i} X_i - b_y\sum_{i} Y_i - b_z\sum_{i} Z_i
  Args:
      L (int): Number of qubits.
      J (float or (float, float, float), optional) – The XX, YY and ZZ interaction strength. Positive is antiferromagnetic.
      b (float or tuple(float, float, float), optional) – Magnetic field, defaults to z-direction only if tuple not given
      periodic (bool): Whether the chain is periodic, default is False (open boundary)
  Returns:
      H (Qobj): The Heisenberg Hamiltonian.
  """

  # Build Hamiltonian
  I = qt.qeye(2)
  X = qt.sigmax()
  Y = qt.sigmay()
  Z = qt.sigmaz()

  if isinstance(J, (int, float)):
      J_X = J_Y = J_Z = J
  else:
      J_X, J_Y, J_Z = J

  if isinstance(b, (int, float)):
      b_X = b_Y = b_Z = b
  else:
      b_X, b_Y, b_Z = b

  H = qt.tensor([I] * L) * 0
  for i in range(L):
      # Interaction terms: J_X X_i X_{i+1} + J_Y Y_i Y_{i+1} + J_Z Z_i Z_{i+1}
      j_next = (i + 1) % L if periodic else i + 1
      if periodic or j_next < L:
          # XX
          ops = [I] * L
          ops[i] = X
          ops[j_next] = X
          H -= J_X * qt.tensor(ops)
          
          # YY
          ops = [I] * L
          ops[i] = Y
          ops[j_next] = Y
          H -= J_Y * qt.tensor(ops)
          
          # ZZ
          ops = [I] * L
          ops[i] = Z
          ops[j_next] = Z
          H -= J_Z * qt.tensor(ops)
      
      # local term: -bx X_i - by Y_i - bz Z_i
      ops = [I] * L
      
      ops[i] = X
      H -= b_X * qt.tensor(ops)

      ops[i] = Y
      H -= b_Y * qt.tensor(ops)

      ops[i] = Z
      H -= b_Z * qt.tensor(ops)

  return H

def multi_kron(ops):
  return reduce(jnp.kron, ops)
  
def get_Ham_Ising_jax(L, J=1.0, bx=0.0, periodic=False):
  """
  Generate the Ising Hamiltonian for a 1D chain of L qubits.
  H_ising = - J\sum_{i} Z_i Z_{i+1} - bx\sum_{i} X_i
  Args:
      L (int): Number of qubits.
      J (float): Coupling strength.
      bx (float): Magnetic field strength in the x-direction.
      periodic (bool): Whether the chain is periodic, default is False (open boundary)
  Returns:
      H (jax.numpy.ndarray): The Ising Hamiltonian as a JAX array.
  """

  I = jnp.eye(2)
  X = jnp.array([[0., 1.], [1., 0.]])
  Z = jnp.array([[1., 0.], [0., -1.]])

  dim = 2 ** L
  H = jnp.zeros((dim, dim))

  for i in range(L):
      j_next = (i + 1) % L if periodic else i + 1
      if periodic or j_next < L:
          ops = [I] * L
          ops[i] = Z
          ops[j_next] = Z
          H -= J * multi_kron(ops)
      
      ops = [I] * L
      ops[i] = X
      H -= bx * multi_kron(ops)
  return H

def get_Ham_Ising_random_all_jax(L, J_s=1.0, bz=0.0, W=1.0):
  """
  Generate the Ising Hamiltonian for a 1D chain of L qubits.
  H_ising = \sum_{i>j} J_{ij}X_i X_j + \sum_{i} (bz/2 + h_i) Z_i
  Args:
      L (int): Number of qubits.
      J_s (float): Magnitude of coupling strength whre J_{ij} are the
        spin-spin couplings, randomly selected from a uniform
        distribution in the interval [-J_s/2, J_s/2].
      bz (float): Magnetic field strength in the z-direction.
      W (float): disorder strength, where h_i are the
        random magnetic fields, randomly selected from a uniform
        distribution in the interval [-W/2, W/2].
  Returns:
      H (jax.numpy.ndarray): The Ising Hamiltonian as a JAX array.
  """

  I = jnp.eye(2)
  X = jnp.array([[0., 1.], [1., 0.]])
  Z = jnp.array([[1., 0.], [0., -1.]])

  dim = 2 ** L
  H = jnp.zeros((dim, dim))

  # Generate random J_ij for all pairs i < j
  J_matrix = np.random.uniform(-J_s / 2, J_s / 2, (L, L))
  for i in range(L):
      for j in range(i + 1, L):
          ops = [I] * L
          ops[i] = X
          ops[j] = X
          H += J_matrix[i, j] * multi_kron(ops)

  # Generate random h_i
  h = np.random.uniform(-W / 2, W / 2, L)
  for i in range(L):
      ops = [I] * L
      ops[i] = Z
      H += (bz/2.0 + h[i]) * multi_kron(ops)

  return H

def get_Ham_HEIS_jax(L, J=1.0, b=0.0, periodic=False):
  """
  Generate the nearest neighbor Heisenberg Hamiltonian for a 1D chain of L qubits
  H_Heis = - \sum_{i} (J_X*X_i*X_{i+1} + J_Y*Y_i*Y_{i+1} + J_Z*Z_i*Z_{i+1}) - b_x\sum_{i} X_i - b_y\sum_{i} Y_i - b_z\sum_{i} Z_i
  Args:
      L (int): Number of qubits.
      J (float or (float, float, float), optional) – The XX, YY and ZZ interaction strength. Positive is antiferromagnetic.
      b (float or tuple(float, float, float), optional) – Magnetic field, defaults to z-direction only if tuple not given
      periodic (bool): Whether the chain is periodic, default is False (open boundary)
  Returns:
      H (jax.numpy.ndarray): The Heisenberg Hamiltonian as a JAX array.
  """

  I = jnp.eye(2, dtype=jnp.complex64)  # Use complex dtype for Y terms
  X = jnp.array([[0., 1.], [1., 0.]], dtype=jnp.complex64)
  Y = jnp.array([[0., -1j], [1j, 0.]], dtype=jnp.complex64)
  Z = jnp.array([[1., 0.], [0., -1.]], dtype=jnp.complex64)

  if isinstance(J, (int, float)):
      J_X = J_Y = J_Z = J
  else:
      J_X, J_Y, J_Z = J

  if isinstance(b, (int, float)):
      b_X = b_Y = 0.0
      b_Z = b
  else:
      b_X, b_Y, b_Z = b

  dim = 2 ** L
  H = jnp.zeros((dim, dim), dtype=jnp.complex64)

  for i in range(L):
      j_next = (i + 1) % L if periodic else i + 1
      if periodic or j_next < L:
          # XX
          ops = [I] * L
          ops[i] = X
          ops[j_next] = X
          H -= J_X * multi_kron(ops)
          
          # YY
          ops = [I] * L
          ops[i] = Y
          ops[j_next] = Y
          H -= J_Y * multi_kron(ops)
          
          # ZZ
          ops = [I] * L
          ops[i] = Z
          ops[j_next] = Z
          H -= J_Z * multi_kron(ops)
      
      # local term: -b_x X_i - b_y Y_i - b_z Z_i
      ops = [I] * L
      
      ops[i] = X
      H -= b_X * multi_kron(ops)

      ops[i] = Y
      H -= b_Y * multi_kron(ops)

      ops[i] = Z
      H -= b_Z * multi_kron(ops)

  return H


import jax.numpy as jnp

def vendi_score(states):
    """
    Compute Vendi Score for an ensemble of pure quantum state vectors using fidelity kernel.

    Args:
    states: list of jnp.ndarray, each a pure state ket vector of shape (d,) or (d,1).

    Returns:
    vendi_score: float, the Vendi Score.
    """
    N = len(states)

    # Ensure states are column vectors of shape (d, 1)
    states = [jnp.reshape(state, (-1, 1)) for state in states]

    # Stack states into a matrix S of shape (d, N)
    S = jnp.hstack(states)

    # Compute the Gram matrix elements <psi_i | psi_j> = psi_i^\dagger psi_j
    gram_matrix = S.conj().T @ S

    # Build similarity matrix K using fidelity: |<psi_i | psi_j>|^2
    K = jnp.abs(gram_matrix)**2

    # Compute eigenvalues (K is Hermitian positive semi-definite)
    eigenvalues = jnp.linalg.eigvalsh(K)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)  # Clip negatives due to numerical errors
    trace_K = jnp.sum(eigenvalues)
    probs = eigenvalues / trace_K

    # Compute Shannon entropy, with safe log to handle zero probs
    log_probs = jnp.log(probs + 1e-20)  # Small epsilon to avoid nan
    entropy = -jnp.sum(probs * log_probs)

    # Vendi Score
    vendi = jnp.exp(entropy)
    return vendi