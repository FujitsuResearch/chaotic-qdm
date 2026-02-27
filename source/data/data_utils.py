import numpy as np
import scipy as sc
import qutip as qt
from scipy.stats import unitary_group
import jax
from jax import numpy as jnp
from jax import random
from utils import *
import warnings

def gen_line_data(nstates, nqubits):
  data_ls = []
  ids = list(range(0, nstates))
  for i in range(nstates):
      t = ((ids[-i - 1]) / (nstates - 1)) * qt.basis(2 ** nqubits, 0) + ((ids[i]) / (nstates - 1)) * qt.basis(2 ** nqubits, 1)
      t = (1 / sc.linalg.norm(t)) * t
      dims1 = [2 for i in range(nqubits)]
      dims2 = [1 for i in range(nqubits)]
      dims = [dims1, dims2]
      t.dims = dims
      data_ls.append(t)
  return data_ls

def gen_rand_state(nqubits):
  dim = 2 ** nqubits
  # Create normalized state
  res = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
  res = (1 / sc.linalg.norm(res)) * res
  res = qt.Qobj(res)
  # Make dims list
  dims1 = [2 for _ in range(nqubits)]
  dims2 = [1 for _ in range(nqubits)]
  dims = [dims1, dims2]
  res.dims = dims
  return res

def gen_rand_line_plus_minus_state(nqubits):
  """
  Generate a random state on the line between |+> and |->
  """
  plus_state = (qt.basis(2 ** nqubits, 0) + qt.basis(2 ** nqubits, 1)).unit()  # |+>
  minus_state = (qt.basis(2 ** nqubits, 0) - qt.basis(2 ** nqubits, 1)).unit()  # |->
  t = np.random.rand()  # Random parameter between 0 and 1
  state = t * plus_state + (1 - t) * minus_state
  state = state.unit()  # Normalize the state
  dims1 = [2 for i in range(nqubits)]
  dims2 = [1 for i in range(nqubits)]
  dims = [dims1, dims2]
  state.dims = dims
  return state

def gen_rand_state_list(nstates, nqubits, input_type='rand'):
  states_ls = []
  for _ in range(nstates):
    if input_type == 'line':
      st = gen_rand_line_plus_minus_state(nqubits)
    else:
      st = gen_rand_state(nqubits)
    states_ls.append(st)
  return states_ls

def gen_Haar_states(nstates, nqubits, seed):
  """
  Generate random Haar states
    nstates: number of samples
    nqubits: number of qubits in each state
    seed: seed of random generator
  """
  np.random.seed(seed)
  states = unitary_group.rvs(dim=2**nqubits, size=nstates)[:,:,0]
  return jnp.array(states)

def gen_Haar_product_states(nstates, nqubits, seed):
  """
  Generate random Haar product states
    nstates: number of samples
    nqubits: number of qubits in each state
    seed: seed of random generator
  """
  np.random.seed(seed)
  states = []

  for _ in range(nstates):
      # Generate Haar-random single-qubit states and take their tensor product
      single_qubit_states = [unitary_group.rvs(2)[:, 0] for _ in range(nqubits)]
      product_state = single_qubit_states[0]
      for psi in single_qubit_states[1:]:
          product_state = np.kron(product_state, psi)  # Tensor product
      
      states.append(product_state)

  return jnp.array(states)  # Convert to JAX array

def gen_rand_basis_states(nstates, nqubits, seed):
  """
  Generate random basis states
    nstates: number of samples
    nqubits: number of qubits in each state
    seed: seed of random generator
  """
  np.random.seed(seed)
  states = []
  for _ in range(nstates):
      # Generate a random integer between 0 and 2**nqubits - 1
      idx = np.random.randint(0, 2**nqubits)
      # Create the basis state corresponding to that index
      state = np.zeros(2**nqubits, dtype=np.complex64)
      state[idx] = 1.0
      states.append(state)
  
  return jnp.array(states)  # Convert to JAX array

def gen_cluster_0(nstates, nqubits, scale, seed):
  """
  Generate cluster states near state zero |0...0>
    nstates: number of samples
    nqubits: number of qubits in each state
    seed: seed of random generator
  """
  remains = (random.normal(random.PRNGKey(seed), shape=(nstates, 2**nqubits-1)) 
            + 1j*random.normal(random.PRNGKey(seed+1), shape=(nstates, 2**nqubits-1)))
  states = jnp.hstack((np.ones((nstates, 1)), scale*remains))
  states /= jnp.tile(jnp.linalg.norm(states, axis=1).reshape((1, nstates)), (2**nqubits, 1)).T
  return states.astype(jnp.complex64)

def gen_line_data_with_jax(nstates, nqubits, seed=-1):
    """
    Generate quantum states on the Bloch sphere using jax.numpy.
    
    nstates: number of states to generate
    nqubits: number of qubits in each state
    """
    if seed < 0:
      ids = jnp.linspace(0, 1, nstates)
    else:
      jnp.random.seed(seed)
      ids = jnp.random.uniform(0, 1, nstates)

    def generate_state(i):
        coeff_0 = ids[-i - 1]
        coeff_1 = ids[i]
        
        state_0 = coeff_0 * jnp.array([1] + [0] * (2**nqubits - 1), dtype=jnp.complex64)
        state_1 = coeff_1 * jnp.array([0] * (2**nqubits - 1) + [1], dtype=jnp.complex64)
        
        state = state_0 + state_1
        
        # Normalize the state
        norm = jnp.linalg.norm(state)
        state = state / norm
        
        return state
    
    states = jax.vmap(generate_state)(jnp.arange(nstates))
    
    return states.astype(jnp.complex64)

def gen_circle_data_with_jax(nstates, nqubits, seed=-1):
    """
    Generate quantum states on the Bloch sphere using jax.numpy.
    
    nstates: number of states to generate
    nqubits: number of qubits in each state
    """
    if seed < 0:
      angles = jnp.linspace(0, 2 * jnp.pi, nstates)
    else:
      jnp.random.seed(seed)
      angles = jnp.random.uniform(0, 2 * jnp.pi, nstates)
    
    
    def generate_state(angle):
        coeff_0 = jnp.cos(angle / 2)
        coeff_1 = jnp.sin(angle / 2)
        
        state_0 = coeff_0 * jnp.array([1] + [0] * (2**nqubits - 1), dtype=jnp.complex64)
        state_1 = coeff_1 * jnp.array([0] * (2**nqubits - 1) + [1], dtype=jnp.complex64)
        
        state = state_0 + state_1
        
        # Normalize the state
        norm = jnp.linalg.norm(state)
        state = state / norm
        
        return state
    
    states = jax.vmap(generate_state)(angles)
    
    return states.astype(jnp.complex64)

def gen_circle_Y(nstates, seed=None):
  """
  Generate random quantum states from RY(\phi)|0> with uniform distribution
  """
  np.random.seed(seed)
  phis = np.random.uniform(0, 2*np.pi, nstates)
  states = np.vstack((np.cos(phis), np.sin(phis))).T
  return states.astype(np.complex64)


def single_qubit_rotation(nqubits, qubit_idx, axis, angle, key):
    """
    Generate a single-qubit rotation operator for the specified qubit and axis.
    
    Parameters:
    - nqubits: number of qubits (int)
    - qubit_idx: index of the qubit to rotate (int)
    - axis: rotation axis ('x', 'y', or 'z')
    - angle: rotation angle in radians (float)
    - key: JAX PRNG key (for JAX compatibility, not used here)
    
    Returns:
    - U: unitary matrix (jnp.ndarray, shape=(2**nqubits, 2**nqubits))
    """
    dim = 2**nqubits
    # Pauli matrices
    sigma = {
        'x': jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),
        'y': jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64),
        'z': jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    }
    # Single-qubit rotation: exp(-i * angle/2 * sigma)
    single_U = jax.scipy.linalg.expm(-1j * (angle / 2) * sigma[axis])
    # Embed into full system
    if qubit_idx == 0:
        U = single_U
        for _ in range(nqubits - 1):
            U = jnp.kron(U, jnp.eye(2, dtype=jnp.complex64))
    else:
        U = jnp.eye(2, dtype=jnp.complex64)
        for i in range(1, nqubits):
            if i == qubit_idx:
                U = jnp.kron(U, single_U)
            else:
                U = jnp.kron(U, jnp.eye(2, dtype=jnp.complex64))
    return U

def random_rotation_unitary(nqubits, scale, key):
    """
    Generate a product unitary of random single-qubit rotations.
    
    Parameters:
    - nqubits: number of qubits (int)
    - scale: noise scale for rotation angles (float)
    - key: JAX PRNG key
    
    Returns:
    - U: unitary matrix (jnp.ndarray, shape=(2**nqubits, 2**nqubits))
    """
    axes = ['x', 'y', 'z']
    U = jnp.eye(2**nqubits, dtype=jnp.complex64)
    for i in range(nqubits):
        key, subkey = jax.random.split(key)
        # Randomly choose axis and angle
        axis_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=3)
        angle = jax.random.normal(subkey, shape=()) * scale
        U_i = single_qubit_rotation(nqubits, i, axes[axis_idx], angle, subkey)
        U = U_i @ U
    return U

def gen_cluster(nstates, nqubits, center, scale, seed):
    """
    Generate cluster of states around a given center state using random rotation noise.
    
    Parameters:
    - nstates: number of samples (int)
    - nqubits: number of qubits (int)
    - center: central state vector (jnp.ndarray, shape=(2**nqubits,))
    - scale: noise scale for rotation angles (float)
    - seed: random seed (int)
    
    Returns:
    - states: array of normalized state vectors (jnp.ndarray, shape=(nstates, 2**nqubits))
    """
    dim = 2**nqubits
    key = jax.random.PRNGKey(seed)
    states = []
    for i in range(nstates):
        key, subkey = jax.random.split(key)
        U = random_rotation_unitary(nqubits, scale, subkey)
        state = U @ center
        states.append(state)
    states = jnp.stack(states)
    return states.astype(jnp.complex64)

def generate_multi_clustered_states(n, seed, N=3000, scale=0.05):
    """
    Generate a dataset of N clustered n-qubit pure states with perturbed rotation noise.
    
    Parameters:
    - n: number of qubits (int)
    - seed: random seed (int)
    - N: total number of states (int, default=3000)
    - scale: noise level for rotation angles (float, default=0.05)
    
    Returns:
    - S: array of states (jnp.ndarray, shape=(N, 2**n))
    """
    dim = 2**n
    # Define center states
    all_zero = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0)  # |0⟩⊗n
    all_one = jnp.zeros((dim,), dtype=jnp.complex64).at[-1].set(1.0)  # |1⟩⊗n
    ghz = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 / jnp.sqrt(2)).at[-1].set(1.0 / jnp.sqrt(2))  # GHZ state
    
    # Cluster sizes
    n_zero = int(0.4 * N)
    n_one = int(0.4 * N)
    n_ghz = N - n_zero - n_one
    
    # Generate clusters with different seeds
    cluster_zero = gen_cluster(n_zero, n, all_zero, scale, seed=seed)
    cluster_one = gen_cluster(n_one, n, all_one, scale, seed=seed+1)
    cluster_ghz = gen_cluster(n_ghz, n, ghz, scale, seed=seed+2)
    
    # Combine clusters
    S = jnp.vstack((cluster_zero, cluster_one, cluster_ghz))
    # Shuffle indices
    key = jax.random.PRNGKey(seed + 100)
    indices = jax.random.permutation(key, N)
    S = S[indices]
    
    return S

def generate_scrooge_ensemble(rho, M=1000):
    d = rho.shape[0]
    sqrt_rho = rho.sqrtm()
    ensemble = []
    for _ in range(M):
        phi = qt.rand_unitary_haar(d) * qt.basis(d, 0)  # Haar ket
        p = (phi.dag() * rho * phi).tr().real
        if p > 1e-10:
            psi = sqrt_rho * phi / np.sqrt(p)
            rho_psi = psi * psi.dag()
            ensemble.append(rho_psi)
    return ensemble

def gen_tfim_ground_states_qt(N, g_range, n, periodic=True, use_dmrg=False, seed=0):
    """
    Generate a dataset of N ground states for the transverse-field Ising model (TFIM)
    with Hamiltonian H = -(1-g) sum_i Z_i Z_{i+1} - g * sum_i X_i, where g is sampled
    uniformly from g_range. Returns a list of tuples (state, g, energy, magnetization, zz_corrs, disorder_corrs),
    where state is a QuTiP Qobj (state vector), energy is the ground state energy,
    magnetization is <sum_i Z_i> / n (always ~0 without field),
    zz_corrs is a list of average <Z_i Z_{i+k}> for k=1 to n-1, and disorder_corrs is list of abs(<\prod_{i=1}^k X_i>) for k=1 to n.

    Parameters:
    -----------
    N : int
        Number of ground states to generate.
    g_range : list
        Range [g_min, g_max] for the transverse field strength g.
    n : int
        Number of qubits 
    periodic : bool
        Whether to use periodic boundary conditions (default: True).
    use_dmrg : bool
        Use DMRG for ground state computation (default: False, but ignored as QuTiP uses exact diagonalization).

    Returns:
    --------
    dataset : list
        List of tuples [(state_1, g_1, energy_1, magnetization_1, zz_corrs_1, disorder_corrs_1), ..., ].
    """

    dataset = []
    np.random.seed(seed)
    # Generate N unique g values evenly spaced in g_range
    g_values = np.linspace(g_range[0], g_range[1], N, endpoint=True)
    
    I = qt.qeye(2)
    X = qt.sigmax()
    Z = qt.sigmaz()
    # Force exact diagonalization (QuTiP does not support DMRG)
    if use_dmrg:
        warnings.warn("QuTiP does not support DMRG; using exact diagonalization.")
        use_dmrg = False

    for g in g_values:
        # Build Hamiltonian
        H = get_Ham_Ising_qt(n, J=1.0-g, bx=g, periodic=periodic)
        # Get ground state
        energy, ground_state = H.groundstate()
        
        # Fix phase to make first component positive
        #first_comp = ground_state.full()[0, 0]
        #phase = np.sign(first_comp.real) if first_comp.real != 0 else 1
        #ground_state *= phase
        
        # Compute magnetization <sum_i Z_i> / n
        magnetization = 0.0
        for i in range(n):
            ops = [I] * n
            ops[i] = Z
            Z_i = qt.tensor(ops)
            expect = qt.expect(Z_i, ground_state)
            magnetization += expect
        magnetization /= n

        # Compute ZZ correlations: list of avg <Z_i Z_{i+k}> for k=1 to n-1
        zz_corrs = []
        for k in range(1, n):
            corr_k = 0.0
            num_pairs = n if periodic else (n - k)
            for i in range(n):
                j = (i + k) % n if periodic else (i + k)
                if not periodic and j >= n:
                    continue
                ops = [I] * n
                ops[i] = Z
                ops[j] = Z
                zz_ij = qt.tensor(ops)
                expect = qt.expect(zz_ij, ground_state)
                corr_k += expect
            zz_corrs.append(corr_k / num_pairs)
        
        # Compute correlation function of disorder abs(<\prod_{i=1}^k X_i>) for k=1 to n
        disorder_corrs = []
        for k in range(1, n + 1):
            ops = [X if idx < k else qt.qeye(2) for idx in range(n)]
            string_op = qt.tensor(ops)
            expect = qt.expect(string_op, ground_state)
            disorder_corrs.append(expect)
        
        # Append to dataset
        dataset.append((ground_state, g, energy, magnetization, zz_corrs, disorder_corrs))
    
    return dataset

def multi_kron(ops):
    return reduce(jnp.kron, ops)

def gen_tfim_ground_states_jax(N, g_range, n, periodic=True, use_dmrg=False, seed=0):
    """
    Generate a dataset of N ground states for the transverse-field Ising model (TFIM)
    with Hamiltonian H = -(1-g) sum_i Z_i Z_{i+1} - g * sum_i X_i, where g is sampled
    uniformly from g_range. Returns a list of tuples (state, g, energy, magnetization, zz_corrs, disorder_corrs),
    where state is a JAX ndarray (state vector), energy is the ground state energy,
    magnetization is <sum_i Z_i> / n (always ~0 without field),
    zz_corrs is a list of average <Z_i Z_{i+k}> for k=1 to n-1, and disorder_corrs is list of <\prod_{i=1}^k X_i> for k=1 to n.

    Parameters:
    -----------
    N : int
        Number of ground states to generate.
    g_range : list
        Range [g_min, g_max] for the transverse field strength g.
    n : int
        Number of qubits 
    periodic : bool
        Whether to use periodic boundary conditions (default: True).
    use_dmrg : bool
        Use DMRG for ground state computation (default: False, but ignored as JAX uses exact diagonalization).

    Returns:
    --------
    dataset : list
        List of tuples [(state_1, g_1, energy_1, magnetization_1, zz_corrs_1, disorder_corrs_1), ..., ].
    """

    dataset = []
    np.random.seed(seed)
    # Generate N unique g values evenly spaced in g_range
    g_values = np.linspace(g_range[0], g_range[1], N, endpoint=True)
    
    I = jnp.eye(2)
    X = jnp.array([[0., 1.], [1., 0.]])
    Z = jnp.array([[1., 0.], [0., -1.]])
    # Force exact diagonalization (JAX does not support DMRG)
    if use_dmrg:
        warnings.warn("JAX does not support DMRG; using exact diagonalization.")
        use_dmrg = False

    for g in g_values:
        # Build Hamiltonian
        H = get_Ham_Ising_jax(n, J= (1-g), bx=g, periodic=periodic)

        # Get ground state
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)
        energy = eigenvalues[0]
        ground_state = eigenvectors[:, 0]
        
        # Fix phase to make first component positive
        #ground_state = ground_state * jnp.sign(jnp.real(ground_state[0]))
        
        # Compute magnetization <sum_i Z_i> / n
        magnetization = 0.0
        for i in range(n):
            ops = [I] * n
            ops[i] = Z
            Z_i = multi_kron(ops)
            expect = jnp.real(ground_state.conj().T @ Z_i @ ground_state)
            magnetization += expect
        magnetization /= n

        # Compute ZZ correlations: list of avg <Z_i Z_{i+k}> for k=1 to n-1
        zz_corrs = []
        for k in range(1, n):
            corr_k = 0.0
            num_pairs = n if periodic else (n - k)
            for i in range(n):
                j = (i + k) % n if periodic else (i + k)
                if not periodic and j >= n:
                    continue
                ops = [I] * n
                ops[i] = Z
                ops[j] = Z
                zz_ij = multi_kron(ops)
                expect = jnp.real(ground_state.conj().T @ zz_ij @ ground_state)
                corr_k += expect
            zz_corrs.append(corr_k / num_pairs)
        
        # Compute correlation function of disorder <\prod_{i=1}^k X_i> for k=1 to n
        disorder_corrs = []
        for k in range(1, n + 1):
            ops = [X if idx < k else I for idx in range(n)]
            string_op = multi_kron(ops)
            expect = jnp.real(ground_state.conj().T @ string_op @ ground_state)
            disorder_corrs.append(expect)
        
        # Append to dataset
        dataset.append((ground_state, g, energy, magnetization, zz_corrs, disorder_corrs))
    
    return dataset