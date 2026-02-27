##########################################################################
## Code to implement the quantum diffusion model class
## in the tensorflow framework
##
## Author: Tran Quoc Hoan, Start date: 2024/09/19
##
##########################################################################


from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp

#import scipy as sp
#from scipy.stats import unitary_group

import tensorcircuit as tc
from utils.tc_utils import *
from utils.utils import *

from opt_einsum import contract

K = tc.set_backend('jax')
tc.set_dtype('complex64')

PAULIS_1Q = ["x", "y", "z"]
PAULIS_2Q = [
    ("x","x"), ("x","y"), ("x","z"),
    ("y","x"), ("y","y"), ("y","z"),
    ("z","x"), ("z","y"), ("z","z"),
]

@partial(jax.jit, static_argnames=('n_ancilla','n_qubits'))
def _random_measure_jax(inputs: jnp.ndarray,
                       n_ancilla: int,
                       n_qubits: int,
                       key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    inputs: [batch, 2**(n_ancilla + n_qubits)] complex amplitudes
    Returns: [batch, 2**n_qubits] post-measurement states (normalized).
    """
    batch = inputs.shape[0]

    # Compute |ψ|^2 over ancilla vs system
    probs = jnp.abs(inputs.reshape((batch, 2**n_ancilla, 2**n_qubits)))**2
    m_probs = jnp.sum(probs, axis=2)       # shape [batch, 2**n_ancilla]

    # Sample ancilla outcome
    rnd = jax.random.uniform(key, (batch,), minval=0.0, maxval=1.0)
    cum = jnp.cumsum(m_probs, axis=1)
    cum = cum / jnp.sum(m_probs, axis=1, keepdims=True)
    # first index where cum > rnd
    m_res = jnp.argmax(cum > rnd[:, None], axis=1).astype(jnp.int32)

    # Pick out the corresponding system amplitudes
    idx = (2**n_qubits) * m_res[:, None] + jnp.arange(2**n_qubits)
    post = jnp.take_along_axis(inputs, idx, axis=1)  # [batch, 2**n_qubits]

    norm = jnp.linalg.norm(post, axis=1, keepdims=True)
    return post / norm

@staticmethod
@partial(jax.jit, static_argnames=('n_ancilla','n_qubits'))
def _random_measure_pure(inputs, n_ancilla, n_qubits, subkey: jax.random.PRNGKey):
  # Take a PRNGKey in as an explicit argument and split insider for jit

  next_key, meas_key = jax.random.split(subkey)
  post_states = _random_measure_jax(inputs, n_ancilla, n_qubits, meas_key)
  return post_states, next_key

@staticmethod
@partial(jax.jit, static_argnames=('n_ancilla','n_qubits','backward_circuit_vmap'))
def backward_output_pure(inputs, params, 
        n_ancilla, n_qubits,
        backward_circuit_vmap: callable,
        subkey: jax.random.PRNGKey):
  """
    Backward denoising process at step t
    Args: 
      inputs: the input data set at step t
      params: the parameters of the backward circuit at step t
      subkey: PRNGKey for random measurement
  """
  # Outputs through quantum circuits before measurement
  output_full = backward_circuit_vmap(inputs, params)
  # Perform the ancilla measurement and get the post-measurement state
  output_t, next_key = _random_measure_pure(output_full, n_ancilla, n_qubits, subkey)
  return output_t, next_key

class ScramblingModel():
  def __init__(self, n_qubits, T, rseed=0, p1=0.0, p2=0.0):
    """
    Scramble quantum circuit to scramble arbitrary set of states to Haar random states
    Args:
      n_qubits: number of data qubits
      T: number of diffusion (scrambling) steps
      rseed: initial seed for the PRNGKey, default is 0
      p1: probability of applying random Pauli gate after each single-qubit rotation
      p2: probability of applying random Pauli gate after each RZZ gate
    """
    super().__init__()
    self.n_qubits = n_qubits
    self.t = 0
    self.T = T
    self.p1 = p1
    self.p2 = p2
    self._key = jax.random.PRNGKey(rseed)
  
  def __maybe_apply_1q_noise(self, circ, q):
    self._key, k1, k2 = jax.random.split(self._key, 3)
    r = jax.random.uniform(k1)
    if r < self.p1:
      pauli_idx = jax.random.randint(k2, (), 0, 3)
      pauli = PAULIS_1Q[pauli_idx]
      getattr(circ, pauli)(q)
      # if PAULIS_1Q[pauli_idx] == "x":
      #   circ.x(q)
      # elif PAULIS_1Q[pauli_idx] == "y":
      #   circ.y(q)
      # elif PAULIS_1Q[pauli_idx] == "z":
      #   circ.z(q)
  
  def __maybe_apply_2q_noise(self, circ, q1, q2):
    self._key, k1, k2 = jax.random.split(self._key, 3)
    r = jax.random.uniform(k1)
    if r < self.p2:
      pauli_idx = jax.random.randint(k2, (), 0, 9)
      p1, p2 = PAULIS_2Q[pauli_idx]
      getattr(circ, p1)(q1)
      getattr(circ, p2)(q2)
      # if p1 == "x":
      #   circ.x(q1)
      # elif p1 == "y":
      #   circ.y(q1)
      # elif p1 == "z":
      #   circ.z(q1)
      # if p2 == "x":
      #   circ.x(q2)
      # elif p2 == "y":
      #   circ.y(q2)
      # elif p2 == "z":
      #   circ.z(q2)
  
  def scramble_circuit(self, inputs, params):
    """
    Obtain the state through the diffusion step
    Args:
      inputs: the input quantum states
      params: params of the diffusion circuit
        for one qubit, params = phis (the single-qubit rotation angle)
        for multi qubits, params = (phis, gs)  (gs: the angle of RZZ gates in diffusion circuit)
    """
    circ = tc.Circuit(nqubits=self.n_qubits, inputs=inputs)
    if self.n_qubits == 1:
      phis = params
      for s in range(self.t):
        circ.rz(0, theta=phis[3*s])
        self.__maybe_apply_1q_noise(circ, 0)

        circ.ry(0, theta=phis[3*s+1])
        self.__maybe_apply_1q_noise(circ, 0)

        circ.rz(0, theta=phis[3*s+2])
        self.__maybe_apply_1q_noise(circ, 0)
    else:
      phis, gs = params
      for s in range(self.t):
        for i in range(self.n_qubits):
          circ.rz(i, theta=phis[3 * self.n_qubits * s + i])
          self.__maybe_apply_1q_noise(circ, i)

          circ.ry(i, theta=phis[3 * self.n_qubits * s + self.n_qubits + i])
          self.__maybe_apply_1q_noise(circ, i)

          circ.rz(i, theta=phis[3 * self.n_qubits * s + 2*self.n_qubits + i])
          self.__maybe_apply_1q_noise(circ, i)
        # homogenous RZZ on every pair of qubits (n_qubits >= 2)
        if self.n_qubits > 1:
          for i, j in combinations(range(self.n_qubits), 2):
              circ.rzz(i, j, theta=gs[s] / (2 * self.n_qubits ** 0.5))
              self.__maybe_apply_2q_noise(circ, i, j)
    return circ.state()
  
  def diffusion_step(self, t:int, inputs: jnp.ndarray, diff_hs: jnp.ndarray) -> jnp.ndarray:
    """
    JAX version of diffusion_step.
    Args:
      t: diffusion step (int)
      inputs: [batch, 2**(n_qubits)] 
      diff_hs: [batch] vector of hyper-parameters 
    Returns:
      states: [batch, ...] post-diffusion states
    """
    self.t = t
    batch = inputs.shape[0]
    newkey, sub1, sub2 = jax.random.split(self._key, 3)
    self._key = newkey

    # --- generate phis ---
    phis = jax.random.uniform(sub1, (batch, 3 * self.n_qubits * t), minval=0.0, maxval=1.0)
    phis = phis * (jnp.pi/4) - (jnp.pi/8)
    # scale per-sample
    diff_repeated = jnp.repeat(diff_hs, 3 * self.n_qubits)   # shape = (3*n_qubits*t,)
    phis = phis * diff_repeated   # broadcasts diff_hs over the param axis

    if self.n_qubits == 1:
        params = phis
    else:
        # also generate RZZ angles gs
        gs = jax.random.uniform(sub2, (batch, t), minval=0.0, maxval=1.0)
        gs = gs * 0.2 + 0.4
        gs = gs * diff_hs

        # pack into a tuple
        params = (phis, gs)

    # vmap over (inputs, params)
    states = jax.vmap(self.scramble_circuit, in_axes=(0, 0))(inputs, params)
    return states
  

class ChaoticScramblingModel():
  def __init__(self, n_qubits, n_ancilla, rand_ancilla, rseed=0, 
              type_ham='Ising_nearest', J=(-1.0, 0.0, 0.0), b=(-0.8090, -0.9045, 0.0), W=1.0, gamma_phi=0.0):
    """
    Scramble arbitrary set of states to Haar random states via a chaotic Hamiltonian
    Args:
      n_qubits: number of data qubits
      n_ancilla: number of ancilla qubits
      rand_ancilla: if True, the ancilla qubits are chosen randomly, otherwise they are fixed in the |0> state
      rseed: initial seed for the PRNGKey, default is 0
      type_ham: type of Ising Hamiltonian
      
    """
    super().__init__()
    self.n_qubits = n_qubits
    self.n_ancilla = n_ancilla
    self.rand_ancilla = rand_ancilla

    self._key = jax.random.PRNGKey(rseed)
    if type_ham == 'Ising_nearest':
      self.H = get_Ham_HEIS_jax(L=self.n_qubits + self.n_ancilla, J=J, b=b, periodic=False)
    elif type_ham == 'Ising_random_all':
      self.H = get_Ham_Ising_random_all_jax(L=self.n_qubits + self.n_ancilla, J_s=J, bz=b, W=W)
    self.gamma_phi = gamma_phi
    self.L = self.n_qubits + self.n_ancilla

  def _apply_Z(self, psi: jnp.ndarray, q: int) -> jnp.ndarray:
    """
    Apply Z on qubit q (0 = most significant in the kron basis) to a column statevector.
    psi shape: (2**L, 1)
    """
    L = self.L
    psi_flat = psi.reshape((2,) * L)  # tensor view
    # Multiply slice where qubit q = 1 by -1
    idx1 = [slice(None)] * L
    idx1[q] = 1
    psi_flat = psi_flat.at[tuple(idx1)].multiply(-1.0)
    return psi_flat.reshape((-1, 1))
  
  def _apply_dephasing_trajectory(self, psi: jnp.ndarray, t_eff: float) -> jnp.ndarray:
    """
    Stochastic unraveling of independent dephasing on each qubit.
    For each qubit, apply Z with prob p = (1 - exp(-gamma_phi * t_eff))/2.
    """
    if self.gamma_phi == None:
      p = 0.5
    elif self.gamma_phi <= 0.0:
      return psi
    else:
      p = 0.5 * (1.0 - jnp.exp(-self.gamma_phi * t_eff))
    
    # Sample a Bernoulli mask for Z errors on each qubit
    self._key, subkey = jax.random.split(self._key)
    zmask = jax.random.bernoulli(subkey, p=p, shape=(self.L,))

    # Apply Z to qubits where zmask[q] = True
    for q in range(self.L):
      if bool(zmask[q]):
        psi = self._apply_Z(psi, q)
    return psi
  
  def scramble_Hamiltonian(self, input_state: jnp.ndarray, t: float) -> jnp.ndarray:
    """
    Obtain the state through the diffusion step
    Args:
      input_state: the input quantum state with n_d qubit
      t: evolution time interval
    Returns:
      evol_state: [2**(n_ancilla + n_qubits), 1] evolved state
    """
    dim_A = 2 ** self.n_ancilla
    psiA = jnp.zeros((dim_A, 1), dtype=jnp.complex64)
    psiA = psiA.at[0, 0].set(1.0 + 0j)

    if self.rand_ancilla > 0:
      # Build random ancilla basis state 
      newkey, subkey = jax.random.split(self._key)
      self._key = newkey
      rand_idx = jax.random.randint(subkey, (), 0, dim_A)
      psiA = psiA.at[rand_idx, 0].set(1.0 + 0j)

    # Combine ancilla ⊗ input_state
    combined = jnp.kron(psiA, input_state).reshape((-1, 1))

    # Time evolution operator U = exp(-i H t Δt)
    U = jsp.expm(-1j * self.H * t).astype(jnp.complex64)

    evol_state = U @ combined
    # --- dephasing trajectory (non-unitary noise, but kept as statevector per trajectory) ---
    evol_state = self._apply_dephasing_trajectory(evol_state, t)

    return evol_state

  def diffusion_step(self, t, inputs):
    """
    Obtain the quantum dataset through diffusion time t
    Args:
      inputs: [batch, 2**n_qubits] batch of the input quantum state with n_qubits
      t: evolution time interval
    Returns:
      pos_mes_states: [batch, 2**n_qubits] post-measurement states
    """
    newkey, subkey = jax.random.split(self._key)
    self._key = newkey
    evolve_states = K.vmap(self.scramble_Hamiltonian, vectorized_argnums=(0))(inputs, t)
    flat_states = jnp.squeeze(evolve_states, axis=-1)
    pos_mes_states = _random_measure_jax(flat_states, self.n_ancilla, self.n_qubits, subkey)
    return pos_mes_states


class QDM():
  def __init__(self, n_qubits, n_ancilla, T, n_layers, backward_circuit_type='rxycz', rseed=0):
    """
      n_qubits: number of data qubits
      n_ancilla: number of ancilla qubits
      T: number of diffusion steps
      n_layers: number of layers in backward circuit
      backward_circuit_type: type of backward circuit, 'rxycz' for rxycz circuit
    """
    self.n_qubits = n_qubits
    self.n_ancilla = n_ancilla
    self.n_tot = n_qubits + n_ancilla
    self.n_layers = n_layers
    self.T = T
    self.backward_circuit_type = backward_circuit_type

    # Modified parameter counting for backward circuit
    if self.backward_circuit_type == 'rxycz' or self.backward_circuit_type == 'ryzcz':
        self.backward_n_params = 2 * self.n_tot * self.n_layers
    elif self.backward_circuit_type == 'rxyzcz':
        self.backward_n_params = 3 * self.n_tot * self.n_layers
    elif self.backward_circuit_type == 'SU2-full':
        ent1_count = self.n_tot // 2
        ent2_count = (self.n_tot - 1) // 2
        params_per_layer = 3 * self.n_tot + 3 * (ent1_count + ent2_count)
        self.backward_n_params = params_per_layer * self.n_layers
    else:
        raise ValueError(f"Unsupported backward_circuit_type: {self.backward_circuit_type}")

    # Bind all static arguments via partial
    batched_circuit = partial(
        generator_circuit,
        total_qubits=self.n_tot,         # your Python int
        n_layers=self.n_layers,          # your Python int
        circuit_type=self.backward_circuit_type  # your Python str
    )

    # Vectorize over the first argument (in_state) 
    # in_axes=(0, None) means:
    #   • “Map over axis 0 of in_state”
    #   • Keep the entire `params` array fixed for the whole batch.
    vmapped_circuit = jax.vmap(batched_circuit, in_axes=(0, None))

    # JIT‐compile that vmap’d function
    self.backward_circuit_vmap = jax.jit(vmapped_circuit)
  
    self._key = jax.random.PRNGKey(rseed)
  
  def set_forward_states_diff(self, forward_states_diff):
    self.forward_states_diff = jnp.asarray(forward_states_diff, dtype=jnp.complex64)

  
  def backward_output_t(self, inputs, params):
    """
      Backward denoising process at step t
      Args: 
        inputs: the input data set at step t
        params: the parameters of the backward circuit at step t
      
      Non‐jit wrapper that:
        1) pulls the stored key,
        2) calls the pure‐JIT version to get (output_t, new_key),
        3) writes back new_key into self._key, and
        4) returns output_t.
    """
    old_key = self._key
    output_t, new_key = backward_output_pure(inputs, params, self.n_ancilla, self.n_qubits, self.backward_circuit_vmap, old_key)
    self._key = new_key  # update the key for the next call
    return output_t
  
  def prepare_input_t(self, inputs_T, params_cul, t, n_dat):
    """
    Prepare the input samples for step t
    Args:
      inputs_T: the input state at the beginning of the backwarad
      params_cul: all circuit parameters till step t+1
      t: step 
      n_dat: number of data
    Ouput: input samples (combined) and not combined input for step t

    """
    dim_anc = 2**self.n_tot - 2**self.n_qubits
    # Tensor with |0>^{n_ancilla} state and inputs_T is 
    # the concat of the input matrix with zeros (remained dimension)
    pad_zeros = jnp.zeros((n_dat, dim_anc), dtype=jnp.complex64)
    self.input_t_plus = jnp.concatenate([inputs_T, pad_zeros], axis=1)
    params_cul = jnp.asarray(params_cul, dtype=jnp.float32)
    output = inputs_T
    for step in range(self.T - 1, t, -1):
      output = self.backward_output_t(self.input_t_plus, params_cul[step])
      self.input_t_plus = jnp.concatenate([output, pad_zeros], axis=1)
    output = output.astype(jnp.complex64)
    return self.input_t_plus, output
  
  def backward_gen_states(self, inputs_T, params_tot):
    """
    Generate the dataset in backward denoising process
    with training data set
    Args:
      inputs_T: the input state at the beginning of the backward
      params_tol: all circuit parameters
    Ouput: generated states from inputs_T after all backward process
    """
    n_dat = len(inputs_T)
    states = [inputs_T]
    dim_anc = 2**self.n_tot - 2**self.n_qubits
    pad_zeros = jnp.zeros((n_dat, dim_anc), dtype=jnp.complex64)

    input_t_plus = jnp.concatenate([inputs_T, pad_zeros], axis=1)
    params_tot = jnp.asarray(params_tot, dtype=jnp.float32)

    for step in range(self.T - 1, -1, -1):
      output = self.backward_output_t(input_t_plus, params_tot[step])
      input_t_plus = jnp.concatenate([output, pad_zeros], axis=1)
      states.append(output)
    states = jnp.stack(states)[::-1]
    return states
  
