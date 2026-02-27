import jax
import jax.numpy as jnp
from jax import random, vmap, jit, value_and_grad, lax
from functools import partial

from numpy import full
import optax
from jax import debug

# ---------- Basic gates ----------
def ry(theta):
    c = jnp.cos(theta/2.0)
    s = jnp.sin(theta/2.0)
    return jnp.array([[c, -s],
                      [s,  c]], dtype=jnp.complex64)

CNOT = jnp.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]], dtype=jnp.complex64)

# ---------- Gate application on state vector (no full 2^n x 2^n op builds) ----------

def apply_1q(state, gate, q, n_qubits):
    # state: (2**n,), complex
    # reshape to (2,)*n, move target axis to front, apply matmul, move back
    psi = state.reshape((2,)*n_qubits)
    perm = (q,) + tuple(i for i in range(n_qubits) if i != q)
    inv = jnp.argsort(jnp.array(perm))
    psi = jnp.transpose(psi, perm)  # (2, 2, ..., 2)

    psi = psi.reshape(2, -1)
    psi = gate @ psi
    psi = psi.reshape((2,) + (2,)*(n_qubits-1))

    psi = jnp.transpose(psi, tuple(inv))
    return psi.reshape(-1)

def apply_2q(state, gate, q1, q2, n_qubits):
    # works for any pair (including wrap-around); keep order (q1,q2) as in 'gate'
    if q1 == q2:
        return state
    # bring q1,q2 to front
    psi = state.reshape((2,)*n_qubits)
    qs = (q1, q2)
    perm_front = qs + tuple(i for i in range(n_qubits) if i not in qs)
    inv = jnp.argsort(jnp.array(perm_front))
    psi = jnp.transpose(psi, perm_front)  # (2,2,2,...)

    psi = psi.reshape(4, -1)
    psi = gate @ psi
    psi = psi.reshape((2,2) + (2,)*(n_qubits-2))

    psi = jnp.transpose(psi, tuple(inv))
    return psi.reshape(-1)

# ---------- Ansatz & encoder ----------

def ansatz_layer(state, params_layer, n_qubits):
    # single-qubit RY rotations
    for q in range(n_qubits):
        state = apply_1q(state, ry(params_layer[q]), q, n_qubits)
    
    # CNOT ring with wrap-around handled by apply_2q
    for q in range(n_qubits):
        ctrl = q
        tgt = (q + 1) % n_qubits
        state = apply_2q(state, CNOT, ctrl, tgt, n_qubits)

    # defensive renormalization to avoid drift
    nrm = jnp.linalg.norm(state)
    state = jnp.where(nrm > 0, state / nrm, state)
    return state

def encoder(state, params, n_qubits, n_layers):
    for l in range(n_layers):
        layer_params = params[l * n_qubits:(l + 1) * n_qubits]
        state = ansatz_layer(state, layer_params, n_qubits)
    return state

# ---------- QAE objective ----------

def extract_latent_state(encoded, n_latent, n_qubits):
    # Assuming "trash" is last k qubits in the reshape (consistent with our apply_* definition)
    k = n_qubits - n_latent
    psi = encoded.reshape((2**n_latent, 2**k))
    latent = psi[:, 0]  # amplitude slice where trash = |0...0>
    nrm = jnp.linalg.norm(latent)
    return jnp.where(nrm > 0, latent / nrm, latent)

def trash_fidelity(encoded, n_latent, n_qubits):
    # Reference trash state is |0...0>, so F = ⟨0...0| ρ_trash |0...0⟩
    k = n_qubits - n_latent
    psi = encoded.reshape((2**n_latent, 2**k))
    # ρ_trash = psi^† psi over latent indices, but we only need the (0,0) element:
    # F = sum_i |psi[i, 0]|^2
    p0 = jnp.sum(jnp.abs(psi[:, 0])**2)
    # If the global state is normalized, p0 ∈ [0,1]. Clamp tiny negatives from roundoff.
    return jnp.clip(p0.real, 0.0, 1.0)

def qae_loss(params, batch_input_states, n_qubits, n_latent, n_layers):
    def single_loss(state):
        encoded = encoder(state, params, n_qubits, n_layers)
        fid = trash_fidelity(encoded, n_latent, n_qubits)
        return 1.0 - fid
    losses = vmap(single_loss)(batch_input_states)
    return jnp.mean(losses)

# ---------- Training step ----------

#@partial(jit, static_argnums=(2,3,4))
def qae_update(params, batch_input_states, n_qubits, n_latent, n_layers, opt_state, optimizer):
    loss_value, grads = value_and_grad(qae_loss)(params, batch_input_states, n_qubits, n_latent, n_layers)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

def train_qae(logger, S_data, n_qubits, n_latent, n_layers, n_epochs, batch_size, lr, key):
    n_params = n_layers * n_qubits
    key, k_init = random.split(key)
    params = random.uniform(k_init, (n_params,), minval=0.0, maxval=2*jnp.pi).astype(jnp.float32)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    n_samples = S_data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    loss_hist = []

    for epoch in range(n_epochs):
        key, k_perm = random.split(key)
        perm = random.permutation(k_perm, n_samples)
        S_epoch = S_data[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            start = i * batch_size
            end = min((i+1) * batch_size, n_samples)
            batch = S_epoch[start:end]

            params, opt_state, loss_value = qae_update(
                params, batch, n_qubits, n_latent, n_layers, opt_state, optimizer
            )
            # weighted average over uneven last batch
            epoch_loss += loss_value * (end - start) / n_samples

        loss_hist.append(epoch_loss)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(f'QAE Epoch {epoch + 1} / {n_epochs}, Loss: {epoch_loss:.6f}')

    return params, jnp.array(loss_hist)

# ---------- Decoder (approximate inverse) ----------
def inverse_ansatz_layer(state, params_layer, n_qubits):
    # First undo the CNOT ring in reverse order
    for q in range(n_qubits - 1, -1, -1):
        ctrl = q
        tgt  = (q + 1) % n_qubits
        state = apply_2q(state, CNOT, ctrl, tgt, n_qubits)  # CNOT is self-inverse

    # Then undo the single-qubit rotations: RY(-theta)
    for q in range(n_qubits):
        state = apply_1q(state, ry(-params_layer[q]), q, n_qubits)

    # (Optional) defensive renorm
    nrm = jnp.linalg.norm(state)
    state = jnp.where(nrm > 0, state / nrm, state)
    return state


def decoder(latent_state, params, n_qubits, n_latent, n_layers):
    k = n_qubits - n_latent
    trash_zero = jnp.zeros((2**k,), dtype=jnp.complex64).at[0].set(1.0)
    recon = jnp.kron(latent_state, trash_zero)

    # Apply U†: reverse layers, each as inverse_ansatz_layer
    for l in range(n_layers - 1, -1, -1):
        layer_params = params[l * n_qubits:(l + 1) * n_qubits]
        recon = inverse_ansatz_layer(recon, layer_params, n_qubits)

    nrm = jnp.linalg.norm(recon)
    return jnp.where(nrm > 0, recon / nrm, recon)

encode_vmap = vmap(lambda psi, theta, nq, nl, nlayers: extract_latent_state(encoder(psi, theta, nq, nlayers), nl, nq),
                   in_axes=(0, None, None, None, None))
decode_vmap = vmap(lambda lat, theta, nq, nl, nlayers: decoder(lat, theta, nq, nl, nlayers),
                   in_axes=(0, None, None, None, None))

class QAEModel:
    def __init__(self, n_qubits: int, n_latent: int, n_layers: int, n_epochs: int, lr: float, theta: jnp.ndarray = None, save_file: str = None):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.theta = theta
        self.save_file = save_file
        
    def encode(self, states: jnp.ndarray) -> jnp.ndarray:
        return encode_vmap(states, self.theta, self.n_qubits, self.n_latent, self.n_layers)

    def decode(self, latent_states: jnp.ndarray) -> jnp.ndarray:
        return decode_vmap(latent_states, self.theta, self.n_qubits, self.n_latent, self.n_layers)
