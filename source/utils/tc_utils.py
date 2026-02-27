#########################################################
# Util functions for tensorcircuit
#########################################################
import jax
from jax import numpy as jnp
import tensorcircuit as tc

def extract_measurement_features(qc, n_qubits, bases=['X', 'Y', 'Z']):
    """
    Extract features from measurement results in specified bases.
    Args:
        qc: Quantum circuit
        n_qubits: Number of qubits
        bases: Measurement bases (e.g., ['X', 'Y', 'Z'])
    Returns:
        processed_features: Flattened and normalized measurement features
    """
    features = []
    for basis in bases:
        if basis == 'X':
            for i in range(n_qubits):
                qc.h(i)  # Rotate to X-basis
        elif basis == 'Y':
            for i in range(n_qubits):
                qc.sdg(i)  # Rotate to Y-basis
                qc.h(i)

        # Collect measurement results in the current basis
        features.append([qc.expectation((tc.gates.z(), [i])) for i in range(n_qubits)])

        # Undo rotations
        if basis == 'X':
            for i in range(n_qubits):
                qc.h(i)
        elif basis == 'Y':
            for i in range(n_qubits):
                qc.h(i)
                qc.s(i)

    # Flatten the features
    flattened_features = [item for sublist in features for item in sublist]

    # Normalize features
    flattened_features = jnp.array(flattened_features)
    min_val = jnp.min(flattened_features)
    max_val = jnp.max(flattened_features)
    processed_features = (flattened_features - min_val) / (max_val - min_val + 1e-8)

    return processed_features

import tensorcircuit as tc
import jax.numpy as jnp

def generator_circuit(in_state, params, total_qubits, n_layers, circuit_type='rzryrz_rxxryyrzz'):
    """
    in_state: input quantum state of total_qubits (e.g., all |0> for QCBM)
    params: parameters of the circuit, initialized in [-1,1]
    total_qubits: number of qubits in the circuit
    n_layers: number of layers in the circuit
    Output: quantum state
    """
    # Parameter count validation
    Q = total_qubits
    ent1_count = Q // 2
    ent2_count = (Q - 1) // 2
    if circuit_type == 'rzryrz_rxxryyrzz':
        params_per_layer = 3 * Q + 3 * (ent1_count + ent2_count)
    elif circuit_type == 'rxycz':
        params_per_layer = 2 * Q
    elif circuit_type == 'ryzcz':
        params_per_layer = 2 * Q
    elif circuit_type == 'rxyzcz':
        params_per_layer = 3 * Q
    else:
        raise ValueError(f"Unsupported circuit_type: {circuit_type}")
    
    expected_params = params_per_layer * n_layers
    if params.shape[0] != expected_params:
        raise ValueError(f"Expected {expected_params} parameters, got {params.shape[0]}")

    qc = tc.Circuit(nqubits=total_qubits, inputs=in_state)
    
    for l in range(n_layers):
        base_idx = l * params_per_layer
        
        # Single-qubit rotations
        for i in range(total_qubits):
            if circuit_type == 'SU2-full':
                # RZ-RY-RZ for full SU(2) coverage
                qc.rz(i, theta=jnp.pi * params[base_idx + i])
                qc.ry(i, theta=jnp.pi * params[base_idx + Q + i])
                qc.rz(i, theta=jnp.pi * params[base_idx + 2 * Q + i])
            elif circuit_type == 'rxycz':
                # Original rxycz circuit
                qc.rx(i, theta=jnp.pi * params[2 * total_qubits * l + i])
                qc.ry(i, theta=jnp.pi * params[2 * total_qubits * l + total_qubits + i])
            elif circuit_type == 'ryzcz':
                qc.ry(i, theta=jnp.pi * params[2 * total_qubits * l + i])
                qc.rz(i, theta=jnp.pi * params[2 * total_qubits * l + total_qubits + i])
            elif circuit_type == 'rxyzcz':
                qc.rx(i, theta=jnp.pi * params[3 * total_qubits * l + i])
                qc.ry(i, theta=jnp.pi * params[3 * total_qubits * l + total_qubits + i])
                qc.rz(i, theta=jnp.pi * params[3 * total_qubits * l + 2 * total_qubits + i])
        
        # Entangling gates
        if circuit_type == 'SU2-full':
            ent_idx = base_idx + 3 * Q
            for i in range(ent1_count):
                qc.rxx(2 * i, 2 * i + 1, theta=jnp.pi * params[ent_idx + 3 * i])
                qc.ryy(2 * i, 2 * i + 1, theta=jnp.pi * params[ent_idx + 3 * i + 1])
                qc.rzz(2 * i, 2 * i + 1, theta=jnp.pi * params[ent_idx + 3 * i + 2])
            
            ent_idx += 3 * ent1_count
            for i in range(ent2_count):
                qc.rxx(2 * i + 1, 2 * i + 2, theta=jnp.pi * params[ent_idx + 3 * i])
                qc.ryy(2 * i + 1, 2 * i + 2, theta=jnp.pi * params[ent_idx + 3 * i + 1])
                qc.rzz(2 * i + 1, 2 * i + 2, theta=jnp.pi * params[ent_idx + 3 * i + 2])
        elif 'cz' in circuit_type:
            for i in range(total_qubits // 2):
                qc.cz(2 * i, 2 * i + 1)
            for i in range((total_qubits - 1) // 2):
                qc.cz(2 * i + 1, 2 * i + 2)
    
    return qc.state()

def discriminator_circuit(in_states, params_d, params_c, c_dist, n_qubits, n_layers, circuit_type = 'rxyzcz'):
    qc = tc.Circuit(n_qubits, inputs=in_states)

    for l in range(n_layers):
        for i in range(n_qubits):
            if circuit_type == 'rxyzcz':
                qc.rx(i, theta=jnp.pi *params_d[3 * n_qubits * l + i])
                qc.ry(i, theta=jnp.pi *params_d[3 * n_qubits * l + n_qubits + i])
                qc.rz(i, theta=jnp.pi *params_d[3 * n_qubits * l + 2 * n_qubits + i])
            elif circuit_type == 'rxycz':
                qc.rx(i, theta=jnp.pi *params_d[2 * n_qubits * l + i])
                qc.ry(i, theta=jnp.pi *params_d[2 * n_qubits * l + n_qubits + i])
        for i in range(n_qubits - 1):
            qc.cz(i, i + 1)

    features = extract_measurement_features(qc, n_qubits)
    scores = c_dist.apply(params_c, features)
    return scores

def random_rx_circuit(in_states, params, n_qubits):
    qc = tc.Circuit(n_qubits, inputs=in_states)
    # random_unitary = tc.gates.random_unitary(2**n_qubits)
    # # Apply the random unitary as a gate to all qubits in the circuit
    # qc.unitary(random_unitary, [i for i in range(n_qubits)])

    # Apply random rotation gates to the qubit
    for i in range(n_qubits):
        #theta_x = np.random.uniform(0, 2 * np.pi)
        qc.rx(i, theta=jnp.pi *params[i])

    final_state = qc.state()
    return final_state

