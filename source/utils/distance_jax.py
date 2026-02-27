# Optimal transfort Tools
# https://ott-jax.readthedocs.io/en/latest/
# from ott.geometry import pointcloud
# from ott.solvers.linear import solve
# from ott.geometry.costs import CostFn

import jax.numpy as jnp
import numpy as np
from opt_einsum import contract

import jax
from jax import numpy as jnp

# POT: Python Optimal Transport
# open source Python library provides several solvers for optimization problems related to Optimal Transport for signal, image processing and machine learning.
# Website and documentation: https://PythonOT.github.io/


import ot

######## JAX FUNCTION ####################
@jax.jit
def natural_distance_jax(Set1, Set2):
    r11 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn', jnp.conj(Set1), Set1))**2)
    r22 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn', jnp.conj(Set2), Set2))**2)
    r12 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn', jnp.conj(Set1), Set2))**2)
    return 2.0 * r12 - r11 - r22


# If you don’t need JIT for OT‐distance:
def wass_distance_jax(Set1, Set2):
    overlap = jnp.conj(Set1) @ jnp.transpose(Set2)
    D = 1.0 - jnp.abs(overlap)**2
    D_np = np.array(D, dtype=np.float32) 
    return ot.emd2([], [], D_np)


# (If you want the JIT‐compatible OT version, install ott: pip install ott-jax)
from ott.geometry import geometry
# from ott.solvers.linear import sinkhorn

from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr

@jax.jit
def wass_distance_ott(Set1: jnp.ndarray,
                      Set2: jnp.ndarray,
                      epsilon: float = 1e-4) -> jnp.ndarray:
    """
    JIT‐able OT cost between two uniform discrete measures on quantum states.

    Args:
      Set1: [m, d] complex64 or complex128  (each row is a normalized state |ψ_i⟩)
      Set2: [n, d] complex64 or complex128  (each row is a normalized state |φ_j⟩)
      epsilon: entropic regularization strength (0.0 => exact OT)

    Returns:
      A scalar JAX DeviceArray = the OT cost ⟨π*, C⟩.
    """
    # Build the cost matrix C_{i,j} = 1 − |⟨ψ_i | φ_j⟩|²
    overlap = jnp.conj(Set1) @ jnp.transpose(Set2)   # shape [m, n]
    C = 1.0 - jnp.abs(overlap) ** 2                  # shape [m, n]
    
    geom = geometry.Geometry(cost_matrix=C, epsilon=epsilon)
    
    # Uniform weights a, b
    m, n = C.shape
    a = jnp.ones((m,)) / m
    b = jnp.ones((n,)) / n

    ot_prob = linear_problem.LinearProblem(a=a, b=b, geom=geom)
    
    # Call sinkhorn.sinkhorn directly (no LinearProblem needed)
    solver = sinkhorn.Sinkhorn()
    out = solver(ot_prob)  # <-- this returns a SinkhornOutput object
    res = out.reg_ot_cost
    return res