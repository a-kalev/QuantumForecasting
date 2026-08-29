#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact diagonalization utilities for the random-field XXZ chain.

Hamiltonian:
    H = J * sum_i (
            Sx_i Sx_{i+1}
          + Sy_i Sy_{i+1}
          + delta * Sz_i Sz_{i+1}
        )
        + sum_i h_i Sz_i

with open boundaries and S^alpha = sigma^alpha / 2.

The calculation is restricted to a fixed total-Sz sector. For the Neel
state used here, L must be even and the default sector has N_up = L / 2.

Observable channels:
    1. Local magnetizations <sigma_i^z>
    2. Nearest-neighbor memory correlations
           -<sigma_i^z sigma_{i+1}^z>
    3. Staggered imbalance
           (1/L) sum_i (-1)^i <sigma_i^z>

All generated disorder fields, seeds, conventions, and channel names are
returned explicitly in the episode dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class HeisenbergModel:
    L: int
    W: float
    seed: int
    J: float = 1.0
    delta: float = 1.0
    n_up: int | None = None

    def __post_init__(self) -> None:
        if self.L < 2:
            raise ValueError("L must be at least 2.")
        if self.L % 2 != 0:
            raise ValueError("The Neel-state benchmark requires even L.")
        if self.W < 0:
            raise ValueError("W must be nonnegative.")

        n_up = self.L // 2 if self.n_up is None else self.n_up
        if not 0 <= n_up <= self.L:
            raise ValueError("n_up must satisfy 0 <= n_up <= L.")

        object.__setattr__(self, "n_up", n_up)


def fixed_magnetization_basis(L: int, n_up: int) -> np.ndarray:
    """
    Return computational-basis states with exactly n_up up spins.

    Site 0 is stored in the most-significant bit.
    Bit 1 denotes spin up, sigma^z = +1.
    Bit 0 denotes spin down, sigma^z = -1.
    """
    expected_size = comb(L, n_up)
    basis = np.fromiter(
        (state for state in range(1 << L) if bin(state).count("1") == n_up),
        dtype=np.int64,
        count=expected_size,
    )

    if basis.size != expected_size:
        raise RuntimeError("Failed to construct the fixed-magnetization basis.")

    return basis


def sample_disorder_fields(L: int, W: float, seed: int) -> np.ndarray:
    """Draw h_i independently and uniformly from [-W, W]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-W, W, size=L)


def _site_bit(state: int, site: int, L: int) -> int:
    return (state >> (L - 1 - site)) & 1


def _spin_z(state: int, site: int, L: int) -> float:
    """Eigenvalue of S_i^z, equal to +/- 1/2."""
    return 0.5 if _site_bit(state, site, L) else -0.5


def build_hamiltonian(
    model: HeisenbergModel,
    fields: np.ndarray,
    basis: np.ndarray,
) -> csr_matrix:
    """Construct the Hamiltonian in the selected total-Sz sector."""
    if fields.shape != (model.L,):
        raise ValueError(f"fields must have shape ({model.L},).")

    state_to_index = {int(state): index for index, state in enumerate(basis)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for row, state_value in enumerate(basis):
        state = int(state_value)
        diagonal = 0.0

        # Random-field term.
        for site in range(model.L):
            diagonal += fields[site] * _spin_z(state, site, model.L)

        # Open-boundary XXZ interaction.
        for site in range(model.L - 1):
            sz_left = _spin_z(state, site, model.L)
            sz_right = _spin_z(state, site + 1, model.L)

            diagonal += model.J * model.delta * sz_left * sz_right

            # SxSx + SySy = 1/2 (S+S- + S-S+).
            # It exchanges neighboring opposite spins with amplitude J/2.
            if sz_left != sz_right:
                left_mask = 1 << (model.L - 1 - site)
                right_mask = 1 << (model.L - 2 - site)
                flipped_state = state ^ left_mask ^ right_mask

                col = state_to_index[flipped_state]
                rows.append(row)
                cols.append(col)
                data.append(0.5 * model.J)

        rows.append(row)
        cols.append(row)
        data.append(diagonal)

    dimension = basis.size
    hamiltonian = coo_matrix(
        (data, (rows, cols)),
        shape=(dimension, dimension),
        dtype=np.float64,
    ).tocsr()

    hermiticity_error = np.max(
        np.abs((hamiltonian - hamiltonian.T).data)
    ) if (hamiltonian - hamiltonian.T).nnz else 0.0

    if hermiticity_error > 1e-12:
        raise RuntimeError(
            f"Hamiltonian is not Hermitian; maximum error = "
            f"{hermiticity_error:.3e}."
        )

    return hamiltonian


def neel_state(
    L: int,
    basis: np.ndarray,
    first_spin_up: bool = True,
) -> np.ndarray:
    """
    Construct |up,down,up,down,...> when first_spin_up=True.
    """
    state_integer = 0

    for site in range(L):
        is_up = (site % 2 == 0) if first_spin_up else (site % 2 == 1)
        if is_up:
            state_integer |= 1 << (L - 1 - site)

    matches = np.flatnonzero(basis == state_integer)
    if matches.size != 1:
        raise ValueError("The Neel state is not contained in the chosen sector.")

    psi0 = np.zeros(basis.size, dtype=np.complex128)
    psi0[matches[0]] = 1.0
    return psi0


def observable_diagonals(
    L: int,
    basis: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """
    Return names and computational-basis diagonals of all observables.

    Output shape:
        diagonals[channel, basis_state]
    """
    dimension = basis.size
    sigma_z = np.empty((L, dimension), dtype=np.float64)

    for site in range(L):
        shift = L - 1 - site
        bits = (basis >> shift) & 1
        sigma_z[site] = 2.0 * bits.astype(np.float64) - 1.0

    bond_memory = -(sigma_z[:-1] * sigma_z[1:])

    staggered_sign = (-1.0) ** np.arange(L)
    imbalance = np.mean(staggered_sign[:, None] * sigma_z, axis=0)

    names = [f"sigma_z_{site}" for site in range(L)]
    names += [
        f"bond_memory_{site}_{site + 1}"
        for site in range(L - 1)
    ]
    names.append("imbalance")

    diagonals = np.vstack(
        [
            sigma_z,
            bond_memory,
            imbalance[None, :],
        ]
    )

    return names, diagonals


def validate_time_grid(times: Sequence[float]) -> np.ndarray:
    times_array = np.asarray(times, dtype=np.float64)

    if times_array.ndim != 1 or times_array.size < 2:
        raise ValueError("times must be a one-dimensional array of length >= 2.")

    if not np.all(np.isfinite(times_array)):
        raise ValueError("times contains non-finite values.")

    if np.any(np.diff(times_array) <= 0):
        raise ValueError("times must be strictly increasing.")

    spacings = np.diff(times_array)
    if not np.allclose(spacings, spacings[0], rtol=1e-12, atol=1e-12):
        raise ValueError("times must be uniformly spaced.")

    return times_array


def evolve_observables(
    hamiltonian: csr_matrix,
    psi0: np.ndarray,
    times: Sequence[float],
    diagonals: np.ndarray,
) -> np.ndarray:
    """
    Compute observable trajectories using exact Krylov propagation.

    Returns:
        values[time, channel]
    """
    times_array = validate_time_grid(times)

    if psi0.shape != (hamiltonian.shape[0],):
        raise ValueError("psi0 dimension does not match the Hamiltonian.")

    if diagonals.ndim != 2 or diagonals.shape[1] != hamiltonian.shape[0]:
        raise ValueError(
            "diagonals must have shape (n_channels, Hilbert_dimension)."
        )

    generator = (-1j) * hamiltonian
    trace_generator = (-1j) * np.sum(hamiltonian.diagonal())

    states = expm_multiply(
        generator,
        psi0,
        start=float(times_array[0]),
        stop=float(times_array[-1]),
        num=times_array.size,
        endpoint=True,
        traceA=trace_generator,
    )

    probabilities = np.abs(states) ** 2
    values = probabilities @ diagonals.T

    norm_error = np.max(np.abs(np.sum(probabilities, axis=1) - 1.0))
    if norm_error > 1e-9:
        raise RuntimeError(
            f"Time evolution lost normalization; maximum error = "
            f"{norm_error:.3e}."
        )

    return np.asarray(values.real, dtype=np.float64)


def _degenerate_blocks(
    eigenvalues: np.ndarray,
    tolerance: float,
) -> list[slice]:
    """Group numerically degenerate adjacent eigenvalues."""
    blocks: list[slice] = []
    start = 0

    for index in range(1, eigenvalues.size):
        scale = max(
            1.0,
            abs(eigenvalues[index - 1]),
            abs(eigenvalues[index]),
        )

        if abs(eigenvalues[index] - eigenvalues[index - 1]) > tolerance * scale:
            blocks.append(slice(start, index))
            start = index

    blocks.append(slice(start, eigenvalues.size))
    return blocks


def diagonal_ensemble_observables(
    hamiltonian: csr_matrix,
    psi0: np.ndarray,
    diagonals: np.ndarray,
    degeneracy_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exact infinite-time averages for diagonal observables.

    Coherences within exactly degenerate energy subspaces are retained.
    For continuous random fields, degeneracies are almost surely absent,
    but the implementation does not assume nondegeneracy.

    Returns:
        eigenvalues
        diagonal_ensemble_values[channel]
    """
    dense_hamiltonian = hamiltonian.toarray()

    eigenvalues, eigenvectors = eigh(
        dense_hamiltonian,
        overwrite_a=True,
        check_finite=False,
        driver="evr",
    )

    coefficients = eigenvectors.conj().T @ psi0
    infinite_time_probabilities = np.zeros(
        hamiltonian.shape[0],
        dtype=np.float64,
    )

    for block in _degenerate_blocks(
        eigenvalues,
        tolerance=degeneracy_tolerance,
    ):
        block_amplitude = (
            eigenvectors[:, block] @ coefficients[block]
        )
        infinite_time_probabilities += np.abs(block_amplitude) ** 2

    probability_error = abs(np.sum(infinite_time_probabilities) - 1.0)
    if probability_error > 1e-9:
        raise RuntimeError(
            f"Diagonal-ensemble probabilities are not normalized; "
            f"error = {probability_error:.3e}."
        )

    values = diagonals @ infinite_time_probabilities

    return (
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(values.real, dtype=np.float64),
    )


def generate_episode(
    L: int,
    W: float,
    seed: int,
    times: Sequence[float],
    J: float = 1.0,
    delta: float = 1.0,
    n_up: int | None = None,
) -> dict:
    """
    Generate one complete ED trajectory and exact diagonal-ensemble target.
    """
    model = HeisenbergModel(
        L=L,
        W=W,
        seed=seed,
        J=J,
        delta=delta,
        n_up=n_up,
    )

    times_array = validate_time_grid(times)
    fields = sample_disorder_fields(model.L, model.W, model.seed)
    basis = fixed_magnetization_basis(model.L, model.n_up)
    hamiltonian = build_hamiltonian(model, fields, basis)
    psi0 = neel_state(model.L, basis, first_spin_up=True)

    channel_names, diagonals = observable_diagonals(model.L, basis)

    trajectory = evolve_observables(
        hamiltonian=hamiltonian,
        psi0=psi0,
        times=times_array,
        diagonals=diagonals,
    )

    eigenvalues, diagonal_ensemble = diagonal_ensemble_observables(
        hamiltonian=hamiltonian,
        psi0=psi0,
        diagonals=diagonals,
    )

    center_left = model.L // 2 - 1
    center_bond_name = f"bond_memory_{center_left}_{center_left + 1}"
    center_bond_index = channel_names.index(center_bond_name)

    return {
        "format_version": 1,
        "model": "open_random_field_xxz",
        "L": model.L,
        "W": model.W,
        "seed": model.seed,
        "J": model.J,
        "delta": model.delta,
        "n_up": model.n_up,
        "hilbert_dimension": int(basis.size),
        "boundary_condition": "open",
        "hamiltonian_convention": (
            "H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} "
            "+ delta Sz_i Sz_{i+1}) + sum_i h_i Sz_i; "
            "S_alpha = sigma_alpha / 2"
        ),
        "basis_convention": (
            "site 0 is the most-significant bit; "
            "bit 1 is spin up with sigma_z=+1"
        ),
        "initial_state": "up_down_up_down",
        "disorder_distribution": "independent_uniform_minus_W_to_W",
        "fields": fields,
        "times": times_array,
        "channel_names": channel_names,
        "trajectory": trajectory,
        "diagonal_ensemble": diagonal_ensemble,
        "center_bond_channel": center_bond_name,
        "center_bond_trajectory": trajectory[:, center_bond_index],
        "center_bond_diagonal_ensemble": float(
            diagonal_ensemble[center_bond_index]
        ),
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
    }