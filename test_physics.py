#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose

from heisenberg_ed import (
    HeisenbergModel,
    build_hamiltonian,
    diagonal_ensemble_observables,
    evolve_observables,
    fixed_magnetization_basis,
    generate_episode,
    neel_state,
    observable_diagonals,
    sample_disorder_fields,
    validate_time_grid,
)


def test_fixed_magnetization_basis_dimension_and_order():
    basis = fixed_magnetization_basis(L=4, n_up=2)

    assert basis.dtype == np.int64
    assert basis.size == 6
    assert np.all(np.diff(basis) > 0)
    assert all(bin(int(state)).count("1") == 2 for state in basis)


def test_disorder_fields_are_reproducible_and_bounded():
    fields_a = sample_disorder_fields(L=8, W=3.0, seed=1234)
    fields_b = sample_disorder_fields(L=8, W=3.0, seed=1234)
    fields_c = sample_disorder_fields(L=8, W=3.0, seed=1235)

    assert_allclose(fields_a, fields_b, rtol=0.0, atol=0.0)
    assert not np.array_equal(fields_a, fields_c)
    assert np.all(fields_a >= -3.0)
    assert np.all(fields_a <= 3.0)


def test_two_site_hamiltonian_has_correct_spin_half_factors():
    model = HeisenbergModel(
        L=2,
        W=0.0,
        seed=0,
        J=1.0,
        delta=1.0,
        n_up=1,
    )

    basis = fixed_magnetization_basis(L=2, n_up=1)
    fields = np.array([0.6, -0.2], dtype=np.float64)

    hamiltonian = build_hamiltonian(
        model=model,
        fields=fields,
        basis=basis,
    ).toarray()

    expected = np.array(
        [
            [
                -0.25 - 0.5 * fields[0] + 0.5 * fields[1],
                0.5,
            ],
            [
                0.5,
                -0.25 + 0.5 * fields[0] - 0.5 * fields[1],
            ],
        ],
        dtype=np.float64,
    )

    assert_allclose(hamiltonian, expected, rtol=0.0, atol=1e-14)
    assert_allclose(hamiltonian, hamiltonian.T, rtol=0.0, atol=1e-14)


def test_two_site_heisenberg_spectrum():
    model = HeisenbergModel(
        L=2,
        W=0.0,
        seed=0,
        J=1.0,
        delta=1.0,
        n_up=1,
    )

    basis = fixed_magnetization_basis(L=2, n_up=1)
    fields = np.zeros(2, dtype=np.float64)

    eigenvalues = np.linalg.eigvalsh(
        build_hamiltonian(model, fields, basis).toarray()
    )

    assert_allclose(
        eigenvalues,
        np.array([-0.75, 0.25]),
        rtol=0.0,
        atol=1e-14,
    )


def test_neel_state_and_observable_conventions():
    L = 4
    basis = fixed_magnetization_basis(L=L, n_up=L // 2)
    psi0 = neel_state(L=L, basis=basis, first_spin_up=True)

    expected_state_integer = int("1010", 2)
    expected_index = int(np.flatnonzero(basis == expected_state_integer)[0])

    assert_allclose(np.linalg.norm(psi0), 1.0, rtol=0.0, atol=1e-14)
    assert psi0[expected_index] == 1.0
    assert np.count_nonzero(psi0) == 1

    names, diagonals = observable_diagonals(L=L, basis=basis)
    initial_values = diagonals @ np.abs(psi0) ** 2

    expected_sigma_z = np.array([1.0, -1.0, 1.0, -1.0])
    expected_bond_memory = np.ones(L - 1)

    assert_allclose(initial_values[:L], expected_sigma_z)
    assert_allclose(initial_values[L : L + L - 1], expected_bond_memory)
    assert_allclose(initial_values[-1], 1.0)

    assert names[L] == "bond_memory_0_1"
    assert names[-1] == "imbalance"


def test_two_site_exact_dynamics():
    model = HeisenbergModel(
        L=2,
        W=0.0,
        seed=0,
        J=1.0,
        delta=1.0,
        n_up=1,
    )

    basis = fixed_magnetization_basis(L=2, n_up=1)
    fields = np.zeros(2, dtype=np.float64)
    hamiltonian = build_hamiltonian(model, fields, basis)
    psi0 = neel_state(L=2, basis=basis, first_spin_up=True)

    names, diagonals = observable_diagonals(L=2, basis=basis)
    times = np.array([0.0, 0.5 * np.pi, np.pi])

    values = evolve_observables(
        hamiltonian=hamiltonian,
        psi0=psi0,
        times=times,
        diagonals=diagonals,
    )

    sigma_z_0 = values[:, names.index("sigma_z_0")]
    sigma_z_1 = values[:, names.index("sigma_z_1")]
    bond_memory = values[:, names.index("bond_memory_0_1")]
    imbalance = values[:, names.index("imbalance")]

    expected = np.cos(times)

    assert_allclose(sigma_z_0, expected, rtol=0.0, atol=1e-12)
    assert_allclose(sigma_z_1, -expected, rtol=0.0, atol=1e-12)
    assert_allclose(bond_memory, 1.0, rtol=0.0, atol=1e-12)
    assert_allclose(imbalance, expected, rtol=0.0, atol=1e-12)


def test_two_site_diagonal_ensemble():
    model = HeisenbergModel(
        L=2,
        W=0.0,
        seed=0,
        J=1.0,
        delta=1.0,
        n_up=1,
    )

    basis = fixed_magnetization_basis(L=2, n_up=1)
    fields = np.zeros(2, dtype=np.float64)
    hamiltonian = build_hamiltonian(model, fields, basis)
    psi0 = neel_state(L=2, basis=basis, first_spin_up=True)

    names, diagonals = observable_diagonals(L=2, basis=basis)

    _, values = diagonal_ensemble_observables(
        hamiltonian=hamiltonian,
        psi0=psi0,
        diagonals=diagonals,
    )

    assert_allclose(
        values[names.index("sigma_z_0")],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        values[names.index("sigma_z_1")],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        values[names.index("bond_memory_0_1")],
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        values[names.index("imbalance")],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_diagonal_ensemble_retains_degenerate_coherences():
    model = HeisenbergModel(
        L=4,
        W=0.0,
        seed=0,
        J=0.0,
        delta=1.0,
        n_up=2,
    )

    basis = fixed_magnetization_basis(L=4, n_up=2)
    fields = np.zeros(4, dtype=np.float64)
    hamiltonian = build_hamiltonian(model, fields, basis)
    psi0 = neel_state(L=4, basis=basis, first_spin_up=True)

    _, diagonals = observable_diagonals(L=4, basis=basis)
    initial_values = diagonals @ np.abs(psi0) ** 2

    _, infinite_time_values = diagonal_ensemble_observables(
        hamiltonian=hamiltonian,
        psi0=psi0,
        diagonals=diagonals,
    )

    assert_allclose(
        infinite_time_values,
        initial_values,
        rtol=0.0,
        atol=1e-12,
    )


def test_generate_episode_metadata_and_center_bond():
    times = np.array([0.0, 0.1, 0.2])

    episode = generate_episode(
        L=4,
        W=2.0,
        seed=17,
        times=times,
    )

    assert episode["model"] == "open_random_field_xxz"
    assert episode["boundary_condition"] == "open"
    assert episode["L"] == 4
    assert episode["W"] == 2.0
    assert episode["seed"] == 17
    assert episode["n_up"] == 2
    assert episode["hilbert_dimension"] == 6

    assert episode["fields"].shape == (4,)
    assert episode["trajectory"].shape == (3, 8)
    assert episode["diagonal_ensemble"].shape == (8,)

    assert episode["center_bond_channel"] == "bond_memory_1_2"
    assert episode["center_bond_trajectory"].shape == (3,)
    assert_allclose(
        episode["center_bond_trajectory"][0],
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    center_index = episode["channel_names"].index(
        episode["center_bond_channel"]
    )

    assert_allclose(
        episode["center_bond_trajectory"],
        episode["trajectory"][:, center_index],
    )
    assert_allclose(
        episode["center_bond_diagonal_ensemble"],
        episode["diagonal_ensemble"][center_index],
    )


def test_time_grid_validation():
    valid = np.array([0.0, 0.2, 0.4, 0.6])
    assert_allclose(validate_time_grid(valid), valid)

    for invalid in (
        np.array([0.0]),
        np.array([0.0, 0.2, 0.1]),
        np.array([0.0, 0.2, 0.5]),
        np.array([0.0, np.nan]),
    ):
        try:
            validate_time_grid(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"validate_time_grid accepted invalid grid: {invalid}"
            )