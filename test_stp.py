#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_allclose

from canonical_stp import CanonicalSTP


def rank_one_training_data():
    latent = np.array([-2.0, -1.0, 1.0, 2.0])

    hindcast_mean = np.array(
        [
            [1.0, -0.5],
            [0.8, -0.2],
            [0.4, 0.1],
        ]
    )
    hindcast_mode = np.array(
        [
            [0.2, -0.1],
            [0.4, 0.3],
            [-0.2, 0.5],
        ]
    )

    forecast_mean = np.array(
        [
            [0.2],
            [0.1],
        ]
    )
    forecast_mode = np.array(
        [
            [0.7],
            [-0.4],
        ]
    )

    hindcasts = (
        hindcast_mean[None, :, :]
        + latent[:, None, None] * hindcast_mode[None, :, :]
    )
    forecasts = (
        forecast_mean[None, :, :]
        + latent[:, None, None] * forecast_mode[None, :, :]
    )

    return (
        latent,
        hindcasts,
        forecasts,
        hindcast_mean,
        hindcast_mode,
        forecast_mean,
        forecast_mode,
    )


def rank_two_training_data():
    latent = np.array(
        [
            [-2.0, -1.0],
            [-1.0, 2.0],
            [0.0, -2.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ]
    )

    hindcast_mean = np.array(
        [
            [1.0, 0.2],
            [0.8, 0.1],
            [0.5, -0.1],
            [0.3, -0.2],
        ]
    )

    hindcast_modes = np.array(
        [
            [
                [0.3, -0.1],
                [0.5, 0.2],
                [0.2, 0.4],
                [-0.1, 0.3],
            ],
            [
                [0.2, 0.4],
                [-0.3, 0.1],
                [0.6, -0.2],
                [0.4, 0.5],
            ],
        ]
    )

    forecast_mean = np.array(
        [
            [0.25],
            [0.15],
            [0.10],
        ]
    )

    forecast_modes = np.array(
        [
            [
                [0.7],
                [0.2],
                [-0.4],
            ],
            [
                [-0.3],
                [0.6],
                [0.5],
            ],
        ]
    )

    hindcasts = (
        hindcast_mean[None, :, :]
        + np.tensordot(latent, hindcast_modes, axes=(1, 0))
    )

    forecasts = (
        forecast_mean[None, :, :]
        + np.tensordot(latent, forecast_modes, axes=(1, 0))
    )

    return (
        latent,
        hindcasts,
        forecasts,
        hindcast_mean,
        hindcast_modes,
        forecast_mean,
        forecast_modes,
    )


def test_rank_one_exact_forecast():
    (
        _,
        hindcasts,
        forecasts,
        hindcast_mean,
        hindcast_mode,
        forecast_mean,
        forecast_mode,
    ) = rank_one_training_data()

    model = CanonicalSTP(rank=1).fit(
        hindcast_training=hindcasts,
        forecast_training=forecasts,
    )

    new_latent = 0.75
    observed = hindcast_mean + new_latent * hindcast_mode
    expected_forecast = forecast_mean + new_latent * forecast_mode

    result = model.predict(observed)

    assert_allclose(
        result["reconstructed_hindcast"],
        observed,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result["forecast"],
        expected_forecast,
        rtol=0.0,
        atol=1e-12,
    )

    assert result["coefficients"].shape == (1,)
    assert result["hindcast_rmse"] < 1e-12

    assert_allclose(
        model.hindcast_modes.conj().T @ model.hindcast_modes,
        np.eye(1),
        rtol=0.0,
        atol=1e-12,
    )


def test_time_dependent_ensemble_means_are_subtracted_and_restored():
    (
        _,
        hindcasts,
        forecasts,
        hindcast_mean,
        _,
        forecast_mean,
        _,
    ) = rank_one_training_data()

    model = CanonicalSTP(rank=1).fit(hindcasts, forecasts)

    assert_allclose(
        model.hindcast_mean,
        hindcast_mean,
        rtol=0.0,
        atol=1e-14,
    )
    assert_allclose(
        model.forecast_mean,
        forecast_mean,
        rtol=0.0,
        atol=1e-14,
    )

    result = model.predict(hindcast_mean)

    assert_allclose(
        result["coefficients"],
        np.zeros(1),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result["reconstructed_hindcast"],
        hindcast_mean,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result["forecast"],
        forecast_mean,
        rtol=0.0,
        atol=1e-12,
    )


def test_rank_two_predict_many_exactly():
    (
        _,
        hindcasts,
        forecasts,
        hindcast_mean,
        hindcast_modes,
        forecast_mean,
        forecast_modes,
    ) = rank_two_training_data()

    model = CanonicalSTP(rank=2).fit(hindcasts, forecasts)

    new_latent = np.array(
        [
            [0.5, -0.75],
            [-1.25, 0.4],
            [1.1, 1.3],
        ]
    )

    observed = (
        hindcast_mean[None, :, :]
        + np.tensordot(new_latent, hindcast_modes, axes=(1, 0))
    )
    expected_forecasts = (
        forecast_mean[None, :, :]
        + np.tensordot(new_latent, forecast_modes, axes=(1, 0))
    )

    result = model.predict_many(observed)

    assert result["coefficients"].shape == (3, 2)
    assert result["hindcast_rmse"].shape == (3,)

    assert_allclose(
        result["reconstructed_hindcast"],
        observed,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result["forecast"],
        expected_forecasts,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result["hindcast_rmse"],
        np.zeros(3),
        rtol=0.0,
        atol=1e-12,
    )

    assert_allclose(
        model.hindcast_modes.conj().T @ model.hindcast_modes,
        np.eye(2),
        rtol=0.0,
        atol=1e-12,
    )

    assert model.numerical_rank == 2
    assert_allclose(
        model.cumulative_explained_energy[-1],
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_forecast_variables_may_differ_from_hindcast_variables():
    (
        latent,
        hindcasts,
        _,
        hindcast_mean,
        hindcast_modes,
        _,
        _,
    ) = rank_two_training_data()

    scalar_targets = (
        0.4
        + 2.0 * latent[:, 0]
        - 0.5 * latent[:, 1]
    )[:, None]

    model = CanonicalSTP(rank=2).fit(
        hindcast_training=hindcasts,
        forecast_training=scalar_targets,
    )

    new_latent = np.array([0.8, -1.2])
    observed = (
        hindcast_mean
        + np.tensordot(new_latent, hindcast_modes, axes=(0, 0))
    )
    expected_target = np.array(
        [0.4 + 2.0 * new_latent[0] - 0.5 * new_latent[1]]
    )

    result = model.predict(observed)

    assert result["forecast"].shape == (1,)
    assert_allclose(
        result["forecast"],
        expected_target,
        rtol=0.0,
        atol=1e-12,
    )


def test_training_episodes_are_reconstructed_at_full_numerical_rank():
    (
        _,
        hindcasts,
        forecasts,
        _,
        _,
        _,
        _,
    ) = rank_two_training_data()

    model = CanonicalSTP(rank=2).fit(hindcasts, forecasts)
    result = model.predict_many(hindcasts)

    assert_allclose(
        result["reconstructed_hindcast"],
        hindcasts,
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        result["forecast"],
        forecasts,
        rtol=0.0,
        atol=1e-12,
    )


def test_trajectory_wrapper_reconstructs_full_episode():
    latent = np.array(
        [
            [-2.0, -1.0],
            [-1.0, 2.0],
            [0.0, -2.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ]
    )

    trajectory_mean = np.array(
        [
            [1.0, -1.0],
            [0.8, -0.7],
            [0.6, -0.4],
            [0.5, -0.2],
            [0.4, -0.1],
            [0.3, 0.0],
        ]
    )

    trajectory_modes = np.array(
        [
            [
                [0.3, -0.2],
                [0.5, 0.1],
                [0.2, 0.4],
                [0.1, 0.6],
                [-0.2, 0.5],
                [-0.4, 0.3],
            ],
            [
                [0.4, 0.2],
                [-0.3, 0.5],
                [0.6, -0.1],
                [0.5, -0.4],
                [0.2, -0.6],
                [-0.1, -0.3],
            ],
        ]
    )

    training_episodes = (
        trajectory_mean[None, :, :]
        + np.tensordot(latent, trajectory_modes, axes=(1, 0))
    )

    split_index = 3
    model = CanonicalSTP(rank=2).fit_trajectory_episodes(
        training_episodes=training_episodes,
        split_index=split_index,
    )

    new_latent = np.array([0.65, -0.35])
    expected_episode = (
        trajectory_mean
        + np.tensordot(
            new_latent,
            trajectory_modes,
            axes=(0, 0),
        )
    )

    result = model.predict_trajectory(
        expected_episode[:split_index]
    )

    assert model.split_index == split_index
    assert model.total_time_points == 6
    assert model.n_channels == 2

    assert result["reconstructed_hindcast"].shape == (3, 2)
    assert result["forecast"].shape == (3, 2)
    assert result["full_trajectory"].shape == (6, 2)

    assert_allclose(
        result["full_trajectory"],
        expected_episode,
        rtol=0.0,
        atol=1e-12,
    )


def test_rank_is_limited_by_hindcast_numerical_rank():
    (
        _,
        hindcasts,
        forecasts,
        _,
        _,
        _,
        _,
    ) = rank_one_training_data()

    with pytest.raises(ValueError, match="numerical hindcast rank"):
        CanonicalSTP(rank=2).fit(hindcasts, forecasts)


def test_zero_variance_hindcast_is_rejected():
    hindcasts = np.ones((4, 3, 2))
    forecasts = np.arange(8, dtype=float).reshape(4, 2, 1)

    with pytest.raises(ValueError, match="zero rank"):
        CanonicalSTP(rank=1).fit(hindcasts, forecasts)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        CanonicalSTP(rank=0)

    model = CanonicalSTP(rank=1)

    with pytest.raises(RuntimeError):
        model.predict(np.zeros((3, 2)))

    hindcasts = np.zeros((4, 3, 2))
    forecasts = np.zeros((5, 2, 1))

    with pytest.raises(ValueError, match="same number of episodes"):
        model.fit(hindcasts, forecasts)


def test_prediction_shape_is_checked():
    (
        _,
        hindcasts,
        forecasts,
        _,
        _,
        _,
        _,
    ) = rank_one_training_data()

    model = CanonicalSTP(rank=1).fit(hindcasts, forecasts)

    with pytest.raises(ValueError, match="hindcast must have shape"):
        model.predict(np.zeros((3, 1)))

    with pytest.raises(ValueError):
        model.predict_many(np.zeros((2, 3, 1)))