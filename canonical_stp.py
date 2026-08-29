#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Canonical Space-Time Projection (STP).

This implementation follows the hindcast-derived extended-POD construction:

1. Subtract the training-ensemble mean.
2. Compute the SVD of the hindcast data only.
3. Construct hindcast modes from the hindcast SVD.
4. Extend those same modes into the forecast variables using the
   hindcast expansion coefficients.
5. Project a new mean-subtracted hindcast onto the orthonormal
   hindcast modes.
6. Reconstruct both the hindcast and forecast, then add the
   training means back.

Uniform Euclidean weighting is used.

Array convention:
    First axis = independent episodes / disorder realizations.
    Remaining axes = arbitrary physical, temporal, or target dimensions.
"""

from __future__ import annotations

import numpy as np


class CanonicalSTP:
    def __init__(self, rank: int):
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer.")

        self.rank = rank
        self.is_fitted = False

    @staticmethod
    def _validate_training_array(
        array: np.ndarray,
        name: str,
    ) -> np.ndarray:
        values = np.asarray(array)

        if values.ndim < 2:
            raise ValueError(
                f"{name} must have shape "
                "(n_episodes, feature_dimensions...)."
            )

        if values.shape[0] < 2:
            raise ValueError(
                f"{name} must contain at least two episodes."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

        return values

    def fit(
        self,
        hindcast_training: np.ndarray,
        forecast_training: np.ndarray,
    ) -> "CanonicalSTP":
        """
        Fit canonical STP.

        Parameters
        ----------
        hindcast_training
            Shape:
                (n_episodes, hindcast_dimensions...)

        forecast_training
            Shape:
                (n_episodes, forecast_dimensions...)

            The forecast variables may differ from the hindcast variables.
            For example, the hindcast may contain short-time multichannel
            dynamics while the forecast target may be a scalar
            diagonal-ensemble observable.
        """
        hindcast = self._validate_training_array(
            hindcast_training,
            "hindcast_training",
        )
        forecast = self._validate_training_array(
            forecast_training,
            "forecast_training",
        )

        if hindcast.shape[0] != forecast.shape[0]:
            raise ValueError(
                "hindcast_training and forecast_training must contain "
                "the same number of episodes."
            )

        self.n_episodes = hindcast.shape[0]
        self.hindcast_shape = hindcast.shape[1:]
        self.forecast_shape = forecast.shape[1:]

        self.hindcast_mean = np.mean(hindcast, axis=0)
        self.forecast_mean = np.mean(forecast, axis=0)

        hindcast_centered = hindcast - self.hindcast_mean
        forecast_centered = forecast - self.forecast_mean

        # Columns are independent episodes.
        Q_hindcast = hindcast_centered.reshape(
            self.n_episodes,
            -1,
        ).T

        Q_forecast = forecast_centered.reshape(
            self.n_episodes,
            -1,
        ).T

        # Hindcast-only SVD:
        #
        # Q_hindcast = U S V^H
        #
        # U gives the orthonormal hindcast modes.
        U, singular_values, Vh = np.linalg.svd(
            Q_hindcast,
            full_matrices=False,
        )

        if singular_values.size == 0 or singular_values[0] == 0:
            raise ValueError(
                "The mean-subtracted hindcast training data have zero rank."
            )

        tolerance = (
            np.finfo(float).eps
            * max(Q_hindcast.shape)
            * singular_values[0]
        )
        numerical_rank = int(
            np.count_nonzero(singular_values > tolerance)
        )

        if self.rank > numerical_rank:
            raise ValueError(
                f"Requested rank {self.rank} exceeds the numerical "
                f"hindcast rank {numerical_rank}."
            )

        retained_singular_values = singular_values[: self.rank]
        retained_V = Vh.conj().T[:, : self.rank]

        # Canonical hindcast modes.
        self.hindcast_modes = U[:, : self.rank]

        # Extended forecast modes:
        #
        # Phi_forecast = Q_forecast V S^{-1}
        self.forecast_modes = (
            Q_forecast
            @ retained_V
            / retained_singular_values[None, :]
        )

        self.singular_values = singular_values
        self.numerical_rank = numerical_rank

        self.mode_energies = (
            retained_singular_values**2 / self.n_episodes
        )

        total_energy = np.sum(singular_values**2)
        self.explained_energy_fraction = (
            retained_singular_values**2 / total_energy
        )
        self.cumulative_explained_energy = np.cumsum(
            self.explained_energy_fraction
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        hindcast: np.ndarray,
    ) -> dict:
        """
        Forecast one new episode from its hindcast data.
        """
        if not self.is_fitted:
            raise RuntimeError("The STP model has not been fitted.")

        observed = np.asarray(hindcast)

        if observed.shape != self.hindcast_shape:
            raise ValueError(
                f"hindcast must have shape {self.hindcast_shape}, "
                f"received {observed.shape}."
            )

        if not np.all(np.isfinite(observed)):
            raise ValueError("hindcast contains non-finite values.")

        centered = observed - self.hindcast_mean
        centered_vector = centered.reshape(-1)

        # Orthogonal projection onto the hindcast modes.
        coefficients = (
            self.hindcast_modes.conj().T @ centered_vector
        )

        reconstructed_hindcast_vector = (
            self.hindcast_modes @ coefficients
        )

        forecast_vector = self.forecast_modes @ coefficients

        reconstructed_hindcast = (
            reconstructed_hindcast_vector.reshape(
                self.hindcast_shape
            )
            + self.hindcast_mean
        )

        forecast = (
            forecast_vector.reshape(self.forecast_shape)
            + self.forecast_mean
        )

        hindcast_rmse = float(
            np.sqrt(
                np.mean(
                    np.abs(
                        reconstructed_hindcast - observed
                    ) ** 2
                )
            )
        )

        return {
            "reconstructed_hindcast": reconstructed_hindcast,
            "forecast": forecast,
            "coefficients": coefficients,
            "hindcast_rmse": hindcast_rmse,
        }

    def predict_many(
        self,
        hindcasts: np.ndarray,
    ) -> dict:
        """
        Forecast multiple independent episodes.

        Parameters
        ----------
        hindcasts
            Shape:
                (n_episodes, *hindcast_shape)
        """
        if not self.is_fitted:
            raise RuntimeError("The STP model has not been fitted.")

        observed = np.asarray(hindcasts)

        expected_shape = (observed.shape[0],) + self.hindcast_shape

        if observed.ndim != len(self.hindcast_shape) + 1:
            raise ValueError(
                "hindcasts must include an episode axis followed by "
                f"the hindcast shape {self.hindcast_shape}."
            )

        if observed.shape != expected_shape:
            raise ValueError(
                f"hindcasts must have shape "
                f"(n_episodes, {self.hindcast_shape}), "
                f"received {observed.shape}."
            )

        if not np.all(np.isfinite(observed)):
            raise ValueError("hindcasts contains non-finite values.")

        n_predictions = observed.shape[0]

        centered_matrix = (
            observed - self.hindcast_mean
        ).reshape(n_predictions, -1).T

        coefficients = (
            self.hindcast_modes.conj().T @ centered_matrix
        )

        reconstructed_hindcast = (
            self.hindcast_modes @ coefficients
        ).T.reshape(
            (n_predictions,) + self.hindcast_shape
        )

        forecasts = (
            self.forecast_modes @ coefficients
        ).T.reshape(
            (n_predictions,) + self.forecast_shape
        )

        reconstructed_hindcast = (
            reconstructed_hindcast + self.hindcast_mean
        )
        forecasts = forecasts + self.forecast_mean

        reduction_axes = tuple(
            range(1, reconstructed_hindcast.ndim)
        )

        hindcast_rmse = np.sqrt(
            np.mean(
                np.abs(
                    reconstructed_hindcast - observed
                ) ** 2,
                axis=reduction_axes,
            )
        )

        return {
            "reconstructed_hindcast": reconstructed_hindcast,
            "forecast": forecasts,
            "coefficients": coefficients.T,
            "hindcast_rmse": hindcast_rmse,
        }

    def fit_trajectory_episodes(
        self,
        training_episodes: np.ndarray,
        split_index: int,
    ) -> "CanonicalSTP":
        """
        Convenience wrapper for forecasting the continuation of trajectories.

        training_episodes shape:
            (n_episodes, n_times, n_channels)

        split_index:
            Number of time points included in the hindcast.
        """
        episodes = self._validate_training_array(
            training_episodes,
            "training_episodes",
        )

        if episodes.ndim != 3:
            raise ValueError(
                "training_episodes must have shape "
                "(n_episodes, n_times, n_channels)."
            )

        n_times = episodes.shape[1]

        if not isinstance(split_index, int):
            raise ValueError("split_index must be an integer.")

        if not 1 <= split_index < n_times:
            raise ValueError(
                "split_index must satisfy "
                "1 <= split_index < n_times."
            )

        self.split_index = split_index
        self.total_time_points = n_times
        self.n_channels = episodes.shape[2]

        return self.fit(
            hindcast_training=episodes[:, :split_index, :],
            forecast_training=episodes[:, split_index:, :],
        )

    def predict_trajectory(
        self,
        observed_hindcast: np.ndarray,
    ) -> dict:
        """
        Forecast a trajectory continuation after fit_trajectory_episodes().
        """
        result = self.predict(observed_hindcast)

        full_trajectory = np.concatenate(
            [
                result["reconstructed_hindcast"],
                result["forecast"],
            ],
            axis=0,
        )

        result["full_trajectory"] = full_trajectory
        return result