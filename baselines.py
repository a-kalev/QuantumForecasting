#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple forecasting baselines.

Array convention:
    hindcasts shape:
        (n_episodes, n_times, n_channels)

    targets shape:
        (n_episodes, target_dimensions...)

Available baselines:
    1. Training-target mean.
    2. Last observed value.
    3. Mean over the full observed interval.
    4. Mean over the final observed time window.
    5. Linear trend extrapolation over a chosen late-time window.
    6. Regularized linear regression from the complete flattened
       hindcast to an arbitrary forecast target.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate_hindcasts(hindcasts: np.ndarray) -> np.ndarray:
    values = np.asarray(hindcasts, dtype=np.float64)

    if values.ndim != 3:
        raise ValueError(
            "hindcasts must have shape "
            "(n_episodes, n_times, n_channels)."
        )

    if values.shape[0] < 1:
        raise ValueError("hindcasts must contain at least one episode.")

    if values.shape[1] < 1:
        raise ValueError("hindcasts must contain at least one time point.")

    if values.shape[2] < 1:
        raise ValueError("hindcasts must contain at least one channel.")

    if not np.all(np.isfinite(values)):
        raise ValueError("hindcasts contain non-finite values.")

    return values


def _validate_channel(channel_index: int, n_channels: int) -> None:
    if not isinstance(channel_index, int):
        raise ValueError("channel_index must be an integer.")

    if not 0 <= channel_index < n_channels:
        raise ValueError(
            f"channel_index must satisfy 0 <= channel_index < {n_channels}."
        )


def last_observed_value(
    hindcasts: np.ndarray,
    channel_index: int,
) -> np.ndarray:
    """Return the final observed value of one channel."""
    values = _validate_hindcasts(hindcasts)
    _validate_channel(channel_index, values.shape[2])

    return values[:, -1, channel_index].copy()


def observed_interval_mean(
    hindcasts: np.ndarray,
    channel_index: int,
) -> np.ndarray:
    """Return the mean over the complete observed interval."""
    values = _validate_hindcasts(hindcasts)
    _validate_channel(channel_index, values.shape[2])

    return np.mean(values[:, :, channel_index], axis=1)


def late_window_mean(
    hindcasts: np.ndarray,
    channel_index: int,
    window_points: int,
) -> np.ndarray:
    """Return the mean over the final window_points observations."""
    values = _validate_hindcasts(hindcasts)
    _validate_channel(channel_index, values.shape[2])

    if not isinstance(window_points, int) or window_points < 1:
        raise ValueError("window_points must be a positive integer.")

    if window_points > values.shape[1]:
        raise ValueError(
            "window_points cannot exceed the number of observed time points."
        )

    return np.mean(
        values[:, -window_points:, channel_index],
        axis=1,
    )


def linear_trend_forecast(
    hindcasts: np.ndarray,
    observed_times: Sequence[float],
    forecast_times: Sequence[float],
    channel_index: int,
    fit_window_points: int,
) -> np.ndarray:
    """
    Fit a least-squares line to the final observed window and evaluate it
    at the requested forecast times.

    Returns:
        forecast shape (n_episodes, n_forecast_times)
    """
    values = _validate_hindcasts(hindcasts)
    _validate_channel(channel_index, values.shape[2])

    observed_times_array = np.asarray(
        observed_times,
        dtype=np.float64,
    )
    forecast_times_array = np.asarray(
        forecast_times,
        dtype=np.float64,
    )

    if observed_times_array.ndim != 1:
        raise ValueError("observed_times must be one-dimensional.")

    if forecast_times_array.ndim != 1:
        raise ValueError("forecast_times must be one-dimensional.")

    if observed_times_array.size != values.shape[1]:
        raise ValueError(
            "observed_times length must match the hindcast time dimension."
        )

    if forecast_times_array.size < 1:
        raise ValueError("forecast_times must contain at least one value.")

    if not np.all(np.isfinite(observed_times_array)):
        raise ValueError("observed_times contain non-finite values.")

    if not np.all(np.isfinite(forecast_times_array)):
        raise ValueError("forecast_times contain non-finite values.")

    if np.any(np.diff(observed_times_array) <= 0):
        raise ValueError("observed_times must be strictly increasing.")

    if not isinstance(fit_window_points, int) or fit_window_points < 2:
        raise ValueError("fit_window_points must be an integer >= 2.")

    if fit_window_points > observed_times_array.size:
        raise ValueError(
            "fit_window_points cannot exceed the observed time-grid length."
        )

    fit_times = observed_times_array[-fit_window_points:]
    centered_times = fit_times - np.mean(fit_times)

    denominator = np.sum(centered_times**2)
    if denominator <= 0:
        raise ValueError("The selected fitting times have zero variance.")

    observed_values = values[
        :,
        -fit_window_points:,
        channel_index,
    ]

    value_means = np.mean(observed_values, axis=1)
    slopes = (
        observed_values - value_means[:, None]
    ) @ centered_times / denominator

    intercept_times = forecast_times_array - np.mean(fit_times)

    return (
        value_means[:, None]
        + slopes[:, None] * intercept_times[None, :]
    )


class TrainingMeanBaseline:
    """Predict the mean target observed in the training set."""

    def __init__(self) -> None:
        self.is_fitted = False

    def fit(self, targets: np.ndarray) -> "TrainingMeanBaseline":
        values = np.asarray(targets, dtype=np.float64)

        if values.ndim < 1:
            raise ValueError("targets must include an episode axis.")

        if values.shape[0] < 1:
            raise ValueError("targets must contain at least one episode.")

        if not np.all(np.isfinite(values)):
            raise ValueError("targets contain non-finite values.")

        self.target_shape = values.shape[1:]
        self.target_mean = np.mean(values, axis=0)
        self.is_fitted = True
        return self

    def predict(self, n_episodes: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The baseline has not been fitted.")

        if not isinstance(n_episodes, int) or n_episodes < 1:
            raise ValueError("n_episodes must be a positive integer.")

        return np.broadcast_to(
            self.target_mean,
            (n_episodes,) + self.target_shape,
        ).copy()


class RidgeRegressionBaseline:
    """
    Regularized linear map from a flattened hindcast to an arbitrary target.

    Feature centering and scaling are learned from the training set only.
    The regression is solved in the episode-space dual form, which is
    efficient when the number of hindcast features exceeds the number of
    training episodes.
    """

    def __init__(
        self,
        alpha: float,
        standardize: bool = True,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")

        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.is_fitted = False

    def fit(
        self,
        hindcast_training: np.ndarray,
        target_training: np.ndarray,
    ) -> "RidgeRegressionBaseline":
        hindcasts = _validate_hindcasts(hindcast_training)
        targets = np.asarray(target_training, dtype=np.float64)

        if targets.ndim < 1:
            raise ValueError(
                "target_training must include an episode axis."
            )

        if targets.shape[0] != hindcasts.shape[0]:
            raise ValueError(
                "hindcast_training and target_training must contain "
                "the same number of episodes."
            )

        if not np.all(np.isfinite(targets)):
            raise ValueError("target_training contains non-finite values.")

        self.hindcast_shape = hindcasts.shape[1:]
        self.target_shape = targets.shape[1:]

        n_episodes = hindcasts.shape[0]

        X = hindcasts.reshape(n_episodes, -1)
        Y = targets.reshape(n_episodes, -1)

        self.feature_mean = np.mean(X, axis=0)
        X_centered = X - self.feature_mean

        if self.standardize:
            feature_scale = np.std(
                X_centered,
                axis=0,
                ddof=0,
            )
            feature_scale[feature_scale < 1e-12] = 1.0
        else:
            feature_scale = np.ones(X.shape[1], dtype=np.float64)

        self.feature_scale = feature_scale
        X_scaled = X_centered / self.feature_scale

        self.target_mean = np.mean(Y, axis=0)
        Y_centered = Y - self.target_mean

        gram = X_scaled @ X_scaled.T
        regularized_gram = (
            gram + self.alpha * np.eye(n_episodes)
        )

        try:
            dual_coefficients = np.linalg.solve(
                regularized_gram,
                Y_centered,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Ridge-regression system could not be solved."
            ) from exc

        self.weights = X_scaled.T @ dual_coefficients
        self.is_fitted = True
        return self

    def predict(self, hindcasts: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The baseline has not been fitted.")

        values = _validate_hindcasts(hindcasts)

        if values.shape[1:] != self.hindcast_shape:
            raise ValueError(
                f"hindcasts must have trailing shape "
                f"{self.hindcast_shape}, received {values.shape[1:]}."
            )

        n_episodes = values.shape[0]
        X = values.reshape(n_episodes, -1)

        X_scaled = (
            X - self.feature_mean
        ) / self.feature_scale

        prediction = (
            X_scaled @ self.weights
            + self.target_mean
        )

        return prediction.reshape(
            (n_episodes,) + self.target_shape
        )