"""Calibrated class-aware heterogeneous ensemble.

The ensemble operates in log-probability space:

    z_c = sum_m softmax(A[:, c])_m * log p_m(c) + b_c
    p(y=c) = softmax(z)_c

Each class therefore learns a separate convex combination of experts.  The
regularizer keeps those combinations close to a shared (class-agnostic)
mixture, which is important when minority validation classes are small.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


EPS = 1e-12


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling when only probabilities are saved."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _softmax(np.log(np.clip(probs, EPS, 1.0)) / temperature, axis=-1)


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    bounds: Tuple[float, float] = (0.25, 5.0),
) -> float:
    """Fit one positive temperature by validation negative log likelihood."""
    labels = np.asarray(labels, dtype=np.int64)

    def objective(log_t: np.ndarray) -> float:
        calibrated = temperature_scale_probs(probs, float(np.exp(log_t[0])))
        return float(
            -np.log(np.clip(calibrated[np.arange(len(labels)), labels], EPS, 1.0)).mean()
        )

    result = minimize(
        objective,
        x0=np.zeros(1),
        method="L-BFGS-B",
        bounds=[(np.log(bounds[0]), np.log(bounds[1]))],
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x[0]))


def validate_expert_probs(expert_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(expert_probs, dtype=np.float64)
    if probs.ndim != 3:
        raise ValueError("expert_probs must have shape [samples, experts, classes]")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("expert_probs contains invalid values")
    row_sums = probs.sum(axis=-1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("At least one expert probability row sums to zero")
    return probs / row_sums


@dataclass
class ClassAwareEnsemble:
    temperatures: np.ndarray
    class_weights: np.ndarray
    class_bias: np.ndarray
    regularization: float = 0.0

    def transform_experts(self, expert_probs: np.ndarray) -> np.ndarray:
        probs = validate_expert_probs(expert_probs)
        if probs.shape[1] != len(self.temperatures):
            raise ValueError("Number of experts does not match fitted temperatures")
        return np.stack(
            [
                temperature_scale_probs(probs[:, m], float(self.temperatures[m]))
                for m in range(probs.shape[1])
            ],
            axis=1,
        )

    def predict_proba(self, expert_probs: np.ndarray) -> np.ndarray:
        probs = self.transform_experts(expert_probs)
        if self.class_weights.shape != probs.shape[1:]:
            raise ValueError(
                f"Expected weights shape {probs.shape[1:]}, got {self.class_weights.shape}"
            )
        scores = (
            self.class_weights[None, :, :]
            * np.log(np.clip(probs, EPS, 1.0))
        ).sum(axis=1)
        scores += self.class_bias[None, :]
        return _softmax(scores, axis=1)

    def to_dict(self) -> dict:
        return {
            "temperatures": self.temperatures.tolist(),
            "class_weights": self.class_weights.tolist(),
            "class_bias": self.class_bias.tolist(),
            "regularization": float(self.regularization),
            "formula": "softmax(sum_m w[m,c] * log(calibrated_p[m,c]) + bias[c])",
        }


def fit_class_aware_ensemble(
    expert_probs: np.ndarray,
    labels: np.ndarray,
    temperatures: Optional[np.ndarray] = None,
    regularization: float = 0.1,
    class_balanced: bool = True,
    maxiter: int = 1000,
) -> ClassAwareEnsemble:
    """Fit non-negative, per-class expert weights and class biases."""
    probs = validate_expert_probs(expert_probs)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples, n_experts, n_classes = probs.shape
    if labels.shape != (n_samples,):
        raise ValueError("labels shape does not match expert_probs")
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("labels are outside the probability class range")

    if temperatures is None:
        temperatures = np.array(
            [fit_temperature(probs[:, m], labels) for m in range(n_experts)]
        )
    temperatures = np.asarray(temperatures, dtype=np.float64)
    calibrated = np.stack(
        [
            temperature_scale_probs(probs[:, m], temperatures[m])
            for m in range(n_experts)
        ],
        axis=1,
    )
    log_probs = np.log(np.clip(calibrated, EPS, 1.0))

    sample_weights = np.ones(n_samples, dtype=np.float64)
    if class_balanced:
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        class_weights = n_samples / (n_classes * np.maximum(counts, 1.0))
        sample_weights = class_weights[labels]
        sample_weights /= sample_weights.mean()

    def unpack(theta: np.ndarray):
        raw = theta[: n_experts * n_classes].reshape(n_experts, n_classes)
        weights = _softmax(raw, axis=0)
        bias = theta[n_experts * n_classes :]
        bias = bias - bias.mean()  # identifiable parameterization
        return weights, bias

    def objective(theta: np.ndarray) -> float:
        weights, bias = unpack(theta)
        scores = (weights[None, :, :] * log_probs).sum(axis=1) + bias[None, :]
        final_probs = _softmax(scores, axis=1)
        nll = -np.log(
            np.clip(final_probs[np.arange(n_samples), labels], EPS, 1.0)
        )
        loss = float(np.average(nll, weights=sample_weights))
        shared = weights.mean(axis=1, keepdims=True)
        # Biases can otherwise become a post-hoc threshold search on a small
        # minority holdout, so regularize them together with expert weights.
        penalty = float(
            np.square(weights - shared).mean() + 0.1 * np.square(bias).mean()
        )
        return loss + regularization * penalty

    result = minimize(
        objective,
        x0=np.zeros(n_experts * n_classes + n_classes),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Class-aware optimization failed: {result.message}")
    fitted_weights, fitted_bias = unpack(result.x)
    return ClassAwareEnsemble(
        temperatures=temperatures,
        class_weights=fitted_weights,
        class_bias=fitted_bias,
        regularization=regularization,
    )
