"""Small, inspectable linear algebra engine from the supplied prototype.

The state is ``[current pixel, previous pixel, second previous pixel, seed]``.
Missing left-hand neighbours repeat the first pixel in each row, avoiding the
three-column black band in the prototype. No statistical model is fitted here:
the coefficients are explicit, fixed inputs to deterministic spatial operators.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class EngineParameters:
    """Parameters for the matrix engine; defaults reproduce the blueprint."""

    phi_1: float = 0.55
    phi_2: float = 0.25
    phi_3: float = 0.10
    theta_1: float = -0.40
    noise_floor: float = 0.02
    edge_gain: float = 4.0
    integration_gain: float = 0.15
    extension_px: int = 40

    def validate(self) -> None:
        """Check user-facing bounds before constructing any operators."""
        bounds = {
            "phi_1": (0.0, 1.0),
            "phi_2": (0.0, 1.0),
            "phi_3": (0.0, 1.0),
            "theta_1": (-1.0, 1.0),
            "noise_floor": (0.0, 1.0),
            "edge_gain": (0.0, 32.0),
            "integration_gain": (0.0, 8.0),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be a finite number between {lower:g} and {upper:g}.")
        if (
            isinstance(self.extension_px, bool)
            or not isinstance(self.extension_px, Integral)
            or not 0 <= self.extension_px <= 256
        ):
            raise ValueError("extension_px must be a whole number between 0 and 256.")


@dataclass(frozen=True)
class Operators:
    """The three 4 by 4 matrices and their numerical diagnostics."""

    transition: np.ndarray
    difference: np.ndarray
    integration: np.ndarray
    spectral_radius: float
    condition_number: float


@dataclass(frozen=True)
class TransformResult:
    """Display-ready grayscale outputs, timing and the operators used."""

    edges: np.ndarray
    baseline: np.ndarray
    extension: np.ndarray
    elapsed_ms: float
    operators: Operators


def build_operators(params: EngineParameters = EngineParameters()) -> Operators:
    """Construct F, I-F and (I-F)^-1, rejecting unstable configurations.

    A spectral radius below one keeps repeated forecasts bounded. The separate
    condition-number guard avoids amplifying rounding errors near singularity.
    """
    if not isinstance(params, EngineParameters):
        raise ValueError("params must be an EngineParameters instance.")
    params.validate()
    transition = np.array(
        [
            [params.phi_1, params.phi_2, params.phi_3, params.theta_1],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if not np.isfinite(spectral_radius) or spectral_radius >= 1.0:
        raise ValueError(
            "Unstable coefficients: the spectral radius of F must be below 1. "
            "Reduce the phi coefficients so their sum is below 1."
        )

    difference = np.eye(4, dtype=np.float64) - transition
    condition_number = float(np.linalg.cond(difference))
    if not np.isfinite(condition_number) or condition_number > 1e8:
        raise ValueError(
            "I - F is too close to singular for a reliable inverse. "
            "Reduce the phi coefficients."
        )
    try:
        integration = np.linalg.inv(difference)
    except np.linalg.LinAlgError as exc:
        raise ValueError("I - F is singular; reduce the phi coefficients.") from exc

    # A cached result can safely expose these matrices without accidental edits.
    for matrix in (transition, difference, integration):
        matrix.setflags(write=False)
    return Operators(
        transition=transition,
        difference=difference,
        integration=integration,
        spectral_radius=spectral_radius,
        condition_number=condition_number,
    )


def _to_uint8(values: np.ndarray, gain: float = 1.0) -> np.ndarray:
    return np.clip(values * (255.0 * gain), 0.0, 255.0).astype(np.uint8)


def process_image(
    gray: np.ndarray, params: EngineParameters = EngineParameters()
) -> TransformResult:
    """Transform a nonempty uint8 grayscale image at the supplied resolution.

    Pixel states are vectorized over the entire image. Right-edge forecasting
    loops only over the requested extension width and advances all image rows
    together; repeated multiplication by F is mathematically F^k. The original
    part of the extension is copied byte-for-byte from the grayscale input.
    """
    started = perf_counter()
    if not isinstance(gray, np.ndarray):
        raise ValueError("The input must be a NumPy grayscale image array.")
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("The input must be a two-dimensional uint8 grayscale image.")
    rows, cols = gray.shape
    if rows < 1 or cols < 3:
        raise ValueError("The image must contain at least one row and three columns.")
    operators = build_operators(params)
    normalized = gray.astype(np.float64) / 255.0
    padded = np.pad(normalized, ((0, 0), (2, 0)), mode="edge")
    current, lag_1, lag_2 = padded[:, 2:], padded[:, 1:-1], padded[:, :-2]

    def apply_first_row(matrix: np.ndarray) -> np.ndarray:
        return (
            matrix[0, 0] * current
            + matrix[0, 1] * lag_1
            + matrix[0, 2] * lag_2
            + matrix[0, 3] * params.noise_floor
        )

    edges = _to_uint8(np.abs(apply_first_row(operators.difference)), params.edge_gain)
    baseline = _to_uint8(apply_first_row(operators.integration), params.integration_gain)
    extension = np.empty((rows, cols + params.extension_px), dtype=np.uint8)
    extension[:, :cols] = gray
    state = np.column_stack(
        (normalized[:, -1], normalized[:, -2], normalized[:, -3], np.zeros(rows))
    )
    for step in range(params.extension_px):
        state = state @ operators.transition.T
        extension[:, cols + step] = _to_uint8(state[:, 0])

    return TransformResult(
        edges=edges,
        baseline=baseline,
        extension=extension,
        elapsed_ms=(perf_counter() - started) * 1000.0,
        operators=operators,
    )


def process_spatial_matrix_image(img_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility entry point matching the attachment's three-image API."""
    result = process_image(img_gray)
    return result.edges, result.baseline, result.extension
