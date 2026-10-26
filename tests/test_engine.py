"""Numerical regressions against an independent, literal matrix implementation."""

from dataclasses import replace

import numpy as np
import pytest

from spatial_lab.engine import (
    EngineParameters,
    build_operators,
    process_image,
    process_spatial_matrix_image,
)


def reference_transform(gray, params):
    """Deliberately slow pixel loop, separate from the vectorized implementation."""
    transition = np.array([
        [params.phi_1, params.phi_2, params.phi_3, params.theta_1],
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],
    ], dtype=float)
    difference = np.eye(4) - transition
    integration = np.linalg.inv(difference)
    normalized = gray.astype(float) / 255
    edges = np.zeros_like(normalized)
    baseline = np.zeros_like(normalized)
    extension = np.zeros((gray.shape[0], gray.shape[1] + params.extension_px))
    extension[:, :gray.shape[1]] = normalized
    for row in range(gray.shape[0]):
        for col in range(gray.shape[1]):
            state = np.array([
                normalized[row, col], normalized[row, max(col - 1, 0)],
                normalized[row, max(col - 2, 0)], params.noise_floor,
            ])
            edges[row, col] = abs((difference @ state)[0])
            baseline[row, col] = (integration @ state)[0]
        seed = np.array([*normalized[row, -3:][::-1], 0.0])
        for step in range(params.extension_px):
            extension[row, gray.shape[1] + step] = (
                np.linalg.matrix_power(transition, step + 1) @ seed
            )[0]
    convert = lambda image, gain: np.clip(image * 255 * gain, 0, 255).astype(np.uint8)
    return convert(edges, params.edge_gain), convert(baseline, params.integration_gain), convert(extension, 1)


@pytest.mark.parametrize("params", [
    EngineParameters(),
    EngineParameters(phi_1=0.3, phi_2=0.2, phi_3=0.15, theta_1=0.25,
                     noise_floor=0.1, edge_gain=2, integration_gain=0.4, extension_px=17),
])
def test_vectorized_engine_matches_literal_pixel_loop(params):
    gray = np.random.default_rng(76).integers(0, 256, (9, 13), dtype=np.uint8)
    actual = process_image(gray, params)
    for output, expected in zip((actual.edges, actual.baseline, actual.extension),
                                reference_transform(gray, params)):
        np.testing.assert_allclose(output.astype(int), expected.astype(int), atol=1)


def test_operators_match_blueprint_and_are_read_only():
    operators = build_operators()
    np.testing.assert_array_equal(operators.transition[0], [0.55, 0.25, 0.1, -0.4])
    np.testing.assert_allclose(operators.difference, np.eye(4) - operators.transition)
    np.testing.assert_allclose(operators.integration @ operators.difference, np.eye(4), atol=1e-14)
    assert 0 < operators.spectral_radius < 1
    assert operators.condition_number > 1
    with pytest.raises(ValueError):
        operators.transition[0, 0] = 9


def test_shapes_types_input_preservation_and_repeatability():
    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    copy = gray.copy()
    first = process_image(gray)
    second = process_image(gray)
    assert first.edges.shape == first.baseline.shape == gray.shape
    assert first.extension.shape == (5, 47)
    for name in ("edges", "baseline", "extension"):
        assert getattr(first, name).dtype == np.uint8
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_array_equal(first.extension[:, :7], gray)
    np.testing.assert_array_equal(gray, copy)
    assert first.elapsed_ms >= 0


def test_left_border_repeats_first_pixel_instead_of_black_band():
    gray = np.full((2, 6), 150, dtype=np.uint8)
    result = process_image(gray)
    assert result.edges[0, 0] > 0
    np.testing.assert_array_equal(result.edges[:, :3], result.edges[:, 3:])
    np.testing.assert_array_equal(result.baseline[:, :3], result.baseline[:, 3:])


def test_zero_extension_and_compatibility_entry_point():
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(process_image(gray, EngineParameters(extension_px=0)).extension, gray)
    result = process_image(gray)
    for actual, expected in zip(process_spatial_matrix_image(gray),
                                (result.edges, result.baseline, result.extension)):
        np.testing.assert_array_equal(actual, expected)


def test_display_gains_and_noise_weight_do_not_change_zero_seed_forecast():
    gray = np.array([[230, 40, 180, 60, 210]], dtype=np.uint8)
    default = process_image(gray)
    adjusted = process_image(gray, EngineParameters(theta_1=0.4, edge_gain=0, integration_gain=0))
    assert not np.any(adjusted.edges)
    assert not np.any(adjusted.baseline)
    np.testing.assert_array_equal(default.extension, adjusted.extension)
    saturated = process_image(gray, EngineParameters(edge_gain=32))
    assert saturated.edges.max() == 255


@pytest.mark.parametrize("changes", [
    {"phi_1": float("nan")}, {"theta_1": float("inf")},
    {"phi_1": -0.1}, {"phi_2": 1.1}, {"theta_1": -1.1},
    {"noise_floor": -0.1}, {"edge_gain": 33}, {"integration_gain": 9},
    {"extension_px": -1}, {"extension_px": 257}, {"extension_px": 2.5},
    {"extension_px": True}, {"phi_1": True},
    {"phi_1": 0.8, "phi_2": 0.4},
    {"phi_1": 1, "phi_2": 0, "phi_3": 0},
    {"phi_1": 0.9999999999, "phi_2": 0, "phi_3": 0},
])
def test_invalid_or_unstable_parameters_fail_before_processing(changes):
    with pytest.raises(ValueError):
        build_operators(replace(EngineParameters(), **changes))


@pytest.mark.parametrize("gray", [
    [[0, 1, 2]], np.zeros((2, 3, 3), dtype=np.uint8),
    np.zeros((2, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8),
    np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 0), dtype=np.uint8),
])
def test_invalid_images_fail_with_clear_exception(gray):
    with pytest.raises(ValueError):
        process_image(gray)


def test_wrong_parameter_type_rejected():
    with pytest.raises(ValueError):
        build_operators({"phi_1": 0.5})
