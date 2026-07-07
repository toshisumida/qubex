"""Tests for experimental GMM-derived linear DSP classification helpers."""

from __future__ import annotations

import pytest

from qubex.contrib import (
    build_gmm_linear_line_param,
    build_gmm_linear_line_param_pair,
    build_gmm_linear_line_params,
    build_gmm_midpoint_band_line_param_maps,
    build_gmm_midpoint_line_param_maps,
    build_gmm_separated_line_param_maps,
    dsp_line_bit_counts,
    logical_repeated_pair_counts,
    raw_dsp_unique_counts,
    summarize_repeated_dsp_classification,
)


def _evaluate_line(line: tuple[float, float, float], point: complex) -> float:
    a, b, c = line
    return a * point.real + b * point.imag + c


def test_build_gmm_linear_line_param_uses_normalized_perpendicular_bisector() -> None:
    """Given g/e centers, helper should return the midpoint bisector with g on the positive side."""
    line = build_gmm_linear_line_param({0: 1.0 + 1.0j, 1: 3.0 + 1.0j})

    assert _evaluate_line(line, 2.0 + 1.0j) == pytest.approx(0.0)
    assert _evaluate_line(line, 1.0 + 1.0j) > 0.0
    assert _evaluate_line(line, 3.0 + 1.0j) < 0.0
    assert line[0] ** 2 + line[1] ** 2 == pytest.approx(1.0)


def test_build_gmm_linear_line_param_pair_offsets_lines_by_state_sigmas() -> None:
    """Given centers and sigmas, helper should return e-side then g-side parallel lines."""
    line0, line1 = build_gmm_linear_line_param_pair(
        {0: 0.0 + 0.0j, 1: 4.0 + 0.0j},
        {0: 0.5, 1: 1.0},
    )

    assert line0[:2] == pytest.approx(line1[:2])
    assert _evaluate_line(line0, 3.0 + 0.0j) == pytest.approx(0.0)
    assert _evaluate_line(line1, 0.5 + 0.0j) == pytest.approx(0.0)
    assert _evaluate_line(line0, 0.0 + 0.0j) > 0.0
    assert _evaluate_line(line1, 4.0 + 0.0j) < 0.0


def test_build_gmm_linear_line_param_pair_honors_sigma_multiplier() -> None:
    """Given a sigma multiplier, helper should scale both center offsets."""
    line0, line1 = build_gmm_linear_line_param_pair(
        {0: 0.0 + 0.0j, 1: 4.0 + 0.0j},
        {0: 0.5, 1: 1.0},
        sigma_multiplier=2.0,
    )

    assert _evaluate_line(line0, 2.0 + 0.0j) == pytest.approx(0.0)
    assert _evaluate_line(line1, 1.0 + 0.0j) == pytest.approx(0.0)


def test_build_gmm_linear_line_params_returns_per_target_mapping() -> None:
    """Given per-target centers, helper should build one separator for each target."""
    line_params = build_gmm_linear_line_params(
        {
            "RQ00": {0: 0.0 + 0.0j, 1: 2.0 + 0.0j},
            "RQ01": {0: 1.0 + 1.0j, 1: 1.0 + 3.0j},
        }
    )

    assert set(line_params) == {"RQ00", "RQ01"}
    assert _evaluate_line(line_params["RQ00"], 0.0 + 0.0j) > 0.0
    assert _evaluate_line(line_params["RQ00"], 2.0 + 0.0j) < 0.0
    assert _evaluate_line(line_params["RQ01"], 1.0 + 1.0j) > 0.0
    assert _evaluate_line(line_params["RQ01"], 1.0 + 3.0j) < 0.0


def test_build_gmm_linear_line_param_pair_rejects_identical_centers() -> None:
    """Given identical centers, helper should reject the degenerate separator."""
    with pytest.raises(ValueError, match="must not be identical"):
        build_gmm_linear_line_param_pair(
            {0: 1.0 + 0.0j, 1: 1.0 + 0.0j},
            {0: 0.1, 1: 0.1},
        )


class _Context:
    def __init__(self) -> None:
        self.classifiers = {}
        self.state_centers = {"Q00": {0: 0.0 + 0.0j, 1: 4.0 + 0.0j}}
        self.state_stddevs = {"Q00": {0: 0.5, 1: 1.0}}

    def resolve_read_label(self, target: str) -> str:
        return f"R{target[1:]}"


def test_build_line_param_maps_from_context() -> None:
    """Given an experiment context, helpers should return readout-keyed line maps."""
    ctx = _Context()

    midpoint0, midpoint1 = build_gmm_midpoint_line_param_maps(ctx, ["Q00"])
    separated0, separated1 = build_gmm_separated_line_param_maps(ctx, ["Q00"])
    band0, band1, distances = build_gmm_midpoint_band_line_param_maps(
        ctx,
        ["Q00"],
        distance_fraction=0.25,
    )

    assert set(midpoint0) == {"R00"}
    assert midpoint0 == midpoint1
    assert separated0["R00"][:2] == pytest.approx(separated1["R00"][:2])
    assert band0["R00"][:2] == pytest.approx(band1["R00"][:2])
    assert distances["Q00"] == pytest.approx(1.0)


def test_logical_repeated_pair_counts_rejects_ambiguous_dsp_outputs() -> None:
    """Given raw DSP labels, summary should drop 01/10 only when requested."""
    summary = logical_repeated_pair_counts(
        [0, 1, 3, 3],
        [0, 3, 1, 3],
        reject_dsp_ambiguous=True,
    )

    assert summary["accepted_total"] == 2
    assert summary["retained_fraction"] == pytest.approx(0.5)
    assert summary["counts"] == {"00": 1, "01": 0, "10": 0, "11": 1}
    assert dsp_line_bit_counts([0, 1, 3, 3]) == {
        "00": 1,
        "01": 1,
        "10": 0,
        "11": 2,
    }
    assert raw_dsp_unique_counts([0, 1, 3, 3]) == {0: 1, 1: 1, 3: 2}

    with pytest.raises(ValueError, match="Ambiguous DSP outputs"):
        logical_repeated_pair_counts(
            [0, 1, 3, 3],
            [0, 3, 1, 3],
            reject_dsp_ambiguous=False,
        )


class _Capture:
    def __init__(self, raw: list[int]) -> None:
        self.raw = raw


class _RepeatedResult:
    def __init__(self) -> None:
        self.data = {"Q00": [_Capture([0, 1, 3, 3]), _Capture([0, 3, 1, 3])]}


def test_summarize_repeated_dsp_classification_returns_per_target_summary() -> None:
    """Given a repeated DSP-classified result, helper should summarize each target."""
    summaries = summarize_repeated_dsp_classification(
        _RepeatedResult(),
        ["Q00"],
        reject_dsp_ambiguous=True,
        verbose=False,
    )

    assert summaries["Q00"]["accepted_total"] == 2
    assert summaries["Q00"]["first_raw_unique_counts"] == {0: 1, 1: 1, 3: 2}
    assert summaries["Q00"]["second_dsp_line_bit_counts"] == {
        "00": 1,
        "01": 1,
        "10": 0,
        "11": 2,
    }
