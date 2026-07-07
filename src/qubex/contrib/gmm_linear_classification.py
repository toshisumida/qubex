"""Helpers for DSP classification lines from GMM state centers."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from qubex.measurement._gmm_linear_classification import (
    LineParam,
    LineParamPair,
    build_gmm_linear_line_param_pair,
)

LineParamMaps = tuple[dict[str, LineParam], dict[str, LineParam]]
LineParamMapsWithDistance = tuple[
    dict[str, LineParam], dict[str, LineParam], dict[str, float]
]

DSP_LINE_BIT_LABELS = ("00", "01", "10", "11")
REPEATED_OUTCOME_LABELS = ("00", "01", "10", "11")
_StateValueT = TypeVar("_StateValueT", complex, float)


def _resolve_state_value(
    values: Mapping[int | str, _StateValueT],
    state: int | str,
) -> _StateValueT:
    if state in values:
        return values[state]
    text_state = str(state)
    if text_state in values:
        return values[text_state]
    raise ValueError(f"State value {state!r} is not available.")


def _resolve_state_center(
    centers: Mapping[int | str, complex],
    state: int | str,
) -> complex:
    return complex(_resolve_state_value(centers, state))


def _resolve_state_stddev(
    stddevs: Mapping[int | str, float],
    state: int | str,
) -> float:
    stddev = float(_resolve_state_value(stddevs, state))
    if not math.isfinite(stddev) or stddev < 0.0:
        raise ValueError(
            "GMM state standard deviations must be finite and non-negative."
        )
    return stddev


def _state_sort_key(state: int | str) -> tuple[int, str]:
    try:
        return (0, f"{int(state):08d}")
    except (TypeError, ValueError):
        return (1, str(state))


def _line_from_normal_and_point(
    normal: tuple[float, float],
    point: complex,
) -> LineParam:
    a, b = normal
    c = -(a * point.real + b * point.imag)
    return (a, b, float(c))


def _line_projection(line: LineParam) -> float:
    """Return the signed point projection for a normalized line normal."""
    return -float(line[2])


def _normalized_ge_axis(
    *,
    g_center: complex,
    e_center: complex,
) -> tuple[float, float]:
    if g_center == e_center:
        raise ValueError("GMM state centers for g/e must not be identical.")
    delta = g_center - e_center
    norm = math.hypot(delta.real, delta.imag)
    if norm == 0:
        raise ValueError("GMM state centers must define a non-degenerate line.")
    return (float(delta.real / norm), float(delta.imag / norm))


def _resolve_targets_and_readouts(
    ctx: Any,
    targets: Sequence[str],
    readout_targets: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    target_list = list(targets)
    if readout_targets is None:
        readout_list = [ctx.resolve_read_label(target) for target in target_list]
    else:
        readout_list = list(readout_targets)
    if len(target_list) != len(readout_list):
        raise ValueError("targets and readout_targets must have the same length.")
    return target_list, readout_list


def resolve_gmm_state_centers_and_stddevs(
    ctx: Any,
    target: str,
) -> tuple[Mapping[int | str, complex], Mapping[int | str, float]]:
    """Return GMM centers/stddevs for one qubit from classifiers or calibration note."""
    classifiers = getattr(ctx, "classifiers", {}) or {}
    classifier = classifiers.get(target)
    if classifier is not None:
        try:
            centers = classifier.centers
            stddevs = classifier.stddevs
            if centers and stddevs:
                return centers, stddevs
        except (AttributeError, NotImplementedError):
            pass

    centers = getattr(ctx, "state_centers", {}).get(target)
    stddevs = getattr(ctx, "state_stddevs", {}).get(target)
    if centers and stddevs:
        return centers, stddevs

    raise RuntimeError(
        f"GMM centers/stddevs are missing for {target}. "
        "Run `build_classifier` with GMM first."
    )


def build_gmm_linear_line_param(
    centers: Mapping[int | str, complex],
    *,
    g_state: int = 0,
    e_state: int = 1,
) -> LineParam:
    """Return the normalized midpoint separator derived from GMM state centers."""
    g_center = _resolve_state_center(centers, g_state)
    e_center = _resolve_state_center(centers, e_state)
    normal = _normalized_ge_axis(g_center=g_center, e_center=e_center)
    midpoint = (g_center + e_center) / 2
    return _line_from_normal_and_point(normal, midpoint)


def build_gmm_midpoint_band_line_param_pair(
    centers: Mapping[int | str, complex],
    *,
    distance_fraction: float,
    g_state: int = 0,
    e_state: int = 1,
) -> tuple[LineParam, LineParam, float]:
    """Return two midpoint-centered parallel lines separated by a center-distance fraction."""
    distance_fraction = float(distance_fraction)
    if not math.isfinite(distance_fraction) or distance_fraction < 0.0:
        raise ValueError("distance_fraction must be finite and non-negative.")

    g_center = _resolve_state_center(centers, g_state)
    e_center = _resolve_state_center(centers, e_state)
    normal = _normalized_ge_axis(g_center=g_center, e_center=e_center)
    midpoint = (g_center + e_center) / 2
    line_distance = distance_fraction * abs(g_center - e_center)
    normal_complex = complex(normal[0], normal[1])
    line_candidates = (
        _line_from_normal_and_point(
            normal,
            midpoint - normal_complex * line_distance / 2,
        ),
        _line_from_normal_and_point(
            normal,
            midpoint + normal_complex * line_distance / 2,
        ),
    )
    line0, line1 = sorted(line_candidates, key=_line_projection)
    return line0, line1, float(abs(line1[2] - line0[2]) / math.hypot(*normal))


def build_gmm_midpoint_line_param_maps(
    ctx: Any,
    targets: Sequence[str],
    *,
    readout_targets: Sequence[str] | None = None,
    g_state: int = 0,
    e_state: int = 1,
) -> LineParamMaps:
    """Return per-readout maps where line0 and line1 are the same midpoint separator."""
    target_list, readout_list = _resolve_targets_and_readouts(
        ctx,
        targets,
        readout_targets,
    )
    line0_by_readout: dict[str, LineParam] = {}
    line1_by_readout: dict[str, LineParam] = {}
    for target, readout_target in zip(target_list, readout_list, strict=True):
        centers, _ = resolve_gmm_state_centers_and_stddevs(ctx, target)
        line = build_gmm_linear_line_param(
            centers,
            g_state=g_state,
            e_state=e_state,
        )
        line0_by_readout[readout_target] = line
        line1_by_readout[readout_target] = line
    return line0_by_readout, line1_by_readout


def build_gmm_separated_line_param_maps(
    ctx: Any,
    targets: Sequence[str],
    *,
    readout_targets: Sequence[str] | None = None,
    sigma_multiplier: float = 1.0,
    g_state: int = 0,
    e_state: int = 1,
) -> LineParamMaps:
    """Return per-readout maps for sigma-offset separated GMM line pairs."""
    target_list, readout_list = _resolve_targets_and_readouts(
        ctx,
        targets,
        readout_targets,
    )
    line0_by_readout: dict[str, LineParam] = {}
    line1_by_readout: dict[str, LineParam] = {}
    for target, readout_target in zip(target_list, readout_list, strict=True):
        centers, stddevs = resolve_gmm_state_centers_and_stddevs(ctx, target)
        line0, line1 = build_gmm_linear_line_param_pair(
            centers,
            stddevs,
            sigma_multiplier=sigma_multiplier,
            g_state=g_state,
            e_state=e_state,
        )
        line0_by_readout[readout_target] = line0
        line1_by_readout[readout_target] = line1
    return line0_by_readout, line1_by_readout


def build_gmm_midpoint_band_line_param_maps(
    ctx: Any,
    targets: Sequence[str],
    *,
    distance_fraction: float,
    readout_targets: Sequence[str] | None = None,
    g_state: int = 0,
    e_state: int = 1,
) -> LineParamMapsWithDistance:
    """Return per-readout maps for a symmetric rejection band around the midpoint."""
    target_list, readout_list = _resolve_targets_and_readouts(
        ctx,
        targets,
        readout_targets,
    )
    line0_by_readout: dict[str, LineParam] = {}
    line1_by_readout: dict[str, LineParam] = {}
    distances: dict[str, float] = {}
    for target, readout_target in zip(target_list, readout_list, strict=True):
        centers, _ = resolve_gmm_state_centers_and_stddevs(ctx, target)
        line0, line1, distance = build_gmm_midpoint_band_line_param_pair(
            centers,
            distance_fraction=distance_fraction,
            g_state=g_state,
            e_state=e_state,
        )
        line0_by_readout[readout_target] = line0
        line1_by_readout[readout_target] = line1
        distances[target] = distance
    return line0_by_readout, line1_by_readout, distances


def build_gmm_linear_line_params(
    state_centers: Mapping[str, Mapping[int | str, complex]],
    *,
    g_state: int = 0,
    e_state: int = 1,
) -> dict[str, LineParam]:
    """Return per-target midpoint separators derived from GMM state centers."""
    return {
        target: build_gmm_linear_line_param(
            centers,
            g_state=g_state,
            e_state=e_state,
        )
        for target, centers in state_centers.items()
    }


def build_gmm_linear_line_param_pairs(
    state_centers: Mapping[str, Mapping[int | str, complex]],
    state_stddevs: Mapping[str, Mapping[int | str, float]],
    *,
    sigma_multiplier: float = 1.0,
    g_state: int = 0,
    e_state: int = 1,
) -> dict[str, LineParamPair]:
    """Return per-target parallel line pairs derived from GMM centers and sigmas."""
    return {
        target: build_gmm_linear_line_param_pair(
            centers,
            state_stddevs[target],
            sigma_multiplier=sigma_multiplier,
            g_state=g_state,
            e_state=e_state,
        )
        for target, centers in state_centers.items()
    }


def raw_dsp_values(result: Any, target: str, capture_index: int) -> NDArray[np.uint8]:
    """Return one capture's raw DSP labels as a flat uint8 array."""
    return np.asarray(
        result.data[target][capture_index].raw,
        dtype=np.uint8,
    ).reshape(-1)


def dsp_line_bit_counts(raw: Any) -> dict[str, int]:
    """Return raw DSP line-bit counts for labels 00, 01, 10, and 11."""
    values = np.asarray(raw, dtype=np.uint8).reshape(-1)
    return {
        label: int(np.count_nonzero(values == value))
        for value, label in enumerate(DSP_LINE_BIT_LABELS)
    }


def raw_dsp_unique_counts(raw: Any) -> dict[int, int]:
    """Return numeric raw DSP label counts."""
    values, counts = np.unique(
        np.asarray(raw, dtype=np.uint8).reshape(-1),
        return_counts=True,
    )
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def logical_repeated_pair_counts(
    first_raw: Any,
    second_raw: Any,
    *,
    reject_dsp_ambiguous: bool,
) -> dict[str, Any]:
    """Summarize repeated-measurement pairs after optional DSP 01/10 rejection."""
    first_raw = np.asarray(first_raw, dtype=np.uint8).reshape(-1)
    second_raw = np.asarray(second_raw, dtype=np.uint8).reshape(-1)
    if first_raw.shape != second_raw.shape:
        raise ValueError("The first and second captures must have the same shot count.")

    valid_first = np.isin(first_raw, [0, 3])
    valid_second = np.isin(second_raw, [0, 3])
    if reject_dsp_ambiguous:
        dsp_accepted = valid_first & valid_second
    else:
        dsp_accepted = np.ones(first_raw.shape, dtype=bool)
        if not np.all(valid_first & valid_second):
            raise ValueError(
                "Ambiguous DSP outputs were found. "
                "Set reject_dsp_ambiguous=True for this result."
            )

    first_logical = (first_raw[dsp_accepted] == 3).astype(np.uint8)
    second_logical = (second_raw[dsp_accepted] == 3).astype(np.uint8)
    counter = Counter(
        f"{int(first)}{int(second)}"
        for first, second in zip(first_logical, second_logical, strict=True)
    )
    counts = {label: int(counter.get(label, 0)) for label in REPEATED_OUTCOME_LABELS}
    accepted_total = int(dsp_accepted.sum())
    total = int(first_raw.size)
    first_retained_fraction = float(np.mean(valid_first)) if total else np.nan
    second_retained_fraction = float(np.mean(valid_second)) if total else np.nan
    single_readout_retained_fraction = (
        first_retained_fraction + second_retained_fraction
    ) / 2
    diagonal = counts["00"] + counts["11"]

    return {
        "counts": counts,
        "accepted_total": accepted_total,
        "total": total,
        "retained_fraction": accepted_total / total if total else np.nan,
        "first_retained_fraction": first_retained_fraction,
        "second_retained_fraction": second_retained_fraction,
        "single_readout_retained_fraction": single_readout_retained_fraction,
        "dsp_line_rejected_total": total - accepted_total,
        "diagonal_probability": diagonal / accepted_total if accepted_total else np.nan,
        "gg_probability": counts["00"] / accepted_total if accepted_total else np.nan,
        "ee_probability": counts["11"] / accepted_total if accepted_total else np.nan,
    }


def summarize_repeated_dsp_classification(
    result: Any,
    targets: Sequence[str],
    *,
    title: str | None = None,
    reject_dsp_ambiguous: bool,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Summarize two-capture DSP-classified repeated measurements."""
    summaries: dict[str, dict[str, Any]] = {}
    if verbose and title:
        print(title)
    for target in targets:
        first_raw = raw_dsp_values(result, target, 0)
        second_raw = raw_dsp_values(result, target, 1)
        summary = logical_repeated_pair_counts(
            first_raw,
            second_raw,
            reject_dsp_ambiguous=reject_dsp_ambiguous,
        )
        summary["first_raw_unique_counts"] = raw_dsp_unique_counts(first_raw)
        summary["second_raw_unique_counts"] = raw_dsp_unique_counts(second_raw)
        summary["first_dsp_line_bit_counts"] = dsp_line_bit_counts(first_raw)
        summary["second_dsp_line_bit_counts"] = dsp_line_bit_counts(second_raw)
        summaries[target] = summary

        if not verbose:
            continue
        print(target)
        print("  first capture raw numeric counts:", summary["first_raw_unique_counts"])
        print(
            "  second capture raw numeric counts:", summary["second_raw_unique_counts"]
        )
        print(
            "  first capture DSP line-bit counts:",
            {
                label: summary["first_dsp_line_bit_counts"][label]
                for label in DSP_LINE_BIT_LABELS
            },
        )
        print(
            "  second capture DSP line-bit counts:",
            {
                label: summary["second_dsp_line_bit_counts"][label]
                for label in DSP_LINE_BIT_LABELS
            },
        )
        print(f"  rejected by DSP line-bit 01/10: {summary['dsp_line_rejected_total']}")
        print(
            "  DSP-post-selected repeated shots: "
            f"{summary['accepted_total']} / {summary['total']} "
            f"({summary['retained_fraction']:.4f})"
        )
        print(
            "  per-capture retained fraction: "
            f"first={summary['first_retained_fraction']:.4f}, "
            f"second={summary['second_retained_fraction']:.4f}, "
            f"avg={summary['single_readout_retained_fraction']:.4f}"
        )
        for label in REPEATED_OUTCOME_LABELS:
            count = summary["counts"][label]
            denom = max(summary["accepted_total"], 1)
            print(f"  repeated P({label}) = {count / denom:.4f} ({count} shots)")
        print(f"  P(gg) + P(ee) = {summary['diagonal_probability']:.4f}")
    return summaries


def summary_unconditional_diagonal_fraction(summary: Mapping[str, Any]) -> float:
    """Return diagonal probability over all shots, not only retained shots."""
    return float(summary["diagonal_probability"]) * float(summary["retained_fraction"])


def single_readout_fidelity_from_repeated_diagonal(
    diagonal_probability: float,
) -> float:
    """Convert P(00)+P(11) to the symmetric single-readout equivalent fidelity."""
    repeated = float(diagonal_probability)
    return (1.0 + math.sqrt(max(2.0 * repeated - 1.0, 0.0))) / 2.0


def single_readout_retained_fraction(summary_or_record: Mapping[str, Any]) -> float:
    """Return per-capture retained fraction from a summary or sweep record."""
    if "single_readout_retained_fraction" in summary_or_record:
        return float(summary_or_record["single_readout_retained_fraction"])
    return math.sqrt(max(float(summary_or_record["retained_fraction"]), 0.0))


def _line_point(line: LineParam) -> complex:
    a, b, c = (float(value) for value in line)
    denom = a * a + b * b
    return complex(-a * c / denom, -b * c / denom)


def _classifier_events(
    ctx: Any,
    target: str,
    *,
    max_points: int,
) -> dict[int | str, NDArray[np.complex128]]:
    classifiers = getattr(ctx, "classifiers", {}) or {}
    classifier = classifiers.get(target)
    if classifier is None or not hasattr(classifier, "dataset"):
        return {}

    phase = float(getattr(classifier, "phase", 0.0))
    events: dict[int | str, NDArray[np.complex128]] = {}
    for state, xy in classifier.dataset.items():
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] < 2:
            continue
        z = (xy[:, 0] + 1j * xy[:, 1]) * np.exp(1j * phase)
        if z.size > max_points:
            indices = np.linspace(0, z.size - 1, max_points, dtype=int)
            z = z[indices]
        events[state] = np.asarray(z, dtype=np.complex128)
    return events


def _draw_line(
    ax: Any,
    line: LineParam,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    a, b, c = (float(value) for value in line)
    xmin, xmax = ax.get_xlim()
    if abs(b) > 1e-15:
        x = np.linspace(xmin, xmax, 200)
        y = -(a * x + c) / b
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    elif abs(a) > 1e-15:
        ax.axvline(
            -c / a,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    else:
        raise ValueError("Invalid line parameters: both a and b are zero.")


def plot_gmm_events_with_lines(
    ctx: Any,
    targets: Sequence[str],
    line0_by_readout: Mapping[str, LineParam],
    line1_by_readout: Mapping[str, LineParam],
    *,
    title: str,
    readout_targets: Sequence[str] | None = None,
    sigma_multiplier: float | None = None,
    max_points: int = 4000,
) -> Any:
    """Plot classifier events, GMM centers, and DSP line params."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    target_list, readout_list = _resolve_targets_and_readouts(
        ctx,
        targets,
        readout_targets,
    )
    fig, axes = plt.subplots(
        1,
        len(target_list),
        figsize=(5.5 * len(target_list), 4.8),
        squeeze=False,
    )
    for ax, target, readout_target in zip(
        axes[0],
        target_list,
        readout_list,
        strict=True,
    ):
        centers, stddevs = resolve_gmm_state_centers_and_stddevs(ctx, target)
        events = _classifier_events(ctx, target, max_points=max_points)
        line0 = line0_by_readout[readout_target]
        line1 = line1_by_readout[readout_target]

        points = [complex(value) for value in centers.values()]
        points.extend([_line_point(line0), _line_point(line1)])
        for z in events.values():
            points.extend(z)
        point_array = np.asarray(points, dtype=np.complex128)
        x_span = max(float(np.ptp(point_array.real)), 1e-6)
        y_span = max(float(np.ptp(point_array.imag)), 1e-6)
        margin = 0.15 * max(x_span, y_span)
        ax.set_xlim(point_array.real.min() - margin, point_array.real.max() + margin)
        ax.set_ylim(point_array.imag.min() - margin, point_array.imag.max() + margin)

        for state, z in sorted(
            events.items(), key=lambda item: _state_sort_key(item[0])
        ):
            ax.scatter(z.real, z.imag, s=6, alpha=0.18, label=f"state {state} events")
        for state, center in sorted(
            centers.items(),
            key=lambda item: _state_sort_key(item[0]),
        ):
            center = complex(center)
            ax.scatter(
                center.real,
                center.imag,
                marker="x",
                s=80,
                linewidths=2,
                label=f"state {state} center",
            )
            if sigma_multiplier is not None:
                radius = _resolve_state_stddev(stddevs, state) * float(sigma_multiplier)
                ax.add_patch(
                    Circle(
                        (center.real, center.imag),
                        radius,
                        fill=False,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                )

        same_line = np.allclose(line0, line1)
        _draw_line(
            ax,
            line0,
            label="line0=line1" if same_line else "line0 (e-side)",
            color="#1f77b4",
        )
        if not same_line:
            _draw_line(
                ax,
                line1,
                label="line1 (g-side)",
                color="#d62728",
                linestyle="--",
            )

        ax.set_title(f"{target} / {readout_target}")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_raw_dsp_output_distribution(
    result: Any,
    targets: Sequence[str],
    *,
    title: str,
) -> Any:
    """Plot raw DSP line-bit distributions for two captures."""
    import matplotlib.pyplot as plt

    target_list = list(targets)
    fig, axes = plt.subplots(
        len(target_list),
        2,
        figsize=(9, 3.4 * len(target_list)),
        squeeze=False,
    )
    for row, target in enumerate(target_list):
        for col, capture_index in enumerate([0, 1]):
            raw = raw_dsp_values(result, target, capture_index)
            counts = dsp_line_bit_counts(raw)
            bar_values = [counts[label] for label in DSP_LINE_BIT_LABELS]
            bars = axes[row, col].bar(
                DSP_LINE_BIT_LABELS,
                bar_values,
                color="#4c78a8",
            )
            max_count = max(bar_values) if bar_values else 0
            axes[row, col].set_ylim(0, max_count * 1.15 + 1)
            label_offset = max(max_count * 0.02, 0.1)
            for bar, value in zip(bars, bar_values, strict=True):
                axes[row, col].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + label_offset,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axes[row, col].set_title(f"{target}: capture {capture_index + 1}")
            axes[row, col].set_xlabel("DSP line bits")
            axes[row, col].set_ylabel("shots")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_repeated_dsp_summary(
    summaries: Mapping[str, Mapping[str, Any]],
    targets: Sequence[str],
    *,
    title: str,
) -> Any:
    """Plot repeated-measurement counts and probability matrix per target."""
    import matplotlib.pyplot as plt

    target_list = list(targets)
    fig, axes = plt.subplots(
        len(target_list),
        2,
        figsize=(10, 4 * len(target_list)),
        squeeze=False,
    )
    for row, target in enumerate(target_list):
        summary = summaries[target]
        counts = summary["counts"]
        accepted_total = max(int(summary["accepted_total"]), 1)
        count_values = np.asarray(
            [counts[label] for label in REPEATED_OUTCOME_LABELS],
            dtype=float,
        )
        probability_matrix = (
            np.asarray(
                [
                    [counts["00"], counts["01"]],
                    [counts["10"], counts["11"]],
                ],
                dtype=float,
            )
            / accepted_total
        )

        axes[row, 0].bar(
            REPEATED_OUTCOME_LABELS,
            count_values,
            color=["#4c78a8", "#f58518", "#e45756", "#54a24b"],
        )
        axes[row, 0].set_title(f"{target}: repeated measurements")
        axes[row, 0].set_xlabel("repeated outcome (0=g, 1=e)")
        axes[row, 0].set_ylabel("accepted shots")

        vmax = max(float(probability_matrix.max()), 1e-12)
        image = axes[row, 1].imshow(probability_matrix, vmin=0.0, vmax=vmax)
        axes[row, 1].set_title(f"{target}: first vs second measurement")
        axes[row, 1].set_xticks([0, 1], labels=["2nd: g", "2nd: e"])
        axes[row, 1].set_yticks([0, 1], labels=["1st: g", "1st: e"])
        for i in range(2):
            for j in range(2):
                color = "white" if probability_matrix[i, j] > 0.5 * vmax else "black"
                axes[row, 1].text(
                    j,
                    i,
                    f"{probability_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                )
        fig.colorbar(
            image,
            ax=axes[row, 1],
            fraction=0.046,
            pad=0.04,
            label="probability",
        )

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _summary_targets(
    summaries: Mapping[str, Any],
    targets: Sequence[str] | None,
) -> list[str]:
    return list(summaries) if targets is None else list(targets)


def plot_repeated_diagonal_comparison(
    baseline_summary: Mapping[str, Mapping[str, Any]],
    separated_summary: Mapping[str, Mapping[str, Any]],
    *,
    targets: Sequence[str] | None = None,
) -> Any:
    """Plot repeated-measurement diagonal probability and retained fraction."""
    import matplotlib.pyplot as plt

    labels = _summary_targets(baseline_summary, targets)
    baseline_diagonal = [
        baseline_summary[target]["diagonal_probability"] for target in labels
    ]
    separated_diagonal = [
        separated_summary[target]["diagonal_probability"] for target in labels
    ]
    baseline_retained = [
        baseline_summary[target]["retained_fraction"] for target in labels
    ]
    separated_retained = [
        separated_summary[target]["retained_fraction"] for target in labels
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.0), sharex=True)
    axes[0].bar(
        x - width / 2,
        baseline_diagonal,
        width,
        label="single line",
        color="#4c78a8",
    )
    axes[0].bar(
        x + width / 2,
        separated_diagonal,
        width,
        label="separated, post-selected",
        color="#54a24b",
    )
    axes[0].set_ylim(0.8, 1.02)
    axes[0].set_ylabel("conditional P(00)+P(11)")
    axes[0].set_title("Repeated-measurement diagonal probability per qubit")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(
        x - width / 2,
        baseline_retained,
        width,
        label="single line",
        color="#4c78a8",
    )
    axes[1].bar(
        x + width / 2,
        separated_retained,
        width,
        label="separated",
        color="#f58518",
    )
    axes[1].set_xticks(x, labels=labels)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("two-capture retained fraction")
    axes[1].set_title("Repeated-measurement pairs retained after DSP 01/10 rejection")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_single_readout_equivalent_comparison(
    baseline_summary: Mapping[str, Mapping[str, Any]],
    separated_summary: Mapping[str, Mapping[str, Any]],
    *,
    targets: Sequence[str] | None = None,
) -> Any:
    """Plot single-readout-equivalent fidelity and per-capture retained fraction."""
    import matplotlib.pyplot as plt

    labels = _summary_targets(baseline_summary, targets)
    baseline_fidelity = [
        single_readout_fidelity_from_repeated_diagonal(
            baseline_summary[target]["diagonal_probability"]
        )
        for target in labels
    ]
    separated_fidelity = [
        single_readout_fidelity_from_repeated_diagonal(
            separated_summary[target]["diagonal_probability"]
        )
        for target in labels
    ]
    baseline_retained = [
        single_readout_retained_fraction(baseline_summary[target]) for target in labels
    ]
    separated_retained = [
        single_readout_retained_fraction(separated_summary[target]) for target in labels
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.0), sharex=True)
    axes[0].bar(
        x - width / 2,
        baseline_fidelity,
        width,
        label="single line",
        color="#4c78a8",
    )
    axes[0].bar(
        x + width / 2,
        separated_fidelity,
        width,
        label="separated, post-selected",
        color="#54a24b",
    )
    axes[0].set_ylim(0.9, 1.01)
    axes[0].set_ylabel("single-readout equivalent fidelity")
    axes[0].set_title("Single-readout equivalent from P(00)+P(11)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(
        x - width / 2,
        baseline_retained,
        width,
        label="single line",
        color="#4c78a8",
    )
    axes[1].bar(
        x + width / 2,
        separated_retained,
        width,
        label="separated",
        color="#f58518",
    )
    axes[1].set_xticks(x, labels=labels)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("per-capture retained fraction")
    axes[1].set_title("Single-readout events retained after DSP 01/10 rejection")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return fig


def _record_targets(
    records: Sequence[Mapping[str, Any]], targets: Sequence[str] | None
) -> list[str]:
    if targets is not None:
        return list(targets)
    return sorted({str(record["target"]) for record in records})


def plot_repeated_line_distance_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str] | None = None,
) -> Any:
    """Plot repeated-measurement line-distance sweep records."""
    import matplotlib.pyplot as plt

    labels = _record_targets(records, targets)
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(index % 10) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for label in labels:
        rows = sorted(
            [record for record in records if record["target"] == label],
            key=lambda record: record["line_distance_fraction"],
        )
        x = [record["line_distance_fraction"] for record in rows]
        fidelity = [record["conditional_fidelity"] for record in rows]
        retained = [record["retained_fraction"] for record in rows]
        axes[0].plot(x, fidelity, marker="o", label=label, color=colors[label])
        axes[1].plot(x, retained, marker="o", label=label, color=colors[label])

    axes[0].set_ylabel("conditional P(00)+P(11)")
    axes[0].set_ylim(0.8, 1.02)
    axes[0].set_title("Repeated-measurement diagonal probability per qubit")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_ylabel("two-capture retained fraction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Repeated-measurement pairs retained per qubit")
    axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("line distance / GMM g-e center distance")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_single_readout_equivalent_line_distance_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str] | None = None,
) -> Any:
    """Plot single-readout-equivalent line-distance sweep records."""
    import matplotlib.pyplot as plt

    labels = _record_targets(records, targets)
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(index % 10) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for label in labels:
        rows = sorted(
            [record for record in records if record["target"] == label],
            key=lambda record: record["line_distance_fraction"],
        )
        x = [record["line_distance_fraction"] for record in rows]
        fidelity = [
            record.get(
                "single_readout_fidelity",
                single_readout_fidelity_from_repeated_diagonal(
                    record["conditional_fidelity"]
                ),
            )
            for record in rows
        ]
        retained = [single_readout_retained_fraction(record) for record in rows]
        axes[0].plot(x, fidelity, marker="o", label=label, color=colors[label])
        axes[1].plot(x, retained, marker="o", label=label, color=colors[label])

    axes[0].set_ylabel("single-readout equivalent fidelity")
    axes[0].set_ylim(0.9, 1.01)
    axes[0].set_title("Single-readout equivalent fidelity per qubit")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_ylabel("per-capture retained fraction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Single-readout events retained per qubit")
    axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("line distance / GMM g-e center distance")
    axes[1].legend()
    fig.tight_layout()
    return fig
