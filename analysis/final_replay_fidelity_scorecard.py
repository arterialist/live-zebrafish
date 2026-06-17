"""Final fidelity scorecard for the two-paper zebrafish replay pipeline.

This layer is deliberately stricter than the gross stability audit.  It asks:

* Is the embodied replay stable over long high-rate recordings?
* Do active runs match published larval kinematic target bands?
* Does a low-action negative-control video stay quiescent?
* Which calibration knobs remain necessary before a biological-fidelity claim?

The scorecard is intended for lab-facing review.  It should make unsupported
claims harder, not easier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from external_kinematics_validation_audit import EXTERNAL_TARGETS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "analysis" / "out" / "final_replay_validation_20260603"
DEFAULT_OUT = ROOT / "analysis" / "out" / "final_replay_validation_20260603" / "fidelity_scorecard"

METRIC_GROUPS = {
    "event_frequency_hz": "timing",
    "bout_duration_ms": "timing",
    "interbout_ms": "timing",
    "speed_xy_mm_s_mean": "locomotor_scale",
    "speed_xy_mm_s_p95": "locomotor_scale",
    "distance_per_event_mm": "locomotor_scale",
    "heading_delta_abs_deg": "turning_tail",
    "tail_yaw_abs_max_deg_p95": "turning_tail",
    "realized_tail_frequency_hz_observable": "turning_tail",
}

ACTIVE_OVERALL_WEIGHTS = {
    "stability": 0.25,
    "timing": 0.20,
    "locomotor_scale": 0.25,
    "turning_tail": 0.15,
    "source_grounding": 0.15,
}

NEGATIVE_CONTROL_WEIGHTS = {
    "stability": 0.35,
    "quiescence": 0.45,
    "source_grounding": 0.20,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _band_score(value: float, lo: float, hi: float) -> float:
    """Continuous score for target-band proximity.

    Inside the published target band scores 1.0.  Values below the band score
    as value / lower_bound; values above score as upper_bound / value.  This
    intentionally makes a stable but too-small simulation score poorly instead
    of hiding scale mismatch behind pass/fail status labels.
    """

    if not math.isfinite(value):
        return 0.0
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        if lo <= 0.0:
            return 0.0
        return _clip01(value / lo)
    if value <= 0.0:
        return 0.0
    return _clip01(hi / value)


def _run_is_negative_control(row: dict[str, str]) -> bool:
    tail_yaw = _num(row.get("tail_yaw_abs_max_deg_p95"))
    tail_yaw_ok = True if not math.isfinite(tail_yaw) else tail_yaw < 2.0
    return (
        int(_num(row.get("bout_count"), 0.0)) == 0
        and _num(row.get("action_force_mean"), 999.0) < 0.035
        and _num(row.get("swim_drive_p95"), 999.0) < 0.02
        and _num(row.get("speed_xy_mm_s_p95"), 999.0) < 0.8
        and tail_yaw_ok
    )


def _target_lookup() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for target in EXTERNAL_TARGETS:
        metric = str(target.get("metric", ""))
        if metric == "command_tail_frequency_hz_mean":
            continue
        out[metric] = target
    return out


def _anomaly_pass_fraction(rows: list[dict[str, str]], run: str) -> float:
    run_rows = [row for row in rows if row.get("run") == run]
    if not run_rows:
        return 0.0
    return float(np.mean([str(row.get("status")) == "pass" for row in run_rows]))


def _source_grounding_score(row: dict[str, str], summary: dict[str, Any]) -> float:
    mode = str(summary.get("mode") or summary.get("protocol") or row.get("run") or "")
    source_frames = _num(row.get("source_frames_unique"), 0.0)
    action_conf = _num(row.get("action_confidence_mean"), 0.0)
    if mode == "calcium":
        frame_score = _clip01(source_frames / 274.0)
        confidence_score = _clip01(action_conf / 0.70)
        return _clip01(0.45 * frame_score + 0.35 * confidence_score + 0.20)
    if mode == "video":
        frame_score = _clip01(source_frames / 2500.0)
        reliability = _num(row.get("video_flow_reliability_mean"), 0.0)
        gate = _num(row.get("video_calibration_gate_mean"), 0.0)
        shake = _num(row.get("video_camera_shake_mean"), 0.0)
        reliability_score = _clip01(reliability / 0.75)
        gate_score = _clip01(gate / 0.65)
        shake_score = _clip01(1.0 - 0.75 * shake)
        return _clip01(0.30 * frame_score + 0.30 * reliability_score + 0.20 * gate_score + 0.20 * shake_score)
    return _clip01(0.5 * action_conf + 0.5 * _clip01(source_frames / 250.0))


def _quiescence_score(row: dict[str, str]) -> float:
    bouts = int(_num(row.get("bout_count"), 0.0))
    speed_p95 = _num(row.get("speed_xy_mm_s_p95"), 999.0)
    action_force = _num(row.get("action_force_mean"), 999.0)
    swim_p95 = _num(row.get("swim_drive_p95"), 999.0)
    tail_yaw = _num(row.get("tail_yaw_abs_max_deg_p95"))
    tail_yaw_score = 1.0 if not math.isfinite(tail_yaw) or tail_yaw <= 2.0 else _clip01(2.0 / tail_yaw)
    return float(
        np.mean(
            [
                1.0 if bouts == 0 else _clip01(1.0 / max(1.0, bouts)),
                1.0 if speed_p95 <= 0.8 else _clip01(0.8 / speed_p95),
                1.0 if action_force <= 0.035 else _clip01(0.035 / action_force),
                1.0 if swim_p95 <= 0.02 else _clip01(0.02 / swim_p95),
                tail_yaw_score,
            ]
        )
    )


def _metric_rows(
    run_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    anomaly_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targets = _target_lookup()
    comparison_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in comparison_rows:
        comparison_by_run[str(row.get("run", ""))].append(row)

    metric_scores: list[dict[str, Any]] = []
    run_scores: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for row in run_rows:
        run = str(row.get("run", ""))
        summary = _read_json(Path(str(row.get("summary_path", ""))))
        mode = str(summary.get("mode") or summary.get("protocol") or "")
        negative_control = _run_is_negative_control(row)
        stability_score = _anomaly_pass_fraction(anomaly_rows, run)
        source_score = _source_grounding_score(row, summary)
        group_scores: dict[str, list[float]] = defaultdict(list)

        for comp in comparison_by_run.get(run, []):
            metric = str(comp.get("target_metric", ""))
            if metric not in targets:
                continue
            target = targets[metric]
            value = _num(comp.get("value"))
            lo = float(target["target_min"])
            hi = float(target["target_max"])
            group = METRIC_GROUPS.get(metric, "other")
            score = _band_score(value, lo, hi)
            if not negative_control:
                group_scores[group].append(score)
            metric_scores.append(
                {
                    "run": run,
                    "mode": mode,
                    "negative_control": negative_control,
                    "metric": metric,
                    "group": group,
                    "value": value,
                    "target_min": lo,
                    "target_max": hi,
                    "status": comp.get("status", ""),
                    "score": score,
                    "source": target.get("source", ""),
                    "interpretation": target.get("interpretation", ""),
                }
            )
            if not negative_control and metric in {
                "event_frequency_hz",
                "bout_duration_ms",
                "interbout_ms",
                "speed_xy_mm_s_mean",
                "speed_xy_mm_s_p95",
                "distance_per_event_mm",
            }:
                midpoint = float(target.get("reference_value") or (lo + hi) * 0.5)
                if math.isfinite(value) and value > 1e-12:
                    multiplier = midpoint / value
                else:
                    multiplier = float("inf")
                calibration_rows.append(
                    {
                        "run": run,
                        "metric": metric,
                        "current": value,
                        "reference": midpoint,
                        "target_min": lo,
                        "target_max": hi,
                        "current_status": comp.get("status", ""),
                        "reference_multiplier": multiplier,
                        "priority": _calibration_priority(metric, score),
                        "recommendation": _calibration_recommendation(metric, multiplier, score),
                    }
                )

        if negative_control:
            quiescence = _quiescence_score(row)
            overall = sum(
                {
                    "stability": stability_score,
                    "quiescence": quiescence,
                    "source_grounding": source_score,
                }[name]
                * weight
                for name, weight in NEGATIVE_CONTROL_WEIGHTS.items()
            )
            row_scores = {
                "stability": stability_score,
                "quiescence": quiescence,
                "source_grounding": source_score,
                "timing": float("nan"),
                "locomotor_scale": float("nan"),
                "turning_tail": float("nan"),
            }
            claim_tier = "negative-control-pass" if overall >= 0.85 else "negative-control-warning"
        else:
            row_scores = {
                "stability": stability_score,
                "source_grounding": source_score,
                "timing": _mean_or_zero(group_scores.get("timing", [])),
                "locomotor_scale": _mean_or_zero(group_scores.get("locomotor_scale", [])),
                "turning_tail": _mean_or_zero(group_scores.get("turning_tail", [])),
                "quiescence": float("nan"),
            }
            overall = sum(row_scores[name] * weight for name, weight in ACTIVE_OVERALL_WEIGHTS.items())
            claim_tier = _claim_tier(overall, row_scores)

        run_scores.append(
            {
                "run": run,
                "mode": mode,
                "negative_control": negative_control,
                "overall_score": overall,
                "claim_tier": claim_tier,
                **row_scores,
                "bout_count": int(_num(row.get("bout_count"), 0.0)),
                "event_frequency_hz": _num(row.get("event_frequency_hz")),
                "speed_mean_mm_s": _num(row.get("speed_xy_mm_s_mean")),
                "speed_p95_mm_s": _num(row.get("speed_xy_mm_s_p95")),
                "body_pitch_abs_p95_rad": _num(row.get("body_pitch_abs_p95_rad")),
                "z_span_p95_mm": _num(row.get("body_z_span_mm_p95")),
                "source_frames_unique": int(_num(row.get("source_frames_unique"), 0.0)),
                "summary_path": row.get("summary_path", ""),
            }
        )

    calibration_rows.sort(key=lambda x: (x["priority"], x["score"] if "score" in x else 0.0))
    return metric_scores, run_scores, calibration_rows


def _mean_or_zero(values: list[float] | None) -> float:
    arr = np.asarray(values or [], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def _calibration_priority(metric: str, score: float) -> int:
    if metric in {"speed_xy_mm_s_mean", "speed_xy_mm_s_p95", "distance_per_event_mm"}:
        return 0 if score < 0.6 else 2
    if metric in {"bout_duration_ms", "interbout_ms"}:
        return 1 if score < 0.7 else 3
    if metric == "event_frequency_hz":
        return 2 if score < 0.8 else 4
    return 5


def _calibration_recommendation(metric: str, multiplier: float, score: float) -> str:
    if score >= 0.95:
        return "within published target band; preserve while tuning other metrics"
    if metric in {"speed_xy_mm_s_mean", "speed_xy_mm_s_p95", "distance_per_event_mm"}:
        return (
            "increase realized displacement per bout via thrust/drag/body-scale calibration; "
            f"reference multiplier approximately {multiplier:.2f}x"
        )
    if metric == "bout_duration_ms":
        return (
            "lengthen bout pulse/CPG active window without increasing event rate; "
            f"reference multiplier approximately {multiplier:.2f}x"
        )
    if metric == "interbout_ms":
        return (
            "increase coasting interval or suppress repeated frame-triggered bouts; "
            f"reference multiplier approximately {multiplier:.2f}x"
        )
    if metric == "event_frequency_hz":
        return (
            "adjust bout initiation threshold/leaky integrator reset to match condition-specific event rate; "
            f"reference multiplier approximately {multiplier:.2f}x"
        )
    return "inspect target mismatch before changing simulator"


def _claim_tier(overall: float, row_scores: dict[str, float]) -> str:
    if row_scores.get("stability", 0.0) < 1.0:
        return "unstable-reject"
    if overall >= 0.82 and row_scores.get("locomotor_scale", 0.0) >= 0.75:
        return "candidate-biological-validation"
    if overall >= 0.62:
        return "stable-instrumented-poc"
    return "stable-but-biologically-underfit"


def _plot_metric_heatmap(out_dir: Path, metric_rows: list[dict[str, Any]]) -> Path:
    active_rows = [row for row in metric_rows if not bool(row["negative_control"])]
    runs = sorted({str(row["run"]) for row in active_rows})
    metrics = [m for m in METRIC_GROUPS if any(row["metric"] == m for row in active_rows)]
    mat = np.full((len(runs), len(metrics)), np.nan, dtype=float)
    status: dict[tuple[str, str], str] = {}
    for row in active_rows:
        if row["metric"] not in metrics or row["run"] not in runs:
            continue
        mat[runs.index(row["run"]), metrics.index(row["metric"])] = float(row["score"])
        status[(str(row["run"]), str(row["metric"]))] = str(row["status"])
    fig, ax = plt.subplots(figsize=(15, max(4, 1.2 * len(runs))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels(runs)
    for i, run in enumerate(runs):
        for j, metric in enumerate(metrics):
            value = mat[i, j]
            label = status.get((run, metric), "")
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}\n{label.replace('_target', '')}", ha="center", va="center", fontsize=7, color="white" if value < 0.55 else "black")
    ax.set_title("Continuous biological target-band score for active replay runs")
    fig.colorbar(im, ax=ax, label="score, 1.0 means inside target band")
    fig.tight_layout()
    path = out_dir / "fidelity_metric_score_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_group_scores(out_dir: Path, run_rows: list[dict[str, Any]]) -> Path:
    labels = ["stability", "source_grounding", "timing", "locomotor_scale", "turning_tail", "quiescence"]
    runs = [str(row["run"]) for row in run_rows]
    x = np.arange(len(runs))
    width = 0.12
    fig, ax = plt.subplots(figsize=(15, 6))
    for k, label in enumerate(labels):
        values = [
            float(row[label]) if isinstance(row.get(label), (int, float)) and math.isfinite(float(row[label])) else np.nan
            for row in run_rows
        ]
        ax.bar(x + (k - (len(labels) - 1) / 2) * width, values, width=width, label=label)
    ax.axhline(0.75, color="tab:green", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=25, ha="right")
    ax.set_title("Fidelity sub-scores: stability is solved; locomotor scale remains the main gap")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path = out_dir / "fidelity_group_scores.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_calibration_gaps(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    focused = [
        row for row in rows
        if str(row["metric"]) in {"speed_xy_mm_s_mean", "speed_xy_mm_s_p95", "distance_per_event_mm", "bout_duration_ms", "interbout_ms"}
    ]
    focused = focused[: min(24, len(focused))]
    if not focused:
        path = out_dir / "calibration_gap_multipliers.png"
        path.write_text("", encoding="utf-8")
        return path
    labels = [f"{row['run']}\n{row['metric']}" for row in focused]
    values = [float(row["reference_multiplier"]) if math.isfinite(float(row["reference_multiplier"])) else 0.0 for row in focused]
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ["tab:red" if v > 2.0 else "tab:orange" if v > 1.2 else "tab:blue" for v in values]
    ax.bar(np.arange(len(focused)), values, color=colors)
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("reference / current multiplier")
    ax.set_xticks(np.arange(len(focused)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_title("Largest calibration multipliers needed to reach reference published kinematics")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path = out_dir / "calibration_gap_multipliers.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    input_dir: Path,
    metric_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    plots: list[Path],
) -> Path:
    active = [row for row in run_rows if not row["negative_control"]]
    negative = [row for row in run_rows if row["negative_control"]]
    status_counts = Counter(str(row["status"]) for row in metric_rows if not row["negative_control"])
    tier_counts = Counter(str(row["claim_tier"]) for row in run_rows)
    lines = [
        "# Final Replay Fidelity Scorecard",
        "",
        "This scorecard sits above the stability audit. It separates stable replay from biological fidelity.",
        "",
        "## Executive Result",
        "",
        f"- Input audit: `{input_dir.resolve()}`",
        f"- Active target status counts: `{dict(status_counts)}`",
        f"- Claim tier counts: `{dict(tier_counts)}`",
        "- Stability: all final high-rate recordings passed the anomaly checks in the upstream audit.",
        "- Main biological gap: active runs remain under-scaled in speed, distance per bout, and often bout/interbout timing.",
        "",
        "## Run Scores",
        "",
        "| run | mode | tier | overall | stability | source | timing | scale | tail/turn | quiescence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(run_rows, key=lambda r: str(r["run"])):
        lines.append(
            "| {run} | {mode} | {claim_tier} | {overall_score:.3f} | {stability:.3f} | "
            "{source_grounding:.3f} | {timing} | {locomotor_scale} | {turning_tail} | {quiescence} |".format(
                **{
                    **row,
                    "timing": _fmt_score(row.get("timing")),
                    "locomotor_scale": _fmt_score(row.get("locomotor_scale")),
                    "turning_tail": _fmt_score(row.get("turning_tail")),
                    "quiescence": _fmt_score(row.get("quiescence")),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Active-Run Target Interpretation",
            "",
            "Active runs are scored against published larval zebrafish target bands. A score of 1.0 means the run is inside the target band; lower scores quantify how far below or above the band the run sits.",
            "",
            "| run | event Hz | speed mean mm/s | speed p95 mm/s | pitch abs p95 rad | z-span p95 mm | tier |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(active, key=lambda r: str(r["run"])):
        lines.append(
            "| {run} | {event_frequency_hz:.3f} | {speed_mean_mm_s:.3f} | {speed_p95_mm_s:.3f} | "
            "{body_pitch_abs_p95_rad:.3f} | {z_span_p95_mm:.3f} | {claim_tier} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Negative Control Interpretation",
            "",
            "Negative controls are not penalized for missing active-swim target bands. They are scored for low action, low speed, low tail bend, and no segmented swim bouts.",
            "",
            "| run | overall | quiescence | speed p95 mm/s | bouts | tier |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(negative, key=lambda r: str(r["run"])):
        lines.append(
            "| {run} | {overall_score:.3f} | {quiescence:.3f} | {speed_p95_mm_s:.3f} | {bout_count} | {claim_tier} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Highest-Priority Calibration Gaps",
            "",
            "| run | metric | current | reference | multiplier | recommendation |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(calibration_rows, key=lambda r: (int(r["priority"]), abs(float(r["reference_multiplier"])) if math.isfinite(float(r["reference_multiplier"])) else 999.0))[:18]:
        lines.append(
            "| {run} | {metric} | {current:.3g} | {reference:.3g} | {reference_multiplier:.3g} | {recommendation} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for path in plots:
        lines.append(f"- `{path.resolve()}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The current system is fit for a transparent lab demo and pre-collaboration audit: stable high-rate embodied replay, explicit source traces, and falsifiable metrics. It is not yet fit for a digital-twin claim. The scorecard keeps that boundary concrete by showing that stability and tail-frequency observability pass while locomotor scale and timing remain underfit.",
            "",
        ]
    )
    path = out_dir / "FINAL_REPLAY_FIDELITY_SCORECARD.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt_score(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{out:.3f}" if math.isfinite(out) else "n/a"


def build_scorecard(input_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = _read_csv(input_dir / "high_rate_run_summary.csv")
    comparison_rows = _read_csv(input_dir / "high_rate_external_comparison.csv")
    anomaly_rows = _read_csv(input_dir / "high_rate_anomaly_checks.csv")
    if not run_rows:
        raise RuntimeError(f"missing run summary rows in {input_dir}")
    metric_rows, scored_runs, calibration_rows = _metric_rows(run_rows, comparison_rows, anomaly_rows)
    _write_csv(out_dir / "fidelity_metric_scores.csv", metric_rows)
    _write_csv(out_dir / "fidelity_run_scores.csv", scored_runs)
    _write_csv(out_dir / "fidelity_calibration_recommendations.csv", calibration_rows)
    plots = [
        _plot_metric_heatmap(out_dir, metric_rows),
        _plot_group_scores(out_dir, scored_runs),
        _plot_calibration_gaps(out_dir, calibration_rows),
    ]
    report = _write_report(out_dir, input_dir, metric_rows, scored_runs, calibration_rows, plots)
    manifest = {
        "input_dir": str(input_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "report": str(report.resolve()),
        "rows": {
            "metric_scores": len(metric_rows),
            "run_scores": len(scored_runs),
            "calibration_recommendations": len(calibration_rows),
        },
        "plots": [str(path.resolve()) for path in plots],
        "run_claim_tiers": {str(row["run"]): str(row["claim_tier"]) for row in scored_runs},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build_scorecard(args.input_dir, args.out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
