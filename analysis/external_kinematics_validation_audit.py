"""Compare realized zebrafish lab kinematics with external larval zebrafish targets.

The existing audits test source alignment, decoded actions, behavior-target
fractions, and command-to-MuJoCo body transfer.  This script adds the next
validation layer: compare the long-run realized body telemetry against published
larval zebrafish kinematic ranges and explicitly mark quantities that the
current telemetry cannot resolve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVITY_ROOT = ROOT / "analysis" / "out" / "comprehensive_activity_study"
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "external_kinematics_validation_audit_20260603"

RUNS = [
    ("verified_calcium_all", DEFAULT_ACTIVITY_ROOT / "20260603_verified_50k" / "calcium_all"),
    (
        "verified_video_tenggol",
        DEFAULT_ACTIVITY_ROOT / "20260603_verified_50k" / "video_commons_tenggol_underwater",
    ),
    ("black_rockfish_calcium_all", DEFAULT_ACTIVITY_ROOT / "20260603_black_rockfish_50k" / "calcium_all"),
    (
        "black_rockfish_video",
        DEFAULT_ACTIVITY_ROOT / "20260603_black_rockfish_50k" / "video_commons_black_rockfish_stereo_dov",
    ),
]


KINEMATIC_SOURCES = [
    {
        "id": "zebrazoom_global",
        "title": "ZebraZoom: automated high-throughput larval zebrafish behavior analysis",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3679480/",
        "usable_targets": [
            "movement frequency 0.4495 Hz",
            "3.19 oscillations per movement",
            "24.29 Hz TBF",
            "189.5 ms movement duration",
            "51.14 deg heading range",
            "2.49 mm distance",
            "13.35 mm/s speed",
        ],
        "notes": "Search/open snippets expose the quantitative global movement statistics for 5-7 dpf larvae.",
    },
    {
        "id": "plos_lexical_bouts",
        "title": "A lexical approach for identifying behavioural action sequences",
        "url": "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009672",
        "usable_targets": [
            "bout duration 150 +/- 50 ms",
            "interbout duration 700 +/- 400 ms",
            "approximately 85,000 bouts from 171 fish",
            "speed, heading change, summed tail angle, and tail-angle PCs as bout features",
        ],
        "notes": "Open PLOS article provides direct text for these targets.",
    },
    {
        "id": "neural_speed_modulation",
        "title": "Neural control and modulation of swimming speed in larval zebrafish",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4126853/",
        "usable_targets": [
            "52,938 bouts from 45 freely swimming larvae",
            "OMR grating speeds 0-40 mm/s",
            "larvae match grating speeds up to about 20 mm/s",
            "interbout duration drops from about 1 s to about 200 ms for fast gratings",
            "fast freely swimming bouts often exceed 65 Hz maximum TBF",
        ],
        "notes": "Search/open snippets expose the core OMR kinematic relationships; direct PMC access was intermittently gated.",
    },
    {
        "id": "prey_capture_vr",
        "title": "Visually driven chaining of elementary swim patterns into prey-capture sequences",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3650304/",
        "usable_targets": [
            "larval TBF range 20-80 Hz",
            "basic burst-like tail beat about 30 Hz",
            "bout duration about 150 ms",
        ],
        "notes": "Search/open snippets expose the relevant kinematic summary; direct PMC access was intermittently gated.",
    },
    {
        "id": "spinal_projection_omr",
        "title": "Control of visually guided behavior by distinct spinal projection neurons",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2894808/",
        "usable_targets": [
            "forward swims around 30 Hz",
            "turns can bend the tail tip up to 180 deg",
            "forward mode heading change about 1.5 deg",
            "turning mode heading change about 38.9 deg",
        ],
        "notes": "Search/open snippets expose the relevant OMR swim-bout kinematics; direct PMC access was intermittently gated.",
    },
    {
        "id": "dryad_biomechanics",
        "title": "Dryad data: body torque and Strouhal number in larval zebrafish",
        "url": "https://datadryad.org/dataset/doi:10.5061/dryad.r503m",
        "usable_targets": [
            "raw video, digitized axes, body models, and tail amplitudes in a 695.95 MB dataset",
            "2 dpf frequency close to 80 Hz",
            "3-5 dpf frequency up to about 95 Hz",
            "tail amplitude 1.8-2.4 mm",
            "swimming speed one-order range roughly 25-250 mm/s",
            "Strouhal number 2.5 to 0.72 over Re 60-1400",
        ],
        "notes": "Dryad page is directly accessible and gives dataset structure plus key numerical abstract targets.",
    },
]


EXTERNAL_TARGETS = [
    {
        "metric": "event_frequency_hz",
        "target_min": 0.30,
        "target_max": 1.20,
        "reference_value": 0.4495,
        "source": "zebrazoom_global;plos_lexical_bouts;neural_speed_modulation",
        "interpretation": "spontaneous/OMR bout rate envelope from movement frequency and interbout timing",
    },
    {
        "metric": "bout_duration_ms",
        "target_min": 100.0,
        "target_max": 250.0,
        "reference_value": 150.0,
        "source": "plos_lexical_bouts;zebrazoom_global;prey_capture_vr",
        "interpretation": "short punctuated larval swim bout duration",
    },
    {
        "metric": "interbout_ms",
        "target_min": 200.0,
        "target_max": 1100.0,
        "reference_value": 700.0,
        "source": "plos_lexical_bouts;neural_speed_modulation",
        "interpretation": "rest/coast interval between bouts",
    },
    {
        "metric": "speed_xy_mm_s_mean",
        "target_min": 8.0,
        "target_max": 20.0,
        "reference_value": 13.35,
        "source": "zebrazoom_global",
        "interpretation": "mean maneuver speed for 5-7 dpf larvae",
    },
    {
        "metric": "speed_xy_mm_s_p95",
        "target_min": 20.0,
        "target_max": 250.0,
        "reference_value": 25.0,
        "source": "neural_speed_modulation;dryad_biomechanics",
        "interpretation": "fast OMR/cyclic swimming speed envelope",
    },
    {
        "metric": "distance_per_event_mm",
        "target_min": 1.0,
        "target_max": 4.0,
        "reference_value": 2.49,
        "source": "zebrazoom_global",
        "interpretation": "distance travelled per movement",
    },
    {
        "metric": "heading_delta_abs_deg",
        "target_min": 1.0,
        "target_max": 60.0,
        "reference_value": 38.9,
        "source": "zebrazoom_global;spinal_projection_omr",
        "interpretation": "forward-to-routine-turn heading range",
    },
    {
        "metric": "tail_yaw_abs_max_deg_p95",
        "target_min": 15.0,
        "target_max": 180.0,
        "reference_value": 51.14,
        "source": "zebrazoom_global;spinal_projection_omr",
        "interpretation": "tail bend amplitude envelope, broad because bout types differ",
    },
    {
        "metric": "command_tail_frequency_hz_mean",
        "target_min": 20.0,
        "target_max": 80.0,
        "reference_value": 30.0,
        "source": "prey_capture_vr;spinal_projection_omr;zebrazoom_global",
        "interpretation": "commanded target frequency should live in larval TBF range",
    },
    {
        "metric": "realized_tail_frequency_hz_observable",
        "target_min": 20.0,
        "target_max": 95.0,
        "reference_value": 30.0,
        "source": "zebrazoom_global;neural_speed_modulation;dryad_biomechanics",
        "interpretation": "actual realized TBF requires telemetry Nyquist above the larval TBF range",
    },
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_state(run_dir: Path) -> dict[str, np.ndarray]:
    rows = _read_csv(run_dir / "state_metrics.csv")
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: np.asarray([_safe_float(row.get(key), float("nan")) for row in rows], dtype=np.float64)
        for key in keys
    }


def _read_matrices(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "state_matrices.npz"
    if not path.exists():
        return {}
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for key in npz.files:
            out[key] = np.asarray(npz[key])
    return out


def _read_action_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state = row.get("state", row)
            time_s = state.get("sim_time_s", state.get("_sim_time_s", state.get("driver_video_time_s", len(rows))))
            bout_type = str(state.get("action_bout_type") or state.get("bout_type") or "coast")
            force = _safe_float(state.get("action_force", state.get("force", 0.0)))
            kick = _safe_float(state.get("action_kick", state.get("kick", 0.0)))
            rows.append(
                {
                    "time_s": _safe_float(time_s),
                    "force": force,
                    "kick": kick,
                    "side_score": _safe_float(state.get("action_side_score", state.get("side_score", 0.0))),
                    "bout_type": bout_type,
                    "noncoast": float(bout_type not in {"coast", "none", ""} and (kick >= 0.5 or force > 0.02)),
                    "tail_frequency_hz": _safe_float(state.get("tail_frequency_hz"), float("nan")),
                    "tail_amplitude": _safe_float(state.get("tail_amplitude"), float("nan")),
                }
            )
    return rows


def _unique_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[float, list[dict[str, Any]]] = {}
    for row in actions:
        buckets.setdefault(round(_safe_float(row.get("time_s")), 6), []).append(row)
    out: list[dict[str, Any]] = []
    for t in sorted(buckets):
        members = buckets[t]
        out.append(
            {
                "time_s": t,
                "force": float(np.mean([_safe_float(row.get("force")) for row in members])),
                "kick": max(_safe_float(row.get("kick")) for row in members),
                "side_score": float(np.mean([_safe_float(row.get("side_score")) for row in members])),
                "noncoast": max(_safe_float(row.get("noncoast")) for row in members),
                "tail_frequency_hz": float(
                    np.nanmean([_safe_float(row.get("tail_frequency_hz"), float("nan")) for row in members])
                ),
                "tail_amplitude": float(
                    np.nanmean([_safe_float(row.get("tail_amplitude"), float("nan")) for row in members])
                ),
                "bout_type": max((str(row.get("bout_type", "")) for row in members), key=len, default=""),
            }
        )
    return out


def _action_spans(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _unique_actions(actions)
    if not rows:
        return []
    times = np.asarray([_safe_float(row.get("time_s")) for row in rows], dtype=np.float64)
    dt = float(np.nanmedian(np.diff(times))) if times.size > 1 else 0.1
    if not math.isfinite(dt) or dt <= 0:
        dt = 0.1
    spans: list[dict[str, Any]] = []
    active = False
    start = 0.0
    end = 0.0
    members: list[dict[str, Any]] = []
    for row in rows:
        is_active = _safe_float(row.get("noncoast")) >= 0.5
        t = _safe_float(row.get("time_s"))
        if is_active and not active:
            active = True
            start = t
            end = t
            members = [row]
        elif is_active:
            end = t
            members.append(row)
        elif active:
            spans.append({"start_s": start, "end_s": end + dt, "rows": members})
            active = False
            members = []
    if active:
        spans.append({"start_s": start, "end_s": end + dt, "rows": members})
    return spans


def _interp(values_t: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    if values_t.size == 0 or values.size == 0:
        return np.asarray([])
    if values.ndim == 1:
        return np.asarray([np.interp(t, values_t, values)], dtype=np.float64)
    return np.asarray([np.interp(t, values_t, values[:, i]) for i in range(values.shape[1])], dtype=np.float64)


def _event_metrics(state: dict[str, np.ndarray], matrices: dict[str, np.ndarray], spans: list[dict[str, Any]]) -> dict[str, Any]:
    t = state.get("sim_time_s", np.asarray([], dtype=np.float64))
    heading = np.unwrap(state.get("heading_rad", np.zeros_like(t)))
    com = matrices.get("com_mm", np.zeros((t.size, 3)))
    duration_ms: list[float] = []
    interbout_ms: list[float] = []
    distance_mm: list[float] = []
    heading_delta_deg: list[float] = []
    tail_event_max_deg: list[float] = []
    prev_end: float | None = None
    for span in spans:
        start = _safe_float(span.get("start_s"))
        end = _safe_float(span.get("end_s"))
        if end <= start:
            continue
        duration_ms.append((end - start) * 1000.0)
        if prev_end is not None and start > prev_end:
            interbout_ms.append((start - prev_end) * 1000.0)
        prev_end = end
        start_com = _interp(t, com, start)
        end_com = _interp(t, com, end)
        if start_com.size and end_com.size:
            distance_mm.append(float(np.linalg.norm(end_com[:2] - start_com[:2])))
        start_heading = _interp(t, heading, start)
        end_heading = _interp(t, heading, end)
        if start_heading.size and end_heading.size:
            heading_delta_deg.append(abs(float(end_heading[0] - start_heading[0])) * 180.0 / math.pi)
        mask = (t >= start) & (t <= end)
        if np.any(mask):
            tail_event_max_deg.append(float(np.nanmax(state.get("tail_yaw_abs_max", np.zeros_like(t))[mask])) * 180.0 / math.pi)
    return {
        "bout_duration_ms_mean": float(np.nanmean(duration_ms)) if duration_ms else float("nan"),
        "bout_duration_ms_p50": float(np.nanmedian(duration_ms)) if duration_ms else float("nan"),
        "bout_duration_ms_p95": float(np.nanquantile(duration_ms, 0.95)) if duration_ms else float("nan"),
        "interbout_ms_mean": float(np.nanmean(interbout_ms)) if interbout_ms else float("nan"),
        "interbout_ms_p50": float(np.nanmedian(interbout_ms)) if interbout_ms else float("nan"),
        "distance_per_event_mm_mean": float(np.nanmean(distance_mm)) if distance_mm else float("nan"),
        "distance_per_event_mm_p50": float(np.nanmedian(distance_mm)) if distance_mm else float("nan"),
        "heading_delta_abs_deg_mean": float(np.nanmean(heading_delta_deg)) if heading_delta_deg else float("nan"),
        "heading_delta_abs_deg_p50": float(np.nanmedian(heading_delta_deg)) if heading_delta_deg else float("nan"),
        "tail_event_abs_max_deg_p95": float(np.nanquantile(tail_event_max_deg, 0.95)) if tail_event_max_deg else float("nan"),
    }


def _dominant_frequency(t: np.ndarray, y: np.ndarray, *, min_hz: float = 0.05) -> float:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 16:
        return float("nan")
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    if unique_t.size != t.size:
        sums = np.zeros(unique_t.size, dtype=np.float64)
        counts = np.zeros(unique_t.size, dtype=np.float64)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1.0)
        t = unique_t
        y = sums / np.maximum(counts, 1.0)
    if t.size < 16:
        return float("nan")
    dt = float(np.nanmedian(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0:
        return float("nan")
    y = y - float(np.nanmean(y))
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(y.size, dt)
    mask = freq >= min_hz
    if not np.any(mask):
        return float("nan")
    return float(freq[mask][int(np.argmax(spec[mask]))])


def _realized_tail_frequency(matrices: dict[str, np.ndarray], state: dict[str, np.ndarray]) -> dict[str, Any]:
    t = state.get("sim_time_s", np.asarray([], dtype=np.float64))
    tail = matrices.get("tail_yaw_rad", np.zeros((t.size, 0)))
    if t.size < 16 or tail.size == 0:
        return {
            "telemetry_sample_hz": float("nan"),
            "telemetry_nyquist_hz": float("nan"),
            "realized_tail_frequency_hz_observable": float("nan"),
            "realized_tail_frequency_status": "missing_tail_matrix",
        }
    positive_dt = np.diff(np.unique(t))
    positive_dt = positive_dt[positive_dt > 1e-9]
    dt = float(np.nanmedian(positive_dt)) if positive_dt.size else float("nan")
    sample_hz = 1.0 / dt if math.isfinite(dt) and dt > 0 else float("nan")
    nyquist = sample_hz / 2.0 if math.isfinite(sample_hz) else float("nan")
    freqs = [_dominant_frequency(t, tail[:, i]) for i in range(tail.shape[1])]
    finite_freqs = [freq for freq in freqs if math.isfinite(freq)]
    freq = float(np.median(finite_freqs)) if finite_freqs else float("nan")
    status = "measurable"
    if not math.isfinite(nyquist) or nyquist < 80.0:
        status = "undersampled_for_20_100hz_larval_tail_beats"
    return {
        "telemetry_sample_hz": sample_hz,
        "telemetry_nyquist_hz": nyquist,
        "realized_tail_frequency_hz_observable": freq,
        "realized_tail_frequency_status": status,
    }


def _percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, q)) if arr.size else float("nan")


def _mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _analyze_run(label: str, run_dir: Path) -> dict[str, Any]:
    state = _read_state(run_dir)
    matrices = _read_matrices(run_dir)
    actions = _read_action_rows(run_dir / "action_samples.jsonl")
    unique_actions = _unique_actions(actions)
    spans = _action_spans(actions)
    t = state.get("sim_time_s", np.asarray([], dtype=np.float64))
    sim_seconds = float(t[-1] - t[0]) if t.size else 0.0
    active_command_rows = [row for row in unique_actions if _safe_float(row.get("noncoast")) >= 0.5]
    command_source_rows = active_command_rows or unique_actions
    command_freqs = np.asarray(
        [_safe_float(row.get("tail_frequency_hz"), float("nan")) for row in command_source_rows],
        dtype=np.float64,
    )
    row: dict[str, Any] = {
        "run": label,
        "source_dir": str(run_dir.resolve()),
        "state_frames": int(t.size),
        "action_samples_raw": len(actions),
        "action_samples_unique": len(unique_actions),
        "event_count": len(spans),
        "sim_seconds": sim_seconds,
        "event_frequency_hz": float(len(spans) / sim_seconds) if sim_seconds > 0 else float("nan"),
        "speed_xy_mm_s_mean": _mean(state.get("speed_xy_mm_s", np.asarray([]))),
        "speed_xy_mm_s_p95": _percentile(state.get("speed_xy_mm_s", np.asarray([])), 0.95),
        "speed_3d_mm_s_mean": _mean(state.get("speed_3d_mm_s", np.asarray([]))),
        "tail_yaw_abs_max_deg_p95": _percentile(state.get("tail_yaw_abs_max", np.asarray([])), 0.95) * 180.0 / math.pi,
        "body_max_local_bend_2d_deg_p95": _percentile(
            state.get("body_max_local_bend_2d_rad", np.asarray([])), 0.95
        )
        * 180.0
        / math.pi,
        "body_z_span_mm_p95": _percentile(state.get("body_z_span_mm", np.asarray([])), 0.95),
        "command_tail_frequency_hz_mean": _mean(command_freqs),
        "command_tail_frequency_hz_p95": _percentile(command_freqs, 0.95),
    }
    row.update(_event_metrics(state, matrices, spans))
    row["bout_duration_ms"] = row.get("bout_duration_ms_p50", float("nan"))
    row["interbout_ms"] = row.get("interbout_ms_p50", float("nan"))
    row["distance_per_event_mm"] = row.get("distance_per_event_mm_p50", float("nan"))
    row["heading_delta_abs_deg"] = row.get("heading_delta_abs_deg_p50", float("nan"))
    row.update(_realized_tail_frequency(matrices, state))
    return row


def _score(metric: str, value: float, target_min: float, target_max: float, run_row: dict[str, Any]) -> str:
    if metric == "realized_tail_frequency_hz_observable" and str(run_row.get("realized_tail_frequency_status", "")) != "measurable":
        return "not_measurable_telemetry_undersampled"
    if not math.isfinite(value):
        return "missing"
    if target_min <= value <= target_max:
        return "inside_target"
    if value < target_min:
        return "below_target"
    return "above_target"


def _comparison_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_row in run_rows:
        for target in EXTERNAL_TARGETS:
            metric = str(target["metric"])
            value = _safe_float(run_row.get(metric), float("nan"))
            target_min = _safe_float(target["target_min"])
            target_max = _safe_float(target["target_max"])
            status = _score(metric, value, target_min, target_max, run_row)
            if math.isfinite(value) and value < target_min:
                signed_error = value - target_min
            elif math.isfinite(value) and value > target_max:
                signed_error = value - target_max
            else:
                signed_error = 0.0 if math.isfinite(value) else float("nan")
            rows.append(
                {
                    "run": run_row["run"],
                    "metric": metric,
                    "value": value,
                    "target_min": target_min,
                    "target_max": target_max,
                    "reference_value": _safe_float(target.get("reference_value"), float("nan")),
                    "status": status,
                    "signed_error_to_band": signed_error,
                    "source": target["source"],
                    "interpretation": target["interpretation"],
                }
            )
    return rows


def _status_code(status: str) -> int:
    return {
        "inside_target": 2,
        "below_target": -1,
        "above_target": 1,
        "not_measurable_telemetry_undersampled": -2,
        "missing": -3,
    }.get(status, 0)


def _plot_scorecard(out_dir: Path, comparison: list[dict[str, Any]]) -> Path:
    path = out_dir / "external_kinematic_target_scorecard.png"
    runs = sorted({str(row["run"]) for row in comparison})
    metrics = [str(target["metric"]) for target in EXTERNAL_TARGETS]
    mat = np.asarray(
        [
            [
                _status_code(
                    next(
                        (
                            str(row["status"])
                            for row in comparison
                            if row["run"] == run and row["metric"] == metric
                        ),
                        "missing",
                    )
                )
                for metric in metrics
            ]
            for run in runs
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(15, 7))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=2)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels(runs)
    ax.set_title("External larval zebrafish kinematic target scorecard")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([-3, -2, -1, 1, 2])
    cbar.set_ticklabels(["missing", "undersampled", "below", "above", "inside"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _target_band(ax: plt.Axes, min_value: float, max_value: float, *, label: str) -> None:
    ax.axhspan(min_value, max_value, color="#74c476", alpha=0.22, label=label)


def _plot_speed_event(out_dir: Path, run_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "speed_and_event_rate_vs_targets.png"
    labels = [str(row["run"]) for row in run_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    speed_target = next(t for t in EXTERNAL_TARGETS if t["metric"] == "speed_xy_mm_s_mean")
    rate_target = next(t for t in EXTERNAL_TARGETS if t["metric"] == "event_frequency_hz")
    _target_band(axes[0], speed_target["target_min"], speed_target["target_max"], label="ZebraZoom maneuver speed band")
    axes[0].bar(x, [_safe_float(row.get("speed_xy_mm_s_mean")) for row in run_rows], color="#2b8cbe")
    axes[0].set_ylabel("mean speed (mm/s)")
    axes[0].legend(fontsize=8)
    _target_band(axes[1], rate_target["target_min"], rate_target["target_max"], label="bout-rate/interbout band")
    axes[1].bar(x, [_safe_float(row.get("event_frequency_hz")) for row in run_rows], color="#7bccc4")
    axes[1].set_ylabel("event frequency (Hz)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    fig.suptitle("Current 50k runs versus external speed and bout-rate targets")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_bout_timing(out_dir: Path, run_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "bout_timing_vs_targets.png"
    labels = [str(row["run"]) for row in run_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    duration_target = next(t for t in EXTERNAL_TARGETS if t["metric"] == "bout_duration_ms")
    interbout_target = next(t for t in EXTERNAL_TARGETS if t["metric"] == "interbout_ms")
    _target_band(axes[0], duration_target["target_min"], duration_target["target_max"], label="bout duration band")
    axes[0].bar(x, [_safe_float(row.get("bout_duration_ms_p50")) for row in run_rows], color="#756bb1")
    axes[0].set_ylabel("median active span (ms)")
    axes[0].legend(fontsize=8)
    _target_band(axes[1], interbout_target["target_min"], interbout_target["target_max"], label="interbout band")
    axes[1].bar(x, [_safe_float(row.get("interbout_ms_p50")) for row in run_rows], color="#9e9ac8")
    axes[1].set_ylabel("median inactive interval (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    fig.suptitle("Action-derived bout timing versus larval zebrafish literature targets")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_tail_sampling(out_dir: Path, run_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "tail_frequency_sampling_limits.png"
    labels = [str(row["run"]) for row in run_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    freq_target = next(t for t in EXTERNAL_TARGETS if t["metric"] == "command_tail_frequency_hz_mean")
    _target_band(axes[0], freq_target["target_min"], freq_target["target_max"], label="larval TBF target range")
    axes[0].bar(x - 0.18, [_safe_float(row.get("command_tail_frequency_hz_mean")) for row in run_rows], width=0.36, label="commanded TBF", color="#31a354")
    axes[0].bar(x + 0.18, [_safe_float(row.get("realized_tail_frequency_hz_observable")) for row in run_rows], width=0.36, label="FFT observable median", color="#fdae6b")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].legend(fontsize=8)
    axes[1].axhspan(80, 100, color="#fb6a4a", alpha=0.18, label="needed to resolve 40-50 Hz safely")
    axes[1].bar(x, [_safe_float(row.get("telemetry_nyquist_hz")) for row in run_rows], color="#de2d26")
    axes[1].set_ylabel("telemetry Nyquist (Hz)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    fig.suptitle("Tail-beat target frequency and telemetry sampling adequacy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_heading_tail(out_dir: Path, run_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "heading_tail_amplitude_vs_targets.png"
    labels = [str(row["run"]) for row in run_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    heading_target = next(t for t in EXTERNAL_TARGETS if t["metric"] == "heading_delta_abs_deg")
    tail_target = next(t for t in EXTERNAL_TARGETS if t["metric"] == "tail_yaw_abs_max_deg_p95")
    _target_band(axes[0], heading_target["target_min"], heading_target["target_max"], label="heading-change target band")
    axes[0].bar(x, [_safe_float(row.get("heading_delta_abs_deg_p50")) for row in run_rows], color="#3182bd")
    axes[0].set_ylabel("median heading change/event (deg)")
    axes[0].legend(fontsize=8)
    _target_band(axes[1], tail_target["target_min"], tail_target["target_max"], label="tail bend broad target band")
    axes[1].bar(x, [_safe_float(row.get("tail_yaw_abs_max_deg_p95")) for row in run_rows], color="#e6550d")
    axes[1].set_ylabel("tail yaw abs-max p95 (deg)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    fig.suptitle("Heading and tail-amplitude comparison against external kinematic anchors")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(path: Path, run_rows: list[dict[str, Any]], comparison: list[dict[str, Any]], plots: list[Path]) -> None:
    status_counts = Counter(str(row.get("status", "")) for row in comparison)
    inside_counts: dict[str, int] = {}
    for row in comparison:
        if row["status"] == "inside_target":
            inside_counts[str(row["run"])] = inside_counts.get(str(row["run"]), 0) + 1
    best_run = max(run_rows, key=lambda row: inside_counts.get(str(row["run"]), 0), default={})
    lines = [
        "# External Larval Zebrafish Kinematics Validation Audit",
        "",
        "## Scope",
        "",
        "This audit compares the current >=50k MuJoCo zebrafish recordings against external larval zebrafish movement targets from ZebraZoom, OMR/prey-capture kinematic papers, and the Dryad larval swimming biomechanics dataset. It is deliberately stricter than command-level behavior matching: the question is whether the simulated body expresses measured-scale zebrafish kinematics.",
        "",
        "## Main Findings",
        "",
        f"- Runs audited: `{len(run_rows)}`.",
        f"- Target comparisons: `{len(comparison)}`.",
        f"- Status counts: `{json.dumps(dict(status_counts), sort_keys=True)}`.",
        f"- Best current run by inside-target count: `{best_run.get('run', '')}` with `{inside_counts.get(str(best_run.get('run', '')), 0)}` inside-target metrics.",
        "- Supported: video runs have behavior-like event frequency and action-derived command frequencies in the larval tail-beat range.",
        "- Not supported: realized swimming speed is far below the external larval speed anchors.",
        "- Not measurable from current telemetry: true realized 20-100 Hz tail-beat frequency, because the state recordings are undersampled for that biological frequency band.",
        "",
        "## External Source Register",
        "",
        "| source | usable target surface | URL |",
        "|---|---|---|",
    ]
    for source in KINEMATIC_SOURCES:
        lines.append(
            f"| `{source['id']}` | {', '.join(source['usable_targets'])} | {source['url']} |"
        )
    lines.extend(
        [
            "",
            "## Run-Level Kinematic Summary",
            "",
            "| run | events | event Hz | bout p50 ms | interbout p50 ms | speed mean/p95 mm/s | distance/event mm | heading/event deg | tail p95 deg | command TBF Hz | telemetry Nyquist Hz |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in run_rows:
        lines.append(
            f"| `{row['run']}` | {int(_safe_float(row.get('event_count')))} | "
            f"{_safe_float(row.get('event_frequency_hz')):.4f} | "
            f"{_safe_float(row.get('bout_duration_ms_p50')):.2f} | "
            f"{_safe_float(row.get('interbout_ms_p50')):.2f} | "
            f"{_safe_float(row.get('speed_xy_mm_s_mean')):.3f}/{_safe_float(row.get('speed_xy_mm_s_p95')):.3f} | "
            f"{_safe_float(row.get('distance_per_event_mm_p50')):.3f} | "
            f"{_safe_float(row.get('heading_delta_abs_deg_p50')):.3f} | "
            f"{_safe_float(row.get('tail_yaw_abs_max_deg_p95')):.3f} | "
            f"{_safe_float(row.get('command_tail_frequency_hz_mean')):.3f} | "
            f"{_safe_float(row.get('telemetry_nyquist_hz')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Target Comparison",
            "",
            "| run | metric | value | target band | status | source |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| `{row['run']}` | `{row['metric']}` | {_safe_float(row.get('value'), float('nan')):.4f} | "
            f"{_safe_float(row.get('target_min')):.4f}-{_safe_float(row.get('target_max')):.4f} | "
            f"`{row['status']}` | `{row['source']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current long-run recordings are adequate for slow whole-body displacement, z-span, action pulse timing, and broad tail-angle envelope checks.",
            "- They are not adequate for realized biological tail-beat validation. A 20-100 Hz larval tail beat requires high-speed body telemetry, ideally at least 200 Hz and preferably with raw segment angles saved every physics step during validation windows.",
            "- Current command target frequencies are biologically plausible, but this is not the same as proving the MuJoCo body executes those frequencies.",
            "- Current event rates for video are closer to spontaneous/OMR larval ranges than calcium replay event rates. Current speeds remain substantially below ZebraZoom/Dryad larval motion anchors, which is a concrete scale/calibration problem.",
            "",
            "## Generated Visualizations",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot.resolve()}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-root", type=Path, default=DEFAULT_ACTIVITY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    for label, default_dir in RUNS:
        run_dir = args.activity_root / default_dir.relative_to(DEFAULT_ACTIVITY_ROOT)
        if run_dir.exists():
            run_rows.append(_analyze_run(label, run_dir))
    comparison = _comparison_rows(run_rows)
    _write_csv(out_dir / "external_kinematic_run_summary.csv", run_rows)
    _write_csv(out_dir / "external_kinematic_target_comparison.csv", comparison)
    (out_dir / "external_kinematic_source_register.json").write_text(
        json.dumps(KINEMATIC_SOURCES, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (out_dir / "external_kinematic_targets.json").write_text(
        json.dumps(EXTERNAL_TARGETS, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    plots = [
        _plot_scorecard(out_dir, comparison),
        _plot_speed_event(out_dir, run_rows),
        _plot_bout_timing(out_dir, run_rows),
        _plot_tail_sampling(out_dir, run_rows),
        _plot_heading_tail(out_dir, run_rows),
    ]
    report = out_dir / "EXTERNAL_KINEMATICS_VALIDATION_AUDIT.md"
    _write_report(report, run_rows, comparison, plots)
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report.resolve()),
        "run_count": len(run_rows),
        "comparison_rows": len(comparison),
        "status_counts": dict(Counter(str(row.get("status", "")) for row in comparison)),
        "source_register": str((out_dir / "external_kinematic_source_register.json").resolve()),
        "target_register": str((out_dir / "external_kinematic_targets.json").resolve()),
        "run_summary_csv": str((out_dir / "external_kinematic_run_summary.csv").resolve()),
        "target_comparison_csv": str((out_dir / "external_kinematic_target_comparison.csv").resolve()),
        "plots": [str(plot.resolve()) for plot in plots],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
