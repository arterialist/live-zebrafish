"""Embodied replay propulsion calibration sweep.

The final fidelity scorecard shows that the replay branches are stable but
under-scaled in realized speed and distance per bout.  This script performs the
next falsifiable step: run real MuJoCo replay candidates under patched
propulsion / bout-pulse constants, score active video, active calcium, and a
low-action video negative control against the same external larval kinematic
target bands, and write an auditable calibration report.

This is intentionally an analysis layer.  It does not edit production constants;
it identifies candidate values that should be promoted only after long-run
validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[1]
ACTIVE_INFERENCE = REPO_ROOT / "active-inference"
if str(ACTIVE_INFERENCE) not in sys.path:
    sys.path.insert(0, str(ACTIVE_INFERENCE))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_kinematics_validation_audit import EXTERNAL_TARGETS  # noqa: E402
from high_rate_replay_recording import (  # noqa: E402
    DEFAULT_REPLAY_PATH,
    DEFAULT_VIDEO_CALIBRATION,
    record_replay_run,
)
from simulations.zebrafish import config as zfc  # noqa: E402


DEFAULT_OUT = ROOT / "analysis" / "out" / "replay_propulsion_calibration_20260603"
ACTIVE_VIDEO = "commons_brycon_rio_prata.mp4"
LOW_ACTION_VIDEO = "noaa_batfish_rov.mp4"
LOWER_DRIVE_VIDEO = "commons_gopro_crabbing.mp4"
DT_S = float(zfc.PHYSICS_TIMESTEP_S)

TARGET_BY_METRIC = {str(row["metric"]): row for row in EXTERNAL_TARGETS}
SCORED_ACTIVE_METRICS = [
    "event_frequency_hz",
    "bout_duration_ms",
    "interbout_ms",
    "speed_xy_mm_s_mean",
    "speed_xy_mm_s_p95",
    "distance_per_event_mm",
    "heading_delta_abs_deg",
    "tail_yaw_abs_max_deg_p95",
]
TIMING_METRICS = {"event_frequency_hz", "bout_duration_ms", "interbout_ms"}
SCALE_METRICS = {"speed_xy_mm_s_mean", "speed_xy_mm_s_p95", "distance_per_event_mm"}
TAIL_METRICS = {"heading_delta_abs_deg", "tail_yaw_abs_max_deg_p95"}


@dataclass(frozen=True)
class Candidate:
    name: str
    max_forward_accel_m_s2: float
    linear_drag_per_s: float
    calcium_action_pulse_ticks: int
    calcium_action_decay_ticks: float
    calcium_force_gain: float = 2.2
    video_action_force_scale: float = 1.0
    video_action_side_scale: float = 1.0
    video_tail_target_scale: float = 1.0
    video_confidence_scale: float = 1.0
    video_motion_scale: float = 1.0
    video_startle_scale: float = 1.0
    video_camera_shake_gate: float = 0.0
    video_reliability_floor: float = 0.35
    video_max_action_force: float = 1.0
    video_min_action_force: float = 0.0

    def patches(self) -> dict[str, float | int]:
        return {
            "MAX_FORWARD_ACCEL_M_S2": float(self.max_forward_accel_m_s2),
            "LINEAR_DRAG_PER_S": float(self.linear_drag_per_s),
            "CALCIUM_ACTION_PULSE_TICKS": int(self.calcium_action_pulse_ticks),
            "CALCIUM_ACTION_DECAY_TICKS": float(self.calcium_action_decay_ticks),
            "CALCIUM_FORCE_GAIN": float(self.calcium_force_gain),
        }

    def video_calibration(self) -> dict[str, float]:
        out = dict(DEFAULT_VIDEO_CALIBRATION)
        out.update(
            {
                "action_force_scale": float(self.video_action_force_scale),
                "action_side_scale": float(self.video_action_side_scale),
                "tail_target_scale": float(self.video_tail_target_scale),
                "confidence_scale": float(self.video_confidence_scale),
                "motion_scale": float(self.video_motion_scale),
                "startle_scale": float(self.video_startle_scale),
                "camera_shake_gate": float(self.video_camera_shake_gate),
                "reliability_floor": float(self.video_reliability_floor),
                "max_action_force": float(self.video_max_action_force),
                "min_action_force": float(self.video_min_action_force),
            }
        )
        return out


DEFAULT_CANDIDATES = [
    Candidate("current", 1.55, 9.0, 42, 22.0),
    Candidate(
        "current_mild_gate",
        1.55,
        9.0,
        42,
        22.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate("mild_thrust", 3.5, 7.5, 50, 26.0),
    Candidate(
        "mild_thrust_mild_gate",
        3.5,
        7.5,
        50,
        26.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate("balanced_thrust", 5.5, 6.5, 56, 30.0),
    Candidate(
        "balanced_thrust_mild_gate",
        5.5,
        6.5,
        56,
        30.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate(
        "balanced_thrust_mild_gate_ca4",
        5.5,
        6.5,
        56,
        30.0,
        calcium_force_gain=4.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate(
        "balanced_thrust_mild_gate_ca6",
        5.5,
        6.5,
        56,
        30.0,
        calcium_force_gain=6.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate("strong_thrust", 8.0, 5.8, 60, 34.0),
    Candidate(
        "strong_thrust_mild_gate",
        8.0,
        5.8,
        60,
        34.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate("high_impulse", 10.5, 5.0, 64, 36.0),
    Candidate(
        "high_impulse_mild_gate",
        10.5,
        5.0,
        64,
        36.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate("high_impulse_damped", 12.0, 7.0, 64, 36.0),
    Candidate(
        "high_impulse_damped_mild_gate",
        12.0,
        7.0,
        64,
        36.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate(
        "high_impulse_damped_mild_gate_ca4",
        12.0,
        7.0,
        64,
        36.0,
        calcium_force_gain=4.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate(
        "high_impulse_damped_mild_gate_ca6",
        12.0,
        7.0,
        64,
        36.0,
        calcium_force_gain=6.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
    Candidate(
        "high_impulse_damped_mild_gate_ca8",
        12.0,
        7.0,
        64,
        36.0,
        calcium_force_gain=8.0,
        video_action_force_scale=0.90,
        video_action_side_scale=0.80,
        video_tail_target_scale=0.75,
        video_confidence_scale=0.95,
        video_motion_scale=0.95,
        video_startle_scale=0.65,
        video_camera_shake_gate=0.65,
        video_reliability_floor=0.30,
        video_max_action_force=0.90,
        video_min_action_force=0.02,
    ),
]


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _stats(values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _quantile(values: Any, q: float, default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else default


def _mean(values: Any, default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else default


def _contiguous_events(mask: np.ndarray) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if bool(value) and start is None:
            start = i
        elif not bool(value) and start is not None:
            events.append((start, i))
            start = None
    if start is not None:
        events.append((start, int(mask.size)))
    return events


def _angle_wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _band_score(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return float(np.clip(value / lo, 0.0, 1.0)) if lo > 0.0 else 0.0
    return float(np.clip(hi / value, 0.0, 1.0)) if value > 0.0 else 0.0


def _metric_score(metric: str, value: float) -> float:
    target = TARGET_BY_METRIC.get(metric)
    if not target:
        return 0.0
    return _band_score(value, float(target["target_min"]), float(target["target_max"]))


def _float_row_value(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        out = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _stability_score(row: dict[str, Any]) -> float:
    """Final-audit-compatible stability score.

    The long-run validation audit treats vertical lock as a distributional
    failure, not as one transient tick.  Use p95 pitch and z-span thresholds
    from ``high_rate_tail_validation_audit.py`` and only penalize severe
    outlier ticks if they occupy a meaningful fraction of the run.
    """

    pitch_p95 = _float_row_value(row, "body_pitch_abs_p95_rad", 999.0)
    z_span_p95 = _float_row_value(row, "z_span_p95_mm", 999.0)
    severe_events = _float_row_value(row, "vertical_instability_events", 0.0)
    ticks = max(1.0, _float_row_value(row, "ticks", 1.0))
    severe_fraction = severe_events / ticks
    pitch_score = 1.0 if pitch_p95 < 0.75 else _band_score(pitch_p95, 0.0, 0.75)
    z_span_score = 1.0 if z_span_p95 < 30.0 else _band_score(z_span_p95, 0.0, 30.0)
    severe_score = 1.0 if severe_fraction < 0.02 else float(np.clip(0.02 / max(1e-9, severe_fraction), 0.0, 1.0))
    return float(np.mean([pitch_score, z_span_score, severe_score]))


@contextmanager
def _patched_config(candidate: Candidate) -> Iterator[None]:
    old: dict[str, Any] = {}
    for name, value in candidate.patches().items():
        old[name] = getattr(zfc, name)
        setattr(zfc, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(zfc, name, value)


def _run_protocol(
    candidate: Candidate,
    *,
    mode: str,
    label: str,
    ticks: int,
    seed: int,
    out_dir: Path,
    video_file_name: str,
    log_every: int,
) -> dict[str, Any]:
    with _patched_config(candidate):
        return record_replay_run(
            mode=mode,
            ticks=ticks,
            seed=seed,
            out_dir=out_dir,
            label=label,
            log_every=log_every,
            calcium_replay_path=DEFAULT_REPLAY_PATH,
            calcium_condition="all",
            calcium_gain=1.0,
            video_file_name=video_file_name,
            video_sample_hz=10.0,
            video_gain=1.0,
            video_loop_source=True,
            use_video_feature_cache=True,
            video_calibration=candidate.video_calibration(),
        )


def _load_npz(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    archive = Path(str(summary["archive_path"]))
    with np.load(archive, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _metrics_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    candidate: Candidate,
    protocol: str,
    mode: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    ticks = np.asarray(arrays.get("ticks", np.arange(0)), dtype=np.float64)
    n = max(1, int(ticks.size))
    duration_s = n * DT_S
    swim = np.asarray(arrays.get("swim_drive", np.zeros(n)), dtype=np.float64)
    speed = np.asarray(arrays.get("speed_mm_s", np.zeros(n)), dtype=np.float64)
    body_pitch = np.asarray(arrays.get("body_pitch_rad", np.zeros(n)), dtype=np.float64)
    z_span = np.asarray(arrays.get("z_span_mm", np.zeros(n)), dtype=np.float64)
    heading = np.unwrap(np.asarray(arrays.get("heading_rad", np.zeros(n)), dtype=np.float64))
    com = np.asarray(arrays.get("com_mm", np.zeros((n, 3))), dtype=np.float64)
    tail = np.asarray(arrays.get("tail_yaw_rad", np.zeros((n, 0))), dtype=np.float64)
    tail_abs = np.nanmax(np.abs(tail), axis=1) if tail.ndim == 2 and tail.size else np.zeros(n)
    action_force = np.asarray(arrays.get("action_force", np.zeros(n)), dtype=np.float64)
    action_conf = np.asarray(arrays.get("action_confidence", np.zeros(n)), dtype=np.float64)
    external_conf = np.asarray(arrays.get("external_tail_confidence", np.zeros(n)), dtype=np.float64)

    bouts = []
    prev_end: int | None = None
    for a, b in _contiguous_events(swim > 0.12):
        if b <= a:
            continue
        end = min(b - 1, n - 1)
        duration_ms = (end - a + 1) * DT_S * 1000.0
        interbout_ms = float("nan") if prev_end is None else (a - prev_end) * DT_S * 1000.0
        prev_end = end
        if com.ndim == 2 and end < com.shape[0]:
            distance_mm = float(np.linalg.norm(com[end, :2] - com[a, :2]))
        else:
            distance_mm = float("nan")
        heading_delta = 0.0
        if end < heading.size:
            heading_delta = abs(float(_angle_wrap(np.asarray([heading[end] - heading[a]]))[0])) * 180.0 / math.pi
        bouts.append(
            {
                "duration_ms": duration_ms,
                "interbout_ms": interbout_ms,
                "distance_mm": distance_mm,
                "heading_delta_abs_deg": heading_delta,
                "tail_yaw_abs_max_deg": float(np.nanmax(tail_abs[a : end + 1]) * 180.0 / math.pi),
            }
        )

    travel_path_mm = 0.0
    if com.ndim == 2 and com.shape[0] > 1:
        travel_path_mm = float(np.sum(np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1)))
    event_frequency = len(bouts) / duration_s if duration_s > 0 else float("nan")
    metrics = {
        "candidate": candidate.name,
        "protocol": protocol,
        "mode": mode,
        "seed": int(summary.get("seed", 0)),
        "ticks": n,
        "simulated_s": duration_s,
        "max_forward_accel_m_s2": candidate.max_forward_accel_m_s2,
        "linear_drag_per_s": candidate.linear_drag_per_s,
        "calcium_action_pulse_ticks": candidate.calcium_action_pulse_ticks,
        "calcium_action_decay_ticks": candidate.calcium_action_decay_ticks,
        "calcium_force_gain": candidate.calcium_force_gain,
        "video_action_force_scale": candidate.video_action_force_scale,
        "video_action_side_scale": candidate.video_action_side_scale,
        "video_tail_target_scale": candidate.video_tail_target_scale,
        "video_confidence_scale": candidate.video_confidence_scale,
        "video_camera_shake_gate": candidate.video_camera_shake_gate,
        "video_reliability_floor": candidate.video_reliability_floor,
        "video_max_action_force": candidate.video_max_action_force,
        "video_min_action_force": candidate.video_min_action_force,
        "bout_count": len(bouts),
        "event_frequency_hz": event_frequency,
        "bout_duration_ms": _quantile([b["duration_ms"] for b in bouts], 0.50),
        "interbout_ms": _quantile([b["interbout_ms"] for b in bouts], 0.50),
        "distance_per_event_mm": _quantile([b["distance_mm"] for b in bouts], 0.50),
        "heading_delta_abs_deg": _quantile([b["heading_delta_abs_deg"] for b in bouts], 0.50),
        "tail_yaw_abs_max_deg_p95": _quantile([b["tail_yaw_abs_max_deg"] for b in bouts], 0.95),
        "speed_xy_mm_s_mean": _mean(speed),
        "speed_xy_mm_s_p95": _quantile(speed, 0.95),
        "speed_xy_mm_s_max": _quantile(speed, 1.0),
        "travel_path_mm": travel_path_mm,
        "body_pitch_abs_p95_rad": _quantile(np.abs(body_pitch), 0.95),
        "z_span_p95_mm": _quantile(z_span, 0.95),
        "vertical_instability_events": int(np.sum((np.abs(body_pitch) > 0.55) | (z_span > 2.5))),
        "swim_drive_mean": _mean(swim),
        "swim_drive_p95": _quantile(swim, 0.95),
        "action_force_mean": _mean(action_force),
        "action_confidence_mean": _mean(action_conf),
        "external_tail_confidence_mean": _mean(external_conf),
        "source_frames_unique": int(np.unique(arrays.get("source_frame_index", np.asarray([-1]))[arrays.get("source_frame_index", np.asarray([-1])) >= 0]).size),
        "archive_path": str(summary.get("archive_path", "")),
        "summary_path": str(summary.get("summary_path", "")),
    }
    for metric in SCORED_ACTIVE_METRICS:
        metrics[f"{metric}_score"] = _metric_score(metric, float(metrics.get(metric, float("nan"))))
    timing = [metrics[f"{m}_score"] for m in TIMING_METRICS]
    scale = [metrics[f"{m}_score"] for m in SCALE_METRICS]
    tailing = [metrics[f"{m}_score"] for m in TAIL_METRICS]
    metrics["timing_score"] = float(np.mean(timing))
    metrics["scale_score"] = float(np.mean(scale))
    metrics["tail_turn_score"] = float(np.mean(tailing))
    metrics["stability_score"] = _stability_score(metrics)
    metrics["active_overall_score"] = (
        0.25 * metrics["stability_score"]
        + 0.25 * metrics["scale_score"]
        + 0.20 * metrics["timing_score"]
        + 0.15 * metrics["tail_turn_score"]
        + 0.15 * min(1.0, metrics["source_frames_unique"] / max(1.0, 0.05 * n))
    )
    metrics["quiescence_score"] = _quiescence_score(metrics)
    return metrics


def _quiescence_score(row: dict[str, Any]) -> float:
    parts = [
        1.0 if int(row.get("bout_count", 0)) == 0 else float(np.clip(1.0 / max(1.0, float(row["bout_count"])), 0.0, 1.0)),
        1.0 if float(row.get("speed_xy_mm_s_p95", 999.0)) <= 0.8 else float(np.clip(0.8 / max(1e-9, float(row["speed_xy_mm_s_p95"])), 0.0, 1.0)),
        1.0 if float(row.get("action_force_mean", 999.0)) <= 0.035 else float(np.clip(0.035 / max(1e-9, float(row["action_force_mean"])), 0.0, 1.0)),
        1.0 if float(row.get("swim_drive_p95", 999.0)) <= 0.02 else float(np.clip(0.02 / max(1e-9, float(row["swim_drive_p95"])), 0.0, 1.0)),
        1.0 if float(row.get("tail_yaw_abs_max_deg_p95", 999.0)) <= 2.0 else float(np.clip(2.0 / max(1e-9, float(row["tail_yaw_abs_max_deg_p95"])), 0.0, 1.0)),
    ]
    return float(np.mean(parts))


def _candidate_score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    for candidate, members in by_candidate.items():
        active = [row for row in members if "negative" not in str(row["protocol"])]
        negative = [row for row in members if "negative" in str(row["protocol"])]
        active_score = float(np.mean([_float_row_value(row, "active_overall_score", 0.0) for row in active])) if active else 0.0
        scale_score = float(np.mean([_float_row_value(row, "scale_score", 0.0) for row in active])) if active else 0.0
        timing_score = float(np.mean([_float_row_value(row, "timing_score", 0.0) for row in active])) if active else 0.0
        for row in members:
            row["stability_score"] = _stability_score(row)
        stability_score = float(np.mean([_float_row_value(row, "stability_score", 0.0) for row in members])) if members else 0.0
        quiescence_score = float(np.mean([_float_row_value(row, "quiescence_score", 0.0) for row in negative])) if negative else 0.0
        final_score = 0.58 * active_score + 0.22 * quiescence_score + 0.20 * stability_score
        out.append(
            {
                "candidate": candidate,
                "final_score": final_score,
                "active_score": active_score,
                "scale_score": scale_score,
                "timing_score": timing_score,
                "stability_score": stability_score,
                "negative_quiescence_score": quiescence_score,
                "active_protocols": len(active),
                "negative_protocols": len(negative),
                "claim": _candidate_claim(final_score, scale_score, timing_score, stability_score, quiescence_score),
            }
        )
    out.sort(key=lambda row: row["final_score"], reverse=True)
    return out


def _candidate_claim(final: float, scale: float, timing: float, stability: float, quiescence: float) -> str:
    if stability < 1.0:
        return "reject-instability"
    if quiescence < 0.85:
        return "reject-negative-control"
    if final >= 0.82 and scale >= 0.65 and timing >= 0.65:
        return "promote-to-50k-validation"
    if final >= 0.68:
        return "promising-underfit"
    return "underfit"


def _plot_candidate_scores(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        path = out_dir / "candidate_scores.png"
        path.write_text("", encoding="utf-8")
        return path
    labels = [str(row["candidate"]) for row in rows]
    x = np.arange(len(labels))
    metrics = [
        ("final_score", "final"),
        ("scale_score", "scale"),
        ("timing_score", "timing"),
        ("negative_quiescence_score", "negative"),
    ]
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (key, label) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [float(row[key]) for row in rows], width=width, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.75, color="tab:green", linestyle="--", linewidth=1)
    ax.set_ylabel("score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Replay propulsion calibration candidate scores")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(ncol=4)
    fig.tight_layout()
    path = out_dir / "candidate_scores.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_protocol_metrics(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    active = [row for row in rows if "negative" not in str(row["protocol"])]
    if not active:
        path = out_dir / "candidate_protocol_metrics.png"
        path.write_text("", encoding="utf-8")
        return path
    metrics = [
        ("speed_xy_mm_s_mean", "mean speed mm/s"),
        ("speed_xy_mm_s_p95", "p95 speed mm/s"),
        ("distance_per_event_mm", "distance/event mm"),
        ("event_frequency_hz", "event Hz"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 12), sharex=True)
    labels = [f"{row['candidate']}:{row['protocol']}" for row in active]
    x = np.arange(len(labels))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        ax.bar(x, [float(row.get(metric, 0.0)) for row in active], color="#3182bd")
        target = TARGET_BY_METRIC.get(metric)
        if target:
            ax.axhspan(float(target["target_min"]), float(target["target_max"]), color="tab:green", alpha=0.14)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.18)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    fig.suptitle("Active replay metrics versus external larval target bands", y=0.995)
    fig.tight_layout()
    path = out_dir / "candidate_protocol_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    plots: list[Path],
    *,
    ticks: int,
) -> Path:
    best = candidate_rows[0] if candidate_rows else {}
    lines = [
        "# Replay Propulsion Calibration Sweep",
        "",
        "This sweep is an embodied MuJoCo calibration layer for the two-paper video/calcium replay pipeline. It patches propulsion and calcium pulse constants in-process, runs real replay branches, and scores the resulting body telemetry against external larval zebrafish target bands.",
        "",
        "## Executive Result",
        "",
        f"- Candidate count: `{len(candidate_rows)}`",
        f"- Ticks per protocol: `{ticks}` (`{ticks * DT_S:.1f}` simulated seconds)",
        f"- Best candidate: `{best.get('candidate', 'n/a')}` with score `{float(best.get('final_score', 0.0)):.3f}` and claim `{best.get('claim', 'n/a')}`.",
        "- Promotion rule: only candidates with stable active runs and preserved low-action negative control are eligible for 50k validation.",
        "",
        "## Candidate Ranking",
        "",
        "| rank | candidate | final | active | scale | timing | stability | negative | claim |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(candidate_rows, start=1):
        lines.append(
            f"| {i} | `{row['candidate']}` | {row['final_score']:.3f} | {row['active_score']:.3f} | "
            f"{row['scale_score']:.3f} | {row['timing_score']:.3f} | {row['stability_score']:.3f} | "
            f"{row['negative_quiescence_score']:.3f} | `{row['claim']}` |"
        )
    lines.extend(
        [
            "",
            "## Protocol Metrics",
            "",
            "| candidate | protocol | bouts | event Hz | bout p50 ms | interbout p50 ms | speed mean/p95 mm/s | distance/event mm | pitch p95 rad | z-span p95 mm |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (str(r["candidate"]), str(r["protocol"]))):
        lines.append(
            f"| `{row['candidate']}` | `{row['protocol']}` | {int(row['bout_count'])} | "
            f"{float(row['event_frequency_hz']):.3f} | {float(row['bout_duration_ms']):.1f} | "
            f"{float(row['interbout_ms']):.1f} | {float(row['speed_xy_mm_s_mean']):.3f}/"
            f"{float(row['speed_xy_mm_s_p95']):.3f} | {float(row['distance_per_event_mm']):.3f} | "
            f"{float(row['body_pitch_abs_p95_rad']):.3f} | {float(row['z_span_p95_mm']):.3f} |"
        )
    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append(f"- `{plot.resolve()}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A score improvement here is evidence for a calibration direction, not final biological validity.",
            "- The next required gate is a 50k-tick high-rate validation run for the best candidate, followed by the final fidelity scorecard.",
            "- If all candidates remain scale-underfit, the bottleneck is likely the current point-force hydrodynamic approximation rather than action decoding alone.",
        ]
    )
    path = out_dir / "REPLAY_PROPULSION_CALIBRATION_SWEEP.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _maybe_delete_archive(summary: dict[str, Any], keep_recordings: bool) -> None:
    if keep_recordings:
        return
    archive = Path(str(summary.get("archive_path", "")))
    if archive.exists():
        archive.unlink()


def _candidate_by_name(name: str) -> Candidate:
    for candidate in DEFAULT_CANDIDATES:
        if candidate.name == name:
            return candidate
    raise ValueError(f"unknown candidate {name!r}; choose from {[c.name for c in DEFAULT_CANDIDATES]}")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    recording_dir = out_dir / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)
    if args.candidates:
        candidates = [_candidate_by_name(name) for name in args.candidates]
    else:
        candidates = list(DEFAULT_CANDIDATES)

    protocol_defs_all = [
        ("video_active_brycon", "video", ACTIVE_VIDEO),
        ("video_lower_drive_gopro", "video", LOWER_DRIVE_VIDEO),
        ("calcium_all", "calcium", ""),
        ("video_negative_batfish", "video", LOW_ACTION_VIDEO),
    ]
    if args.quick:
        protocol_defs_all = [
            ("video_active_brycon", "video", ACTIVE_VIDEO),
            ("calcium_all", "calcium", ""),
            ("video_negative_batfish", "video", LOW_ACTION_VIDEO),
        ]
    if args.protocol:
        wanted = set(args.protocol)
        protocol_defs = [item for item in protocol_defs_all if item[0] in wanted]
        missing = wanted - {item[0] for item in protocol_defs_all}
        if missing:
            raise ValueError(f"unknown protocol(s): {sorted(missing)}")
    else:
        protocol_defs = protocol_defs_all

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for ci, candidate in enumerate(candidates, start=1):
        for pi, (protocol, mode, video_name) in enumerate(protocol_defs, start=1):
            seed = int(args.seed + ci * 100 + pi)
            label = f"{candidate.name}_{protocol}"
            print(
                json.dumps(
                    {
                        "event": "candidate_protocol_start",
                        "candidate": candidate.name,
                        "protocol": protocol,
                        "mode": mode,
                        "seed": seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            summary = _run_protocol(
                candidate,
                mode=mode,
                label=label,
                ticks=args.ticks,
                seed=seed,
                out_dir=recording_dir,
                video_file_name=video_name or ACTIVE_VIDEO,
                log_every=args.log_every,
            )
            arrays = _load_npz(summary)
            row = _metrics_from_arrays(
                arrays,
                candidate=candidate,
                protocol=protocol,
                mode=mode,
                summary=summary,
            )
            rows.append(row)
            summaries.append(summary)
            _maybe_delete_archive(summary, bool(args.keep_recordings))
            print(
                json.dumps(
                    {
                        "event": "candidate_protocol_done",
                        "candidate": candidate.name,
                        "protocol": protocol,
                        "active_score": round(float(row["active_overall_score"]), 4),
                        "quiescence": round(float(row["quiescence_score"]), 4),
                        "speed_mean": round(float(row["speed_xy_mm_s_mean"]), 3),
                        "speed_p95": round(float(row["speed_xy_mm_s_p95"]), 3),
                        "event_hz": round(float(row["event_frequency_hz"]), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    candidate_rows = _candidate_score(rows)
    plots = [
        _plot_candidate_scores(out_dir, candidate_rows),
        _plot_protocol_metrics(out_dir, rows),
    ]
    _write_csv(out_dir / "protocol_metrics.csv", rows)
    _write_csv(out_dir / "candidate_scores.csv", candidate_rows)
    (out_dir / "run_summaries.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report = _write_report(out_dir, rows, candidate_rows, plots, ticks=int(args.ticks))
    manifest = {
        "out_dir": str(out_dir),
        "report": str(report),
        "protocol_metrics": str(out_dir / "protocol_metrics.csv"),
        "candidate_scores": str(out_dir / "candidate_scores.csv"),
        "plots": [str(path) for path in plots],
        "ticks": int(args.ticks),
        "candidates": [candidate.name for candidate in candidates],
        "wall_seconds": time.perf_counter() - t0,
        "best_candidate": candidate_rows[0] if candidate_rows else {},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return manifest


def rescore_existing(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    rows = _read_csv(args.rescore_from)
    for row in rows:
        for metric in SCORED_ACTIVE_METRICS:
            row[f"{metric}_score"] = _metric_score(metric, _float_row_value(row, metric))
        row["timing_score"] = float(np.mean([_float_row_value(row, f"{m}_score", 0.0) for m in TIMING_METRICS]))
        row["scale_score"] = float(np.mean([_float_row_value(row, f"{m}_score", 0.0) for m in SCALE_METRICS]))
        row["tail_turn_score"] = float(np.mean([_float_row_value(row, f"{m}_score", 0.0) for m in TAIL_METRICS]))
        row["stability_score"] = _stability_score(row)
        row["active_overall_score"] = (
            0.25 * _float_row_value(row, "stability_score", 0.0)
            + 0.25 * _float_row_value(row, "scale_score", 0.0)
            + 0.20 * _float_row_value(row, "timing_score", 0.0)
            + 0.15 * _float_row_value(row, "tail_turn_score", 0.0)
            + 0.15 * min(1.0, _float_row_value(row, "source_frames_unique", 0.0) / max(1.0, 0.05 * _float_row_value(row, "ticks", 1.0)))
        )
        row["quiescence_score"] = _quiescence_score(row)
    candidate_rows = _candidate_score(rows)
    plots = [
        _plot_candidate_scores(out_dir, candidate_rows),
        _plot_protocol_metrics(out_dir, rows),
    ]
    _write_csv(out_dir / "protocol_metrics.csv", rows)
    _write_csv(out_dir / "candidate_scores.csv", candidate_rows)
    report = _write_report(out_dir, rows, candidate_rows, plots, ticks=int(args.ticks))
    manifest = {
        "out_dir": str(out_dir),
        "report": str(report),
        "protocol_metrics": str(out_dir / "protocol_metrics.csv"),
        "candidate_scores": str(out_dir / "candidate_scores.csv"),
        "plots": [str(path) for path in plots],
        "ticks": int(args.ticks),
        "source_metrics": str(args.rescore_from.resolve()),
        "best_candidate": candidate_rows[0] if candidate_rows else {},
    }
    (out_dir / "manifest_rescored.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ticks", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--candidate", dest="candidates", action="append", help="candidate name; repeat to subset")
    parser.add_argument("--quick", action="store_true", help="skip the lower-drive active video protocol")
    parser.add_argument("--keep-recordings", action="store_true", help="retain heavy candidate npz archives")
    parser.add_argument("--rescore-from", type=Path, help="recompute scores from an existing protocol_metrics.csv")
    parser.add_argument("--protocol", action="append", help="protocol name to run; repeat to subset")
    args = parser.parse_args()
    manifest = rescore_existing(args) if args.rescore_from else run_sweep(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
