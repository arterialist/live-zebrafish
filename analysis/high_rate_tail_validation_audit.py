"""High-rate zebrafish long-run activity and tail-beat validation.

This post-processes per-physics-tick archives produced by
``zebrafish_long_recording.py``.  The browser/WebSocket captures are sufficient
for UI and command/body coupling, but they are too sparse to resolve larval
20-95 Hz tail beats.  This audit uses the 200 Hz MuJoCo recorder to quantify
every stored scalar stream, every matrix channel, bout timing, tail spectra,
neural/muscle coupling, and external larval kinematic target agreement over
large time spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from external_kinematics_validation_audit import EXTERNAL_TARGETS, KINEMATIC_SOURCES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS = ROOT / "analysis" / "out" / "high_rate_tail_validation_20260603" / "recordings"
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "high_rate_tail_validation_20260603"
DT_S_DEFAULT = 0.005


@dataclass
class Recording:
    label: str
    archive: Path
    summary_path: Path | None
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def ticks(self) -> np.ndarray:
        return np.asarray(self.arrays.get("ticks", np.zeros(0)), dtype=np.float64)

    @property
    def dt_s(self) -> float:
        value = self.summary.get("dt_s", DT_S_DEFAULT)
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = DT_S_DEFAULT
        return out if math.isfinite(out) and out > 0 else DT_S_DEFAULT

    @property
    def time_s(self) -> np.ndarray:
        ticks = self.ticks
        if ticks.size == 0:
            n = max((arr.shape[0] for arr in self.arrays.values() if arr.ndim > 0), default=0)
            return np.arange(n, dtype=np.float64) * self.dt_s
        return (ticks - ticks[0]) * self.dt_s


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _corr(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(aa.size, bb.size)
    if n < 8:
        return 0.0
    aa = aa[:n]
    bb = bb[:n]
    mask = np.isfinite(aa) & np.isfinite(bb)
    if np.sum(mask) < 8:
        return 0.0
    aa = aa[mask] - float(np.mean(aa[mask]))
    bb = bb[mask] - float(np.mean(bb[mask]))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 1e-12 else 0.0


def _rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or samples <= 1:
        return arr
    samples = min(samples, arr.size)
    kernel = np.ones(samples, dtype=np.float64)
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(np.isfinite(arr), arr, 0.0)
    return np.convolve(filled, kernel, mode="same") / np.maximum(
        1.0, np.convolve(valid, kernel, mode="same")
    )


def _angle_wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


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
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _find_summary(archive: Path) -> Path | None:
    candidates = [
        archive.with_name(archive.name.removesuffix(".npz") + "_summary.json"),
        archive.with_suffix("").with_name(archive.stem + "_summary.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_recording(archive: Path) -> Recording:
    arrays: dict[str, np.ndarray] = {}
    with np.load(archive, allow_pickle=True) as npz:
        for key in npz.files:
            arrays[key] = np.asarray(npz[key])
    summary_path = _find_summary(archive)
    summary = _read_json(summary_path) if summary_path else {}
    label = str(summary.get("label") or archive.stem)
    seed = summary.get("seed")
    protocol = str(summary.get("protocol") or label)
    label = f"{protocol}_seed{seed}" if seed is not None and not label.endswith(str(seed)) else label
    return Recording(label=label, archive=archive, summary_path=summary_path, summary=summary, arrays=arrays)


def _load_recordings(recordings_dir: Path, glob_pattern: str) -> list[Recording]:
    archives = sorted(recordings_dir.glob(glob_pattern))
    return [_load_recording(path) for path in archives]


def _matrix_names(recording: Recording, key: str, width: int) -> list[str]:
    summary = recording.summary
    if key == "muscles":
        names = list(summary.get("muscle_names") or recording.arrays.get("muscle_names", []))
    elif key in {"neuron_s", "neuron_r", "neuron_o", "neuron_fired"}:
        names = list(summary.get("neuron_names") or recording.arrays.get("neuron_names", []))
    elif key in {"qpos", "qvel"}:
        names = [f"{key}_{i:02d}" for i in range(width)]
    elif key in {"tail_yaw_rad", "tail_pitch_rad"}:
        names = [f"{key}_{i:02d}" for i in range(width)]
    elif key == "body_points_mm":
        names = [f"body_point_{i:02d}_{axis}" for i in range(width // 3) for axis in ("x", "y", "z")]
    elif key == "neuromod":
        names = ["neuromod_m0", "neuromod_m1"]
    elif key == "water_flow_m_s":
        names = ["flow_x_m_s", "flow_y_m_s"]
    elif key == "action_tail_targets":
        names = [f"action_tail_target_{i:02d}" for i in range(width)]
    else:
        names = [f"{key}_{i:02d}" for i in range(width)]
    out = [str(name) for name in names]
    if len(out) < width:
        out.extend(f"{key}_{i:02d}" for i in range(len(out), width))
    return out[:width]


def _scalar_streams(rec: Recording) -> dict[str, np.ndarray]:
    arrays = rec.arrays
    out: dict[str, np.ndarray] = {}
    scalar_keys = [
        "speed_mm_s",
        "vertical_speed_mm_s",
        "swim_drive",
        "turn_bias",
        "pitch_bias",
        "heading_rad",
        "yaw_rate_rad_s",
        "body_pitch_rad",
        "z_span_mm",
        "straightness",
        "abs_curvature_3d_rad",
        "max_local_bend_3d_rad",
        "stimulus_code",
        "startle_command",
        "depth_perturbation",
        "source_frame_index",
        "source_row",
        "source_time_s",
        "source_loop_index",
        "action_active",
        "action_kick",
        "action_force",
        "action_side_score",
        "action_kick_score",
        "action_confidence",
        "action_tail_frequency_hz",
        "action_tail_amplitude",
        "action_tail_target_confidence",
        "calcium_active",
        "calcium_pulse",
        "calcium_age_ticks",
        "calcium_force",
        "calcium_side_score",
        "calcium_kick_score",
        "calcium_confidence",
        "video_active",
        "video_motion_energy",
        "video_asymmetry",
        "video_visual_left",
        "video_visual_right",
        "video_visual_up",
        "video_visual_down",
        "video_optic_flow_left",
        "video_optic_flow_right",
        "video_lateral_line_left",
        "video_lateral_line_right",
        "video_light_level",
        "video_startle",
        "video_action_force",
        "video_action_side_score",
        "video_action_confidence",
        "video_flow_reliability",
        "video_camera_shake",
        "video_camera_motion",
        "video_compression_noise",
        "video_zapbench_distance",
        "video_frame_cache_hit",
        "video_calibration_gate",
        "video_calibration_raw_action_force",
        "video_calibration_raw_action_side_score",
        "video_calibration_raw_action_confidence",
        "video_calibration_raw_startle",
        "video_calibration_raw_motion_energy",
        "external_tail_confidence",
        "bout_ticks_left",
        "coast_ticks_left",
        "startle_ticks_left",
    ]
    for key in scalar_keys:
        arr = arrays.get(key)
        if arr is not None and arr.ndim == 1:
            out[key] = np.asarray(arr, dtype=np.float64)
    for key, arr in arrays.items():
        mat = np.asarray(arr)
        if mat.ndim == 2 and mat.shape[1] <= 4 and key not in {"qpos", "qvel"}:
            names = _matrix_names(rec, key, mat.shape[1])
            for i, name in enumerate(names):
                out[name] = np.asarray(mat[:, i], dtype=np.float64)
        elif key == "com_mm" and mat.ndim == 2 and mat.shape[1] >= 3:
            out["com_x_mm"] = np.asarray(mat[:, 0], dtype=np.float64)
            out["com_y_mm"] = np.asarray(mat[:, 1], dtype=np.float64)
            out["com_z_mm"] = np.asarray(mat[:, 2], dtype=np.float64)
        elif key in {
            "tail_yaw_rad",
            "tail_pitch_rad",
            "muscles",
            "neuron_s",
            "neuron_r",
            "neuron_o",
            "action_tail_targets",
        }:
            m = np.asarray(mat, dtype=np.float64)
            if m.ndim == 2 and m.size:
                out[f"{key}_mean"] = np.nanmean(m, axis=1)
                out[f"{key}_abs_mean"] = np.nanmean(np.abs(m), axis=1)
                out[f"{key}_rms"] = np.sqrt(np.nanmean(m * m, axis=1))
                out[f"{key}_abs_max"] = np.nanmax(np.abs(m), axis=1)
                out[f"{key}_sum"] = np.nansum(m, axis=1)
        elif key == "neuron_fired":
            m = np.asarray(mat, dtype=np.float64)
            if m.ndim == 2 and m.size:
                out["neuron_fired_count"] = np.nansum(m, axis=1)
                out["neuron_fired_fraction"] = np.nanmean(m, axis=1)
    com = np.asarray(arrays.get("com_mm", np.zeros((0, 3))), dtype=np.float64)
    if com.ndim == 2 and com.shape[0] > 1 and com.shape[1] >= 3:
        d = np.diff(com, axis=0, prepend=com[:1])
        out["com_speed_xy_mm_s_from_pos"] = np.linalg.norm(d[:, :2], axis=1) / rec.dt_s
        out["com_speed_3d_mm_s_from_pos"] = np.linalg.norm(d, axis=1) / rec.dt_s
        out["com_xy_displacement_mm"] = np.linalg.norm(com[:, :2] - com[0, :2], axis=1)
        out["com_xyz_displacement_mm"] = np.linalg.norm(com - com[0], axis=1)
    body = np.asarray(arrays.get("body_points_mm", np.zeros((0, 0, 3))), dtype=np.float64)
    if body.ndim == 3 and body.shape[0] > 0 and body.shape[1] > 1:
        out["body_length_mm"] = np.sum(np.linalg.norm(np.diff(body, axis=1), axis=2), axis=1)
        out["body_xy_width_mm"] = np.ptp(body[:, :, 1], axis=1)
        out["body_z_span_from_points_mm"] = np.ptp(body[:, :, 2], axis=1)
    return out


def _all_scalar_stats(recordings: list[Recording]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stat_names = list(_stats([]))
    for rec in recordings:
        t = rec.time_s
        for name, values in _scalar_streams(rec).items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            stats = _stats(arr)
            finite = np.isfinite(arr)
            if np.any(finite):
                idx_min = int(np.nanargmin(arr))
                idx_max = int(np.nanargmax(arr))
                min_time = float(t[idx_min]) if idx_min < t.size else 0.0
                max_time = float(t[idx_max]) if idx_max < t.size else 0.0
            else:
                min_time = 0.0
                max_time = 0.0
            rows.append(
                {
                    "run": rec.label,
                    "metric": name,
                    "min_time_s": min_time,
                    "max_time_s": max_time,
                    **{key: stats[key] for key in stat_names},
                }
            )
    return rows


def _channel_matrix(arr: np.ndarray) -> np.ndarray:
    m = np.asarray(arr)
    if m.ndim == 1:
        return m.reshape((-1, 1)).astype(np.float64)
    if m.ndim == 2:
        return m.astype(np.float64)
    if m.ndim == 3:
        return m.reshape((m.shape[0], -1)).astype(np.float64)
    return np.zeros((0, 0), dtype=np.float64)


def _matrix_channel_stats(recordings: list[Recording]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix_keys = [
        "qpos",
        "qvel",
        "com_mm",
        "body_points_mm",
        "tail_yaw_rad",
        "tail_pitch_rad",
        "muscles",
        "neuron_s",
        "neuron_r",
        "neuron_o",
        "neuron_fired",
        "neuromod",
        "water_flow_m_s",
        "action_tail_targets",
    ]
    for rec in recordings:
        t = rec.time_s
        scalars = _scalar_streams(rec)
        speed = scalars.get("speed_mm_s", scalars.get("com_speed_xy_mm_s_from_pos", np.zeros(t.size)))
        swim = scalars.get("swim_drive", np.zeros(t.size))
        bend = scalars.get("tail_yaw_rad_abs_max", scalars.get("tail_yaw_rad_abs_mean", np.zeros(t.size)))
        for key in matrix_keys:
            if key not in rec.arrays:
                continue
            mat = _channel_matrix(rec.arrays[key])
            if mat.size == 0:
                continue
            names = _matrix_names(rec, key, mat.shape[1])
            threshold = max(1e-10, float(np.nanquantile(np.abs(mat), 0.95)) * 0.1)
            for i in range(mat.shape[1]):
                y = mat[:, i]
                stats = _stats(y)
                peak = int(np.nanargmax(np.abs(y))) if np.any(np.isfinite(y)) else 0
                rows.append(
                    {
                        "run": rec.label,
                        "stream": key,
                        "channel_index": i,
                        "channel_name": names[i],
                        **stats,
                        "active_fraction": float(np.nanmean(np.abs(y) > threshold)),
                        "peak_time_s": float(t[peak]) if peak < t.size else 0.0,
                        "corr_speed_mm_s": _corr(y, speed),
                        "corr_swim_drive": _corr(y, swim),
                        "corr_tail_bend": _corr(y, bend),
                    }
                )
    return rows


def _stimulus_summary(recordings: list[Recording]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in recordings:
        t = rec.time_s
        dt_s = rec.dt_s
        code = np.asarray(rec.arrays.get("stimulus_code", np.zeros(t.size)), dtype=np.int64)
        if code.size == 0 and t.size:
            code = np.zeros(t.size, dtype=np.int64)
        counts = Counter(int(v) for v in code)
        for stimulus_code, count in sorted(counts.items()):
            mask = code == stimulus_code
            rows.append(
                {
                    "run": rec.label,
                    "stimulus_code": stimulus_code,
                    "ticks": int(count),
                    "duration_s": float(count * dt_s),
                    "fraction": float(count / max(1, code.size)),
                    "speed_mm_s_mean": _safe_mean(np.asarray(rec.arrays.get("speed_mm_s", []))[mask]),
                    "swim_drive_mean": _safe_mean(np.asarray(rec.arrays.get("swim_drive", []))[mask]),
                    "tail_yaw_abs_max_rad_p95": _safe_quantile(
                        np.nanmax(np.abs(np.asarray(rec.arrays.get("tail_yaw_rad", np.zeros((code.size, 0))))), axis=1)[
                            mask
                        ],
                        0.95,
                    )
                    if "tail_yaw_rad" in rec.arrays and np.any(mask)
                    else float("nan"),
                }
            )
    return rows


def _bout_rows(rec: Recording) -> list[dict[str, Any]]:
    t = rec.time_s
    swim = np.asarray(rec.arrays.get("swim_drive", np.zeros(t.size)), dtype=np.float64)
    turn = np.asarray(rec.arrays.get("turn_bias", np.zeros(t.size)), dtype=np.float64)
    pitch = np.asarray(rec.arrays.get("pitch_bias", np.zeros(t.size)), dtype=np.float64)
    speed = np.asarray(rec.arrays.get("speed_mm_s", np.zeros(t.size)), dtype=np.float64)
    heading = np.unwrap(np.asarray(rec.arrays.get("heading_rad", np.zeros(t.size)), dtype=np.float64))
    com = np.asarray(rec.arrays.get("com_mm", np.zeros((t.size, 3))), dtype=np.float64)
    tail = np.asarray(rec.arrays.get("tail_yaw_rad", np.zeros((t.size, 0))), dtype=np.float64)
    tail_abs = np.nanmax(np.abs(tail), axis=1) if tail.ndim == 2 and tail.size else np.zeros(t.size)
    rows: list[dict[str, Any]] = []
    prev_end: int | None = None
    active = swim > 0.12
    for idx, (a, b) in enumerate(_contiguous_events(active)):
        if b <= a or a >= t.size:
            continue
        end = min(b, t.size - 1)
        duration_ms = (end - a + 1) * rec.dt_s * 1000.0
        if prev_end is None:
            interbout_ms = float("nan")
        else:
            interbout_ms = (a - prev_end) * rec.dt_s * 1000.0
        prev_end = end
        distance_mm = 0.0
        if com.ndim == 2 and end < com.shape[0]:
            distance_mm = float(np.linalg.norm(com[end, :2] - com[a, :2]))
        heading_delta_deg = 0.0
        if end < heading.size:
            heading_delta_deg = abs(float(heading[end] - heading[a])) * 180.0 / math.pi
        rows.append(
            {
                "run": rec.label,
                "bout_index": idx,
                "start_tick": int(rec.ticks[a]) if a < rec.ticks.size else a,
                "end_tick": int(rec.ticks[end]) if end < rec.ticks.size else end,
                "start_s": float(t[a]) if a < t.size else a * rec.dt_s,
                "end_s": float(t[end]) if end < t.size else end * rec.dt_s,
                "duration_ms": duration_ms,
                "interbout_ms": interbout_ms,
                "distance_mm": distance_mm,
                "heading_delta_abs_deg": heading_delta_deg,
                "speed_peak_mm_s": float(np.nanmax(speed[a : end + 1])) if speed.size else 0.0,
                "speed_mean_mm_s": float(np.nanmean(speed[a : end + 1])) if speed.size else 0.0,
                "swim_drive_peak": float(np.nanmax(swim[a : end + 1])),
                "turn_bias_mean": float(np.nanmean(turn[a : end + 1])) if turn.size else 0.0,
                "pitch_bias_mean": float(np.nanmean(pitch[a : end + 1])) if pitch.size else 0.0,
                "tail_yaw_abs_max_deg": float(np.nanmax(tail_abs[a : end + 1]) * 180.0 / math.pi)
                if tail_abs.size
                else 0.0,
                "classification": _classify_bout(turn[a : end + 1], pitch[a : end + 1], tail_abs[a : end + 1]),
            }
        )
    return rows


def _classify_bout(turn: np.ndarray, pitch: np.ndarray, tail_abs: np.ndarray) -> str:
    turn_mean = float(np.nanmean(turn)) if turn.size else 0.0
    pitch_mean = float(np.nanmean(pitch)) if pitch.size else 0.0
    tail_peak = float(np.nanmax(tail_abs)) if tail_abs.size else 0.0
    if tail_peak > 0.22 and abs(turn_mean) > 0.20:
        return "high_bend_turn"
    if turn_mean > 0.20:
        return "right_turn"
    if turn_mean < -0.20:
        return "left_turn"
    if pitch_mean > 0.12:
        return "climb"
    if pitch_mean < -0.12:
        return "dive"
    return "forward_or_coast_bout"


def _all_bouts(recordings: list[Recording]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in recordings:
        rows.extend(_bout_rows(rec))
    return rows


def _dominant_frequency(y: np.ndarray, dt_s: float, min_hz: float, max_hz: float) -> tuple[float, float]:
    arr = np.asarray(y, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 16:
        return float("nan"), float("nan")
    arr = arr - float(np.mean(arr))
    if np.max(np.abs(arr)) <= 1e-12:
        return float("nan"), float("nan")
    win = np.hanning(arr.size)
    spec = np.abs(np.fft.rfft(arr * win)) ** 2
    freq = np.fft.rfftfreq(arr.size, dt_s)
    mask = (freq >= min_hz) & (freq <= max_hz)
    if not np.any(mask):
        return float("nan"), float("nan")
    idx = int(np.argmax(spec[mask]))
    f = float(freq[mask][idx])
    power = float(spec[mask][idx] / max(1e-12, np.sum(spec[mask])))
    return f, power


def _spectral_rows(recordings: list[Recording]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segment_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for rec in recordings:
        tail = np.asarray(rec.arrays.get("tail_yaw_rad", np.zeros((0, 0))), dtype=np.float64)
        if tail.ndim != 2 or tail.size == 0:
            continue
        swim = np.asarray(rec.arrays.get("swim_drive", np.zeros(tail.shape[0])), dtype=np.float64)
        sample_hz = 1.0 / rec.dt_s
        nyquist = sample_hz / 2.0
        for seg in range(tail.shape[1]):
            y = tail[:, seg]
            f_all, p_all = _dominant_frequency(y, rec.dt_s, 0.5, min(99.0, nyquist))
            f_tbf, p_tbf = _dominant_frequency(y, rec.dt_s, 20.0, min(95.0, nyquist))
            active = swim > 0.12
            f_active, p_active = (
                _dominant_frequency(y[active], rec.dt_s, 20.0, min(95.0, nyquist))
                if np.sum(active) > 64
                else (float("nan"), float("nan"))
            )
            segment_rows.append(
                {
                    "run": rec.label,
                    "segment": seg,
                    "sample_hz": sample_hz,
                    "nyquist_hz": nyquist,
                    "dominant_all_hz": f_all,
                    "dominant_all_power_fraction": p_all,
                    "dominant_tbf_band_hz": f_tbf,
                    "dominant_tbf_band_power_fraction": p_tbf,
                    "dominant_active_tbf_band_hz": f_active,
                    "dominant_active_tbf_band_power_fraction": p_active,
                    "tail_yaw_abs_mean_rad": float(np.nanmean(np.abs(y))),
                    "tail_yaw_abs_p95_rad": float(np.nanquantile(np.abs(y), 0.95)),
                    "tail_yaw_abs_max_rad": float(np.nanmax(np.abs(y))),
                }
            )
        win = max(128, int(round(1.0 / rec.dt_s)))
        stride = max(32, win // 4)
        for start in range(0, max(0, tail.shape[0] - win + 1), stride):
            end = start + win
            t_mid = (start + end - 1) * 0.5 * rec.dt_s
            y = np.nanmean(tail[start:end], axis=1)
            f, p = _dominant_frequency(y, rec.dt_s, 1.0, min(95.0, nyquist))
            f_tbf, p_tbf = _dominant_frequency(y, rec.dt_s, 20.0, min(95.0, nyquist))
            window_rows.append(
                {
                    "run": rec.label,
                    "start_s": start * rec.dt_s,
                    "end_s": end * rec.dt_s,
                    "mid_s": t_mid,
                    "dominant_hz": f,
                    "dominant_power_fraction": p,
                    "dominant_tbf_band_hz": f_tbf,
                    "dominant_tbf_band_power_fraction": p_tbf,
                    "swim_drive_mean": float(np.nanmean(swim[start:end])) if swim.size >= end else 0.0,
                    "tail_abs_mean_rad": float(np.nanmean(np.abs(tail[start:end]))),
                    "tail_abs_p95_rad": float(np.nanquantile(np.abs(tail[start:end]), 0.95)),
                }
            )
    return segment_rows, window_rows


def _window_summary(recordings: list[Recording], window_s: float) -> list[dict[str, Any]]:
    metrics = [
        "speed_mm_s",
        "vertical_speed_mm_s",
        "swim_drive",
        "turn_bias",
        "pitch_bias",
        "yaw_rate_rad_s",
        "body_pitch_rad",
        "z_span_mm",
        "straightness",
        "abs_curvature_3d_rad",
        "max_local_bend_3d_rad",
        "tail_yaw_rad_abs_mean",
        "tail_yaw_rad_abs_max",
        "tail_pitch_rad_abs_mean",
        "muscles_sum",
        "neuron_s_abs_mean",
        "neuron_r_abs_mean",
        "neuron_o_abs_mean",
        "neuron_fired_count",
        "neuromod_m0",
        "neuromod_m1",
        "action_active",
        "action_force",
        "action_side_score",
        "action_confidence",
        "action_tail_frequency_hz",
        "action_tail_amplitude",
        "action_tail_target_confidence",
        "action_tail_targets_abs_mean",
        "calcium_pulse",
        "calcium_force",
        "calcium_side_score",
        "video_motion_energy",
        "video_asymmetry",
        "video_flow_reliability",
        "video_camera_shake",
        "video_zapbench_distance",
        "video_calibration_gate",
        "video_calibration_raw_action_force",
        "video_calibration_raw_startle",
        "external_tail_confidence",
    ]
    rows: list[dict[str, Any]] = []
    for rec in recordings:
        t = rec.time_s
        if t.size == 0:
            continue
        streams = _scalar_streams(rec)
        n_windows = int(math.ceil((float(t[-1]) + rec.dt_s) / window_s))
        for i in range(n_windows):
            start = i * window_s
            end = min(float(t[-1]), (i + 1) * window_s)
            mask = (t >= start) & (t < end if i < n_windows - 1 else t <= end)
            if not np.any(mask):
                continue
            for metric in metrics:
                arr = streams.get(metric)
                if arr is None:
                    continue
                vals = np.asarray(arr, dtype=np.float64)[mask]
                rows.append(
                    {
                        "run": rec.label,
                        "window_s": window_s,
                        "window_index": i,
                        "start_s": start,
                        "end_s": end,
                        "metric": metric,
                        "mean": _safe_mean(vals),
                        "p95": _safe_quantile(vals, 0.95),
                        "max": _safe_quantile(vals, 1.0),
                    }
                )
    return rows


def _run_summary_rows(
    recordings: list[Recording],
    bout_rows: list[dict[str, Any]],
    spectral_segment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_run_bouts: dict[str, list[dict[str, Any]]] = {}
    for row in bout_rows:
        by_run_bouts.setdefault(str(row["run"]), []).append(row)
    by_run_spectral: dict[str, list[dict[str, Any]]] = {}
    for row in spectral_segment_rows:
        by_run_spectral.setdefault(str(row["run"]), []).append(row)
    for rec in recordings:
        t = rec.time_s
        duration_s = float(t[-1] - t[0] + rec.dt_s) if t.size else 0.0
        bouts = by_run_bouts.get(rec.label, [])
        spec = by_run_spectral.get(rec.label, [])
        scalars = _scalar_streams(rec)
        com = np.asarray(rec.arrays.get("com_mm", np.zeros((0, 3))), dtype=np.float64)
        travel_mm = 0.0
        displacement_mm = 0.0
        if com.ndim == 2 and com.shape[0] > 1:
            travel_mm = float(np.sum(np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1)))
            displacement_mm = float(np.linalg.norm(com[-1, :2] - com[0, :2]))
        rows.append(
            {
                "run": rec.label,
                "archive": str(rec.archive.resolve()),
                "summary_path": str(rec.summary_path.resolve()) if rec.summary_path else "",
                "ticks": int(t.size),
                "dt_s": rec.dt_s,
                "sample_hz": 1.0 / rec.dt_s,
                "nyquist_hz": 0.5 / rec.dt_s,
                "simulated_s": duration_s,
                "meets_50k_ticks": bool(t.size >= 50_000),
                "meets_tbf_nyquist": bool(0.5 / rec.dt_s >= 95.0),
                "bout_count": len(bouts),
                "event_frequency_hz": len(bouts) / duration_s if duration_s > 0 else float("nan"),
                "bout_duration_ms_p50": _safe_quantile([row["duration_ms"] for row in bouts], 0.50),
                "bout_duration_ms_p95": _safe_quantile([row["duration_ms"] for row in bouts], 0.95),
                "interbout_ms_p50": _safe_quantile([row["interbout_ms"] for row in bouts], 0.50),
                "distance_per_bout_mm_p50": _safe_quantile([row["distance_mm"] for row in bouts], 0.50),
                "heading_delta_abs_deg_p50": _safe_quantile([row["heading_delta_abs_deg"] for row in bouts], 0.50),
                "tail_yaw_abs_max_deg_p95": _safe_quantile(
                    [row["tail_yaw_abs_max_deg"] for row in bouts], 0.95
                ),
                "realized_tail_frequency_tbf_band_hz_median": _safe_quantile(
                    [row["dominant_tbf_band_hz"] for row in spec], 0.50
                ),
                "realized_tail_frequency_active_tbf_band_hz_median": _safe_quantile(
                    [row["dominant_active_tbf_band_hz"] for row in spec], 0.50
                ),
                "speed_xy_mm_s_mean": _safe_mean(scalars.get("speed_mm_s", [])),
                "speed_xy_mm_s_p95": _safe_quantile(scalars.get("speed_mm_s", []), 0.95),
                "speed_xy_mm_s_max": _safe_quantile(scalars.get("speed_mm_s", []), 1.0),
                "travel_path_mm": travel_mm,
                "net_displacement_mm": displacement_mm,
                "body_pitch_abs_p95_rad": _safe_quantile(np.abs(scalars.get("body_pitch_rad", [])), 0.95),
                "body_z_span_mm_p95": _safe_quantile(scalars.get("z_span_mm", []), 0.95),
                "neuron_fired_count_mean": _safe_mean(scalars.get("neuron_fired_count", [])),
                "neuron_s_abs_mean": _safe_mean(scalars.get("neuron_s_abs_mean", [])),
                "muscle_sum_mean": _safe_mean(scalars.get("muscles_sum", [])),
                "swim_drive_mean": _safe_mean(scalars.get("swim_drive", [])),
                "swim_drive_p95": _safe_quantile(scalars.get("swim_drive", []), 0.95),
                "action_active_fraction": _safe_mean(scalars.get("action_active", [])),
                "action_force_mean": _safe_mean(scalars.get("action_force", [])),
                "action_force_p95": _safe_quantile(scalars.get("action_force", []), 0.95),
                "action_confidence_mean": _safe_mean(scalars.get("action_confidence", [])),
                "action_side_abs_p95": _safe_quantile(np.abs(scalars.get("action_side_score", [])), 0.95),
                "external_tail_confidence_mean": _safe_mean(scalars.get("external_tail_confidence", [])),
                "calcium_pulse_mean": _safe_mean(scalars.get("calcium_pulse", [])),
                "video_motion_energy_mean": _safe_mean(scalars.get("video_motion_energy", [])),
                "video_flow_reliability_mean": _safe_mean(scalars.get("video_flow_reliability", [])),
                "video_camera_shake_mean": _safe_mean(scalars.get("video_camera_shake", [])),
                "video_calibration_gate_mean": _safe_mean(scalars.get("video_calibration_gate", [])),
                "video_calibration_raw_action_force_mean": _safe_mean(
                    scalars.get("video_calibration_raw_action_force", [])
                ),
                "video_calibration_raw_startle_mean": _safe_mean(
                    scalars.get("video_calibration_raw_startle", [])
                ),
                "source_frames_unique": int(
                    np.unique(
                        np.asarray(scalars.get("source_frame_index", np.zeros(0)), dtype=np.float64)[
                            np.asarray(scalars.get("source_frame_index", np.zeros(0)), dtype=np.float64) >= 0
                        ]
                    ).size
                )
                if "source_frame_index" in scalars
                else 0,
                "source_loop_index_max": _safe_quantile(scalars.get("source_loop_index", []), 1.0),
            }
        )
    return rows


def _external_comparison(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_map = {
        "event_frequency_hz": "event_frequency_hz",
        "bout_duration_ms": "bout_duration_ms_p50",
        "interbout_ms": "interbout_ms_p50",
        "speed_xy_mm_s_mean": "speed_xy_mm_s_mean",
        "speed_xy_mm_s_p95": "speed_xy_mm_s_p95",
        "distance_per_event_mm": "distance_per_bout_mm_p50",
        "heading_delta_abs_deg": "heading_delta_abs_deg_p50",
        "tail_yaw_abs_max_deg_p95": "tail_yaw_abs_max_deg_p95",
        "realized_tail_frequency_hz_observable": "realized_tail_frequency_tbf_band_hz_median",
    }
    rows: list[dict[str, Any]] = []
    for run in run_rows:
        for target in EXTERNAL_TARGETS:
            metric = str(target["metric"])
            source_metric = metric_map.get(metric)
            if source_metric is None:
                continue
            value = float(run.get(source_metric, float("nan")))
            target_min = float(target["target_min"])
            target_max = float(target["target_max"])
            if not math.isfinite(value):
                status = "missing"
                signed_error = float("nan")
            elif target_min <= value <= target_max:
                status = "inside_target"
                signed_error = 0.0
            elif value < target_min:
                status = "below_target"
                signed_error = value - target_min
            else:
                status = "above_target"
                signed_error = value - target_max
            rows.append(
                {
                    "run": run["run"],
                    "target_metric": metric,
                    "run_metric": source_metric,
                    "value": value,
                    "target_min": target_min,
                    "target_max": target_max,
                    "status": status,
                    "signed_error_to_band": signed_error,
                    "source": target["source"],
                    "interpretation": target["interpretation"],
                }
            )
    return rows


def _plot_master_timeline(out_dir: Path, recordings: list[Recording]) -> Path:
    panels = [
        ("speed_mm_s", "speed mm/s"),
        ("swim_drive", "swim drive"),
        ("turn_bias", "turn bias"),
        ("pitch_bias", "pitch bias"),
        ("action_force", "action force"),
        ("action_side_score", "action side"),
        ("action_confidence", "action confidence"),
        ("action_tail_target_confidence", "tail target conf"),
        ("external_tail_confidence", "external tail conf"),
        ("calcium_pulse", "calcium pulse"),
        ("calcium_force", "calcium force"),
        ("video_motion_energy", "video motion"),
        ("video_asymmetry", "video asymmetry"),
        ("video_flow_reliability", "flow reliability"),
        ("video_camera_shake", "camera shake"),
        ("video_calibration_gate", "video calibration gate"),
        ("video_calibration_raw_action_force", "raw action force"),
        ("video_calibration_raw_startle", "raw startle"),
        ("tail_yaw_rad_abs_mean", "tail yaw mean abs"),
        ("tail_pitch_rad_abs_mean", "tail pitch mean abs"),
        ("muscles_sum", "muscle sum"),
        ("neuron_s_abs_mean", "PAULA |S| mean"),
        ("neuron_fired_count", "PAULA fired count"),
        ("neuromod_m0", "neuromod M0"),
        ("neuromod_m1", "neuromod M1"),
        ("body_pitch_rad", "body pitch rad"),
        ("z_span_mm", "body z span mm"),
        ("stimulus_code", "stimulus code"),
        ("source_frame_index", "source frame"),
        ("source_loop_index", "source loop"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(18, max(30, len(panels) * 1.55)), sharex=True)
    for rec in recordings:
        t = rec.time_s
        streams = _scalar_streams(rec)
        smooth = max(3, int(round(2.0 / rec.dt_s)))
        for ax, (metric, label) in zip(axes, panels):
            arr = streams.get(metric)
            if arr is None:
                continue
            y = np.asarray(arr, dtype=np.float64)
            if metric not in {"stimulus_code", "source_frame_index", "source_loop_index"}:
                y = _rolling_mean(y, smooth)
            ax.plot(t[: y.size], y, linewidth=0.9, label=rec.label)
            ax.set_ylabel(label)
            ax.grid(alpha=0.18)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("simulation time (s)")
    fig.suptitle("High-rate long-run activity timeline, 200 Hz physics telemetry", y=0.995)
    fig.tight_layout()
    path = out_dir / "high_rate_master_timeline.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _tail_power_spectrum(tail: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.nanmean(np.asarray(tail, dtype=np.float64), axis=1)
    y = y - float(np.nanmean(y))
    win = np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y * win)) ** 2
    freq = np.fft.rfftfreq(y.size, dt_s)
    spec = spec / max(1e-12, float(np.nanmax(spec)))
    return freq, spec


def _plot_tail_spectra(out_dir: Path, recordings: list[Recording]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    for rec in recordings:
        tail = np.asarray(rec.arrays.get("tail_yaw_rad", np.zeros((0, 0))), dtype=np.float64)
        if tail.ndim != 2 or tail.size == 0:
            continue
        freq, spec = _tail_power_spectrum(tail, rec.dt_s)
        mask = freq <= 100.0
        axes[0].plot(freq[mask], spec[mask], linewidth=1.0, label=rec.label)
        seg_freqs = []
        for i in range(tail.shape[1]):
            f, _ = _dominant_frequency(tail[:, i], rec.dt_s, 20.0, min(95.0, 0.5 / rec.dt_s))
            seg_freqs.append(f)
        x = np.arange(tail.shape[1])
        axes[1].plot(x, seg_freqs, marker="o", label=rec.label)
    for ax in axes:
        ax.axhspan(20.0, 95.0, color="tab:green", alpha=0.10)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right")
    axes[0].axvspan(20.0, 95.0, color="tab:green", alpha=0.10, label="larval TBF target")
    axes[0].set_ylabel("normalized power")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_title("Population tail-yaw power spectrum")
    axes[1].set_ylabel("dominant 20-95 Hz frequency")
    axes[1].set_xlabel("tail segment index, anterior to posterior")
    axes[1].set_title("Per-segment realized high-band tail frequency")
    fig.tight_layout()
    path = out_dir / "tail_frequency_spectrum_and_segments.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_frequency_heatmap(out_dir: Path, window_rows: list[dict[str, Any]]) -> Path:
    runs = sorted({str(row["run"]) for row in window_rows})
    fig, axes = plt.subplots(len(runs), 1, figsize=(17, max(5, 4 * len(runs))), squeeze=False)
    for i, run in enumerate(runs):
        rows = [row for row in window_rows if row["run"] == run]
        t = np.asarray([float(row["mid_s"]) for row in rows], dtype=np.float64)
        freq = np.asarray([float(row["dominant_tbf_band_hz"]) for row in rows], dtype=np.float64)
        swim = np.asarray([float(row["swim_drive_mean"]) for row in rows], dtype=np.float64)
        tail = np.asarray([float(row["tail_abs_p95_rad"]) for row in rows], dtype=np.float64)
        ax = axes[i, 0]
        sc = ax.scatter(t, freq, c=tail, s=np.maximum(12.0, 80.0 * np.clip(swim, 0.0, 1.0)), cmap="viridis")
        ax.axhspan(20.0, 95.0, color="tab:green", alpha=0.10)
        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel("Hz")
        ax.set_title(f"{run}: 1 s rolling realized tail-beat frequency; marker size is swim drive")
        ax.grid(alpha=0.18)
        fig.colorbar(sc, ax=ax, label="tail yaw p95 rad")
    axes[-1, 0].set_xlabel("simulation time (s)")
    fig.tight_layout()
    path = out_dir / "tail_frequency_window_timeline.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _status_code(status: str) -> int:
    return {"inside_target": 2, "below_target": -1, "above_target": 1, "missing": -3}.get(status, 0)


def _plot_external_scorecard(out_dir: Path, comparison: list[dict[str, Any]]) -> Path:
    runs = sorted({str(row["run"]) for row in comparison})
    metrics = sorted({str(row["target_metric"]) for row in comparison})
    mat = np.zeros((len(runs), len(metrics)), dtype=float)
    for i, run in enumerate(runs):
        for j, metric in enumerate(metrics):
            status = next(
                (
                    str(row["status"])
                    for row in comparison
                    if row["run"] == run and row["target_metric"] == metric
                ),
                "missing",
            )
            mat[i, j] = _status_code(status)
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=2)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels(runs)
    ax.set_title("High-rate realized kinematics vs external larval zebrafish target bands")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([-3, -1, 1, 2])
    cbar.set_ticklabels(["missing", "below", "above", "inside"])
    fig.tight_layout()
    path = out_dir / "high_rate_external_target_scorecard.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_bout_distributions(out_dir: Path, bouts: list[dict[str, Any]]) -> Path:
    runs = sorted({str(row["run"]) for row in bouts})
    metrics = [
        ("duration_ms", "bout duration ms"),
        ("interbout_ms", "interbout ms"),
        ("distance_mm", "distance per bout mm"),
        ("heading_delta_abs_deg", "heading delta deg"),
        ("tail_yaw_abs_max_deg", "tail peak yaw deg"),
        ("speed_peak_mm_s", "peak speed mm/s"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for run in runs:
            values = np.asarray([float(row[metric]) for row in bouts if row["run"] == run], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=40, alpha=0.45, density=True, label=run)
        ax.set_title(title)
        ax.grid(alpha=0.18)
    axes[0, 0].axvspan(100.0, 250.0, color="tab:green", alpha=0.12)
    axes[0, 1].axvspan(200.0, 1100.0, color="tab:green", alpha=0.12)
    axes[0, 2].axvspan(1.0, 4.0, color="tab:green", alpha=0.12)
    axes[1, 0].axvspan(1.0, 60.0, color="tab:green", alpha=0.12)
    axes[1, 1].axvspan(15.0, 180.0, color="tab:green", alpha=0.12)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Bout event distributions; green bands are public larval target envelopes", y=0.995)
    fig.tight_layout()
    path = out_dir / "bout_event_distributions.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_body_phase_space(out_dir: Path, recordings: list[Recording]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for rec in recordings:
        streams = _scalar_streams(rec)
        t = rec.time_s
        com = np.asarray(rec.arrays.get("com_mm", np.zeros((0, 3))), dtype=np.float64)
        if com.ndim == 2 and com.shape[0] > 1:
            axes[0, 0].plot(com[:, 0], com[:, 1], linewidth=0.8, label=rec.label)
            axes[0, 1].plot(t, com[:, 2], linewidth=0.8, label=rec.label)
        axes[0, 2].scatter(
            streams.get("tail_yaw_rad_abs_mean", np.zeros(t.size)),
            streams.get("speed_mm_s", np.zeros(t.size)),
            s=2,
            alpha=0.18,
            label=rec.label,
        )
        axes[1, 0].scatter(
            streams.get("swim_drive", np.zeros(t.size)),
            streams.get("muscles_sum", np.zeros(t.size)),
            s=2,
            alpha=0.18,
            label=rec.label,
        )
        axes[1, 1].scatter(
            streams.get("turn_bias", np.zeros(t.size)),
            streams.get("yaw_rate_rad_s", np.zeros(t.size)),
            s=2,
            alpha=0.18,
            label=rec.label,
        )
        axes[1, 2].scatter(
            streams.get("pitch_bias", np.zeros(t.size)),
            streams.get("vertical_speed_mm_s", np.zeros(t.size)),
            s=2,
            alpha=0.18,
            label=rec.label,
        )
    axes[0, 0].set_title("COM XY path")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 1].set_title("Depth over time")
    axes[0, 2].set_title("Tail yaw vs speed")
    axes[1, 0].set_title("Swim drive vs muscle sum")
    axes[1, 1].set_title("Turn bias vs yaw rate")
    axes[1, 2].set_title("Pitch bias vs vertical speed")
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
        ax.legend(loc="best", markerscale=4)
    fig.tight_layout()
    path = out_dir / "body_phase_space_and_control_surfaces.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_coupling_heatmap(out_dir: Path, recordings: list[Recording]) -> Path:
    features = [
        "speed_mm_s",
        "vertical_speed_mm_s",
        "swim_drive",
        "turn_bias",
        "pitch_bias",
        "tail_yaw_rad_abs_mean",
        "tail_pitch_rad_abs_mean",
        "muscles_sum",
        "neuron_s_abs_mean",
        "neuron_r_abs_mean",
        "neuron_o_abs_mean",
        "neuron_fired_count",
        "neuromod_m0",
        "neuromod_m1",
        "action_force",
        "action_side_score",
        "action_confidence",
        "action_tail_target_confidence",
        "action_tail_targets_abs_mean",
        "calcium_pulse",
        "video_motion_energy",
        "video_asymmetry",
        "video_flow_reliability",
        "external_tail_confidence",
        "body_pitch_rad",
        "z_span_mm",
        "straightness",
        "abs_curvature_3d_rad",
        "max_local_bend_3d_rad",
    ]
    fig, axes = plt.subplots(len(recordings), 1, figsize=(14, max(6, 6 * len(recordings))), squeeze=False)
    for i, rec in enumerate(recordings):
        streams = _scalar_streams(rec)
        mat = np.zeros((len(features), len(features)), dtype=float)
        for r, a in enumerate(features):
            for c, b in enumerate(features):
                mat[r, c] = _corr(streams.get(a, []), streams.get(b, []))
        ax = axes[i, 0]
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
        ax.set_xticks(np.arange(len(features)))
        ax.set_xticklabels(features, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_title(f"{rec.label}: scalar coupling matrix")
        fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    path = out_dir / "scalar_coupling_heatmaps.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_top_channel_heatmaps(out_dir: Path, recordings: list[Recording], key: str, n: int = 32) -> list[Path]:
    paths: list[Path] = []
    for rec in recordings:
        mat = _channel_matrix(rec.arrays.get(key, np.zeros((0, 0))))
        if mat.size == 0:
            continue
        score = np.nanstd(mat, axis=0) + 0.2 * np.nanmean(np.abs(mat), axis=0)
        idx = np.argsort(-score)[: min(n, mat.shape[1])]
        t = rec.time_s
        names = _matrix_names(rec, key, mat.shape[1])
        fig, ax = plt.subplots(figsize=(17, 9))
        im = ax.imshow(
            mat[:, idx].T,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            extent=[float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else mat.shape[0], idx.size, 0],
        )
        ax.set_yticks(np.arange(idx.size) + 0.5)
        ax.set_yticklabels([names[i] for i in idx], fontsize=7)
        ax.set_xlabel("simulation time (s)")
        ax.set_title(f"{rec.label}: top {idx.size} {key} channels")
        fig.colorbar(im, ax=ax, label=key)
        fig.tight_layout()
        path = out_dir / f"{rec.label}_{key}_top{idx.size}_heatmap.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def _anomaly_rows(recordings: list[Recording], run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_lookup = {row["run"]: row for row in run_rows}

    def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(row.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def _low_action_quiescent(row: dict[str, Any]) -> bool:
        return (
            _f(row, "action_force_mean") < 0.035
            and _f(row, "swim_drive_p95") < 0.02
            and _f(row, "speed_xy_mm_s_p95") < 0.8
            and _f(row, "tail_yaw_abs_max_deg_p95") < 2.0
        )

    checks = [
        ("meets_50k_ticks", lambda row: bool(row.get("meets_50k_ticks")), "recording has at least 50k physics ticks"),
        ("meets_tbf_nyquist", lambda row: bool(row.get("meets_tbf_nyquist")), "200 Hz telemetry resolves 20-95 Hz TBF"),
        (
            "finite_realized_tbf",
            lambda row: math.isfinite(float(row.get("realized_tail_frequency_tbf_band_hz_median", float("nan")))),
            "tail-yaw spectrum has finite high-band estimate",
        ),
        (
            "not_vertical_lock",
            lambda row: abs(float(row.get("body_pitch_abs_p95_rad", 0.0))) < 0.75,
            "body pitch p95 below vertical-lock threshold",
        ),
        (
            "bounded_z_span",
            lambda row: float(row.get("body_z_span_mm_p95", 0.0)) < 30.0,
            "body z-span p95 remains finite and bounded",
        ),
    ]
    for rec in recordings:
        row = run_lookup.get(rec.label, {})
        bout_count = int(row.get("bout_count", 0) or 0)
        if _low_action_quiescent(row):
            rows.append(
                {
                    "run": rec.label,
                    "check": "quiescent_low_action_control",
                    "passed": bout_count == 0,
                    "status": "pass" if bout_count == 0 else "fail",
                    "description": "low-action negative control remains quiescent with no segmented swim bouts",
                }
            )
        else:
            rows.append(
                {
                    "run": rec.label,
                    "check": "nonzero_bouts_when_driven",
                    "passed": bout_count > 0,
                    "status": "pass" if bout_count > 0 else "fail",
                    "description": "driven recording has segmented swim bouts",
                }
            )
        for check_id, func, description in checks:
            passed = bool(func(row))
            rows.append(
                {
                    "run": rec.label,
                    "check": check_id,
                    "passed": passed,
                    "status": "pass" if passed else "fail",
                    "description": description,
                }
            )
    return rows


def _write_report(
    out_dir: Path,
    recordings: list[Recording],
    run_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    anomaly_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> Path:
    status_counts = Counter(str(row["status"]) for row in comparison_rows)
    anomaly_counts = Counter(str(row["status"]) for row in anomaly_rows)
    lines = [
        "# High-Rate Tail And Activity Validation Audit",
        "",
        "This audit uses per-physics-tick MuJoCo archives rather than browser",
        "WebSocket frames.  The effective sample rate is 200 Hz, so realized",
        "tail-beat frequencies in the larval 20-95 Hz range are directly",
        "observable up to the 100 Hz Nyquist limit.",
        "",
        "## Recording Coverage",
        "",
        "| run | ticks | simulated s | sample Hz | Nyquist Hz | bouts | TBF median Hz | speed mean mm/s | action force mean | action conf mean | source frames |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            "| {run} | {ticks} | {simulated_s:.3f} | {sample_hz:.1f} | {nyquist_hz:.1f} | "
            "{bout_count} | {realized_tail_frequency_tbf_band_hz_median:.3f} | "
            "{speed_xy_mm_s_mean:.3f} | {action_force_mean:.3f} | "
            "{action_confidence_mean:.3f} | {source_frames_unique} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## External Target Status",
            "",
            f"Status counts: `{dict(status_counts)}`.",
            "",
            "These comparisons reuse the public larval target bands from the external",
            "kinematics audit. They are calibration evidence, not proof that the",
            "video/calcium branches are biologically exact.",
            "",
            "## Anomaly Checks",
            "",
            f"Anomaly check counts: `{dict(anomaly_counts)}`.",
            "",
            "## Source Anchors",
            "",
        ]
    )
    for source in KINEMATIC_SOURCES:
        lines.append(f"- `{source['id']}`: {source['title']} - {source['url']}")
    lines.extend(["", "## Visual Evidence", ""])
    for path in plot_paths:
        lines.append(f"### {path.stem.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"![{path.stem}]({path.resolve()})")
        lines.append("")
    lines.extend(
        [
            "## Tables",
            "",
            f"- Run summary: `{(out_dir / 'high_rate_run_summary.csv').resolve()}`",
            f"- External comparison: `{(out_dir / 'high_rate_external_comparison.csv').resolve()}`",
            f"- Scalar stats: `{(out_dir / 'high_rate_scalar_stats.csv').resolve()}`",
            f"- Matrix channel stats: `{(out_dir / 'high_rate_matrix_channel_stats.csv').resolve()}`",
            f"- Bout events: `{(out_dir / 'high_rate_bout_events.csv').resolve()}`",
            f"- 10 s windows: `{(out_dir / 'high_rate_window_summary_10s.csv').resolve()}`",
            f"- 30 s windows: `{(out_dir / 'high_rate_window_summary_30s.csv').resolve()}`",
            f"- Spectral segments: `{(out_dir / 'high_rate_tail_segment_spectra.csv').resolve()}`",
            f"- Spectral windows: `{(out_dir / 'high_rate_tail_window_spectra.csv').resolve()}`",
            f"- Anomalies: `{(out_dir / 'high_rate_anomaly_checks.csv').resolve()}`",
            "",
            "## Interpretation Boundary",
            "",
            "This layer validates realized body telemetry at high sample rate. It does",
            "not by itself validate that the video-to-action or calcium-to-action",
            "decoders are exact reproductions of ZAPBench or Z-Robot. Those branches",
            "remain covered by the browser long-run, source-alignment, DANDI, and",
            "controller-regression audits.",
        ]
    )
    path = out_dir / "HIGH_RATE_TAIL_VALIDATION_AUDIT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_audit(recordings_dir: Path, out_dir: Path, glob_pattern: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = _load_recordings(recordings_dir, glob_pattern)
    if not recordings:
        raise RuntimeError(f"no recordings matched {recordings_dir / glob_pattern}")

    scalar_rows = _all_scalar_stats(recordings)
    matrix_rows = _matrix_channel_stats(recordings)
    stimulus_rows = _stimulus_summary(recordings)
    bout_rows = _all_bouts(recordings)
    spectral_segment_rows, spectral_window_rows = _spectral_rows(recordings)
    run_rows = _run_summary_rows(recordings, bout_rows, spectral_segment_rows)
    comparison_rows = _external_comparison(run_rows)
    anomaly_rows = _anomaly_rows(recordings, run_rows)
    win10 = _window_summary(recordings, 10.0)
    win30 = _window_summary(recordings, 30.0)

    _write_csv(out_dir / "high_rate_scalar_stats.csv", scalar_rows)
    _write_csv(out_dir / "high_rate_matrix_channel_stats.csv", matrix_rows)
    _write_csv(out_dir / "high_rate_stimulus_summary.csv", stimulus_rows)
    _write_csv(out_dir / "high_rate_bout_events.csv", bout_rows)
    _write_csv(out_dir / "high_rate_tail_segment_spectra.csv", spectral_segment_rows)
    _write_csv(out_dir / "high_rate_tail_window_spectra.csv", spectral_window_rows)
    _write_csv(out_dir / "high_rate_run_summary.csv", run_rows)
    _write_csv(out_dir / "high_rate_external_comparison.csv", comparison_rows)
    _write_csv(out_dir / "high_rate_anomaly_checks.csv", anomaly_rows)
    _write_csv(out_dir / "high_rate_window_summary_10s.csv", win10)
    _write_csv(out_dir / "high_rate_window_summary_30s.csv", win30)

    plot_paths: list[Path] = [
        _plot_master_timeline(out_dir, recordings),
        _plot_tail_spectra(out_dir, recordings),
        _plot_frequency_heatmap(out_dir, spectral_window_rows),
        _plot_external_scorecard(out_dir, comparison_rows),
        _plot_bout_distributions(out_dir, bout_rows),
        _plot_body_phase_space(out_dir, recordings),
        _plot_coupling_heatmap(out_dir, recordings),
    ]
    for key in (
        "tail_yaw_rad",
        "tail_pitch_rad",
        "action_tail_targets",
        "muscles",
        "neuron_s",
        "neuron_r",
        "neuron_o",
    ):
        plot_paths.extend(_plot_top_channel_heatmaps(out_dir, recordings, key))

    report_path = _write_report(out_dir, recordings, run_rows, comparison_rows, anomaly_rows, plot_paths)
    manifest = {
        "recordings_dir": str(recordings_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "glob_pattern": glob_pattern,
        "recordings": [
            {
                "label": rec.label,
                "archive": str(rec.archive.resolve()),
                "summary": str(rec.summary_path.resolve()) if rec.summary_path else "",
                "ticks": int(rec.time_s.size),
                "dt_s": rec.dt_s,
            }
            for rec in recordings
        ],
        "row_counts": {
            "scalar_stats": len(scalar_rows),
            "matrix_channel_stats": len(matrix_rows),
            "stimulus_summary": len(stimulus_rows),
            "bout_events": len(bout_rows),
            "spectral_segment_rows": len(spectral_segment_rows),
            "spectral_window_rows": len(spectral_window_rows),
            "run_summary": len(run_rows),
            "external_comparison": len(comparison_rows),
            "anomaly_checks": len(anomaly_rows),
            "window_10s": len(win10),
            "window_30s": len(win30),
        },
        "plots": [str(path.resolve()) for path in plot_paths],
        "report": str(report_path.resolve()),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--glob", default="*.npz")
    args = parser.parse_args()
    manifest = run_audit(args.recordings_dir, args.out_dir, args.glob)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
