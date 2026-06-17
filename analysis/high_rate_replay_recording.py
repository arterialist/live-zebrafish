"""Exact high-rate recorder for zebrafish lab replay branches.

This records the calcium and video stimulus regimes at the MuJoCo physics
tick rate.  It mirrors the lab replay contracts instead of sampling browser
WebSocket snapshots:

* calcium: ZAPBench decoded action frames advance at 1.093 Hz, and the
  environment converts each frame into a short decaying motor pulse.
* video: backend OpenCV extraction produces one frame-level latent action,
  then the simulation advances the exact rounded number of physics ticks for
  that sampled video frame.

The archive intentionally includes both controller inputs and realized body /
PAULA / muscle telemetry so downstream audits can trace every transition from
source stimulus to motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ACTIVE_INFERENCE = REPO_ROOT / "active-inference"
if str(ACTIVE_INFERENCE) not in sys.path:
    sys.path.insert(0, str(ACTIVE_INFERENCE))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simulations.zebrafish import config as zfc  # noqa: E402
from simulations.zebrafish.environment import AquaticArenaEnvironment  # noqa: E402
from simulations.zebrafish.simulation import build_zebrafish_simulation  # noqa: E402

from lab.video_pipeline import BackendVideoPipeline  # noqa: E402
from zebrafish_long_recording import (  # noqa: E402
    DT_S,
    _angle_wrap,
    _heading_from_points,
    _motif_summary,
    _stats,
)
from zebrafish_reverse_engineering import _path_metrics  # noqa: E402


ZAPBENCH_CALCIUM_FRAME_HZ = 1.093
MODE_CODE = {"baseline": 0, "calcium": 1, "video": 2}
DEFAULT_REPLAY_PATH = SCRIPT_DIR / "cache" / "zapbench" / "zapbench_calcium_action_replay.npz"
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "cache" / "video_stimuli" / "clips"
DEFAULT_UPLOAD_DIR = SCRIPT_DIR / "cache" / "video_stimuli" / "uploads"
DEFAULT_FRAME_CACHE = SCRIPT_DIR / "cache" / "high_rate_replay_video_frames"
DEFAULT_VIDEO_CALIBRATION = {
    "action_force_scale": 1.0,
    "action_side_scale": 1.0,
    "tail_target_scale": 1.0,
    "startle_scale": 1.0,
    "motion_scale": 1.0,
    "confidence_scale": 1.0,
    "camera_shake_gate": 0.0,
    "reliability_floor": 0.35,
    "max_action_force": 1.0,
    "min_action_force": 0.0,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _stats_2d_mean(matrix: np.ndarray) -> dict[str, float]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 2 and arr.size:
        return _stats(np.mean(arr, axis=1))
    return _stats([])


def _load_calcium_replay(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing calcium replay artifact: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {
            "rows": np.asarray(npz["rows"], dtype=np.int32),
            "calcium_time_s": np.asarray(npz["calcium_time_s"], dtype=np.float32),
            "condition_id": np.asarray(npz["condition_id"], dtype=np.int32),
            "kick": np.asarray(npz["kick"], dtype=np.float32),
            "side_score": np.asarray(npz["side_score"], dtype=np.float32),
            "force": np.asarray(npz["force"], dtype=np.float32),
            "kick_score": np.asarray(npz["kick_score"], dtype=np.float32),
            "confidence": np.asarray(npz["confidence"], dtype=np.float32),
            "metadata": json.loads(str(npz["metadata_json"])),
        }


def _calcium_indices(data: dict[str, Any], condition: str) -> np.ndarray:
    rows = np.asarray(data["rows"], dtype=np.int32)
    if condition == "all":
        return np.arange(rows.shape[0], dtype=np.int32)
    names = list(data["metadata"].get("condition_names", []))
    try:
        cond_id = names.index(condition)
    except ValueError as exc:
        raise ValueError(f"unknown calcium condition {condition!r}; choose from {names}") from exc
    return np.flatnonzero(np.asarray(data["condition_id"], dtype=np.int32) == cond_id).astype(np.int32)


def _video_duration_s(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0.0 and frames > 0.0:
            return frames / fps
        duration_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return max(0.0, duration_ms / 1000.0)
    finally:
        cap.release()


def _video_source_path(file_name: str) -> Path:
    safe_name = Path(file_name).name
    sample = (DEFAULT_VIDEO_DIR / safe_name).resolve()
    if sample.exists():
        return sample
    upload = (DEFAULT_UPLOAD_DIR / safe_name).resolve()
    if upload.exists():
        return upload
    raise FileNotFoundError(f"unknown video source {file_name!r}")


def _manual_video_ticks_for_frame(frame_index: int, sample_hz: float, last_frame_index: int) -> tuple[int, int]:
    hz = max(1.0, float(sample_hz))
    idx = max(0, int(frame_index))
    if idx <= last_frame_index:
        last_frame_index = idx - 1
    ticks_per_frame = 1.0 / (hz * zfc.PHYSICS_TIMESTEP_S)
    prev_boundary = int(round(idx * ticks_per_frame))
    next_boundary = int(round((idx + 1) * ticks_per_frame))
    return max(1, next_boundary - prev_boundary), idx


def _frame_cache_key(path: Path, file_name: str, frame_index: int, video_time_s: float, sample_hz: float) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "file_name": Path(file_name).name,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "frame_index": int(frame_index),
        "video_time_s": round(float(video_time_s), 6),
        "sample_hz": round(float(sample_hz), 6),
        "pipeline": "BackendVideoPipeline.extract",
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_or_extract_video_frame(
    pipeline: BackendVideoPipeline,
    *,
    path: Path,
    file_name: str,
    frame_index: int,
    video_time_s: float,
    sample_hz: float,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _frame_cache_key(path, file_name, frame_index, video_time_s, sample_hz)
    cache_path = cache_dir / f"{key}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return dict(cached["features"]), dict(cached["diagnostics"]), True
    extracted = pipeline.extract(
        path=path,
        file_name=Path(file_name).name,
        frame_index=frame_index,
        video_time_s=video_time_s,
        sample_hz=sample_hz,
    )
    payload = {"features": extracted.features, "diagnostics": extracted.diagnostics}
    if use_cache:
        cache_path.write_text(json.dumps(payload, sort_keys=True, default=_json_default), encoding="utf-8")
    return dict(extracted.features), dict(extracted.diagnostics), False


def _video_calibration_gate(diagnostics: dict[str, Any], calibration: dict[str, float]) -> float:
    reliability = float(np.clip(float(diagnostics.get("flow_reliability", 0.0)), 0.0, 1.0))
    shake = float(np.clip(float(diagnostics.get("camera_shake", 0.0)), 0.0, 1.0))
    reliability_floor = float(np.clip(calibration.get("reliability_floor", 0.35), 0.0, 1.0))
    shake_gate = max(0.0, float(calibration.get("camera_shake_gate", 0.0)))
    reliability_term = reliability_floor + (1.0 - reliability_floor) * reliability
    shake_term = float(np.clip(1.0 - shake_gate * shake, 0.05, 1.0))
    return float(np.clip(reliability_term * shake_term, 0.05, 1.0))


def _calibrate_video_features(
    features: dict[str, Any],
    diagnostics: dict[str, Any],
    calibration: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if all(
        abs(float(calibration[key]) - float(DEFAULT_VIDEO_CALIBRATION[key])) < 1e-12
        for key in DEFAULT_VIDEO_CALIBRATION
    ):
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "calibration_gate": 1.0,
                "calibration_action_force_scale": 1.0,
                "calibration_startle_scale": 1.0,
                "calibration_tail_target_scale": 1.0,
                "calibration_camera_shake_gate": 0.0,
            }
        )
        return features, diagnostics

    out = dict(features)
    diag = dict(diagnostics)
    gate = _video_calibration_gate(diag, calibration)
    force_scale = max(0.0, float(calibration.get("action_force_scale", 1.0)))
    side_scale = max(0.0, float(calibration.get("action_side_scale", 1.0)))
    tail_scale = max(0.0, float(calibration.get("tail_target_scale", 1.0)))
    startle_scale = max(0.0, float(calibration.get("startle_scale", 1.0)))
    motion_scale = max(0.0, float(calibration.get("motion_scale", 1.0)))
    confidence_scale = max(0.0, float(calibration.get("confidence_scale", 1.0)))
    max_force = float(np.clip(float(calibration.get("max_action_force", 1.0)), 0.0, 1.0))
    min_force = float(np.clip(float(calibration.get("min_action_force", 0.0)), 0.0, 1.0))

    raw_force = float(np.clip(float(out.get("action_force", 0.0)), 0.0, 1.0))
    raw_side = float(np.clip(float(out.get("action_side_score", 0.0)), -1.0, 1.0))
    raw_conf = float(np.clip(float(out.get("action_confidence", 0.0)), 0.0, 1.0))
    raw_startle = float(np.clip(float(out.get("startle", 0.0)), 0.0, 1.0))
    raw_motion = float(np.clip(float(out.get("motion_energy", 0.0)), 0.0, 1.0))

    force = float(np.clip(raw_force * force_scale * gate, 0.0, max_force))
    side = float(np.clip(raw_side * side_scale * gate, -1.0, 1.0))
    confidence = float(np.clip(raw_conf * confidence_scale * gate, 0.0, 1.0))
    startle = float(np.clip(raw_startle * startle_scale * gate, 0.0, 1.0))
    motion = float(np.clip(raw_motion * motion_scale * gate, 0.0, 1.0))
    if force < min_force:
        force = 0.0
        side = 0.0
        confidence = min(confidence, 0.25)
        out["action_kick"] = 0.0
        out["action_kick_score"] = min(float(out.get("action_kick_score", 0.0)), 0.35)
    else:
        out["action_kick"] = 1.0 if force > 0.02 and float(out.get("action_kick", 0.0)) >= 0.5 else 0.0

    out["action_force"] = force
    out["action_side_score"] = side
    out["action_confidence"] = confidence
    out["startle"] = startle
    out["motion_energy"] = motion
    out["tail_amplitude"] = float(np.clip(float(out.get("tail_amplitude", raw_force)) * tail_scale * gate, 0.0, 1.0))
    if force > 0.0:
        out["tail_frequency_hz"] = float(
            zfc.TAIL_BEAT_FREQ_MIN_HZ + force * (zfc.TAIL_BEAT_FREQ_MAX_HZ - zfc.TAIL_BEAT_FREQ_MIN_HZ)
        )
    else:
        out["tail_frequency_hz"] = 0.0
    try:
        raw_targets = list(out.get("tail_targets", []))
    except TypeError:
        raw_targets = []
    out["tail_targets"] = [
        float(np.clip(float(value) * tail_scale * gate, -1.0, 1.0))
        for value in raw_targets[: zfc.N_BODY_SEGMENTS]
    ]
    while len(out["tail_targets"]) < zfc.N_BODY_SEGMENTS:
        out["tail_targets"].append(0.0)

    diag.update(
        {
            "calibration_gate": gate,
            "calibration_action_force_scale": force_scale,
            "calibration_action_side_scale": side_scale,
            "calibration_tail_target_scale": tail_scale,
            "calibration_startle_scale": startle_scale,
            "calibration_motion_scale": motion_scale,
            "calibration_confidence_scale": confidence_scale,
            "calibration_camera_shake_gate": float(calibration.get("camera_shake_gate", 0.0)),
            "calibration_reliability_floor": float(calibration.get("reliability_floor", 0.35)),
            "calibration_max_action_force": max_force,
            "calibration_min_action_force": min_force,
            "calibration_raw_action_force": raw_force,
            "calibration_raw_action_side_score": raw_side,
            "calibration_raw_action_confidence": raw_conf,
            "calibration_raw_startle": raw_startle,
            "calibration_raw_motion_energy": raw_motion,
        }
    )
    return out, diag


def _alloc_recording_arrays(ticks: int, model: Any, body: Any, ns: Any) -> dict[str, np.ndarray]:
    neuron_names = ns.get_neuron_names_paula_order()
    muscle_names = list(body.muscle_names)
    joint_names = list(body.joint_names)
    n_body_points = 1 + len([name for name in joint_names if name.startswith("tail_yaw_")])
    n_segments = zfc.N_BODY_SEGMENTS
    arrays: dict[str, np.ndarray] = {
        "ticks": np.zeros(ticks, dtype=np.int32),
        "qpos": np.zeros((ticks, model.nq), dtype=np.float32),
        "qvel": np.zeros((ticks, model.nv), dtype=np.float32),
        "com_mm": np.zeros((ticks, 3), dtype=np.float32),
        "body_points_mm": np.zeros((ticks, n_body_points, 3), dtype=np.float32),
        "tail_yaw_rad": np.zeros((ticks, n_body_points - 1), dtype=np.float32),
        "tail_pitch_rad": np.zeros((ticks, n_body_points - 1), dtype=np.float32),
        "muscles": np.zeros((ticks, len(muscle_names)), dtype=np.float32),
        "neuron_s": np.zeros((ticks, len(neuron_names)), dtype=np.float32),
        "neuron_r": np.zeros((ticks, len(neuron_names)), dtype=np.float32),
        "neuron_o": np.zeros((ticks, len(neuron_names)), dtype=np.float32),
        "neuron_fired": np.zeros((ticks, len(neuron_names)), dtype=np.bool_),
        "neuromod": np.zeros((ticks, 2), dtype=np.float32),
        "swim_drive": np.zeros(ticks, dtype=np.float32),
        "turn_bias": np.zeros(ticks, dtype=np.float32),
        "pitch_bias": np.zeros(ticks, dtype=np.float32),
        "external_tail_confidence": np.zeros(ticks, dtype=np.float32),
        "bout_ticks_left": np.zeros(ticks, dtype=np.int16),
        "coast_ticks_left": np.zeros(ticks, dtype=np.int16),
        "startle_ticks_left": np.zeros(ticks, dtype=np.int16),
        "speed_mm_s": np.zeros(ticks, dtype=np.float32),
        "vertical_speed_mm_s": np.zeros(ticks, dtype=np.float32),
        "heading_rad": np.zeros(ticks, dtype=np.float32),
        "yaw_rate_rad_s": np.zeros(ticks, dtype=np.float32),
        "body_pitch_rad": np.zeros(ticks, dtype=np.float32),
        "z_span_mm": np.zeros(ticks, dtype=np.float32),
        "straightness": np.zeros(ticks, dtype=np.float32),
        "abs_curvature_3d_rad": np.zeros(ticks, dtype=np.float32),
        "max_local_bend_3d_rad": np.zeros(ticks, dtype=np.float32),
        "stimulus_code": np.zeros(ticks, dtype=np.int16),
        "source_frame_index": np.full(ticks, -1, dtype=np.int32),
        "source_row": np.full(ticks, -1, dtype=np.int32),
        "source_time_s": np.zeros(ticks, dtype=np.float32),
        "source_loop_index": np.zeros(ticks, dtype=np.int16),
        "action_active": np.zeros(ticks, dtype=np.float32),
        "action_kick": np.zeros(ticks, dtype=np.float32),
        "action_force": np.zeros(ticks, dtype=np.float32),
        "action_side_score": np.zeros(ticks, dtype=np.float32),
        "action_kick_score": np.zeros(ticks, dtype=np.float32),
        "action_confidence": np.zeros(ticks, dtype=np.float32),
        "action_tail_frequency_hz": np.zeros(ticks, dtype=np.float32),
        "action_tail_amplitude": np.zeros(ticks, dtype=np.float32),
        "action_tail_target_confidence": np.zeros(ticks, dtype=np.float32),
        "action_tail_targets": np.zeros((ticks, n_segments), dtype=np.float32),
        "calcium_active": np.zeros(ticks, dtype=np.float32),
        "calcium_pulse": np.zeros(ticks, dtype=np.float32),
        "calcium_age_ticks": np.zeros(ticks, dtype=np.float32),
        "calcium_force": np.zeros(ticks, dtype=np.float32),
        "calcium_side_score": np.zeros(ticks, dtype=np.float32),
        "calcium_kick_score": np.zeros(ticks, dtype=np.float32),
        "calcium_confidence": np.zeros(ticks, dtype=np.float32),
        "video_active": np.zeros(ticks, dtype=np.float32),
        "video_motion_energy": np.zeros(ticks, dtype=np.float32),
        "video_asymmetry": np.zeros(ticks, dtype=np.float32),
        "video_visual_left": np.zeros(ticks, dtype=np.float32),
        "video_visual_right": np.zeros(ticks, dtype=np.float32),
        "video_visual_up": np.zeros(ticks, dtype=np.float32),
        "video_visual_down": np.zeros(ticks, dtype=np.float32),
        "video_optic_flow_left": np.zeros(ticks, dtype=np.float32),
        "video_optic_flow_right": np.zeros(ticks, dtype=np.float32),
        "video_lateral_line_left": np.zeros(ticks, dtype=np.float32),
        "video_lateral_line_right": np.zeros(ticks, dtype=np.float32),
        "video_light_level": np.zeros(ticks, dtype=np.float32),
        "video_startle": np.zeros(ticks, dtype=np.float32),
        "video_action_force": np.zeros(ticks, dtype=np.float32),
        "video_action_side_score": np.zeros(ticks, dtype=np.float32),
        "video_action_confidence": np.zeros(ticks, dtype=np.float32),
        "video_flow_reliability": np.zeros(ticks, dtype=np.float32),
        "video_camera_shake": np.zeros(ticks, dtype=np.float32),
        "video_camera_motion": np.zeros(ticks, dtype=np.float32),
        "video_compression_noise": np.zeros(ticks, dtype=np.float32),
        "video_zapbench_distance": np.zeros(ticks, dtype=np.float32),
        "video_frame_cache_hit": np.zeros(ticks, dtype=np.float32),
        "video_calibration_gate": np.zeros(ticks, dtype=np.float32),
        "video_calibration_raw_action_force": np.zeros(ticks, dtype=np.float32),
        "video_calibration_raw_action_side_score": np.zeros(ticks, dtype=np.float32),
        "video_calibration_raw_action_confidence": np.zeros(ticks, dtype=np.float32),
        "video_calibration_raw_startle": np.zeros(ticks, dtype=np.float32),
        "video_calibration_raw_motion_energy": np.zeros(ticks, dtype=np.float32),
    }
    arrays["neuron_names"] = np.asarray(neuron_names, dtype=object)
    arrays["muscle_names"] = np.asarray(muscle_names, dtype=object)
    arrays["joint_names"] = np.asarray(joint_names, dtype=object)
    return arrays


def _copy_tail_targets(extra: dict[str, Any], n_segments: int) -> np.ndarray:
    return np.asarray(
        [float(np.clip(extra.get(f"action_tail_target_{i:02d}", 0.0), -1.0, 1.0)) for i in range(n_segments)],
        dtype=np.float32,
    )


def _record_tick(
    arrays: dict[str, np.ndarray],
    i: int,
    step: Any,
    *,
    body: Any,
    ns: Any,
    mode: str,
    current_video_diag: dict[str, Any],
    current_video_cache_hit: bool,
    current_source_loop_index: int,
) -> None:
    st = step.body_state
    obs_extra = step.observation.extra or {}
    body_extra = st.extra or {}
    points = np.asarray(body_extra.get("tail_points", body.get_body_shape()), dtype=np.float64) * 1000.0
    expected_points = arrays["body_points_mm"].shape[1]
    if points.shape != (expected_points, 3):
        fixed = np.zeros((expected_points, 3), dtype=np.float64)
        fixed[: min(expected_points, points.shape[0])] = points[:expected_points]
        points = fixed
    metrics = _path_metrics(points)
    heading = _heading_from_points(points)
    if i == 0:
        yaw_rate = 0.0
    else:
        yaw_rate = float(_angle_wrap(np.asarray([heading - float(arrays["heading_rad"][i - 1])]))[0] / DT_S)

    arrays["ticks"][i] = int(step.tick)
    arrays["qpos"][i] = np.asarray(body.data.qpos, dtype=np.float32)
    arrays["qvel"][i] = np.asarray(body.data.qvel, dtype=np.float32)
    arrays["com_mm"][i] = np.asarray(st.position, dtype=np.float64) * 1000.0
    arrays["body_points_mm"][i] = points
    arrays["tail_yaw_rad"][i] = np.asarray(body_extra.get("tail_angles", []), dtype=np.float32)
    arrays["tail_pitch_rad"][i] = np.asarray(body_extra.get("tail_pitch_angles", []), dtype=np.float32)
    arrays["muscles"][i] = np.asarray([step.motor_outputs.get(name, 0.0) for name in body.muscle_names], dtype=np.float32)
    arrays["neuromod"][i] = np.asarray(ns.neuromod_levels, dtype=np.float32)
    behavior = ns.behavior_state
    arrays["swim_drive"][i] = float(body_extra.get("swim_drive", behavior.get("swim_drive", 0.0)))
    arrays["turn_bias"][i] = float(body_extra.get("turn_bias", behavior.get("turn_bias", 0.0)))
    arrays["pitch_bias"][i] = float(body_extra.get("pitch_bias", behavior.get("pitch_bias", 0.0)))
    arrays["external_tail_confidence"][i] = float(behavior.get("external_tail_confidence", 0.0))
    arrays["bout_ticks_left"][i] = int(behavior.get("bout_ticks_left", 0))
    arrays["coast_ticks_left"][i] = int(behavior.get("coast_ticks_left", 0))
    arrays["startle_ticks_left"][i] = int(behavior.get("startle_ticks_left", 0))
    arrays["speed_mm_s"][i] = float(body_extra.get("speed_m_s", 0.0)) * 1000.0
    arrays["vertical_speed_mm_s"][i] = float(body_extra.get("vertical_velocity_m_s", 0.0)) * 1000.0
    arrays["heading_rad"][i] = heading
    arrays["yaw_rate_rad_s"][i] = yaw_rate
    arrays["body_pitch_rad"][i] = float(metrics.get("body_pitch_rad", 0.0))
    arrays["z_span_mm"][i] = float(metrics.get("z_span_mm", 0.0))
    arrays["straightness"][i] = float(metrics.get("straightness", 0.0))
    arrays["abs_curvature_3d_rad"][i] = float(metrics.get("abs_curvature_3d_rad", 0.0))
    arrays["max_local_bend_3d_rad"][i] = float(metrics.get("max_local_bend_3d_rad", 0.0))
    arrays["stimulus_code"][i] = MODE_CODE[mode]
    arrays["source_loop_index"][i] = int(current_source_loop_index)

    for name in (
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
    ):
        arrays[name][i] = float(obs_extra.get(name, 0.0))

    arrays["action_tail_targets"][i] = _copy_tail_targets(obs_extra, zfc.N_BODY_SEGMENTS)
    arrays["source_frame_index"][i] = int(
        obs_extra.get("calcium_frame_index", obs_extra.get("video_frame_index", -1))
    )
    arrays["source_row"][i] = int(obs_extra.get("calcium_row", obs_extra.get("video_zapbench_row", -1)))
    arrays["source_time_s"][i] = float(obs_extra.get("calcium_time_s", obs_extra.get("video_time_s", 0.0)))

    video_key_map = {
        "video_visual_left": "visual_left",
        "video_visual_right": "visual_right",
        "video_visual_up": "visual_up",
        "video_visual_down": "visual_down",
        "video_optic_flow_left": "optic_flow_left",
        "video_optic_flow_right": "optic_flow_right",
        "video_lateral_line_left": "lateral_line_left",
        "video_lateral_line_right": "lateral_line_right",
        "video_light_level": "light_level",
        "video_startle": "startle",
        "video_action_force": "video_action_force",
        "video_action_side_score": "video_action_side_score",
        "video_action_confidence": "video_action_confidence",
    }
    for out_key, extra_key in video_key_map.items():
        arrays[out_key][i] = float(obs_extra.get(extra_key, 0.0))
    arrays["video_flow_reliability"][i] = float(current_video_diag.get("flow_reliability", 0.0))
    arrays["video_camera_shake"][i] = float(current_video_diag.get("camera_shake", 0.0))
    arrays["video_camera_motion"][i] = float(current_video_diag.get("camera_motion", 0.0))
    arrays["video_compression_noise"][i] = float(current_video_diag.get("compression_noise", 0.0))
    arrays["video_zapbench_distance"][i] = float(current_video_diag.get("zapbench_distance", 0.0))
    arrays["video_frame_cache_hit"][i] = 1.0 if current_video_cache_hit else 0.0
    arrays["video_calibration_gate"][i] = float(current_video_diag.get("calibration_gate", 0.0))
    arrays["video_calibration_raw_action_force"][i] = float(
        current_video_diag.get("calibration_raw_action_force", obs_extra.get("video_action_force", 0.0))
    )
    arrays["video_calibration_raw_action_side_score"][i] = float(
        current_video_diag.get("calibration_raw_action_side_score", obs_extra.get("video_action_side_score", 0.0))
    )
    arrays["video_calibration_raw_action_confidence"][i] = float(
        current_video_diag.get("calibration_raw_action_confidence", obs_extra.get("video_action_confidence", 0.0))
    )
    arrays["video_calibration_raw_startle"][i] = float(
        current_video_diag.get("calibration_raw_startle", obs_extra.get("startle", 0.0))
    )
    arrays["video_calibration_raw_motion_energy"][i] = float(
        current_video_diag.get("calibration_raw_motion_energy", obs_extra.get("video_motion_energy", 0.0))
    )

    if ns._network is not None:  # type: ignore[attr-defined]
        neurons = ns._network.network.neurons  # type: ignore[attr-defined]
        for nid, neuron in neurons.items():
            j = int(nid)
            if j >= arrays["neuron_s"].shape[1]:
                continue
            arrays["neuron_s"][i, j] = float(neuron.S)
            arrays["neuron_r"][i, j] = float(neuron.r)
            arrays["neuron_o"][i, j] = float(neuron.O)
            arrays["neuron_fired"][i, j] = float(neuron.O) > 0.0


def _summarize(
    arrays: dict[str, np.ndarray],
    *,
    label: str,
    mode: str,
    seed: int,
    ticks: int,
    wall_s: float,
    source: dict[str, Any],
) -> dict[str, Any]:
    max_tail_yaw_rad = (
        np.max(np.abs(arrays["tail_yaw_rad"]), axis=1) if arrays["tail_yaw_rad"].size else np.zeros(ticks)
    )
    com = np.asarray(arrays["com_mm"], dtype=np.float64)
    travel_mm = float(np.sum(np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1))) if com.shape[0] > 1 else 0.0
    summary: dict[str, Any] = {
        "label": label,
        "protocol": mode,
        "mode": mode,
        "seed": int(seed),
        "ticks": int(ticks),
        "dt_s": DT_S,
        "simulated_seconds": float(ticks * DT_S),
        "wall_seconds": float(wall_s),
        "wall_ticks_per_second": float(ticks / max(1e-12, wall_s)),
        "source": source,
        "neuron_names": [str(v) for v in arrays["neuron_names"].tolist()],
        "muscle_names": [str(v) for v in arrays["muscle_names"].tolist()],
        "joint_names": [str(v) for v in arrays["joint_names"].tolist()],
        "kinematics": {
            "speed_mm_s": _stats(arrays["speed_mm_s"]),
            "vertical_speed_mm_s": _stats(arrays["vertical_speed_mm_s"]),
            "yaw_rate_rad_s": _stats(arrays["yaw_rate_rad_s"]),
            "body_pitch_rad": _stats(arrays["body_pitch_rad"]),
            "z_span_mm": _stats(arrays["z_span_mm"]),
            "straightness": _stats(arrays["straightness"]),
            "abs_curvature_3d_rad": _stats(arrays["abs_curvature_3d_rad"]),
            "max_local_bend_3d_rad": _stats(arrays["max_local_bend_3d_rad"]),
            "max_abs_tail_yaw_rad": _stats(max_tail_yaw_rad),
            "depth_mm": _stats(arrays["com_mm"][:, 2]),
            "travel_path_mm": travel_mm,
            "net_displacement_mm": float(np.linalg.norm(com[-1, :2] - com[0, :2])) if com.shape[0] > 1 else 0.0,
        },
        "control": {
            "swim_drive": _stats(arrays["swim_drive"]),
            "turn_bias": _stats(arrays["turn_bias"]),
            "pitch_bias": _stats(arrays["pitch_bias"]),
            "external_tail_confidence": _stats(arrays["external_tail_confidence"]),
            "neuromod_m0": _stats(arrays["neuromod"][:, 0]),
            "neuromod_m1": _stats(arrays["neuromod"][:, 1]),
        },
        "action": {
            "active": _stats(arrays["action_active"]),
            "kick": _stats(arrays["action_kick"]),
            "force": _stats(arrays["action_force"]),
            "side_score": _stats(arrays["action_side_score"]),
            "confidence": _stats(arrays["action_confidence"]),
            "tail_frequency_hz": _stats(arrays["action_tail_frequency_hz"]),
            "tail_amplitude": _stats(arrays["action_tail_amplitude"]),
            "tail_target_confidence": _stats(arrays["action_tail_target_confidence"]),
        },
        "neural": {
            "mean_S": _stats(np.mean(arrays["neuron_s"], axis=1)),
            "max_S": _stats(np.max(arrays["neuron_s"], axis=1)),
            "fired_count": _stats(np.sum(arrays["neuron_fired"], axis=1)),
            "mean_r": _stats(np.mean(arrays["neuron_r"], axis=1)),
            "top_active": [],
        },
        "muscle": {
            "mean_activation": _stats_2d_mean(arrays["muscles"]),
            "sum_activation": _stats(np.sum(arrays["muscles"], axis=1)),
        },
        "motifs": _motif_summary(
            arrays["ticks"],
            arrays["swim_drive"],
            arrays["turn_bias"],
            arrays["pitch_bias"],
            arrays["speed_mm_s"],
            arrays["body_pitch_rad"],
            max_tail_yaw_rad,
        ),
    }
    neuron_names = [str(v) for v in arrays["neuron_names"].tolist()]
    if neuron_names:
        fire_fraction = np.mean(arrays["neuron_fired"], axis=0)
        mean_s = np.mean(arrays["neuron_s"], axis=0)
        order = np.argsort(-(fire_fraction + 0.001 * mean_s))[:16]
        summary["neural"]["top_active"] = [
            {
                "name": neuron_names[int(i)],
                "fire_fraction": float(fire_fraction[int(i)]),
                "mean_S": float(mean_s[int(i)]),
            }
            for i in order
        ]
    if mode == "calcium":
        summary["calcium"] = {
            "pulse": _stats(arrays["calcium_pulse"]),
            "force": _stats(arrays["calcium_force"]),
            "side_score": _stats(arrays["calcium_side_score"]),
            "unique_source_frames": int(np.unique(arrays["source_frame_index"][arrays["source_frame_index"] >= 0]).size),
        }
    if mode == "video":
        summary["video"] = {
            "motion_energy": _stats(arrays["video_motion_energy"]),
            "asymmetry": _stats(arrays["video_asymmetry"]),
            "flow_reliability": _stats(arrays["video_flow_reliability"]),
            "camera_shake": _stats(arrays["video_camera_shake"]),
            "zapbench_distance": _stats(arrays["video_zapbench_distance"]),
            "calibration_gate": _stats(arrays["video_calibration_gate"]),
            "calibration_raw_action_force": _stats(arrays["video_calibration_raw_action_force"]),
            "calibration_raw_startle": _stats(arrays["video_calibration_raw_startle"]),
            "cache_hit_fraction": float(np.mean(arrays["video_frame_cache_hit"] > 0.5)),
            "unique_source_frames": int(np.unique(arrays["source_frame_index"][arrays["source_frame_index"] >= 0]).size),
            "max_source_loop_index": int(np.max(arrays["source_loop_index"])) if arrays["source_loop_index"].size else 0,
        }
    return summary


def record_replay_run(
    *,
    mode: str,
    ticks: int,
    seed: int,
    out_dir: Path,
    label: str,
    log_every: int,
    calcium_replay_path: Path,
    calcium_condition: str,
    calcium_gain: float,
    video_file_name: str,
    video_sample_hz: float,
    video_gain: float,
    video_loop_source: bool,
    use_video_feature_cache: bool,
    video_calibration: dict[str, float] | None = None,
) -> dict[str, Any]:
    if mode not in {"calcium", "video"}:
        raise ValueError("mode must be calcium or video")
    out_dir.mkdir(parents=True, exist_ok=True)

    engine, _loop = build_zebrafish_simulation(
        food_positions=[],
        log_level="ERROR",
        record_neural_states=False,
        max_history=8,
        suppress_connectome_summary=True,
        seed=seed,
    )
    engine.reset(nervous_rebuild=False)
    body = engine.body
    ns = engine.nervous_system
    env = engine.environment
    if not isinstance(env, AquaticArenaEnvironment):
        raise RuntimeError("zebrafish environment missing")

    arrays = _alloc_recording_arrays(ticks, body.model, body, ns)
    source: dict[str, Any] = {}
    calcium_data: dict[str, Any] | None = None
    calcium_selected = np.zeros(0, dtype=np.int32)
    calcium_last_position = -1
    if mode == "calcium":
        calcium_data = _load_calcium_replay(calcium_replay_path)
        calcium_selected = _calcium_indices(calcium_data, calcium_condition)
        if calcium_selected.shape[0] == 0:
            raise RuntimeError(f"no calcium frames for condition {calcium_condition!r}")
        env.set_calcium_stimulus(
            enabled=True,
            gain=calcium_gain,
            source=f"zapbench-calcium:{calcium_condition}",
        )
        source = {
            "type": "zapbench_calcium_action_replay",
            "path": str(calcium_replay_path.resolve()),
            "condition": calcium_condition,
            "gain": float(calcium_gain),
            "frame_hz": ZAPBENCH_CALCIUM_FRAME_HZ,
            "selected_frames": int(calcium_selected.shape[0]),
        }

    pipeline: BackendVideoPipeline | None = None
    video_path: Path | None = None
    video_duration_s = 0.0
    video_last_frame_index = -1
    current_video_diag: dict[str, Any] = {}
    current_video_cache_hit = False
    current_source_loop_index = 0
    calibration = dict(DEFAULT_VIDEO_CALIBRATION)
    if video_calibration:
        calibration.update({key: float(value) for key, value in video_calibration.items() if key in calibration})
    if mode == "video":
        pipeline = BackendVideoPipeline(
            cache_root=SCRIPT_DIR / "cache",
            upload_root=DEFAULT_UPLOAD_DIR,
            action_calibration={},
        )
        video_path = _video_source_path(video_file_name)
        video_duration_s = _video_duration_s(video_path)
        if video_duration_s <= 0.0:
            raise RuntimeError(f"could not determine video duration for {video_path}")
        env.set_video_stimulus(enabled=True, gain=video_gain, file_name=Path(video_file_name).name)
        source = {
            "type": "backend_video_stimulus",
            "path": str(video_path.resolve()),
            "file_name": Path(video_file_name).name,
            "sample_hz": float(video_sample_hz),
            "gain": float(video_gain),
            "loop_source": bool(video_loop_source),
            "duration_s": float(video_duration_s),
            "feature_cache": str(DEFAULT_FRAME_CACHE.resolve()),
            "use_feature_cache": bool(use_video_feature_cache),
            "calibration": calibration,
        }

    t0 = time.perf_counter()
    tick_i = 0
    global_video_frame = 0
    while tick_i < ticks:
        if mode == "calcium":
            assert calcium_data is not None
            elapsed_s = max(0.0, engine.tick * zfc.PHYSICS_TIMESTEP_S)
            elapsed_frames = int(math.floor(elapsed_s * ZAPBENCH_CALCIUM_FRAME_HZ + 1e-9))
            position = int(elapsed_frames % calcium_selected.shape[0])
            if position != calcium_last_position:
                calcium_last_position = position
                row_index = int(calcium_selected[position])
                force = float(calcium_data["force"][row_index])
                env.push_calcium_action_frame(
                    {
                        "enabled": True,
                        "source": f"zapbench-calcium:{calcium_condition}",
                        "row": int(calcium_data["rows"][row_index]),
                        "frame_index": position,
                        "calcium_time_s": float(calcium_data["calcium_time_s"][row_index]),
                        "kick": float(calcium_data["kick"][row_index]),
                        "side_score": float(calcium_data["side_score"][row_index]),
                        "force": force,
                        "kick_score": float(calcium_data["kick_score"][row_index]),
                        "confidence": float(calcium_data["confidence"][row_index]),
                        "tail_phase": float(
                            zfc.TWO_PI
                            * elapsed_s
                            * (
                                zfc.TAIL_BEAT_FREQ_MIN_HZ
                                + force * (zfc.TAIL_BEAT_FREQ_MAX_HZ - zfc.TAIL_BEAT_FREQ_MIN_HZ)
                            )
                        ),
                    }
                )
            step = engine.step()
            _record_tick(
                arrays,
                tick_i,
                step,
                body=body,
                ns=ns,
                mode=mode,
                current_video_diag={},
                current_video_cache_hit=False,
                current_source_loop_index=0,
            )
            tick_i += 1
        else:
            assert pipeline is not None and video_path is not None
            source_elapsed_s = global_video_frame / max(1.0, float(video_sample_hz))
            if video_loop_source:
                current_source_loop_index = int(source_elapsed_s // video_duration_s)
                video_time_s = float(source_elapsed_s % video_duration_s)
            else:
                current_source_loop_index = 0
                video_time_s = min(float(source_elapsed_s), max(0.0, video_duration_s - 1e-3))
            features, current_video_diag, current_video_cache_hit = _load_or_extract_video_frame(
                pipeline,
                path=video_path,
                file_name=Path(video_file_name).name,
                frame_index=global_video_frame,
                video_time_s=video_time_s,
                sample_hz=video_sample_hz,
                cache_dir=DEFAULT_FRAME_CACHE,
                use_cache=use_video_feature_cache,
            )
            features, current_video_diag = _calibrate_video_features(features, current_video_diag, calibration)
            features["enabled"] = True
            env.push_video_stimulus_frame(features)
            ticks_to_advance, video_last_frame_index = _manual_video_ticks_for_frame(
                global_video_frame,
                video_sample_hz,
                video_last_frame_index,
            )
            for _ in range(ticks_to_advance):
                if tick_i >= ticks:
                    break
                step = engine.step()
                _record_tick(
                    arrays,
                    tick_i,
                    step,
                    body=body,
                    ns=ns,
                    mode=mode,
                    current_video_diag=current_video_diag,
                    current_video_cache_hit=current_video_cache_hit,
                    current_source_loop_index=current_source_loop_index,
                )
                tick_i += 1
            global_video_frame += 1

        if log_every > 0 and tick_i > 0 and (tick_i % log_every == 0 or tick_i == ticks):
            elapsed = time.perf_counter() - t0
            print(
                json.dumps(
                    {
                        "label": label,
                        "mode": mode,
                        "tick_i": int(tick_i),
                        "engine_tick": int(engine.tick),
                        "elapsed_s": round(elapsed, 3),
                        "speed_mm_s": round(float(arrays["speed_mm_s"][tick_i - 1]), 3),
                        "action_force": round(float(arrays["action_force"][tick_i - 1]), 4),
                        "body_pitch_rad": round(float(arrays["body_pitch_rad"][tick_i - 1]), 4),
                        "z_span_mm": round(float(arrays["z_span_mm"][tick_i - 1]), 4),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    wall_s = time.perf_counter() - t0
    summary = _summarize(
        arrays,
        label=label,
        mode=mode,
        seed=seed,
        ticks=ticks,
        wall_s=wall_s,
        source=source,
    )
    archive_path = out_dir / f"{label}_seed{seed}_{ticks}ticks.npz"
    summary_path = out_dir / f"{label}_seed{seed}_{ticks}ticks_summary.json"
    arrays_to_save = {key: value for key, value in arrays.items()}
    np.savez_compressed(archive_path, **arrays_to_save)
    summary["archive_path"] = str(archive_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calcium", "video"], required=True)
    parser.add_argument("--ticks", type=int, default=50_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", default="highrate_replay")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/out/high_rate_replay_validation_20260603/recordings"))
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--calcium-replay-path", type=Path, default=DEFAULT_REPLAY_PATH)
    parser.add_argument("--calcium-condition", default="all")
    parser.add_argument("--calcium-gain", type=float, default=1.0)
    parser.add_argument("--video-file-name", default="commons_tenggol_underwater.mp4")
    parser.add_argument("--video-sample-hz", type=float, default=10.0)
    parser.add_argument("--video-gain", type=float, default=1.0)
    parser.add_argument("--video-loop-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-feature-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-action-force-scale", type=float, default=DEFAULT_VIDEO_CALIBRATION["action_force_scale"])
    parser.add_argument("--video-action-side-scale", type=float, default=DEFAULT_VIDEO_CALIBRATION["action_side_scale"])
    parser.add_argument("--video-tail-target-scale", type=float, default=DEFAULT_VIDEO_CALIBRATION["tail_target_scale"])
    parser.add_argument("--video-startle-scale", type=float, default=DEFAULT_VIDEO_CALIBRATION["startle_scale"])
    parser.add_argument("--video-motion-scale", type=float, default=DEFAULT_VIDEO_CALIBRATION["motion_scale"])
    parser.add_argument("--video-confidence-scale", type=float, default=DEFAULT_VIDEO_CALIBRATION["confidence_scale"])
    parser.add_argument("--video-camera-shake-gate", type=float, default=DEFAULT_VIDEO_CALIBRATION["camera_shake_gate"])
    parser.add_argument("--video-reliability-floor", type=float, default=DEFAULT_VIDEO_CALIBRATION["reliability_floor"])
    parser.add_argument("--video-max-action-force", type=float, default=DEFAULT_VIDEO_CALIBRATION["max_action_force"])
    parser.add_argument("--video-min-action-force", type=float, default=DEFAULT_VIDEO_CALIBRATION["min_action_force"])
    args = parser.parse_args()

    summary = record_replay_run(
        mode=args.mode,
        ticks=args.ticks,
        seed=args.seed,
        out_dir=args.out_dir,
        label=args.label,
        log_every=args.log_every,
        calcium_replay_path=args.calcium_replay_path,
        calcium_condition=args.calcium_condition,
        calcium_gain=args.calcium_gain,
        video_file_name=args.video_file_name,
        video_sample_hz=args.video_sample_hz,
        video_gain=args.video_gain,
        video_loop_source=args.video_loop_source,
        use_video_feature_cache=args.video_feature_cache,
        video_calibration={
            "action_force_scale": args.video_action_force_scale,
            "action_side_scale": args.video_action_side_scale,
            "tail_target_scale": args.video_tail_target_scale,
            "startle_scale": args.video_startle_scale,
            "motion_scale": args.video_motion_scale,
            "confidence_scale": args.video_confidence_scale,
            "camera_shake_gate": args.video_camera_shake_gate,
            "reliability_floor": args.video_reliability_floor,
            "max_action_force": args.video_max_action_force,
            "min_action_force": args.video_min_action_force,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
