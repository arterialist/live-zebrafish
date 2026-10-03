"""Backend video-to-action extraction for zebrafish lab replay.

The browser is only a controller.  This module owns deterministic frame
decoding, stabilized optical-flow extraction, and ZAPBench-conditioned action
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from simulations.zebrafish import config as zfc
from simulations.zebrafish.simzfish_omr import SimZFishOMRActionAdapter


FRAME_W = 160
FRAME_H = 90
RAW_VIDEO_ACTION_CALIBRATION = {
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
DEFAULT_VIDEO_ACTION_CALIBRATION = {
    "action_force_scale": 0.90,
    "action_side_scale": 0.80,
    "tail_target_scale": 0.75,
    "startle_scale": 0.65,
    "motion_scale": 0.95,
    "confidence_scale": 0.95,
    "camera_shake_gate": 0.65,
    "reliability_floor": 0.30,
    "max_action_force": 0.90,
    "min_action_force": 0.02,
}


@dataclass(frozen=True)
class BackendVideoFrame:
    features: dict[str, Any]
    diagnostics: dict[str, Any]


class VideoCalibrationUnavailableError(RuntimeError):
    """Calibrated video extraction requires locally retained ZAPBench data."""


def safe_upload_name(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    stem = Path(file_name).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)[:80]
    safe_suffix = suffix if suffix in {".mp4", ".mov", ".m4v", ".webm", ".ogv", ".avi"} else ".mp4"
    digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:12]
    return f"upload_{digest}_{safe_stem or 'video'}{safe_suffix}"


class BackendVideoPipeline:
    """Extract robust visual features and ZAPBench-calibrated action per frame."""

    def __init__(
        self,
        *,
        cache_root: Path,
        upload_root: Path,
        action_calibration: dict[str, float] | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.upload_root = upload_root
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._calibration: ZapbenchVideoCalibrator | None = None
        self._calibration_lock = Lock()
        self._simzfish = SimZFishOMRActionAdapter()
        self._action_calibration = dict(
            DEFAULT_VIDEO_ACTION_CALIBRATION if action_calibration is None else RAW_VIDEO_ACTION_CALIBRATION
        )
        if action_calibration is not None:
            for key, value in action_calibration.items():
                if key in self._action_calibration:
                    self._action_calibration[key] = float(value)

    def uploaded_path(self, file_name: str) -> Path:
        return self.upload_root / safe_upload_name(file_name)

    def _require_calibration(self) -> ZapbenchVideoCalibrator:
        with self._calibration_lock:
            if self._calibration is None:
                try:
                    self._calibration = _load_zapbench_calibration(self.cache_root)
                except FileNotFoundError as exc:
                    raise VideoCalibrationUnavailableError(
                        f"calibrated video extraction is unavailable: {exc}"
                    ) from exc
            return self._calibration

    def extract(
        self,
        *,
        path: Path,
        file_name: str,
        frame_index: int,
        video_time_s: float,
        sample_hz: float,
    ) -> BackendVideoFrame:
        calibration = self._require_calibration()
        sample_hz = max(1.0, float(sample_hz))
        current = _read_frame(path, video_time_s)
        previous = _read_frame(path, max(0.0, float(video_time_s) - 1.0 / sample_hz))
        if current is None:
            raise ValueError(f"could not decode video frame at {video_time_s:.3f}s from {path}")
        if previous is None:
            previous = current
        current_gray, current_eq = self._preprocess(current)
        previous_gray, previous_eq = self._preprocess(previous)
        flow_raw = cv2.calcOpticalFlowFarneback(
            previous_eq,
            current_eq,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=19,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        camera_flow, affine_ok, inlier_ratio, global_rotation = _global_camera_flow(previous_eq, current_eq)
        residual_flow = flow_raw - camera_flow
        features, diagnostics = self._features_from_flow(
            current_gray=current_gray,
            current_eq=current_eq,
            previous_gray=previous_gray,
            flow_raw=flow_raw,
            camera_flow=camera_flow,
            residual_flow=residual_flow,
            affine_ok=affine_ok,
            inlier_ratio=inlier_ratio,
            global_rotation=global_rotation,
        )
        latent, simzfish = self._simzfish.action_from_frame(
            current_gray=current_gray,
            previous_gray=previous_gray,
            residual_flow=residual_flow,
            features=features,
            diagnostics=diagnostics,
            file_name=file_name,
            frame_index=frame_index,
            video_time_s=video_time_s,
            sample_hz=sample_hz,
        )
        zap_action, zapbench = calibration.action_from_features(features, diagnostics)
        features.update(latent.to_legacy_action_fields())
        features.update(
            {
                "file_name": file_name,
                "frame_index": int(frame_index),
                "video_time_s": float(video_time_s),
                "backend_extracted": True,
                "zapbench_estimated_action_kick": zap_action["action_kick"],
                "zapbench_estimated_action_force": zap_action["action_force"],
                "zapbench_estimated_action_side_score": zap_action["action_side_score"],
                "zapbench_estimated_action_kick_score": zap_action["action_kick_score"],
                "zapbench_estimated_action_confidence": zap_action["action_confidence"],
                "zapbench_estimated_row": zap_action["zapbench_row"],
            }
        )
        features, calibration_diag = _apply_video_action_calibration(
            features,
            diagnostics,
            self._action_calibration,
        )
        diagnostics.update(
            {
                "backend": "opencv-farneback-ransac-simzfish-omr",
                "primary_action_model": "simzfish_retina_omr",
                "zapbench_adapter_role": "auxiliary_estimated_covariates_not_primary_action",
            }
        )
        diagnostics.update(simzfish)
        diagnostics.update(zapbench)
        diagnostics.update(calibration_diag)
        return BackendVideoFrame(features=features, diagnostics=diagnostics)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame = cv2.resize(frame_bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        eq = self._clahe.apply(gray)
        return gray.astype(np.float32) / 255.0, eq

    def _features_from_flow(
        self,
        *,
        current_gray: np.ndarray,
        current_eq: np.ndarray,
        previous_gray: np.ndarray,
        flow_raw: np.ndarray,
        camera_flow: np.ndarray,
        residual_flow: np.ndarray,
        affine_ok: bool,
        inlier_ratio: float,
        global_rotation: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        h, w = current_gray.shape
        left = np.s_[:, : w // 2]
        right = np.s_[:, w // 2 :]
        up = np.s_[: h // 2, :]
        down = np.s_[h // 2 :, :]
        frame_delta = np.abs(current_gray - previous_gray)
        residual_mag = np.linalg.norm(residual_flow, axis=2)
        camera_mag = np.linalg.norm(camera_flow, axis=2)
        raw_mag = np.linalg.norm(flow_raw, axis=2)
        raw_u = flow_raw[:, :, 0]
        raw_v = flow_raw[:, :, 1]
        residual_u = residual_flow[:, :, 0]
        residual_v = residual_flow[:, :, 1]

        light = float(np.mean(current_gray))
        contrast = float(np.quantile(current_gray, 0.95) - np.quantile(current_gray, 0.05))
        compression_noise = _high_frequency_noise(current_eq)
        camera_motion = _robust_mean(camera_mag) / max(1.0, w * 0.045)
        residual_motion = _robust_mean(residual_mag) / max(1.0, w * 0.035)
        raw_motion = _robust_mean(raw_mag) / max(1.0, w * 0.04)
        motion_energy = float(np.clip(0.58 * residual_motion + 0.27 * raw_motion + 2.2 * np.mean(frame_delta), 0.0, 1.0))
        flow_coherence = float(np.clip(_vector_coherence(flow_raw), 0.0, 1.0))
        residual_coherence = float(np.clip(_vector_coherence(residual_flow), 0.0, 1.0))
        camera_shake = float(np.clip(camera_motion * (1.0 - inlier_ratio) + 0.35 * compression_noise, 0.0, 1.0))
        reliability = float(np.clip(0.40 + 0.42 * flow_coherence + 0.18 * inlier_ratio - 0.35 * camera_shake, 0.0, 1.0))

        left_salience = _salience(current_gray[left])
        right_salience = _salience(current_gray[right])
        up_salience = _salience(current_gray[up])
        down_salience = _salience(current_gray[down])
        left_motion = _robust_mean(residual_mag[left])
        right_motion = _robust_mean(residual_mag[right])
        up_motion = _robust_mean(residual_mag[up])
        down_motion = _robust_mean(residual_mag[down])

        visual_left = float(np.clip(0.32 * np.mean(current_gray[left]) + 1.35 * left_salience + 1.8 * left_motion / w, 0.0, 1.0))
        visual_right = float(np.clip(0.32 * np.mean(current_gray[right]) + 1.35 * right_salience + 1.8 * right_motion / w, 0.0, 1.0))
        visual_up = float(np.clip(0.28 * np.mean(current_gray[up]) + 1.1 * up_salience + 1.2 * up_motion / h, 0.0, 1.0))
        visual_down = float(np.clip(0.28 * np.mean(current_gray[down]) + 1.1 * down_salience + 1.2 * down_motion / h, 0.0, 1.0))

        left_flow = _robust_mean(np.linalg.norm(flow_raw[left], axis=2)) / max(1.0, w * 0.035)
        right_flow = _robust_mean(np.linalg.norm(flow_raw[right], axis=2)) / max(1.0, w * 0.035)
        lateral_left = _robust_mean(np.maximum(0.0, residual_u[right])) / max(1.0, w * 0.022)
        lateral_right = _robust_mean(np.maximum(0.0, -residual_u[left])) / max(1.0, w * 0.022)
        mean_u = float(_robust_mean(raw_u) / max(1.0, w * 0.025))
        mean_v = float(_robust_mean(raw_v) / max(1.0, h * 0.035))
        residual_yaw = float((_robust_mean(residual_u[right]) - _robust_mean(residual_u[left])) / max(1.0, w * 0.025))
        visual_asym = (visual_left - visual_right) / max(0.05, visual_left + visual_right)
        flow_asym = (right_flow - left_flow) / max(0.05, right_flow + left_flow)
        asymmetry = float(np.clip(0.42 * visual_asym + 0.40 * flow_asym + 0.18 * residual_yaw, -1.0, 1.0))
        startle = float(np.clip((motion_energy - 0.15) * 3.5 + max(0.0, camera_motion - 0.45) * 0.35, 0.0, 1.0))

        return (
            {
                "visual_left": visual_left,
                "visual_right": visual_right,
                "optic_flow_left": float(np.clip(left_flow * reliability, 0.0, 1.0)),
                "optic_flow_right": float(np.clip(right_flow * reliability, 0.0, 1.0)),
                "lateral_line_left": float(np.clip(lateral_left * reliability, 0.0, 1.0)),
                "lateral_line_right": float(np.clip(lateral_right * reliability, 0.0, 1.0)),
                "visual_up": visual_up,
                "visual_down": visual_down,
                "light_level": float(np.clip(light, 0.0, 1.0)),
                "startle": startle,
                "motion_energy": motion_energy,
                "asymmetry": asymmetry,
            },
            {
                "backend": "opencv-farneback-ransac-zapbench",
                "true_optical_flow": True,
                "camera_stabilized": bool(affine_ok),
                "flow_coherence": flow_coherence,
                "residual_flow_coherence": residual_coherence,
                "camera_motion": float(np.clip(camera_motion, 0.0, 1.0)),
                "camera_shake": camera_shake,
                "compression_noise": float(np.clip(compression_noise, 0.0, 1.0)),
                "contrast": float(np.clip(contrast, 0.0, 1.0)),
                "flow_reliability": reliability,
                "global_horizontal_flow": float(np.clip(mean_u, -1.0, 1.0)),
                "global_vertical_flow": float(np.clip(mean_v, -1.0, 1.0)),
                "global_rotation": float(np.clip(global_rotation, -1.0, 1.0)),
                "affine_inlier_ratio": float(np.clip(inlier_ratio, 0.0, 1.0)),
            },
        )


def _video_action_calibration_gate(diagnostics: dict[str, Any], calibration: dict[str, float]) -> float:
    reliability = float(np.clip(float(diagnostics.get("flow_reliability", 0.0)), 0.0, 1.0))
    shake = float(np.clip(float(diagnostics.get("camera_shake", 0.0)), 0.0, 1.0))
    reliability_floor = float(np.clip(calibration.get("reliability_floor", 0.30), 0.0, 1.0))
    shake_gate = max(0.0, float(calibration.get("camera_shake_gate", 0.0)))
    reliability_term = reliability_floor + (1.0 - reliability_floor) * reliability
    shake_term = float(np.clip(1.0 - shake_gate * shake, 0.05, 1.0))
    return float(np.clip(reliability_term * shake_term, 0.05, 1.0))


def _apply_video_action_calibration(
    features: dict[str, Any],
    diagnostics: dict[str, Any],
    calibration: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(features)
    gate = _video_action_calibration_gate(diagnostics, calibration)
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
    raw_confidence = float(np.clip(float(out.get("action_confidence", 0.0)), 0.0, 1.0))
    raw_startle = float(np.clip(float(out.get("startle", 0.0)), 0.0, 1.0))
    raw_motion = float(np.clip(float(out.get("motion_energy", 0.0)), 0.0, 1.0))

    force = float(np.clip(raw_force * force_scale * gate, 0.0, max_force))
    side = float(np.clip(raw_side * side_scale * gate, -1.0, 1.0))
    confidence = float(np.clip(raw_confidence * confidence_scale * gate, 0.0, 1.0))
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
    out["tail_frequency_hz"] = (
        float(zfc.TAIL_BEAT_FREQ_MIN_HZ + force * (zfc.TAIL_BEAT_FREQ_MAX_HZ - zfc.TAIL_BEAT_FREQ_MIN_HZ))
        if force > 0.0
        else 0.0
    )
    try:
        targets = list(out.get("tail_targets", []))
    except TypeError:
        targets = []
    calibrated_targets = [
        float(np.clip(float(value) * tail_scale * gate, -1.0, 1.0))
        for value in targets[: zfc.N_BODY_SEGMENTS]
    ]
    while len(calibrated_targets) < zfc.N_BODY_SEGMENTS:
        calibrated_targets.append(0.0)
    out["tail_targets"] = calibrated_targets
    out["action_calibration_gate"] = gate
    out["action_raw_force"] = raw_force
    out["action_raw_side_score"] = raw_side
    out["action_raw_confidence"] = raw_confidence
    out["action_raw_startle"] = raw_startle
    out["action_raw_motion_energy"] = raw_motion

    return out, {
        "video_action_calibrated": True,
        "video_action_calibration": dict(calibration),
        "calibration_gate": gate,
        "calibration_raw_action_force": raw_force,
        "calibration_raw_action_side_score": raw_side,
        "calibration_raw_action_confidence": raw_confidence,
        "calibration_raw_startle": raw_startle,
        "calibration_raw_motion_energy": raw_motion,
    }


class ZapbenchVideoCalibrator:
    def __init__(self, *, stimuli: np.ndarray, labels: dict[str, np.ndarray]) -> None:
        self.stimuli = np.asarray(stimuli, dtype=np.float32)
        self.mean = self.stimuli.mean(axis=0).astype(np.float32)
        self.std = self.stimuli.std(axis=0).astype(np.float32)
        self.std = np.where(self.std < 1e-5, 1.0, self.std).astype(np.float32)
        self.zstimuli = ((self.stimuli - self.mean) / self.std).astype(np.float32)
        self.labels = labels

    def action_from_features(
        self, features: dict[str, float], diagnostics: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        vector = _features_to_zapbench_vector(features, diagnostics)
        zvec = (vector - self.mean) / self.std
        dist = np.linalg.norm(self.zstimuli - zvec.reshape(1, -1), axis=1)
        k = min(64, dist.shape[0])
        idx = np.argpartition(dist, k - 1)[:k]
        nearest = idx[np.argsort(dist[idx])]
        sigma = max(0.35, float(np.median(dist[nearest])) + 1e-6)
        weights = np.exp(-0.5 * (dist[nearest] / sigma) ** 2).astype(np.float32)
        weights /= max(1e-8, float(np.sum(weights)))
        force = float(np.dot(weights, self.labels["force"][nearest]))
        kick_score = float(np.dot(weights, self.labels["kick"][nearest]))
        side_score = float(np.dot(weights, self.labels["side"][nearest]))
        kick = 1.0 if kick_score >= 0.42 or force >= 0.24 or features.get("startle", 0.0) > 0.55 else 0.0
        confidence = float(np.clip(1.0 / (1.0 + float(np.mean(dist[nearest]))), 0.0, 1.0))
        confidence = float(np.clip(0.65 * confidence + 0.35 * diagnostics.get("flow_reliability", 0.0), 0.0, 1.0))
        return (
            {
                "action_kick": kick,
                "action_force": float(np.clip(force, 0.0, 1.0)),
                "action_side_score": float(np.clip(side_score, -1.0, 1.0)),
                "action_kick_score": float(np.clip(kick_score, 0.0, 1.0)),
                "action_confidence": confidence,
                "zapbench_row": int(nearest[0]),
            },
            {
                "zapbench_grounded": True,
                "zapbench_row": int(nearest[0]),
                "zapbench_distance": float(dist[nearest[0]]),
                "zapbench_neighbor_rows": [int(v) for v in nearest[:8]],
                "zapbench_neighbor_distance_mean": float(np.mean(dist[nearest])),
                "zapbench_vector": [float(v) for v in vector.tolist()],
            },
        )


@lru_cache(maxsize=4)
def _load_zapbench_calibration(cache_root: Path) -> ZapbenchVideoCalibrator:
    stimulus_path = cache_root / "zapbench" / "zapbench_stimulus_features.npz"
    labels_path = cache_root.parent / "out" / "zapbench_ephys_action_decoder" / "direct_ephys_labels.npz"
    if not stimulus_path.exists():
        raise FileNotFoundError(f"missing ZAPBench stimulus cache: {stimulus_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing ZAPBench direct ephys labels: {labels_path}")
    with np.load(stimulus_path, allow_pickle=False) as data:
        stimuli = np.asarray(data["features"], dtype=np.float32)
    with np.load(labels_path, allow_pickle=False) as data:
        rows = np.asarray(data["rows"], dtype=np.int32)
        keep = rows < stimuli.shape[0]
        full_force = np.zeros(stimuli.shape[0], dtype=np.float32)
        full_kick = np.zeros(stimuli.shape[0], dtype=np.float32)
        full_side = np.zeros(stimuli.shape[0], dtype=np.float32)
        full_force[rows[keep]] = np.asarray(data["force"], dtype=np.float32)[keep]
        full_kick[rows[keep]] = np.asarray(data["kick"], dtype=np.float32)[keep]
        full_side[rows[keep]] = np.asarray(data["side"], dtype=np.float32)[keep]
    return ZapbenchVideoCalibrator(
        stimuli=stimuli,
        labels={"force": full_force, "kick": full_kick, "side": full_side},
    )


def _read_frame(path: Path, time_s: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps > 0:
            frame_index = max(0, int(round(float(time_s) * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_s) * 1000.0))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _global_camera_flow(prev_eq: np.ndarray, curr_eq: np.ndarray) -> tuple[np.ndarray, bool, float, float]:
    h, w = prev_eq.shape
    zero = np.zeros((h, w, 2), dtype=np.float32)
    pts0 = cv2.goodFeaturesToTrack(
        prev_eq,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    if pts0 is None or pts0.shape[0] < 12:
        return zero, False, 0.0, 0.0
    pts1, status, _ = cv2.calcOpticalFlowPyrLK(prev_eq, curr_eq, pts0, None)
    if pts1 is None or status is None:
        return zero, False, 0.0, 0.0
    good = status.reshape(-1).astype(bool)
    p0 = pts0.reshape(-1, 2)[good]
    p1 = pts1.reshape(-1, 2)[good]
    if p0.shape[0] < 12:
        return zero, False, 0.0, 0.0
    affine, inliers = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.4,
        maxIters=500,
        confidence=0.98,
    )
    if affine is None:
        median = np.median(p1 - p0, axis=0).astype(np.float32)
        zero[:, :, 0] = median[0]
        zero[:, :, 1] = median[1]
        return zero, False, 0.0, 0.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped_x = affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]
    warped_y = affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]
    out = np.zeros((h, w, 2), dtype=np.float32)
    out[:, :, 0] = warped_x - xx
    out[:, :, 1] = warped_y - yy
    inlier_ratio = float(np.mean(inliers)) if inliers is not None and inliers.size else 0.0
    rotation = float(np.arctan2(float(affine[1, 0]), float(affine[0, 0])) / np.pi)
    return out, True, inlier_ratio, rotation


def _features_to_zapbench_vector(features: dict[str, float], diagnostics: dict[str, Any]) -> np.ndarray:
    visual_left = float(features.get("visual_left", 0.0))
    visual_right = float(features.get("visual_right", 0.0))
    flow_left = float(features.get("optic_flow_left", 0.0))
    flow_right = float(features.get("optic_flow_right", 0.0))
    motion = float(features.get("motion_energy", 0.0))
    startle = float(features.get("startle", 0.0))
    asymmetry = float(features.get("asymmetry", 0.0))
    light = float(features.get("light_level", 0.0))
    vertical = float(features.get("visual_up", 0.0) - features.get("visual_down", 0.0))
    global_u = float(diagnostics.get("global_horizontal_flow", 0.0))
    global_rotation = float(diagnostics.get("global_rotation", 0.0))
    flow_strength = max(flow_left, flow_right, motion)
    vec = np.zeros(26, dtype=np.float32)
    # Appendix B.6 covariate schema.  These are not raw projected video frames;
    # this is an out-of-distribution estimate of the controlled VR stimulus
    # covariates from arbitrary video.  Columns 22-25 are specimen identity in
    # ZAPBench and remain zero for this single-specimen bridge.
    vec[0] = -1.0 if global_u < -0.08 else 1.0 if global_u > 0.08 else 0.0
    vec[1] = 1.0 if flow_strength > 0.10 and diagnostics.get("flow_coherence", 0.0) > 0.35 else 0.0
    vec[2] = -1.0 if global_u + 0.35 * asymmetry < -0.05 else 1.0 if global_u + 0.35 * asymmetry > 0.05 else 0.0
    vec[3] = 1.0 if motion > 0.05 and diagnostics.get("contrast", 0.0) > 0.08 else 0.0
    vec[4] = -1.0 if light < 0.28 else 1.0 if light > 0.62 or startle > 0.35 else 0.0
    vec[5] = 1.0 if startle > 0.15 or abs(light - 0.5) > 0.22 else 0.0
    vec[6] = -1.0 if visual_left < 0.10 else 1.0 if visual_left > 0.18 else 0.0
    vec[7] = -1.0 if visual_right < 0.10 else 1.0 if visual_right > 0.18 else 0.0
    vec[8] = 1.0 if max(visual_left, visual_right) > 0.10 else 0.0
    vec[9] = 1.0 if abs(asymmetry) + 0.4 * motion > 0.12 else 0.0
    vec[10] = -1.0 if asymmetry > 0.05 else 1.0 if asymmetry < -0.05 else 0.0
    vec[11] = 1.0 if abs(asymmetry) < 0.05 and motion > 0.08 else 0.0
    vec[12] = 1.0 if abs(asymmetry) > 0.05 or motion > 0.12 else 0.0
    vec[13] = 1.0 if vertical > 0.10 else 0.0
    vec[14] = 1.0 if vertical < -0.10 else 0.0
    vec[15] = 1.0 if abs(vertical) <= 0.10 and motion > 0.10 else 0.0
    vec[16] = float(np.clip(round(np.clip(motion, 0.0, 0.9) * 10.0) / 10.0, 0.0, 0.9))
    vec[17] = 1.0 if abs(vertical) > 0.08 else 0.0
    vec[18] = 1.0 if flow_strength > 0.12 else 0.0
    vec[19] = -1.0 if global_rotation < -0.015 else 1.0 if global_rotation > 0.015 else 0.0
    vec[20] = 1.0 if abs(global_rotation) > 0.015 else 0.0
    vec[21] = 1.0 if light < 0.18 else 0.0
    vec[22:26] = 0.0
    return vec


def _salience(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(np.abs(values - float(np.mean(values)))))


def _robust_mean(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    hi = float(np.quantile(flat, 0.92))
    clipped = flat[flat <= hi]
    return float(np.mean(clipped)) if clipped.size else float(np.mean(flat))


def _vector_coherence(flow: np.ndarray) -> float:
    vectors = flow.reshape(-1, 2).astype(np.float32)
    mag = np.linalg.norm(vectors, axis=1)
    mean_mag = float(np.mean(mag))
    if mean_mag < 1e-6:
        return 0.0
    mean_vec_mag = float(np.linalg.norm(np.mean(vectors, axis=0)))
    return mean_vec_mag / mean_mag


def _high_frequency_noise(gray_u8: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    return float(np.clip(np.std(lap) / 160.0, 0.0, 1.0))
