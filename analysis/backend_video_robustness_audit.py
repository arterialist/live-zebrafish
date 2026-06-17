"""Offline robustness audit for the backend zebrafish video-to-action path.

This script runs every cached selected video through the same backend OpenCV
flow, camera-stabilization, simZFish-style retinal OMR adapter, and auxiliary
ZAPBench covariate mapper used by the lab REST endpoint.  It does not advance
MuJoCo; it isolates the video decoder/action layer so source-video quality and
action emission can be evaluated across many clips before expensive embodied
50k-tick recordings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab.video_pipeline import BackendVideoPipeline, _global_camera_flow


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "out" / "backend_video_robustness"
CLIP_DIR = SCRIPT_DIR / "cache" / "video_stimuli" / "clips"
UPLOAD_DIR = SCRIPT_DIR / "cache" / "video_stimuli" / "uploads"
PUBLISHED_LOCOMOTION = (
    SCRIPT_DIR
    / "out"
    / "simzfish_calibration_targets_20260603"
    / "published_locomotion_targets.csv"
)


QUALITY_THRESHOLDS = {
    "low_flow_reliability": ("flow_reliability", 0.25, "lt"),
    "high_camera_shake": ("camera_shake", 0.75, "gt"),
    "high_compression_noise": ("compression_noise", 0.30, "gt"),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _duration_s(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0, 0.0, 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if fps > 0.0 else 0.0
        return duration, fps, frames
    finally:
        cap.release()


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_numeric(f"{prefix}_{key}" if prefix else str(key), nested, out)
        return
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        out[prefix] = float(value)
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[f"{prefix}_mean"] = float(np.mean(arr))
            out[f"{prefix}_rms"] = float(np.sqrt(np.mean(arr * arr)))
            out[f"{prefix}_abs_max"] = float(np.max(np.abs(arr)))


def _read_sampled_frame(cap: cv2.VideoCapture, fps: float, time_s: float) -> np.ndarray | None:
    if fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_s * fps))))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s * 1000.0))
    ok, frame = cap.read()
    return frame if ok else None


def _extract_clip(
    *,
    path: Path,
    sample_hz: float,
    seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duration, fps, frame_count = _duration_s(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"error": "could not open clip", "duration_s": duration}
    pipeline = BackendVideoPipeline(cache_root=SCRIPT_DIR / "cache", upload_root=UPLOAD_DIR)
    rows: list[dict[str, Any]] = []
    errors = 0
    usable_s = min(float(seconds), duration) if duration > 0 else float(seconds)
    sample_hz = max(1.0, float(sample_hz))
    times = np.arange(0.0, max(0.0, usable_s), 1.0 / sample_hz, dtype=np.float64)
    previous_bgr: np.ndarray | None = None
    try:
        for frame_index, video_time_s in enumerate(times):
            current = _read_sampled_frame(cap, fps, float(video_time_s))
            if current is None:
                errors += 1
                continue
            previous = previous_bgr if previous_bgr is not None else current
            previous_bgr = current

            try:
                current_gray, current_eq = pipeline._preprocess(current)  # noqa: SLF001
                previous_gray, previous_eq = pipeline._preprocess(previous)  # noqa: SLF001
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
                camera_flow, affine_ok, inlier_ratio, global_rotation = _global_camera_flow(
                    previous_eq, current_eq
                )
                residual_flow = flow_raw - camera_flow
                features, diagnostics = pipeline._features_from_flow(  # noqa: SLF001
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
                latent, simzfish = pipeline._simzfish.action_from_frame(  # noqa: SLF001
                    current_gray=current_gray,
                    previous_gray=previous_gray,
                    residual_flow=residual_flow,
                    features=features,
                    diagnostics=diagnostics,
                    file_name=path.name,
                    frame_index=frame_index,
                    video_time_s=float(video_time_s),
                    sample_hz=sample_hz,
                )
                zap_action, zapbench = pipeline._calibration.action_from_features(  # noqa: SLF001
                    features, diagnostics
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                rows.append(
                    {
                        "clip": path.stem,
                        "frame_index": frame_index,
                        "video_time_s": float(video_time_s),
                        "error": repr(exc),
                    }
                )
                continue

            features.update(latent.to_legacy_action_fields())
            features.update(
                {
                    "file_name": path.name,
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
            diagnostics.update(
                {
                    "backend": "opencv-farneback-ransac-simzfish-omr",
                    "primary_action_model": "simzfish_retina_omr",
                    "zapbench_adapter_role": "auxiliary_estimated_covariates_not_primary_action",
                }
            )
            diagnostics.update(simzfish)
            diagnostics.update(zapbench)

            row: dict[str, Any] = {
                "clip": path.stem,
                "frame_index": frame_index,
                "video_time_s": float(video_time_s),
                "action_bout_type": str(features.get("action_bout_type", "none")),
            }
            for source in (features, diagnostics):
                for key, value in source.items():
                    if key == "action_bout_type":
                        continue
                    _flatten_numeric(str(key), value, row)
            for event, (key, threshold, direction) in QUALITY_THRESHOLDS.items():
                value = float(row.get(key, 0.0))
                row[event] = float(value < threshold if direction == "lt" else value > threshold)
            rows.append(row)
    finally:
        cap.release()
    meta = {
        "duration_s": duration,
        "fps": fps,
        "source_frame_count": frame_count,
        "sampled_seconds": usable_s,
        "sample_hz": sample_hz,
        "decode_errors": errors,
        "sampled_frames": len(rows),
    }
    return rows, meta


def _event_count(bouts: list[str]) -> int:
    count = 0
    previous = "coast"
    for bout in bouts:
        noncoast = bout not in {"", "none", "coast"}
        previous_noncoast = previous not in {"", "none", "coast"}
        if noncoast and (not previous_noncoast or bout != previous):
            count += 1
        previous = bout
    return count


def _load_published_targets() -> dict[str, float]:
    if not PUBLISHED_LOCOMOTION.exists():
        return {}
    rows: list[dict[str, float]] = []
    with PUBLISHED_LOCOMOTION.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "bout_frequency_hz": float(row.get("bout_frequency_hz") or 0.0),
                        "left_fraction": float(row.get("left_fraction") or 0.0),
                        "forward_fraction": float(row.get("forward_fraction") or 0.0),
                        "right_fraction": float(row.get("right_fraction") or 0.0),
                    }
                )
            except ValueError:
                continue
    out: dict[str, float] = {}
    for key in ("bout_frequency_hz", "left_fraction", "forward_fraction", "right_fraction"):
        stats = _stats([row[key] for row in rows])
        for stat_key, value in stats.items():
            out[f"published_{key}_{stat_key}"] = value
    return out


def _summarize_clip(
    *,
    clip: str,
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    published: dict[str, float],
) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    duration = max(1e-9, float(meta.get("sampled_seconds", 0.0)))
    bouts = [str(row.get("action_bout_type", "none")) for row in valid]
    counts = Counter(bouts)
    event_count = _event_count(bouts)
    noncoast_frames = sum(1 for b in bouts if b not in {"", "none", "coast"})
    coast_nonzero_force = sum(
        1
        for row, bout in zip(valid, bouts)
        if bout in {"", "none", "coast"} and float(row.get("action_force", 0.0)) > 1e-6
    )
    event_hz = event_count / duration
    command_hz = noncoast_frames / duration
    published_mean = float(published.get("published_bout_frequency_hz_mean", 0.0))
    published_p05 = float(published.get("published_bout_frequency_hz_p05", 0.0))
    published_p95 = float(published.get("published_bout_frequency_hz_p95", 0.0))
    return {
        "clip": clip,
        **meta,
        "valid_frames": len(valid),
        "error_rate": float(meta.get("decode_errors", 0)) / max(1, len(rows)),
        "noncoast_frames": noncoast_frames,
        "noncoast_frame_fraction": noncoast_frames / max(1, len(valid)),
        "noncoast_event_count": event_count,
        "noncoast_event_frequency_hz": event_hz,
        "noncoast_command_frequency_hz": command_hz,
        "event_frequency_ratio_to_published_mean": event_hz / published_mean if published_mean > 0 else 0.0,
        "event_frequency_inside_published_p05_p95": float(published_p05 <= event_hz <= published_p95),
        "coast_nonzero_force_frames": coast_nonzero_force,
        "coast_nonzero_force_fraction": coast_nonzero_force / max(1, len(valid)),
        "coast_frames": counts.get("coast", 0),
        "startle_c_bend_frames": counts.get("startle_c_bend", 0),
        "omr_forward_bout_frames": counts.get("omr_forward_bout", 0),
        "omr_turn_left_frames": counts.get("omr_turn_left", 0),
        "omr_turn_right_frames": counts.get("omr_turn_right", 0),
        "coast_fraction": counts.get("coast", 0) / max(1, len(valid)),
        "startle_fraction": counts.get("startle_c_bend", 0) / max(1, len(valid)),
        "forward_fraction": counts.get("omr_forward_bout", 0) / max(1, len(valid)),
        "left_turn_fraction": counts.get("omr_turn_left", 0) / max(1, len(valid)),
        "right_turn_fraction": counts.get("omr_turn_right", 0) / max(1, len(valid)),
        "action_force_mean": _stats([float(row.get("action_force", 0.0)) for row in valid])["mean"],
        "action_force_p95": _stats([float(row.get("action_force", 0.0)) for row in valid])["p95"],
        "action_force_max": _stats([float(row.get("action_force", 0.0)) for row in valid])["max"],
        "action_confidence_mean": _stats([float(row.get("action_confidence", 0.0)) for row in valid])["mean"],
        "flow_reliability_mean": _stats([float(row.get("flow_reliability", 0.0)) for row in valid])["mean"],
        "flow_reliability_p05": _stats([float(row.get("flow_reliability", 0.0)) for row in valid])["p05"],
        "camera_shake_mean": _stats([float(row.get("camera_shake", 0.0)) for row in valid])["mean"],
        "compression_noise_mean": _stats([float(row.get("compression_noise", 0.0)) for row in valid])["mean"],
        "zapbench_distance_mean": _stats([float(row.get("zapbench_distance", 0.0)) for row in valid])["mean"],
        "zapbench_distance_p95": _stats([float(row.get("zapbench_distance", 0.0)) for row in valid])["p95"],
        "low_flow_reliability_frames": int(sum(float(row.get("low_flow_reliability", 0.0)) for row in valid)),
        "high_camera_shake_frames": int(sum(float(row.get("high_camera_shake", 0.0)) for row in valid)),
        "high_compression_noise_frames": int(sum(float(row.get("high_compression_noise", 0.0)) for row in valid)),
        "quality_event_fraction": float(
            np.mean(
                [
                    max(
                        float(row.get("low_flow_reliability", 0.0)),
                        float(row.get("high_camera_shake", 0.0)),
                        float(row.get("high_compression_noise", 0.0)),
                    )
                    for row in valid
                ]
            )
        )
        if valid
        else 0.0,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_bout_rate(summary: list[dict[str, Any]], published: dict[str, float], out_dir: Path) -> Path:
    labels = [str(row["clip"]) for row in summary]
    rates = [float(row["noncoast_event_frequency_hz"]) for row in summary]
    fig, ax = plt.subplots(figsize=(15, 6))
    y = np.arange(len(labels))
    ax.barh(y, rates, color="#2696d8")
    mean = float(published.get("published_bout_frequency_hz_mean", 0.0))
    p05 = float(published.get("published_bout_frequency_hz_p05", 0.0))
    p95 = float(published.get("published_bout_frequency_hz_p95", 0.0))
    ax.axvspan(p05, p95, color="#8bbf6a", alpha=0.16, label="published p05-p95")
    ax.axvline(mean, color="#8bbf6a", linewidth=2.0, label="published mean")
    ax.set_yticks(y, labels)
    ax.set_xlabel("transition-defined noncoast event frequency (Hz)")
    ax.set_title("Backend video action event rate vs published simZFish/Z-Robot locomotion target")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = out_dir / "bout_rate_vs_published.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_bout_fractions(summary: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [str(row["clip"]) for row in summary]
    keys = [
        ("coast_fraction", "coast", "#909090"),
        ("startle_fraction", "startle", "#d6604d"),
        ("forward_fraction", "forward", "#4393c3"),
        ("left_turn_fraction", "left", "#8073ac"),
        ("right_turn_fraction", "right", "#4dac26"),
    ]
    fig, ax = plt.subplots(figsize=(16, 7))
    bottom = np.zeros(len(summary), dtype=np.float64)
    x = np.arange(len(summary))
    for key, label, color in keys:
        vals = np.asarray([float(row[key]) for row in summary], dtype=np.float64)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x, labels, rotation=50, ha="right")
    ax.set_ylabel("frame fraction")
    ax.set_title("Backend action state occupancy across selected videos")
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = out_dir / "bout_type_fractions.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_quality(summary: list[dict[str, Any]], out_dir: Path) -> Path:
    x = np.asarray([float(row["quality_event_fraction"]) for row in summary], dtype=np.float64)
    y = np.asarray([float(row["noncoast_event_frequency_hz"]) for row in summary], dtype=np.float64)
    force = np.asarray([float(row["action_force_mean"]) for row in summary], dtype=np.float64)
    labels = [str(row["clip"]) for row in summary]
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x, y, c=force, cmap="viridis", s=80, edgecolor="white", linewidth=0.8)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=7, alpha=0.82, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("quality event frame fraction")
    ax.set_ylabel("noncoast event frequency (Hz)")
    ax.set_title("Source-video quality vs emitted action events")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="mean action force")
    fig.tight_layout()
    path = out_dir / "quality_vs_action.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_metric_heatmap(summary: list[dict[str, Any]], out_dir: Path) -> Path:
    metrics = [
        "noncoast_event_frequency_hz",
        "noncoast_frame_fraction",
        "action_force_mean",
        "action_force_p95",
        "flow_reliability_mean",
        "camera_shake_mean",
        "compression_noise_mean",
        "quality_event_fraction",
        "zapbench_distance_mean",
        "coast_nonzero_force_fraction",
    ]
    labels = [str(row["clip"]) for row in summary]
    mat = np.asarray([[float(row.get(metric, 0.0)) for metric in metrics] for row in summary])
    scaled = mat.copy()
    for j in range(scaled.shape[1]):
        col = scaled[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        scaled[:, j] = 0.0 if hi <= lo else (col - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(scaled, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(metrics)), metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2g}", ha="center", va="center", fontsize=7, color="white")
    ax.set_title("Per-clip backend video robustness metrics")
    fig.colorbar(im, ax=ax, label="column-normalized value")
    fig.tight_layout()
    path = out_dir / "clip_metric_heatmap.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_force_timelines(frame_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    clips = sorted({str(row["clip"]) for row in frame_rows})
    cols = 2
    rows_n = int(math.ceil(len(clips) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(16, max(4, rows_n * 3.1)), sharex=False)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, clip in zip(axes_arr, clips):
        rows = [row for row in frame_rows if str(row["clip"]) == clip and "error" not in row]
        t = np.asarray([float(row["video_time_s"]) for row in rows])
        force = np.asarray([float(row.get("action_force", 0.0)) for row in rows])
        reliability = np.asarray([float(row.get("flow_reliability", 0.0)) for row in rows])
        quality = np.asarray(
            [
                max(
                    float(row.get("low_flow_reliability", 0.0)),
                    float(row.get("high_camera_shake", 0.0)),
                    float(row.get("high_compression_noise", 0.0)),
                )
                for row in rows
            ]
        )
        ax.plot(t, force, label="force", color="#1f78b4")
        ax.plot(t, reliability, label="flow reliability", color="#33a02c", alpha=0.65)
        if quality.size:
            ax.fill_between(t, 0, quality, color="#e31a1c", alpha=0.16, label="quality event")
        ax.set_title(clip, fontsize=9)
        ax.set_ylim(-0.04, 1.05)
        ax.grid(alpha=0.18)
    for ax in axes_arr[len(clips) :]:
        ax.axis("off")
    axes_arr[0].legend(ncols=3, loc="upper center", bbox_to_anchor=(0.55, 1.35), fontsize=8)
    fig.supxlabel("video time (s)")
    fig.suptitle("Backend action-force and quality timelines for all selected videos", y=0.995)
    fig.tight_layout()
    path = out_dir / "action_force_timelines.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _bootstrap_mean_ci(values: list[float], *, seed: int = 17, n_boot: int = 10000) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "ci05": 0.0, "ci95": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = np.mean(arr[idx], axis=1)
    return {
        "mean": float(np.mean(arr)),
        "ci05": float(np.quantile(means, 0.05)),
        "ci95": float(np.quantile(means, 0.95)),
    }


def _render_report(
    *,
    out_dir: Path,
    summary: list[dict[str, Any]],
    published: dict[str, float],
    artifacts: dict[str, str],
    config: dict[str, Any],
) -> str:
    rates = [float(row["noncoast_event_frequency_hz"]) for row in summary]
    quality = [float(row["quality_event_fraction"]) for row in summary]
    coast_leaks = [float(row["coast_nonzero_force_fraction"]) for row in summary]
    inside = sum(int(float(row["event_frequency_inside_published_p05_p95"]) > 0.5) for row in summary)
    ci = _bootstrap_mean_ci(rates)
    published_mean = float(published.get("published_bout_frequency_hz_mean", 0.0))
    published_p05 = float(published.get("published_bout_frequency_hz_p05", 0.0))
    published_p95 = float(published.get("published_bout_frequency_hz_p95", 0.0))
    quality_sorted = sorted(summary, key=lambda row: float(row["quality_event_fraction"]), reverse=True)
    rate_sorted = sorted(summary, key=lambda row: float(row["noncoast_event_frequency_hz"]), reverse=True)
    lines = [
        "# Backend Video Robustness Audit",
        "",
        f"- output directory: `{out_dir}`",
        f"- clips audited: `{len(summary)}`",
        f"- sample Hz: `{config['sample_hz']}`",
        f"- max seconds per clip: `{config['seconds']}`",
        "- extraction path: backend OpenCV Farneback flow + RANSAC stabilization + simZFish-style retina/OMR adapter + auxiliary ZAPBench covariate mapper",
        "",
        "## Headline Results",
        "",
        f"- Mean transition-defined noncoast event rate across clips: `{ci['mean']:.4f}` Hz (bootstrap 90% CI `{ci['ci05']:.4f}`-`{ci['ci95']:.4f}`).",
        f"- Published simZFish/Z-Robot locomotion target: mean `{published_mean:.4f}` Hz, p05-p95 `{published_p05:.4f}`-`{published_p95:.4f}` Hz.",
        f"- Clips inside published p05-p95 event-rate band: `{inside}/{len(summary)}`.",
        f"- Mean quality-event fraction: `{float(np.mean(quality)):.4f}`; max `{float(np.max(quality)):.4f}`.",
        f"- Coast-force leak regression: max coast nonzero-force fraction `{float(np.max(coast_leaks)):.6f}`.",
        "",
        "## Interpretation",
        "",
        "This audit is not an embodied MuJoCo stability run. It isolates the current selected-video action extractor across every cached selected clip, which is the right regression layer for camera shake, compression artifacts, flow reliability, ZAPBench distance, action occupancy, and coast-frame force leakage.",
        "",
        "A strong result here means the backend decoder behaves consistently across source videos. It does not prove biological accuracy by itself; the full 50k embodied run remains the evidence for MuJoCo/body stability.",
        "",
        "## Highest Quality-Risk Clips",
        "",
        "| clip | quality fraction | low flow | camera shake | compression | event Hz | mean force |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in quality_sorted[:8]:
        lines.append(
            f"| `{row['clip']}` | {float(row['quality_event_fraction']):.3f} | "
            f"{int(row['low_flow_reliability_frames'])} | {int(row['high_camera_shake_frames'])} | "
            f"{int(row['high_compression_noise_frames'])} | "
            f"{float(row['noncoast_event_frequency_hz']):.3f} | {float(row['action_force_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Highest Action-Event Clips",
            "",
            "| clip | event Hz | published ratio | noncoast frame fraction | startle frac | forward frac | turn frac |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rate_sorted[:8]:
        turn_frac = float(row["left_turn_fraction"]) + float(row["right_turn_fraction"])
        lines.append(
            f"| `{row['clip']}` | {float(row['noncoast_event_frequency_hz']):.3f} | "
            f"{float(row['event_frequency_ratio_to_published_mean']):.3f} | "
            f"{float(row['noncoast_frame_fraction']):.3f} | {float(row['startle_fraction']):.3f} | "
            f"{float(row['forward_fraction']):.3f} | {turn_frac:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Research-Grade Caveats",
            "",
            "- Event rate is transition-defined, not command-frame occupancy; command occupancy is reported separately and must not be read as bout rate.",
            "- The ZAPBench mapper is auxiliary for arbitrary video because ZAPBench released covariates, not the raw projected visual movies.",
            "- Clips with high quality-event fractions should be excluded or down-weighted before claiming a video-to-motion behavior result.",
            "- This audit confirms action-extractor behavior and regression properties. It must be paired with long embodied recordings for final claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir) if args.output_dir else OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(CLIP_DIR.glob("*.mp4"))
    if args.clips:
        requested = {slug.strip() for slug in args.clips.split(",") if slug.strip()}
        clips = [path for path in clips if path.stem in requested]
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    if not clips:
        raise RuntimeError(f"no clips found under {CLIP_DIR}")

    published = _load_published_targets()
    frame_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    clip_meta: dict[str, Any] = {}
    for path in clips:
        rows, meta = _extract_clip(path=path, sample_hz=args.sample_hz, seconds=args.seconds)
        frame_rows.extend(rows)
        clip_meta[path.stem] = meta
        summary.append(_summarize_clip(clip=path.stem, rows=rows, meta=meta, published=published))
        print(
            json.dumps(
                {
                    "clip": path.stem,
                    "frames": len(rows),
                    "event_hz": summary[-1]["noncoast_event_frequency_hz"],
                    "quality_fraction": summary[-1]["quality_event_fraction"],
                    "coast_force_leaks": summary[-1]["coast_nonzero_force_frames"],
                },
                default=_json_default,
            ),
            flush=True,
        )

    frame_csv = out_dir / "backend_video_frame_metrics.csv"
    summary_csv = out_dir / "backend_video_robustness_summary.csv"
    _write_csv(frame_rows, frame_csv)
    _write_csv(summary, summary_csv)
    artifacts = {
        "frame_metrics": str(frame_csv),
        "summary": str(summary_csv),
        "bout_rate_vs_published": str(_plot_bout_rate(summary, published, out_dir)),
        "bout_type_fractions": str(_plot_bout_fractions(summary, out_dir)),
        "quality_vs_action": str(_plot_quality(summary, out_dir)),
        "clip_metric_heatmap": str(_plot_metric_heatmap(summary, out_dir)),
        "action_force_timelines": str(_plot_force_timelines(frame_rows, out_dir)),
    }
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": vars(args),
        "clip_count": len(clips),
        "frame_count": len(frame_rows),
        "published_targets": published,
        "clip_meta": clip_meta,
        "artifacts": artifacts,
        "summary": summary,
    }
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "BACKEND_VIDEO_ROBUSTNESS_AUDIT.md"
    artifacts["manifest"] = str(manifest_path)
    artifacts["report"] = str(report_path)
    manifest["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        _render_report(
            out_dir=out_dir,
            summary=summary,
            published=published,
            artifacts=artifacts,
            config=vars(args),
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--clips", default="", help="Comma-separated clip slugs; defaults to every cached selected clip.")
    parser.add_argument("--max-clips", type=int, default=0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"out_dir": result["artifacts"]["report"], "clip_count": result["clip_count"], "frame_count": result["frame_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
