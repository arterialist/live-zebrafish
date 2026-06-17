"""Align selected-video OMR estimates with the DANDI 001076 calcium atlas.

The selected-video branch cannot be treated as if it had the exact projector
stimuli used in the DANDI OMR recordings.  This audit therefore performs a
bounded check: every backend-decoded video frame is projected into coarse OMR
axes, mapped to the nearest DANDI OMR stimulus class, and scored against the
class-level calcium response atlas extracted by
``dandi_omr_neural_validation.py``.

This is a plausibility/traceability layer, not a supervised validation of
natural video -> whole-brain calcium -> motion.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DANDI_SUMMARY = ROOT / "analysis" / "out" / "dandi_omr_neural_validation_20260603" / "stimulus_response_summary.csv"
DEFAULT_VIDEO_FRAMES = (
    ROOT
    / "analysis"
    / "out"
    / "backend_video_robustness"
    / "20260603_all_selected_60s"
    / "backend_video_frame_metrics.csv"
)
DEFAULT_BACKEND_SUMMARY = (
    ROOT
    / "analysis"
    / "out"
    / "backend_video_robustness"
    / "20260603_all_selected_60s"
    / "backend_video_robustness_summary.csv"
)
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "dandi_video_omr_alignment_20260603"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_dandi_stimuli(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "stimulus": row["stimulus"],
                "primary_axis": row.get("primary_axis", ""),
                "forward_drive": _safe_float(row.get("forward_drive")),
                "side_drive": _safe_float(row.get("side_drive")),
                "expansion_drive": _safe_float(row.get("expansion_drive")),
                "population_response_mean": _safe_float(row.get("population_response_mean")),
                "population_response_sem": _safe_float(row.get("population_response_sem_across_files")),
                "selective_active_roi_fraction": _safe_float(row.get("selective_active_roi_fraction_mean")),
                "positive_roi_fraction": _safe_float(row.get("positive_roi_fraction_mean")),
            }
        )
    if not out:
        raise ValueError(f"No DANDI stimulus rows loaded from {path}")
    return out


def _dandi_matrix(stimuli: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = np.asarray(
        [[row["forward_drive"], row["side_drive"], row["expansion_drive"]] for row in stimuli],
        dtype=np.float64,
    )
    response = np.asarray([row["population_response_mean"] for row in stimuli], dtype=np.float64)
    active = np.asarray([row["selective_active_roi_fraction"] for row in stimuli], dtype=np.float64)
    return axes, response, active


def _quality_weight(row: dict[str, str]) -> tuple[float, int]:
    high_camera = _safe_float(row.get("high_camera_shake")) > 0.5
    high_compression = _safe_float(row.get("high_compression_noise")) > 0.5
    low_flow = _safe_float(row.get("low_flow_reliability")) > 0.5
    flags = int(high_camera) + int(high_compression) + int(low_flow)
    reliability = max(0.0, min(1.0, _safe_float(row.get("flow_reliability"), 0.0)))
    inlier = max(0.0, min(1.0, _safe_float(row.get("affine_inlier_ratio"), 1.0)))
    coherence = max(0.0, min(1.0, _safe_float(row.get("residual_flow_coherence"), _safe_float(row.get("flow_coherence"), 0.0))))
    artifact_penalty = 0.0
    if high_camera:
        artifact_penalty += 0.35
    if high_compression:
        artifact_penalty += 0.25
    if low_flow:
        artifact_penalty += 0.35
    quality = (0.45 * reliability) + (0.30 * inlier) + (0.25 * coherence)
    return max(0.0, min(1.0, quality * (1.0 - artifact_penalty))), flags


def _frame_omr_axes(row: dict[str, str]) -> tuple[np.ndarray, dict[str, float]]:
    """Project video features into DANDI-like OMR axes.

    Axis signs are estimator conventions, not calibrated projector coordinates.
    The audit keeps the raw components and reports a confidence/distance score
    so this is not mistaken for exact stimulus reconstruction.
    """

    visual_up = _safe_float(row.get("visual_up"))
    visual_down = _safe_float(row.get("visual_down"))
    visual_left = _safe_float(row.get("visual_left"))
    visual_right = _safe_float(row.get("visual_right"))
    optic_left = _safe_float(row.get("optic_flow_left"))
    optic_right = _safe_float(row.get("optic_flow_right"))
    global_horizontal = _safe_float(row.get("global_horizontal_flow"))
    global_vertical = _safe_float(row.get("global_vertical_flow"))
    camera_motion = _safe_float(row.get("camera_motion"))
    motion_energy = _safe_float(row.get("motion_energy"))
    off_left = _safe_float(row.get("simzfish_retinal_counters_off_left"))
    off_right = _safe_float(row.get("simzfish_retinal_counters_off_right"))
    lower_field = _safe_float(row.get("simzfish_retinal_counters_lower_field_fraction"), 0.0)

    side_visual = visual_right - visual_left
    forward_visual = visual_up - visual_down
    side_flow = math.tanh(global_horizontal)
    forward_flow = math.tanh(-global_vertical)

    # Symmetric optic-flow energy is the only available expansion/contraction
    # proxy in arbitrary POV videos.  Direction is ambiguous; retinal OFF
    # asymmetry and lower-field bias provide a weak sign convention.
    symmetric_flow = min(optic_left, optic_right)
    asymmetric_flow = abs(optic_right - optic_left)
    off_asymmetry = off_right - off_left
    expansion_mag = math.tanh(max(0.0, symmetric_flow - 0.4 * asymmetric_flow) + 0.15 * motion_energy)
    expansion_sign = 1.0 if (off_asymmetry + 0.5 * lower_field - 0.08 * camera_motion) >= 0 else -1.0
    expansion_drive = expansion_sign * expansion_mag

    forward_drive = max(-1.0, min(1.0, 0.65 * forward_visual + 0.35 * forward_flow))
    side_drive = max(-1.0, min(1.0, 0.65 * side_visual + 0.35 * side_flow))
    expansion_drive = max(-1.0, min(1.0, expansion_drive))
    axes = np.asarray([forward_drive, side_drive, expansion_drive], dtype=np.float64)
    raw = {
        "visual_forward_drive": forward_visual,
        "visual_side_drive": side_visual,
        "flow_forward_drive": forward_flow,
        "flow_side_drive": side_flow,
        "estimated_forward_drive": float(forward_drive),
        "estimated_side_drive": float(side_drive),
        "estimated_expansion_drive": float(expansion_drive),
        "symmetric_flow": float(symmetric_flow),
        "asymmetric_flow": float(asymmetric_flow),
        "expansion_sign_proxy": float(expansion_sign),
    }
    return axes, raw


def _nearest_stimulus(
    axes: np.ndarray,
    stimuli: list[dict[str, Any]],
    dandi_axes: np.ndarray,
    dandi_response: np.ndarray,
    dandi_active: np.ndarray,
) -> dict[str, Any]:
    distances = np.sqrt(np.sum((dandi_axes - axes.reshape(1, 3)) ** 2, axis=1))
    order = np.argsort(distances)
    nearest_i = int(order[0])
    # Soft assignment gives a smoother neural-response score than a hard label.
    weights = np.exp(-distances * 2.0)
    weights = weights / max(1e-12, float(np.sum(weights)))
    response_min = float(np.min(dandi_response))
    response_max = float(np.max(dandi_response))
    soft_response = float(np.sum(weights * dandi_response))
    soft_active = float(np.sum(weights * dandi_active))
    plausibility = (soft_response - response_min) / max(1e-9, response_max - response_min)
    return {
        "nearest_stimulus": stimuli[nearest_i]["stimulus"],
        "nearest_primary_axis": stimuli[nearest_i]["primary_axis"],
        "nearest_distance": float(distances[nearest_i]),
        "nearest_dandi_response": float(dandi_response[nearest_i]),
        "nearest_dandi_active_fraction": float(dandi_active[nearest_i]),
        "soft_dandi_response": soft_response,
        "soft_dandi_active_fraction": soft_active,
        "dandi_neural_plausibility": max(0.0, min(1.0, float(plausibility))),
        "second_stimulus": stimuli[int(order[1])]["stimulus"] if len(order) > 1 else "",
        "second_distance": float(distances[int(order[1])]) if len(order) > 1 else 0.0,
    }


def _frame_alignment_rows(
    video_rows: list[dict[str, str]],
    stimuli: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dandi_axes, dandi_response, dandi_active = _dandi_matrix(stimuli)
    rows: list[dict[str, Any]] = []
    for row in video_rows:
        axes, raw = _frame_omr_axes(row)
        match = _nearest_stimulus(axes, stimuli, dandi_axes, dandi_response, dandi_active)
        quality_weight, quality_flags = _quality_weight(row)
        action_force = _safe_float(row.get("action_force"))
        action_kick = _safe_float(row.get("action_kick"))
        action_confidence = _safe_float(row.get("action_confidence"))
        frame = {
            "clip": row.get("clip", ""),
            "frame_index": _safe_int(row.get("frame_index")),
            "video_time_s": _safe_float(row.get("video_time_s")),
            "action_bout_type": row.get("action_bout_type", ""),
            "action_force": action_force,
            "action_kick": action_kick,
            "action_confidence": action_confidence,
            "action_side_score": _safe_float(row.get("action_side_score")),
            "camera_shake": _safe_float(row.get("camera_shake")),
            "compression_noise": _safe_float(row.get("compression_noise")),
            "flow_reliability": _safe_float(row.get("flow_reliability")),
            "quality_weight": quality_weight,
            "quality_flags": quality_flags,
            **raw,
            **match,
        }
        frame["quality_weighted_dandi_response"] = quality_weight * frame["soft_dandi_response"]
        frame["quality_weighted_plausibility"] = quality_weight * frame["dandi_neural_plausibility"]
        frame["drive_magnitude"] = float(np.linalg.norm(axes))
        frame["neural_drive_product"] = frame["dandi_neural_plausibility"] * action_force * action_confidence
        rows.append(frame)
    return rows


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([count / total for count in counts.values() if count > 0], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs)))


def _corr(x: list[float], y: list[float]) -> float:
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(ax) & np.isfinite(ay)
    if np.sum(keep) < 3:
        return 0.0
    ax = ax[keep]
    ay = ay[keep]
    if float(np.std(ax)) <= 1e-12 or float(np.std(ay)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(ax, ay)[0, 1])


def _clip_summary_rows(frame_rows: list[dict[str, Any]], backend_summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    backend_by_clip = {row.get("clip", ""): row for row in backend_summary_rows}
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_clip[str(row["clip"])].append(row)
    summaries: list[dict[str, Any]] = []
    for clip, rows in sorted(by_clip.items()):
        counts = Counter(str(row["nearest_stimulus"]) for row in rows)
        primary_counts = Counter(str(row["nearest_primary_axis"]) for row in rows)
        quality = np.asarray([_safe_float(row["quality_weight"]) for row in rows], dtype=np.float64)
        response = np.asarray([_safe_float(row["soft_dandi_response"]) for row in rows], dtype=np.float64)
        plaus = np.asarray([_safe_float(row["dandi_neural_plausibility"]) for row in rows], dtype=np.float64)
        action_force = np.asarray([_safe_float(row["action_force"]) for row in rows], dtype=np.float64)
        action_kick = np.asarray([_safe_float(row["action_kick"]) for row in rows], dtype=np.float64)
        distance = np.asarray([_safe_float(row["nearest_distance"]) for row in rows], dtype=np.float64)
        backend = backend_by_clip.get(clip, {})
        top_stim, top_stim_count = counts.most_common(1)[0]
        top_axis, top_axis_count = primary_counts.most_common(1)[0]
        summaries.append(
            {
                "clip": clip,
                "frames": len(rows),
                "duration_s": float(np.nanmax([_safe_float(row["video_time_s"]) for row in rows])) if rows else 0.0,
                "top_stimulus": top_stim,
                "top_stimulus_fraction": top_stim_count / max(1, len(rows)),
                "top_primary_axis": top_axis,
                "top_primary_axis_fraction": top_axis_count / max(1, len(rows)),
                "stimulus_entropy_bits": _entropy(counts),
                "mean_dandi_response": float(np.nanmean(response)),
                "p95_dandi_response": float(np.nanquantile(response, 0.95)),
                "mean_neural_plausibility": float(np.nanmean(plaus)),
                "quality_weighted_neural_plausibility": float(np.nanmean(plaus * quality)),
                "mean_quality_weight": float(np.nanmean(quality)),
                "quality_event_fraction": _safe_float(backend.get("quality_event_fraction"), float(np.nanmean([row["quality_flags"] > 0 for row in rows]))),
                "mean_nearest_distance": float(np.nanmean(distance)),
                "p95_nearest_distance": float(np.nanquantile(distance, 0.95)),
                "mean_action_force": float(np.nanmean(action_force)),
                "kick_fraction": float(np.nanmean(action_kick > 0.5)),
                "force_vs_plausibility_corr": _corr(action_force.tolist(), plaus.tolist()),
                "quality_vs_plausibility_corr": _corr(quality.tolist(), plaus.tolist()),
                "backend_event_frequency_hz": _safe_float(backend.get("noncoast_event_frequency_hz")),
                "backend_inside_published_band": _safe_float(backend.get("event_frequency_inside_published_p05_p95")),
            }
        )
    return sorted(summaries, key=lambda row: row["quality_weighted_neural_plausibility"], reverse=True)


def _stimulus_fraction_matrix(frame_rows: list[dict[str, Any]], stimuli: list[dict[str, Any]]) -> tuple[list[str], list[str], np.ndarray]:
    clips = sorted({str(row["clip"]) for row in frame_rows})
    labels = [str(row["stimulus"]) for row in stimuli]
    mat = np.zeros((len(clips), len(labels)), dtype=np.float64)
    label_i = {label: i for i, label in enumerate(labels)}
    clip_i = {clip: i for i, clip in enumerate(clips)}
    counts = Counter(str(row["clip"]) for row in frame_rows)
    for row in frame_rows:
        clip = str(row["clip"])
        stim = str(row["nearest_stimulus"])
        if stim in label_i:
            mat[clip_i[clip], label_i[stim]] += 1.0 / max(1, counts[clip])
    return clips, labels, mat


def _save_clip_rank(out_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    path = out_dir / "clip_neural_alignment_rank.png"
    labels = [row["clip"] for row in summaries]
    scores = np.asarray([row["quality_weighted_neural_plausibility"] for row in summaries], dtype=np.float64)
    quality = np.asarray([row["mean_quality_weight"] for row in summaries], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, scores, width=0.36, label="quality-weighted DANDI plausibility", color="#2b8cbe")
    ax.bar(x + 0.18, quality, width=0.36, label="mean quality weight", color="#a1d99b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Selected-video DANDI OMR neural-alignment score by clip")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_stimulus_distribution(out_dir: Path, frame_rows: list[dict[str, Any]], stimuli: list[dict[str, Any]]) -> Path:
    path = out_dir / "clip_stimulus_assignment_heatmap.png"
    clips, labels, mat = _stimulus_fraction_matrix(frame_rows, stimuli)
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(mat, aspect="auto", cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0, vmax=max(0.05, float(np.max(mat))))
    ax.set_yticks(np.arange(len(clips)))
    ax.set_yticklabels(clips, fontsize=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Nearest DANDI OMR stimulus assignment fractions per selected video")
    fig.colorbar(im, ax=ax, label="fraction of backend-decoded frames")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_axis_space(out_dir: Path, frame_rows: list[dict[str, Any]], stimuli: list[dict[str, Any]]) -> Path:
    path = out_dir / "video_omr_axis_space.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sample = frame_rows[:: max(1, len(frame_rows) // 2500)]
    x = np.asarray([row["estimated_side_drive"] for row in sample], dtype=np.float64)
    y = np.asarray([row["estimated_forward_drive"] for row in sample], dtype=np.float64)
    c = np.asarray([row["dandi_neural_plausibility"] for row in sample], dtype=np.float64)
    sc = axes[0].scatter(x, y, c=c, s=7, alpha=0.35, cmap="viridis", rasterized=True)
    for stim in stimuli:
        axes[0].scatter(stim["side_drive"], stim["forward_drive"], s=80, marker="x", color="#111111")
        axes[0].text(stim["side_drive"], stim["forward_drive"], stim["stimulus"], fontsize=7)
    axes[0].set_xlabel("side drive")
    axes[0].set_ylabel("forward drive")
    axes[0].set_title("Video frames projected into DANDI OMR side/forward axes")
    fig.colorbar(sc, ax=axes[0], label="DANDI neural plausibility")

    x2 = np.asarray([row["estimated_expansion_drive"] for row in sample], dtype=np.float64)
    y2 = np.asarray([row["drive_magnitude"] for row in sample], dtype=np.float64)
    axes[1].scatter(x2, y2, c=c, s=7, alpha=0.35, cmap="viridis", rasterized=True)
    for stim in stimuli:
        axes[1].scatter(stim["expansion_drive"], np.linalg.norm([stim["forward_drive"], stim["side_drive"], stim["expansion_drive"]]), s=80, marker="x", color="#111111")
    axes[1].set_xlabel("expansion/contraction proxy")
    axes[1].set_ylabel("OMR drive magnitude")
    axes[1].set_title("Video frames projected into expansion/magnitude axes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_response_force(out_dir: Path, frame_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "dandi_response_vs_action_force.png"
    sample = frame_rows[:: max(1, len(frame_rows) // 4000)]
    response = np.asarray([row["soft_dandi_response"] for row in sample], dtype=np.float64)
    force = np.asarray([row["action_force"] for row in sample], dtype=np.float64)
    quality = np.asarray([row["quality_weight"] for row in sample], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(response, force, c=quality, s=9, alpha=0.35, cmap="plasma", rasterized=True)
    axes[0].set_xlabel("DANDI soft calcium response score")
    axes[0].set_ylabel("backend action force")
    axes[0].set_title("Neural motif score versus motor output")
    fig.colorbar(sc, ax=axes[0], label="quality weight")
    axes[1].scatter(quality, response, c=force, s=9, alpha=0.35, cmap="viridis", rasterized=True)
    axes[1].set_xlabel("quality weight")
    axes[1].set_ylabel("DANDI soft calcium response score")
    axes[1].set_title("Quality dependence of neural motif score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_timelines(out_dir: Path, frame_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> Path:
    path = out_dir / "selected_clip_dandi_alignment_timelines.png"
    chosen = [row["clip"] for row in summaries[:4]]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, max(4, 2.6 * len(chosen))), sharex=False)
    if len(chosen) == 1:
        axes = [axes]
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        if row["clip"] in chosen:
            by_clip[row["clip"]].append(row)
    for ax, clip in zip(axes, chosen, strict=False):
        rows = sorted(by_clip[clip], key=lambda row: row["video_time_s"])
        t = np.asarray([row["video_time_s"] for row in rows], dtype=np.float64)
        plaus = np.asarray([row["dandi_neural_plausibility"] for row in rows], dtype=np.float64)
        force = np.asarray([row["action_force"] for row in rows], dtype=np.float64)
        quality = np.asarray([row["quality_weight"] for row in rows], dtype=np.float64)
        ax.plot(t, plaus, label="DANDI plausibility", color="#2b8cbe", linewidth=1.3)
        ax.plot(t, force, label="action force", color="#e6550d", linewidth=1.0, alpha=0.9)
        ax.plot(t, quality, label="quality weight", color="#31a354", linewidth=0.9, alpha=0.8)
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel(clip, fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel("video time (s)")
    fig.suptitle("DANDI OMR neural-alignment timelines for top selected videos", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_primary_axis_plot(out_dir: Path, frame_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "primary_axis_fractions.png"
    by_clip_axis: dict[str, Counter[str]] = defaultdict(Counter)
    for row in frame_rows:
        by_clip_axis[str(row["clip"])][str(row["nearest_primary_axis"])] += 1
    clips = sorted(by_clip_axis)
    axes_labels = sorted({axis for counter in by_clip_axis.values() for axis in counter})
    mat = np.zeros((len(clips), len(axes_labels)), dtype=np.float64)
    for i, clip in enumerate(clips):
        total = sum(by_clip_axis[clip].values())
        for j, axis in enumerate(axes_labels):
            mat[i, j] = by_clip_axis[clip][axis] / max(1, total)
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(clips), dtype=np.float64)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(axes_labels))))
    for j, axis in enumerate(axes_labels):
        ax.bar(np.arange(len(clips)), mat[:, j], bottom=bottom, label=axis, color=colors[j])
        bottom += mat[:, j]
    ax.set_xticks(np.arange(len(clips)))
    ax.set_xticklabels(clips, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("frame fraction")
    ax.set_title("Primary DANDI OMR axis assignment per selected video")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    frame_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    stimuli: list[dict[str, Any]],
    plots: list[Path],
) -> None:
    total_frames = len(frame_rows)
    clips = sorted({str(row["clip"]) for row in frame_rows})
    quality_events = sum(1 for row in frame_rows if _safe_float(row["quality_flags"]) > 0)
    mean_plaus = float(np.mean([_safe_float(row["dandi_neural_plausibility"]) for row in frame_rows])) if frame_rows else 0.0
    mean_quality_weighted = float(np.mean([_safe_float(row["quality_weighted_plausibility"]) for row in frame_rows])) if frame_rows else 0.0
    top = summaries[0] if summaries else {}
    lines = [
        "# DANDI-Grounded Video OMR Alignment Audit",
        "",
        "## Scope",
        "",
        "This audit projects backend-decoded selected-video frames into coarse OMR axes and maps each frame to the nearest DANDI 001076 OMR calcium stimulus class. It scores neural motif plausibility using the DANDI calcium-response atlas.",
        "",
        "This is not a direct proof that arbitrary videos recreate the projector stimuli used in DANDI. It is a class-level plausibility and traceability test that exposes where the current selected-video inputs do or do not look like known OMR calcium motifs.",
        "",
        "## Coverage",
        "",
        f"- Backend-decoded video frames scored: `{total_frames}`",
        f"- Selected clips: `{len(clips)}`",
        f"- DANDI OMR stimulus classes: `{len(stimuli)}`",
        f"- Frames with any quality flag: `{quality_events}` (`{quality_events / max(1, total_frames):.3f}`)",
        f"- Mean DANDI neural plausibility: `{mean_plaus:.4f}`",
        f"- Mean quality-weighted DANDI plausibility: `{mean_quality_weighted:.4f}`",
        f"- Highest quality-weighted clip: `{top.get('clip', '')}` with score `{_safe_float(top.get('quality_weighted_neural_plausibility')):.4f}`",
        "",
        "## Clip Ranking",
        "",
        "| rank | clip | frames | top stimulus | top axis | neural plausibility | quality-weighted | quality weight | mean force | force/plaus corr |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summaries, start=1):
        lines.append(
            f"| {rank} | `{row['clip']}` | {int(row['frames'])} | `{row['top_stimulus']}` | `{row['top_primary_axis']}` | "
            f"{_safe_float(row['mean_neural_plausibility']):.4f} | {_safe_float(row['quality_weighted_neural_plausibility']):.4f} | "
            f"{_safe_float(row['mean_quality_weight']):.4f} | {_safe_float(row['mean_action_force']):.4f} | "
            f"{_safe_float(row['force_vs_plausibility_corr']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Generated Visualizations",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot.resolve()}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Clips with high quality-weighted plausibility are better candidates for long embodied runs because their estimated OMR axes map onto DANDI calcium motifs while also avoiding severe camera/compression artifacts.",
            "- Clips with high raw plausibility but low quality weight should be treated as stress tests; artifact-driven optical flow can look like an OMR motif but should not be used as validation evidence.",
            "- The audit maps selected-video frames to DANDI stimulus labels at the motif level. It does not infer actual DANDI calcium traces frame by frame.",
            "- Movement validity still depends on separate checks against simZFish/Z-Robot controller targets, ZAPBench/ephys motor labels, and long-run MuJoCo body metrics.",
            "",
            "## Files Written",
            "",
            "- `video_frame_neural_alignment.csv`: one row per backend-decoded frame.",
            "- `clip_neural_alignment_summary.csv`: one row per selected clip.",
            "- `manifest.json`: machine-readable run summary and plot list.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dandi-summary", type=Path, default=DEFAULT_DANDI_SUMMARY)
    parser.add_argument("--video-frames", type=Path, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument("--backend-summary", type=Path, default=DEFAULT_BACKEND_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli = _load_dandi_stimuli(args.dandi_summary)
    video_rows = _read_csv(args.video_frames)
    backend_rows = _read_csv(args.backend_summary) if args.backend_summary.exists() else []

    frame_rows = _frame_alignment_rows(video_rows, stimuli)
    summaries = _clip_summary_rows(frame_rows, backend_rows)

    frame_path = out_dir / "video_frame_neural_alignment.csv"
    summary_path = out_dir / "clip_neural_alignment_summary.csv"
    _write_csv(frame_path, frame_rows)
    _write_csv(summary_path, summaries)
    plots = [
        _save_clip_rank(out_dir, summaries),
        _save_stimulus_distribution(out_dir, frame_rows, stimuli),
        _save_axis_space(out_dir, frame_rows, stimuli),
        _save_response_force(out_dir, frame_rows),
        _save_timelines(out_dir, frame_rows, summaries),
        _save_primary_axis_plot(out_dir, frame_rows),
    ]
    report_path = out_dir / "DANDI_VIDEO_OMR_ALIGNMENT.md"
    _write_report(report_path, frame_rows=frame_rows, summaries=summaries, stimuli=stimuli, plots=plots)
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "dandi_summary": str(args.dandi_summary.resolve()),
        "video_frames": str(args.video_frames.resolve()),
        "backend_summary": str(args.backend_summary.resolve()) if args.backend_summary.exists() else "",
        "frame_rows": len(frame_rows),
        "clip_count": len(summaries),
        "stimulus_count": len(stimuli),
        "quality_flagged_frames": int(sum(1 for row in frame_rows if _safe_float(row["quality_flags"]) > 0)),
        "mean_dandi_neural_plausibility": float(np.mean([row["dandi_neural_plausibility"] for row in frame_rows])) if frame_rows else 0.0,
        "mean_quality_weighted_plausibility": float(np.mean([row["quality_weighted_plausibility"] for row in frame_rows])) if frame_rows else 0.0,
        "top_clip": summaries[0]["clip"] if summaries else "",
        "top_clip_quality_weighted_plausibility": summaries[0]["quality_weighted_neural_plausibility"] if summaries else 0.0,
        "csv": {
            "video_frame_neural_alignment": str(frame_path.resolve()),
            "clip_neural_alignment_summary": str(summary_path.resolve()),
        },
        "plots": [str(plot.resolve()) for plot in plots],
        "limitations": [
            "Coarse OMR-axis projection, not exact DANDI projector reconstruction.",
            "Expansion/contraction is estimated from arbitrary-video optical-flow proxies.",
            "No frame-exact DANDI calcium trace prediction or synchronized kinematics.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
