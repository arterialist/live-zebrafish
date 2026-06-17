"""Build an OMR calcium-response atlas from public DANDI 001076 NWB files.

This script validates the neural-resource side of the Z-Robot/simZFish branch:
it does not train the lab controller and it does not claim paired
video/calcium/kinematics.  It extracts trial-aligned whole-brain calcium
responses to the OMR stimulus labels that are actually present in the public
NWB files, then emits summaries and plots that can be compared with the
current lab's internal OMR state and action traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NWB_DIR = ROOT / "analysis" / "cache" / "z_robot" / "dandi_001076_nwb"
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "dandi_omr_neural_validation_20260603"

DECONVOLVED_PATH = "processing/ophys/Fluorescence/Deconvolved/data"
DECONVOLVED_START_PATH = "processing/ophys/Fluorescence/Deconvolved/starting_time"
ACCEPTED_PATH = "processing/ophys/ImageSegmentation/PlaneSegmentation/Accepted"
CENTROID_PATH = "processing/ophys/ImageSegmentation/PlaneSegmentation/ROICentroids"
TRIAL_START_PATH = "intervals/trials/start_time"
TRIAL_STOP_PATH = "intervals/trials/stop_time"
TRIAL_STIM_PATH = "intervals/trials/stim"

TRACE_TIMES_S = np.arange(-5.0, 11.0, 1.0, dtype=np.float64)


@dataclass
class FileAnalysis:
    summary: dict[str, Any]
    stimulus_rows: list[dict[str, Any]]
    trial_rows: list[dict[str, Any]]
    roi_rows: list[dict[str, Any]]
    file_stim_vector: dict[str, float]
    stim_trace_sum: dict[str, np.ndarray]
    stim_trace_sumsq: dict[str, np.ndarray]
    stim_trace_count: dict[str, int]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _stimulus_axes(stim: str) -> dict[str, Any]:
    tokens = stim.lower().split("_")
    forward_drive = float(tokens.count("forward") - tokens.count("backward"))
    side_drive = float(tokens.count("right") - tokens.count("left"))
    expansion_drive = 0.0
    if "diverging" in tokens:
        expansion_drive += 1.0
    if "converging" in tokens:
        expansion_drive -= 1.0
    if "medial" in tokens:
        expansion_drive -= 0.5
    if "lateral" in tokens:
        expansion_drive += 0.5
    if abs(forward_drive) >= abs(side_drive) and abs(forward_drive) > 0:
        primary = "forward" if forward_drive > 0 else "backward"
    elif abs(side_drive) > 0:
        primary = "right" if side_drive > 0 else "left"
    elif abs(expansion_drive) > 0:
        primary = "expansion" if expansion_drive > 0 else "contraction"
    else:
        primary = "mixed_or_x"
    return {
        "forward_drive": forward_drive,
        "side_drive": side_drive,
        "expansion_drive": expansion_drive,
        "primary_axis": primary,
    }


def _read_rate_and_start(f: h5py.File) -> tuple[float, float]:
    start_ds = f[DECONVOLVED_START_PATH]
    rate = _safe_float(start_ds.attrs.get("rate"), default=float("nan"))
    start = _safe_float(start_ds[()], default=0.0)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid acquisition rate in NWB file")
    return rate, start


def _time_to_index(time_s: float, *, rate_hz: float, start_time_s: float, frames: int) -> int:
    idx = int(round((time_s - start_time_s) * rate_hz))
    return max(0, min(frames - 1, idx))


def _window_indices(
    start_s: float,
    next_start_s: float | None,
    *,
    rate_hz: float,
    data_start_s: float,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    baseline_start_s = max(data_start_s, start_s - 5.0)
    baseline_end_s = max(baseline_start_s + 0.5, start_s - 1.0)
    response_limit_s = start_s + 10.0
    if next_start_s is not None and math.isfinite(next_start_s):
        response_limit_s = min(response_limit_s, next_start_s - 1.0)
    response_limit_s = max(start_s + 1.0, response_limit_s)

    b0 = _time_to_index(baseline_start_s, rate_hz=rate_hz, start_time_s=data_start_s, frames=frames)
    b1 = _time_to_index(baseline_end_s, rate_hz=rate_hz, start_time_s=data_start_s, frames=frames)
    r0 = _time_to_index(start_s, rate_hz=rate_hz, start_time_s=data_start_s, frames=frames)
    r1 = _time_to_index(response_limit_s, rate_hz=rate_hz, start_time_s=data_start_s, frames=frames)
    baseline_idx = np.arange(min(b0, b1), max(b0, b1) + 1, dtype=np.int64)
    response_idx = np.arange(min(r0, r1), max(r0, r1) + 1, dtype=np.int64)
    return baseline_idx, response_idx, response_limit_s - start_s


def _trace_indices(
    start_s: float,
    *,
    rate_hz: float,
    data_start_s: float,
    frames: int,
) -> np.ndarray:
    return np.asarray(
        [
            _time_to_index(start_s + rel, rate_hz=rate_hz, start_time_s=data_start_s, frames=frames)
            for rel in TRACE_TIMES_S
        ],
        dtype=np.int64,
    )


def _subject_value(f: h5py.File, key: str) -> str:
    path = f"subject/{key}"
    if path not in f:
        return ""
    obj = f[path]
    try:
        return _decode(obj[()])
    except Exception:  # noqa: BLE001
        return ""


def _accepted_mask(f: h5py.File, rois: int) -> np.ndarray:
    if ACCEPTED_PATH not in f:
        return np.ones(rois, dtype=bool)
    raw = np.asarray(f[ACCEPTED_PATH][:]).reshape(-1)
    if raw.size != rois:
        return np.ones(rois, dtype=bool)
    mask = raw.astype(np.float64) > 0.0
    if not np.any(mask):
        return np.ones(rois, dtype=bool)
    return mask


def _centroids(f: h5py.File, mask: np.ndarray) -> np.ndarray:
    if CENTROID_PATH not in f:
        return np.full((int(np.sum(mask)), 2), np.nan, dtype=np.float64)
    raw = np.asarray(f[CENTROID_PATH][:], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != mask.size:
        return np.full((int(np.sum(mask)), 2), np.nan, dtype=np.float64)
    cent = raw[mask, :2]
    return cent


def _normalize_centroids(centroids: np.ndarray) -> np.ndarray:
    if centroids.size == 0:
        return centroids
    out = np.asarray(centroids, dtype=np.float64).copy()
    for col in range(min(2, out.shape[1])):
        values = out[:, col]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        lo = float(np.min(values[finite]))
        hi = float(np.max(values[finite]))
        if hi > lo:
            out[:, col] = (values - lo) / (hi - lo)
    return out


def _analyze_file(path: Path, file_index: int) -> FileAnalysis:
    with h5py.File(path, "r") as f:
        if DECONVOLVED_PATH not in f:
            raise KeyError(f"{DECONVOLVED_PATH} missing from {path}")
        data = np.asarray(f[DECONVOLVED_PATH][:], dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D deconvolved data in {path}, got {data.shape}")
        frames, rois = data.shape
        rate_hz, data_start_s = _read_rate_and_start(f)
        accepted = _accepted_mask(f, rois)
        data = data[:, accepted]
        accepted_rois = int(data.shape[1])
        centroids = _centroids(f, accepted)
        norm_centroids = _normalize_centroids(centroids)
        starts = np.asarray(f[TRIAL_START_PATH][:], dtype=np.float64)
        stops = np.asarray(f[TRIAL_STOP_PATH][:], dtype=np.float64) if TRIAL_STOP_PATH in f else np.full_like(starts, np.nan)
        stims = [_decode(x) for x in f[TRIAL_STIM_PATH][:]]
        species = _subject_value(f, "species")
        age = _subject_value(f, "age")

    stim_sum: dict[str, np.ndarray] = {}
    stim_sumsq: dict[str, np.ndarray] = {}
    stim_count: dict[str, int] = defaultdict(int)
    stim_trial_means: dict[str, list[float]] = defaultdict(list)
    stim_trial_stds: dict[str, list[float]] = defaultdict(list)
    stim_response_duration: dict[str, list[float]] = defaultdict(list)
    trial_rows: list[dict[str, Any]] = []
    trace_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(TRACE_TIMES_S, dtype=np.float64))
    trace_sumsq: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(TRACE_TIMES_S, dtype=np.float64))
    trace_count: dict[str, int] = defaultdict(int)

    order = np.argsort(starts)
    sorted_starts = starts[order]
    for trial_order, trial_i in enumerate(order):
        start_s = _safe_float(starts[trial_i])
        if not math.isfinite(start_s):
            continue
        next_start = _safe_float(sorted_starts[trial_order + 1]) if trial_order + 1 < len(sorted_starts) else float("nan")
        next_start_s = next_start if math.isfinite(next_start) else None
        stim = stims[trial_i]
        baseline_idx, response_idx, response_duration_s = _window_indices(
            start_s,
            next_start_s,
            rate_hz=rate_hz,
            data_start_s=data_start_s,
            frames=frames,
        )
        baseline = np.nanmean(data[baseline_idx, :], axis=0)
        response = np.nanmean(data[response_idx, :], axis=0) - baseline
        response = np.asarray(response, dtype=np.float64)
        response[~np.isfinite(response)] = 0.0
        if stim not in stim_sum:
            stim_sum[stim] = np.zeros(accepted_rois, dtype=np.float64)
            stim_sumsq[stim] = np.zeros(accepted_rois, dtype=np.float64)
        stim_sum[stim] += response
        stim_sumsq[stim] += response * response
        stim_count[stim] += 1
        pop_mean = float(np.mean(response)) if response.size else float("nan")
        pop_std = float(np.std(response)) if response.size else float("nan")
        positive_fraction = float(np.mean(response > 0.0)) if response.size else float("nan")
        stim_trial_means[stim].append(pop_mean)
        stim_trial_stds[stim].append(pop_std)
        stim_response_duration[stim].append(response_duration_s)
        trial_rows.append(
            {
                "file_index": file_index,
                "file": path.name,
                "trial_index": int(trial_i),
                "stimulus": stim,
                "start_s": start_s,
                "stop_s": _safe_float(stops[trial_i]),
                "response_duration_s": response_duration_s,
                "baseline_frames": int(baseline_idx.size),
                "response_frames": int(response_idx.size),
                "population_response_mean": pop_mean,
                "population_response_std": pop_std,
                "positive_roi_fraction": positive_fraction,
            }
        )
        t_idx = _trace_indices(start_s, rate_hz=rate_hz, data_start_s=data_start_s, frames=frames)
        trace = np.nanmean(data[t_idx, :], axis=1) - float(np.mean(baseline)) if accepted_rois else np.zeros_like(TRACE_TIMES_S)
        trace = np.asarray(trace, dtype=np.float64)
        trace[~np.isfinite(trace)] = 0.0
        trace_sum[stim] += trace
        trace_sumsq[stim] += trace * trace
        trace_count[stim] += 1

    stimuli = sorted(stim_count)
    stim_roi_means = np.vstack([stim_sum[stim] / max(1, stim_count[stim]) for stim in stimuli]) if stimuli else np.zeros((0, accepted_rois))
    roi_baseline = np.nanmedian(stim_roi_means, axis=0) if stim_roi_means.size else np.zeros(accepted_rois)
    roi_mad = np.nanmedian(np.abs(stim_roi_means - roi_baseline), axis=0) if stim_roi_means.size else np.ones(accepted_rois)
    roi_threshold = roi_baseline + 0.5 * 1.4826 * roi_mad

    stimulus_rows: list[dict[str, Any]] = []
    file_stim_vector: dict[str, float] = {}
    for stim_i, stim in enumerate(stimuli):
        roi_mean = stim_roi_means[stim_i]
        pop_values = np.asarray(stim_trial_means[stim], dtype=np.float64)
        active_fraction = float(np.mean(roi_mean > roi_threshold)) if roi_mean.size else float("nan")
        positive_fraction = float(np.mean(roi_mean > 0.0)) if roi_mean.size else float("nan")
        row = {
            "file_index": file_index,
            "file": path.name,
            "stimulus": stim,
            **_stimulus_axes(stim),
            "trials": int(stim_count[stim]),
            "accepted_rois": accepted_rois,
            "population_response_mean": float(np.mean(pop_values)) if pop_values.size else float("nan"),
            "population_response_sem": _sem(pop_values),
            "roi_response_mean": float(np.mean(roi_mean)) if roi_mean.size else float("nan"),
            "roi_response_std": float(np.std(roi_mean)) if roi_mean.size else float("nan"),
            "roi_response_p05": float(np.quantile(roi_mean, 0.05)) if roi_mean.size else float("nan"),
            "roi_response_p50": float(np.quantile(roi_mean, 0.50)) if roi_mean.size else float("nan"),
            "roi_response_p95": float(np.quantile(roi_mean, 0.95)) if roi_mean.size else float("nan"),
            "selective_active_roi_fraction": active_fraction,
            "positive_roi_fraction": positive_fraction,
            "mean_response_duration_s": float(np.mean(stim_response_duration[stim])) if stim_response_duration[stim] else float("nan"),
        }
        stimulus_rows.append(row)
        file_stim_vector[stim] = row["population_response_mean"]

    roi_rows: list[dict[str, Any]] = []
    if stimuli and stim_roi_means.size:
        best_idx = np.nanargmax(stim_roi_means, axis=0)
        med = np.nanmedian(stim_roi_means, axis=0)
        for roi_i, stim_i in enumerate(best_idx):
            best_stim = stimuli[int(stim_i)]
            best = float(stim_roi_means[int(stim_i), roi_i])
            median = float(med[roi_i])
            selectivity = (best - median) / (abs(best) + abs(median) + 1e-9)
            axes = _stimulus_axes(best_stim)
            cx = float(centroids[roi_i, 0]) if centroids.size else float("nan")
            cy = float(centroids[roi_i, 1]) if centroids.size else float("nan")
            nx = float(norm_centroids[roi_i, 0]) if norm_centroids.size else float("nan")
            ny = float(norm_centroids[roi_i, 1]) if norm_centroids.size else float("nan")
            roi_rows.append(
                {
                    "file_index": file_index,
                    "file": path.name,
                    "roi_index_accepted": roi_i,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "centroid_x_norm": nx,
                    "centroid_y_norm": ny,
                    "best_stimulus": best_stim,
                    "best_primary_axis": axes["primary_axis"],
                    "best_forward_drive": axes["forward_drive"],
                    "best_side_drive": axes["side_drive"],
                    "best_expansion_drive": axes["expansion_drive"],
                    "best_response": best,
                    "median_response": median,
                    "selectivity_index": float(selectivity),
                }
            )

    top_stim = ""
    top_response = float("nan")
    if stimulus_rows:
        top = max(stimulus_rows, key=lambda row: _safe_float(row["population_response_mean"], default=-1e99))
        top_stim = str(top["stimulus"])
        top_response = float(top["population_response_mean"])
    all_trial_means = np.asarray([row["population_response_mean"] for row in trial_rows], dtype=np.float64)
    summary = {
        "file_index": file_index,
        "file": path.name,
        "path": str(path.resolve()),
        "species": species,
        "age": age,
        "frames": frames,
        "total_rois": rois,
        "accepted_rois": accepted_rois,
        "acquisition_rate_hz": rate_hz,
        "recording_duration_s": frames / rate_hz,
        "trials": len(trial_rows),
        "unique_stimuli": len(stimuli),
        "top_stimulus": top_stim,
        "top_population_response_mean": top_response,
        "mean_population_response": float(np.mean(all_trial_means)) if all_trial_means.size else float("nan"),
        "std_population_response": float(np.std(all_trial_means)) if all_trial_means.size else float("nan"),
        "min_trial_start_s": float(np.nanmin(starts)) if starts.size else float("nan"),
        "max_trial_start_s": float(np.nanmax(starts)) if starts.size else float("nan"),
    }
    return FileAnalysis(
        summary=summary,
        stimulus_rows=stimulus_rows,
        trial_rows=trial_rows,
        roi_rows=roi_rows,
        file_stim_vector=file_stim_vector,
        stim_trace_sum=dict(trace_sum),
        stim_trace_sumsq=dict(trace_sumsq),
        stim_trace_count=dict(trace_count),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_stimulus_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stim[str(row["stimulus"])].append(row)
    out: list[dict[str, Any]] = []
    for stim, stim_rows in sorted(by_stim.items()):
        pop = np.asarray([_safe_float(r["population_response_mean"]) for r in stim_rows], dtype=np.float64)
        active = np.asarray([_safe_float(r["selective_active_roi_fraction"]) for r in stim_rows], dtype=np.float64)
        positive = np.asarray([_safe_float(r["positive_roi_fraction"]) for r in stim_rows], dtype=np.float64)
        axes = _stimulus_axes(stim)
        out.append(
            {
                "stimulus": stim,
                **axes,
                "files": len(stim_rows),
                "trials": int(sum(_safe_float(r["trials"], 0.0) for r in stim_rows)),
                "accepted_rois_mean": float(np.mean([_safe_float(r["accepted_rois"], 0.0) for r in stim_rows])),
                "population_response_mean": float(np.nanmean(pop)),
                "population_response_sem_across_files": _sem(pop),
                "population_response_std_across_files": float(np.nanstd(pop)),
                "file_positive_fraction": float(np.nanmean(pop > 0.0)),
                "selective_active_roi_fraction_mean": float(np.nanmean(active)),
                "positive_roi_fraction_mean": float(np.nanmean(positive)),
                "mean_response_duration_s": float(
                    np.nanmean([_safe_float(r["mean_response_duration_s"]) for r in stim_rows])
                ),
            }
        )
    return sorted(out, key=lambda row: _safe_float(row["population_response_mean"], default=-1e99), reverse=True)


def _matrix_from_file_vectors(file_summaries: list[dict[str, Any]], vectors: list[dict[str, float]]) -> tuple[list[str], list[str], np.ndarray]:
    stimuli = sorted({stim for vector in vectors for stim in vector})
    files = [str(row["file"]) for row in file_summaries]
    mat = np.full((len(stimuli), len(vectors)), np.nan, dtype=np.float64)
    for j, vector in enumerate(vectors):
        for i, stim in enumerate(stimuli):
            if stim in vector:
                mat[i, j] = vector[stim]
    return stimuli, files, mat


def _save_stimulus_heatmap(out_dir: Path, stimuli: list[str], files: list[str], mat: np.ndarray) -> Path:
    path = out_dir / "stimulus_response_heatmap.png"
    if mat.size == 0:
        return path
    row_mean = np.nanmean(mat, axis=1)
    order = np.argsort(row_mean)[::-1]
    display = mat[order]
    sorted_stimuli = [stimuli[i] for i in order]
    col_mean = np.nanmean(display, axis=0, keepdims=True)
    col_std = np.nanstd(display, axis=0, keepdims=True)
    z = (display - col_mean) / np.maximum(col_std, 1e-9)
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(sorted_stimuli)))
    ax.set_yticklabels(sorted_stimuli, fontsize=8)
    ax.set_xlabel("NWB file index")
    ax.set_ylabel("OMR stimulus label")
    ax.set_title("DANDI 001076 OMR calcium response by stimulus and file (per-file z-score)")
    ax.set_xticks(np.arange(len(files))[:: max(1, len(files) // 12)])
    ax.set_xticklabels([str(i) for i in np.arange(len(files))[:: max(1, len(files) // 12)]])
    fig.colorbar(im, ax=ax, label="within-file z-scored population response")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_rank_plot(out_dir: Path, aggregate_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "stimulus_response_rank.png"
    labels = [str(row["stimulus"]) for row in aggregate_rows]
    means = np.asarray([_safe_float(row["population_response_mean"]) for row in aggregate_rows], dtype=np.float64)
    sem = np.asarray([_safe_float(row["population_response_sem_across_files"], 0.0) for row in aggregate_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(np.arange(len(labels)), means, yerr=sem, color="#2c7fb8", edgecolor="#111111", linewidth=0.4)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("baseline-subtracted deconvolved response")
    ax.set_title("OMR stimulus response rank across DANDI 001076 NWB files")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_similarity_matrix(out_dir: Path, stimuli: list[str], mat: np.ndarray) -> Path:
    path = out_dir / "stimulus_similarity_matrix.png"
    if mat.size == 0:
        return path
    filled = np.where(np.isfinite(mat), mat, np.nanmean(mat, axis=1, keepdims=True))
    corr = np.corrcoef(filled)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="vlag" if "vlag" in plt.colormaps() else "coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(stimuli)))
    ax.set_yticks(np.arange(len(stimuli)))
    ax.set_xticklabels(stimuli, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(stimuli, fontsize=7)
    ax.set_title("Stimulus response similarity across files")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_selectivity_plot(out_dir: Path, roi_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "roi_selectivity_distribution.png"
    if not roi_rows:
        return path
    selectivity = np.asarray([_safe_float(row["selectivity_index"]) for row in roi_rows], dtype=np.float64)
    counts = Counter(str(row["best_stimulus"]) for row in roi_rows)
    labels, values = zip(*counts.most_common(20))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(selectivity[np.isfinite(selectivity)], bins=60, color="#41ab5d", edgecolor="#111111", linewidth=0.3)
    axes[0].set_xlabel("selectivity index")
    axes[0].set_ylabel("accepted ROI count")
    axes[0].set_title("Accepted ROI stimulus selectivity")
    axes[1].bar(np.arange(len(labels)), values, color="#fdae6b", edgecolor="#111111", linewidth=0.4)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("ROIs with stimulus as best response")
    axes[1].set_title("Top best-stimulus assignments")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_centroid_map(out_dir: Path, roi_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "centroid_selectivity_maps.png"
    if not roi_rows:
        return path
    axes_order = ["forward", "backward", "left", "right", "expansion", "contraction", "mixed_or_x"]
    colors = {
        "forward": "#1b9e77",
        "backward": "#d95f02",
        "left": "#7570b3",
        "right": "#e7298a",
        "expansion": "#66a61e",
        "contraction": "#e6ab02",
        "mixed_or_x": "#666666",
    }
    xs = np.asarray([_safe_float(row["centroid_x_norm"]) for row in roi_rows], dtype=np.float64)
    ys = np.asarray([_safe_float(row["centroid_y_norm"]) for row in roi_rows], dtype=np.float64)
    cats = [str(row["best_primary_axis"]) for row in roi_rows]
    sel = np.asarray([_safe_float(row["selectivity_index"]) for row in roi_rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for cat in axes_order:
        mask = np.asarray([x == cat for x in cats], dtype=bool) & np.isfinite(xs) & np.isfinite(ys)
        if np.any(mask):
            axes[0].scatter(xs[mask], ys[mask], s=2.0, alpha=0.22, color=colors[cat], label=cat, rasterized=True)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("normalized centroid x")
    axes[0].set_ylabel("normalized centroid y")
    axes[0].set_title("Accepted ROI centroids by best OMR response axis")
    axes[0].legend(markerscale=4, fontsize=8, loc="upper right")
    sc = axes[1].scatter(xs, ys, c=sel, s=2.0, alpha=0.22, cmap="viridis", vmin=0, vmax=1, rasterized=True)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("normalized centroid x")
    axes[1].set_ylabel("normalized centroid y")
    axes[1].set_title("Accepted ROI centroids by selectivity strength")
    fig.colorbar(sc, ax=axes[1], label="selectivity index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_trial_trace_plot(
    out_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    trace_sum: dict[str, np.ndarray],
    trace_sumsq: dict[str, np.ndarray],
    trace_count: dict[str, int],
) -> Path:
    path = out_dir / "trial_triggered_population_traces.png"
    top_stims = [str(row["stimulus"]) for row in aggregate_rows[:8]]
    fig, ax = plt.subplots(figsize=(12, 6))
    for stim in top_stims:
        n = trace_count.get(stim, 0)
        if n <= 0:
            continue
        mean = trace_sum[stim] / n
        var = np.maximum(0.0, trace_sumsq[stim] / n - mean * mean)
        sem = np.sqrt(var / max(1, n))
        ax.plot(TRACE_TIMES_S, mean, linewidth=1.8, label=f"{stim} (n={n})")
        ax.fill_between(TRACE_TIMES_S, mean - sem, mean + sem, alpha=0.15)
    ax.axvline(0.0, color="#111111", linewidth=1, linestyle="--")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("time from OMR trial start (s)")
    ax.set_ylabel("population deconvolved response minus baseline")
    ax.set_title("Trial-triggered population calcium traces for top OMR stimuli")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_pca_plot(out_dir: Path, stimuli: list[str], mat: np.ndarray) -> Path:
    path = out_dir / "stimulus_response_pca.png"
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return path
    x = np.where(np.isfinite(mat), mat, np.nanmean(mat, axis=1, keepdims=True))
    x = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s * s) / max(1e-12, float(np.sum(s * s)))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, stim in enumerate(stimuli):
        axes = _stimulus_axes(stim)
        ax.scatter(coords[i, 0], coords[i, 1], s=60)
        ax.text(coords[i, 0], coords[i, 1], stim, fontsize=8, ha="left", va="bottom")
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.axvline(0.0, color="#888888", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("Stimulus response embedding across DANDI files")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    nwb_dir: Path,
    out_dir: Path,
    files: list[Path],
    file_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    roi_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    plots: list[Path],
) -> None:
    accepted_total = int(sum(_safe_float(row["accepted_rois"], 0.0) for row in file_rows))
    trials_total = len(trial_rows)
    top = aggregate_rows[0] if aggregate_rows else {}
    top_stim = top.get("stimulus", "")
    top_response = _safe_float(top.get("population_response_mean"), 0.0)
    stim_count = len(aggregate_rows)
    rate_values = np.asarray([_safe_float(row["acquisition_rate_hz"]) for row in file_rows], dtype=np.float64)
    roi_counts = np.asarray([_safe_float(row["accepted_rois"]) for row in file_rows], dtype=np.float64)
    best_counts = Counter(str(row["best_stimulus"]) for row in roi_rows)
    best_axis_counts = Counter(str(row["best_primary_axis"]) for row in roi_rows)
    lines = [
        "# DANDI 001076 OMR Calcium Validation Atlas",
        "",
        "## Scope",
        "",
        "This analysis reads the public DANDI 001076 NWB files associated with the Z-Robot/simZFish OMR calcium-imaging resource. It extracts trial-aligned, baseline-subtracted deconvolved calcium responses for the exact OMR stimulus labels stored in the NWB trials table.",
        "",
        "This validates the availability and structure of a neural OMR target surface. It does not prove a paired natural-video -> calcium -> muscle/kinematics chain, because these NWB files contain OMR stimulus labels and calcium responses, not arbitrary selected videos and not synchronized MuJoCo or free-swimming tail kinematics.",
        "",
        "## Dataset Coverage",
        "",
        f"- NWB source directory: `{nwb_dir.resolve()}`",
        f"- Output directory: `{out_dir.resolve()}`",
        f"- NWB files parsed: `{len(files)}`",
        f"- Trial rows analyzed: `{trials_total}`",
        f"- Unique stimulus labels: `{stim_count}`",
        f"- Accepted ROIs analyzed: `{accepted_total}`",
        f"- Acquisition rate range: `{np.nanmin(rate_values):.6f}`-`{np.nanmax(rate_values):.6f}` Hz",
        f"- Accepted ROI range per file: `{int(np.nanmin(roi_counts))}`-`{int(np.nanmax(roi_counts))}`",
        "",
        "## Response Summary",
        "",
        f"- Top population-response stimulus: `{top_stim}` with mean response `{top_response:.6g}`.",
        f"- Most common best-stimulus assignments: `{dict(best_counts.most_common(8))}`.",
        f"- Best-axis ROI counts: `{dict(best_axis_counts)}`.",
        "",
        "| rank | stimulus | primary axis | files | trials | mean response | SEM | active ROI fraction | positive ROI fraction |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate_rows[:20], start=1):
        lines.append(
            f"| {rank} | `{row['stimulus']}` | `{row['primary_axis']}` | {int(row['files'])} | "
            f"{int(row['trials'])} | {_safe_float(row['population_response_mean']):.6g} | "
            f"{_safe_float(row['population_response_sem_across_files']):.6g} | "
            f"{_safe_float(row['selective_active_roi_fraction_mean']):.4f} | "
            f"{_safe_float(row['positive_roi_fraction_mean']):.4f} |"
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
            "## Interpretation For The Current Lab Pipeline",
            "",
            "- The DANDI OMR files can be used as a neural target surface for OMR stimulus classes such as forward, backward, left/right, converging/diverging, medial, and lateral patterns.",
            "- The current video branch can compare its inferred stimulus covariates and simZFish-style OMR internal states against these class-level calcium motifs.",
            "- This resource cannot directly supervise the selected-video branch unless selected videos are first converted into the same OMR stimulus coordinate system, and even then the comparison is class-level rather than frame-exact.",
            "- This resource cannot directly train MuJoCo muscle forces without a separate tail/ephys/kinematics bridge.",
            "",
            "## Files Written",
            "",
            f"- `nwb_file_summary.csv`: one row per NWB file.",
            f"- `stimulus_response_by_file.csv`: one row per file/stimulus.",
            f"- `stimulus_response_summary.csv`: aggregate response statistics per stimulus.",
            f"- `trial_response_summary.csv`: one row per NWB trial.",
            f"- `roi_selectivity_summary.csv`: one row per accepted ROI.",
            f"- `manifest.json`: machine-readable run summary and plot list.",
            "",
            "## Limitation Boundary",
            "",
            "Supported: public Z-Robot/DANDI OMR calcium recordings contain enough information to extract class-level whole-brain calcium response targets for the visual stimulus labels used in those recordings.",
            "",
            "Not supported: the public files alone do not establish a precise natural-video -> whole-brain calcium -> ephys/muscle -> free-swimming kinematics mapping for the current MuJoCo lab.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-dir", type=Path, default=DEFAULT_NWB_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=_positive_int, default=None, help="Optional file limit for smoke tests.")
    args = parser.parse_args()

    nwb_dir = args.nwb_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(nwb_dir.glob("**/*.nwb"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No NWB files found under {nwb_dir}")

    file_rows: list[dict[str, Any]] = []
    stim_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    file_vectors: list[dict[str, float]] = []
    trace_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(TRACE_TIMES_S, dtype=np.float64))
    trace_sumsq: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(TRACE_TIMES_S, dtype=np.float64))
    trace_count: dict[str, int] = defaultdict(int)
    errors: list[dict[str, Any]] = []

    for file_index, path in enumerate(files):
        try:
            result = _analyze_file(path, file_index)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file_index": file_index, "file": str(path), "error": repr(exc)})
            continue
        file_rows.append(result.summary)
        stim_rows.extend(result.stimulus_rows)
        trial_rows.extend(result.trial_rows)
        roi_rows.extend(result.roi_rows)
        file_vectors.append(result.file_stim_vector)
        for stim, arr in result.stim_trace_sum.items():
            trace_sum[stim] += arr
        for stim, arr in result.stim_trace_sumsq.items():
            trace_sumsq[stim] += arr
        for stim, count in result.stim_trace_count.items():
            trace_count[stim] += count
        print(
            json.dumps(
                {
                    "file_index": file_index,
                    "file": path.name,
                    "accepted_rois": result.summary["accepted_rois"],
                    "trials": result.summary["trials"],
                    "top_stimulus": result.summary["top_stimulus"],
                }
            ),
            flush=True,
        )

    aggregate_rows = _aggregate_stimulus_rows(stim_rows)
    stimuli, file_names, matrix = _matrix_from_file_vectors(file_rows, file_vectors)

    nwb_file_summary_path = out_dir / "nwb_file_summary.csv"
    stimulus_by_file_path = out_dir / "stimulus_response_by_file.csv"
    stimulus_summary_path = out_dir / "stimulus_response_summary.csv"
    trial_summary_path = out_dir / "trial_response_summary.csv"
    roi_summary_path = out_dir / "roi_selectivity_summary.csv"
    errors_path = out_dir / "parse_errors.json"
    _write_csv(nwb_file_summary_path, file_rows)
    _write_csv(stimulus_by_file_path, stim_rows)
    _write_csv(stimulus_summary_path, aggregate_rows)
    _write_csv(trial_summary_path, trial_rows)
    _write_csv(roi_summary_path, roi_rows)
    errors_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    plots = [
        _save_stimulus_heatmap(out_dir, stimuli, file_names, matrix),
        _save_rank_plot(out_dir, aggregate_rows),
        _save_similarity_matrix(out_dir, stimuli, matrix),
        _save_selectivity_plot(out_dir, roi_rows),
        _save_centroid_map(out_dir, roi_rows),
        _save_trial_trace_plot(out_dir, aggregate_rows, trace_sum, trace_sumsq, trace_count),
        _save_pca_plot(out_dir, stimuli, matrix),
    ]

    report_path = out_dir / "DANDI_OMR_NEURAL_VALIDATION.md"
    _write_report(
        report_path,
        nwb_dir=nwb_dir,
        out_dir=out_dir,
        files=files,
        file_rows=file_rows,
        aggregate_rows=aggregate_rows,
        roi_rows=roi_rows,
        trial_rows=trial_rows,
        plots=plots,
    )
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "nwb_dir": str(nwb_dir.resolve()),
        "requested_files": len(files),
        "parsed_files": len(file_rows),
        "parse_errors": len(errors),
        "trial_rows": len(trial_rows),
        "stimulus_rows": len(stim_rows),
        "unique_stimuli": len(aggregate_rows),
        "roi_rows": len(roi_rows),
        "total_accepted_rois": int(sum(_safe_float(row["accepted_rois"], 0.0) for row in file_rows)),
        "top_stimulus": aggregate_rows[0]["stimulus"] if aggregate_rows else "",
        "top_population_response_mean": aggregate_rows[0]["population_response_mean"] if aggregate_rows else float("nan"),
        "csv": {
            "nwb_file_summary": str(nwb_file_summary_path.resolve()),
            "stimulus_response_by_file": str(stimulus_by_file_path.resolve()),
            "stimulus_response_summary": str(stimulus_summary_path.resolve()),
            "trial_response_summary": str(trial_summary_path.resolve()),
            "roi_selectivity_summary": str(roi_summary_path.resolve()),
            "parse_errors": str(errors_path.resolve()),
        },
        "plots": [str(plot.resolve()) for plot in plots],
        "limitations": [
            "OMR calcium response classes only; no arbitrary selected video frames.",
            "No synchronized free-swimming kinematics in these NWB files.",
            "No direct PAULA neuron identity mapping from DANDI ROIs.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
