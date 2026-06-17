"""Train a ZAPBench brain-activity decoder against direct raw tail ephys.

This is the direct-label companion to ``zapbench_action_decoder.py``.  It reads
the public 10-channel ZAPBench stimulus/ephys recording, aligns motor nerve
activity to the imaging-volume TTLs, derives left/right fictive tail-action
labels, and trains the same compact linear decoder from calcium trace windows to
tail action:

* kick/no kick
* side: none/left/right
* force magnitude in [0, 1]

The raw recording is not checked into the repository.  Download it once with:

```
curl -L --continue-at - \
  --output zebrafish_live_demo_v2/analysis/cache/zapbench/stimuli_and_ephys.10chFlt \
  https://storage.googleapis.com/zapbench-release/volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ACTIVE_INFERENCE = REPO_ROOT / "active-inference"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ACTIVE_INFERENCE) not in sys.path:
    sys.path.insert(0, str(ACTIVE_INFERENCE))

from zapbench_action_decoder import (  # noqa: E402
    CONDITION_NAMES,
    CONDITION_OFFSETS,
    TRACE_SPEC,
    _condition_for_row,
    _labels_subset,
    _load_selected_traces,
    _metrics,
    _predict,
    _read_tensorstore,
    _ridge_fit,
    _train_test_mask,
    _window_features,
)


RAW_URL = (
    "https://storage.googleapis.com/zapbench-release/volumes/20240930/"
    "stimuli_raw/stimuli_and_ephys.10chFlt"
)
N_RAW_CHANNELS = 10
N_TRACE_FRAMES = 7879
KNOWN_STIMULUS_CHANNELS = {
    2: "ttl",
    3: "stimParam4",
    4: "condition_index",
    6: "stimParam3",
    8: "visual_velocity",
}
KNOWN_FRAME_RATE_HZ = 1.093


@dataclass(frozen=True)
class EphysDecoderRun:
    raw_path: Path
    output_dir: Path
    artifact_path: Path
    neurons_per_block: int
    neuron_blocks: int
    projection_components: int
    context: int
    ridge_lambda: float
    seed: int
    split: str
    left_channel: int | None
    right_channel: int | None
    force_threshold_quantile: float | None
    active_weight: float
    side_weight: float
    label_lag: int


def _raw_memmap(path: Path) -> np.memmap:
    if not path.exists():
        raise SystemExit(
            f"Missing raw ephys file: {path}\n"
            f"Download from: {RAW_URL}"
        )
    size = path.stat().st_size
    if size % (np.dtype(np.float32).itemsize * N_RAW_CHANNELS):
        raise ValueError(
            f"{path} size does not divide into {N_RAW_CHANNELS} float32 channels"
        )
    samples = size // (np.dtype(np.float32).itemsize * N_RAW_CHANNELS)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(samples, N_RAW_CHANNELS))


def _find_peaks_min_distance(x: np.ndarray, *, height: float, distance: int) -> np.ndarray:
    above = np.asarray(x > float(height), dtype=bool)
    if not np.any(above):
        return np.zeros(0, dtype=np.int64)
    padded = np.concatenate([[False], above, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    peaks: list[int] = []
    for start, end in zip(starts, ends):
        segment = x[start:end]
        peak = int(start + int(np.argmax(segment)))
        if not peaks or peak - peaks[-1] >= distance:
            peaks.append(peak)
        elif float(x[peak]) > float(x[peaks[-1]]):
            peaks[-1] = peak
    return np.asarray(peaks, dtype=np.int64)


def _remove_peaks_near(peaks: np.ndarray, reference: np.ndarray, tolerance: int = 5) -> np.ndarray:
    if peaks.size == 0 or reference.size == 0:
        return peaks
    idx = np.searchsorted(reference, peaks)
    keep = np.ones(peaks.shape[0], dtype=bool)
    in_next = idx < reference.shape[0]
    keep[in_next] &= np.abs(peaks[in_next] - reference[idx[in_next]]) > tolerance
    in_prev = idx > 0
    keep[in_prev] &= np.abs(peaks[in_prev] - reference[idx[in_prev] - 1]) > tolerance
    return peaks[keep]


def _channel_stats(raw: np.memmap, *, sample_stride: int = 1000) -> list[dict[str, Any]]:
    sample = np.asarray(raw[::sample_stride, :], dtype=np.float32)
    stats: list[dict[str, Any]] = []
    for ch in range(sample.shape[1]):
        x = sample[:, ch]
        finite = x[np.isfinite(x)]
        rounded_unique = int(np.unique(np.round(finite[: min(finite.shape[0], 10000)], 4)).shape[0])
        stats.append(
            {
                "channel": ch,
                "known_role": KNOWN_STIMULUS_CHANNELS.get(ch, "unknown_candidate"),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "diff_std": float(np.std(np.diff(finite))),
                "min": float(np.min(finite)),
                "p01": float(np.quantile(finite, 0.01)),
                "p50": float(np.quantile(finite, 0.50)),
                "p99": float(np.quantile(finite, 0.99)),
                "max": float(np.max(finite)),
                "rounded_unique_first_10000": rounded_unique,
            }
        )
    return stats


def _infer_ephys_channels(stats: list[dict[str, Any]]) -> tuple[int, int]:
    candidates = [
        item
        for item in stats
        if int(item["channel"]) not in KNOWN_STIMULUS_CHANNELS
        and int(item["rounded_unique_first_10000"]) > 100
    ]
    if len(candidates) < 2:
        candidates = [item for item in stats if int(item["channel"]) not in KNOWN_STIMULUS_CHANNELS]
    ranked = sorted(candidates, key=lambda item: float(item["diff_std"]), reverse=True)
    channels = sorted(int(item["channel"]) for item in ranked[:2])
    if len(channels) != 2:
        raise ValueError(f"could not infer two ephys channels from stats: {stats}")
    return channels[0], channels[1]


def _align_imaging_ttls(raw: np.memmap) -> dict[str, Any]:
    ttls = raw[:, 2]
    high_peaks = _find_peaks_min_distance(ttls, height=3.55, distance=500)
    low_peaks = _find_peaks_min_distance(ttls, height=1.0, distance=50)
    low_peaks = _remove_peaks_near(low_peaks, high_peaks)
    frame_count = min(N_TRACE_FRAMES, max(0, high_peaks.shape[0] - 1))
    starts = high_peaks[:frame_count]
    ends = high_peaks[1 : frame_count + 1]
    condition_index = raw[:, 4]
    raw_condition_onsets = np.flatnonzero(np.diff(condition_index) != 0) + 1
    onset_frames = np.searchsorted(high_peaks, raw_condition_onsets)
    median_samples_per_volume = float(np.median(np.diff(high_peaks))) if high_peaks.shape[0] > 1 else 0.0
    sample_rate_hz = median_samples_per_volume * KNOWN_FRAME_RATE_HZ
    return {
        "high_peaks": high_peaks,
        "low_peaks": low_peaks,
        "starts": starts,
        "ends": ends,
        "frame_count": int(frame_count),
        "low_per_high": float(low_peaks.shape[0] / max(1, high_peaks.shape[0])),
        "median_samples_per_volume": median_samples_per_volume,
        "estimated_sample_rate_hz": float(sample_rate_hz),
        "raw_condition_onsets": raw_condition_onsets,
        "raw_condition_onset_frames": onset_frames,
    }


def _frame_windowed_power(
    raw: np.memmap,
    *,
    channel: int,
    starts: np.ndarray,
    ends: np.ndarray,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean_power = np.zeros(starts.shape[0], dtype=np.float32)
    peak_power = np.zeros(starts.shape[0], dtype=np.float32)
    window = max(4, int(window_samples))
    for i, (start, end) in enumerate(zip(starts, ends)):
        segment = np.asarray(raw[int(start) : int(end), channel], dtype=np.float32)
        if segment.size == 0:
            continue
        segment = segment - float(np.median(segment))
        n_windows = segment.size // window
        if n_windows <= 0:
            stds = np.asarray([float(np.std(segment))], dtype=np.float32)
        else:
            trimmed = segment[: n_windows * window].reshape(n_windows, window)
            stds = np.std(trimmed, axis=1).astype(np.float32)
        mean_power[i] = float(np.mean(stds))
        peak_power[i] = float(np.quantile(stds, 0.95))
    return mean_power, peak_power


def _condition_ids(rows: np.ndarray) -> np.ndarray:
    return np.asarray([_condition_for_row(int(row)) for row in rows], dtype=np.int32)


def _weighted_ridge_fit(
    x: np.ndarray, y: np.ndarray, lam: float, sample_weight: np.ndarray
) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    sw = np.sqrt(np.asarray(sample_weight, dtype=np.float32)).reshape((-1, 1))
    xw = xb * sw
    yw = y * sw
    gram = xw.T @ xw
    reg = np.eye(gram.shape[0], dtype=np.float32) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(gram + reg, xw.T @ yw).astype(np.float32)


def _direct_extra_metrics(
    y: dict[str, np.ndarray],
    pred: np.ndarray,
    *,
    side_abs: float = 0.15,
    force_none: float = 0.05,
) -> dict[str, float]:
    true_kick = y["kick"].astype(bool)
    pred_kick = pred[:, 0] >= 0.5
    true_side = y["side_class"].astype(np.int32)
    pred_side = np.zeros(true_side.shape[0], dtype=np.int32)
    pred_side[pred_kick & (pred[:, 2] >= force_none) & (pred[:, 1] < -side_abs)] = 1
    pred_side[pred_kick & (pred[:, 2] >= force_none) & (pred[:, 1] > side_abs)] = 2
    tp = float(np.sum(pred_kick & true_kick))
    fp = float(np.sum(pred_kick & ~true_kick))
    fn = float(np.sum(~pred_kick & true_kick))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    active_side = true_side != 0
    side_active_accuracy = (
        float(np.mean(pred_side[active_side] == true_side[active_side]))
        if np.any(active_side)
        else 1.0
    )
    side_none_accuracy = (
        float(np.mean(pred_side[~active_side] == 0)) if np.any(~active_side) else 1.0
    )
    return {
        "kick_precision": float(precision),
        "kick_recall": float(recall),
        "kick_f1": float(f1),
        "side_active_accuracy": float(side_active_accuracy),
        "side_none_accuracy": float(side_none_accuracy),
        "side_macro_accuracy": float(0.5 * (side_active_accuracy + side_none_accuracy)),
        "side_non_none_rate_pred_direct": float(np.mean(pred_side != 0)),
        "side_non_none_rate_true_direct": float(np.mean(active_side)),
    }


def _threshold_metrics(
    y: dict[str, np.ndarray],
    pred: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, float]:
    true_kick = y["kick"].astype(bool)
    true_side = y["side_class"].astype(np.int32)
    kick = pred[:, 0] >= float(thresholds["kick"])
    side = np.zeros(true_side.shape[0], dtype=np.int32)
    force_none = float(thresholds["force_none"])
    side_abs = float(thresholds["side_abs"])
    side[kick & (pred[:, 2] >= force_none) & (pred[:, 1] < -side_abs)] = 1
    side[kick & (pred[:, 2] >= force_none) & (pred[:, 1] > side_abs)] = 2
    active_side = true_side != 0
    tp = float(np.sum(kick & true_kick))
    fp = float(np.sum(kick & ~true_kick))
    fn = float(np.sum(~kick & true_kick))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    side_active_accuracy = (
        float(np.mean(side[active_side] == true_side[active_side]))
        if np.any(active_side)
        else 1.0
    )
    side_none_accuracy = (
        float(np.mean(side[~active_side] == 0)) if np.any(~active_side) else 1.0
    )
    return {
        "kick_accuracy": float(np.mean(kick == true_kick)),
        "kick_precision": float(precision),
        "kick_recall": float(recall),
        "kick_f1": float(f1),
        "side_accuracy": float(np.mean(side == true_side)),
        "side_active_accuracy": float(side_active_accuracy),
        "side_none_accuracy": float(side_none_accuracy),
        "side_macro_accuracy": float(0.5 * (side_active_accuracy + side_none_accuracy)),
        "side_non_none_rate_pred": float(np.mean(side != 0)),
        "side_non_none_rate_true": float(np.mean(active_side)),
    }


def _calibrate_thresholds(
    y_train: dict[str, np.ndarray],
    pred_train: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    best: tuple[float, dict[str, float], dict[str, float]] | None = None
    for kick in np.linspace(0.30, 0.68, 77):
        for side_abs in np.linspace(0.02, 0.16, 57):
            for force_none in (0.0, 0.02, 0.05, 0.08, 0.12, 0.15):
                thresholds = {
                    "kick": float(kick),
                    "side_abs": float(side_abs),
                    "force_none": float(force_none),
                }
                metrics = _threshold_metrics(y_train, pred_train, thresholds)
                score = (
                    metrics["kick_accuracy"]
                    + 0.20 * metrics["kick_f1"]
                    + 0.25 * metrics["side_active_accuracy"]
                    + 0.55 * metrics["side_accuracy"]
                    - 0.15
                    * abs(
                        metrics["side_non_none_rate_pred"]
                        - metrics["side_non_none_rate_true"]
                    )
                )
                if best is None or score > best[0]:
                    best = (float(score), thresholds, metrics)
    if best is None:
        raise ValueError("threshold calibration failed")
    return best[1], best[2]


def _build_direct_ephys_labels(
    raw: np.memmap,
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    left_channel: int,
    right_channel: int,
    sample_rate_hz: float,
    force_threshold_quantile: float | None,
) -> dict[str, Any]:
    window_samples = max(4, int(round(sample_rate_hz * 0.010)))
    left_mean, left_peak = _frame_windowed_power(
        raw, channel=left_channel, starts=starts, ends=ends, window_samples=window_samples
    )
    right_mean, right_peak = _frame_windowed_power(
        raw, channel=right_channel, starts=starts, ends=ends, window_samples=window_samples
    )

    left = 0.35 * left_mean + 0.65 * left_peak
    right = 0.35 * right_mean + 0.65 * right_peak
    total = left + right
    valid_rows = np.arange(starts.shape[0], dtype=np.int32)
    cond_ids = _condition_ids(valid_rows)
    dark_mask = cond_ids == CONDITION_NAMES.index("dark")
    baseline = total[dark_mask] if np.any(dark_mask) else total
    baseline_threshold = float(np.mean(baseline) + 2.5 * np.std(baseline))
    if force_threshold_quantile is None:
        force_threshold = baseline_threshold
    else:
        force_threshold = float(np.quantile(total, float(force_threshold_quantile)))

    total_p10 = float(np.quantile(total, 0.10))
    total_p995 = float(np.quantile(total, 0.995))
    denom = max(total_p995 - total_p10, 1e-9)
    force = np.clip((total - total_p10) / denom, 0.0, 1.0).astype(np.float32)
    kick = (total > force_threshold).astype(np.float32)

    left_p10 = float(np.quantile(left, 0.10))
    left_p99 = float(np.quantile(left, 0.99))
    right_p10 = float(np.quantile(right, 0.10))
    right_p99 = float(np.quantile(right, 0.99))
    left_norm = np.clip((left - left_p10) / max(left_p99 - left_p10, 1e-9), 0.0, 1.0)
    right_norm = np.clip((right - right_p10) / max(right_p99 - right_p10, 1e-9), 0.0, 1.0)
    side = np.clip((right_norm - left_norm) / (right_norm + left_norm + 1e-6), -1.0, 1.0)
    side = np.where(kick > 0, side, 0.0).astype(np.float32)
    side_class = np.zeros(side.shape[0], dtype=np.int32)
    side_class[(kick > 0) & (side < -0.15)] = 1
    side_class[(kick > 0) & (side > 0.15)] = 2

    labels = {
        "kick": kick.astype(np.float32),
        "side": side.astype(np.float32),
        "force": force.astype(np.float32),
        "side_class": side_class,
    }
    details = {
        "labels": labels,
        "left_power": left.astype(np.float32),
        "right_power": right.astype(np.float32),
        "total_power": total.astype(np.float32),
        "left_mean_power": left_mean,
        "right_mean_power": right_mean,
        "left_peak_power": left_peak,
        "right_peak_power": right_peak,
        "window_samples_10ms": int(window_samples),
        "baseline_threshold_raw": baseline_threshold,
        "force_threshold_raw": force_threshold,
        "force_threshold_quantile": force_threshold_quantile,
        "left_channel": int(left_channel),
        "right_channel": int(right_channel),
        "side_convention": "negative means left channel stronger; positive means right channel stronger",
    }
    return details


def _write_report(
    *,
    run: EphysDecoderRun,
    report_path: Path,
    metrics: dict[str, Any],
    channel_stats: list[dict[str, Any]],
    alignment: dict[str, Any],
    label_details: dict[str, Any],
    target_rows: np.ndarray,
    feature_rows: np.ndarray,
    block_ranges: list[tuple[int, int]],
) -> None:
    label_summary = {
        "kick_positive_rate": float(np.mean(label_details["labels"]["kick"][target_rows])),
        "force_mean": float(np.mean(label_details["labels"]["force"][target_rows])),
        "force_p95": float(np.quantile(label_details["labels"]["force"][target_rows], 0.95)),
        "left_channel": int(label_details["left_channel"]),
        "right_channel": int(label_details["right_channel"]),
        "force_threshold_raw": float(label_details["force_threshold_raw"]),
        "baseline_threshold_raw": float(label_details["baseline_threshold_raw"]),
        "window_samples_10ms": int(label_details["window_samples_10ms"]),
    }
    align_summary = {
        "raw_samples": int(_raw_memmap(run.raw_path).shape[0]),
        "high_ttl_peaks": int(alignment["high_peaks"].shape[0]),
        "low_ttl_peaks": int(alignment["low_peaks"].shape[0]),
        "aligned_frames": int(alignment["frame_count"]),
        "low_per_high": float(alignment["low_per_high"]),
        "median_samples_per_volume": float(alignment["median_samples_per_volume"]),
        "estimated_sample_rate_hz": float(alignment["estimated_sample_rate_hz"]),
        "raw_condition_onset_frames": [
            int(v) for v in np.asarray(alignment["raw_condition_onset_frames"]).tolist()
        ],
    }

    lines: list[str] = []
    lines.append("# ZAPBench Direct Ephys Tail-Action Decoder")
    lines.append("")
    lines.append("Generated: 2026-05-13")
    lines.append("")
    lines.append("## What Changed")
    lines.append("")
    lines.append("This run trains on direct raw fictive motor activity from the public 10-channel stimulus/ephys file, instead of stimulus-implied labels from the 26-column covariate matrix.")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("- ZAPBench landing page: https://zapbench-release.storage.googleapis.com/landing.html")
    lines.append("- Dataset README: https://zapbench-release.storage.googleapis.com/volumes/README.html")
    lines.append("- ICLR/OpenReview paper: https://openreview.net/forum?id=oCHsDpyawq")
    lines.append("- Official code/notebooks: https://github.com/google-research/zapbench")
    lines.append(f"- Raw ephys object: {RAW_URL}")
    lines.append("")
    lines.append("## Alignment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(align_summary, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Raw Channel Stats")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(channel_stats, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Label Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(label_summary, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Training Data")
    lines.append("")
    lines.append(f"- Trace source: `{TRACE_SPEC['kvstore']}`")
    lines.append(f"- Selected neuron blocks: {', '.join(f'{a}:{b}' for a, b in block_ranges)}")
    lines.append(f"- Feature rows used: {int(feature_rows.shape[0])}")
    lines.append(f"- Target rows used: {int(target_rows.shape[0])}")
    lines.append(f"- Label lag: `{run.label_lag}` imaging volumes (`target_row = feature_row + label_lag`).")
    lines.append(f"- Split: `{run.split}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The decoder input is calcium activity only. Labels are direct left/right tail motor nerve activity summarized per imaging volume. Because the calcium volume rate is about 1 Hz and ZAPBench uses nuclear GCaMP, this is a low-pass classifier, not a millisecond-resolved spike-to-tail controller.")
    if run.label_lag < 0:
        lines.append("")
        lines.append("This run uses a negative label lag, meaning the calcium window is matched to motor activity that occurred earlier. This is the expected direction when calcium fluorescence lags fast motor-electrode activity. Use a `--label-lag 0` run for strict same-volume causal decoding.")
    lines.append("")
    lines.append("The side convention is inherited from raw channel order: negative means the inferred left ephys channel is stronger, positive means the inferred right ephys channel is stronger.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_direct_ephys_decoder(run: EphysDecoderRun) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.artifact_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _raw_memmap(run.raw_path)
    channel_stats = _channel_stats(raw)
    if run.left_channel is None or run.right_channel is None:
        left_channel, right_channel = _infer_ephys_channels(channel_stats)
    else:
        left_channel, right_channel = int(run.left_channel), int(run.right_channel)

    alignment = _align_imaging_ttls(raw)
    frame_count = int(alignment["frame_count"])
    starts = alignment["starts"]
    ends = alignment["ends"]
    label_details = _build_direct_ephys_labels(
        raw,
        starts=starts,
        ends=ends,
        left_channel=left_channel,
        right_channel=right_channel,
        sample_rate_hz=float(alignment["estimated_sample_rate_hz"]),
        force_threshold_quantile=run.force_threshold_quantile,
    )
    labels_all = label_details["labels"]
    feature_rows = np.arange(run.context, frame_count, dtype=np.int32)
    target_rows = feature_rows + int(run.label_lag)
    valid = (target_rows >= 0) & (target_rows < frame_count)
    feature_rows = feature_rows[valid].astype(np.int32)
    target_rows = target_rows[valid].astype(np.int32)

    train_mask, test_mask = _train_test_mask(target_rows, split=run.split, seed=run.seed)

    trace_ds = _read_tensorstore(TRACE_SPEC)
    traces, block_ranges = _load_selected_traces(
        trace_ds, run.neuron_blocks, run.neurons_per_block
    )
    traces = traces[:frame_count]

    rng = np.random.default_rng(run.seed)
    train_feature_rows = feature_rows[train_mask]
    train_trace_samples = traces[train_feature_rows]
    trace_mean = train_trace_samples.mean(axis=0).astype(np.float32)
    trace_std = train_trace_samples.std(axis=0).astype(np.float32)
    trace_std = np.where(trace_std < 1e-4, 1.0, trace_std).astype(np.float32)
    normalized = (traces - trace_mean) / trace_std
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(normalized.shape[1]),
        size=(normalized.shape[1], run.projection_components),
    ).astype(np.float32)
    projected = normalized @ projection
    features = _window_features(projected, feature_rows, run.context)
    feature_mean = features[train_mask].mean(axis=0).astype(np.float32)
    feature_std = features[train_mask].std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-4, 1.0, feature_std).astype(np.float32)
    features_z = (features - feature_mean) / feature_std

    labels = {key: value[target_rows] for key, value in labels_all.items()}
    y_mat = np.stack([labels["kick"], labels["side"], labels["force"]], axis=1)
    sample_weight = (
        1.0
        + run.active_weight * labels["kick"]
        + run.side_weight * (labels["side_class"] != 0).astype(np.float32)
        + labels["force"]
    ).astype(np.float32)
    weights = _weighted_ridge_fit(
        features_z[train_mask],
        y_mat[train_mask],
        run.ridge_lambda,
        sample_weight[train_mask],
    )
    pred_test = _predict(features_z[test_mask], weights)
    metrics = _metrics(target_rows[test_mask], _labels_subset(labels, test_mask), pred_test)
    metrics["direct_extra"] = _direct_extra_metrics(_labels_subset(labels, test_mask), pred_test)
    pred_train = _predict(features_z[train_mask], weights)
    metrics["train"] = _metrics(target_rows[train_mask], _labels_subset(labels, train_mask), pred_train)
    metrics["train"]["direct_extra"] = _direct_extra_metrics(
        _labels_subset(labels, train_mask), pred_train
    )
    default_thresholds = {"kick": 0.5, "side_abs": 0.15, "force_none": 0.05}
    thresholds, train_threshold_metrics = _calibrate_thresholds(
        _labels_subset(labels, train_mask), pred_train
    )
    metrics["thresholds"] = {
        "default": {
            "thresholds": default_thresholds,
            "test": _threshold_metrics(
                _labels_subset(labels, test_mask), pred_test, default_thresholds
            ),
        },
        "calibrated": {
            "thresholds": thresholds,
            "train": train_threshold_metrics,
            "test": _threshold_metrics(
                _labels_subset(labels, test_mask), pred_test, thresholds
            ),
        },
        "objective": (
            "kick accuracy + 0.20*kick F1 + 0.25*active-side accuracy "
            "+ 0.55*side accuracy - side-rate penalty"
        ),
    }
    metrics["split"] = {
        "train_rows": int(np.sum(train_mask)),
        "test_rows": int(np.sum(test_mask)),
        "split_rule": (
            "stratified random 70/30 within each condition"
            if run.split == "stratified_random"
            else "first 70 percent of each condition for train, final 30 percent for test"
        ),
    }

    metadata = {
        "source": "ZAPBench public raw stimulus/ephys release",
        "raw_url": RAW_URL,
        "trace_spec": TRACE_SPEC["kvstore"],
        "condition_offsets": CONDITION_OFFSETS,
        "condition_names": CONDITION_NAMES,
        "selected_neuron_blocks": block_ranges,
        "projection_components": run.projection_components,
        "split": run.split,
        "label_type": "direct raw tail motor nerve ephys aligned to imaging frames",
        "left_channel": int(left_channel),
        "right_channel": int(right_channel),
        "side_convention": label_details["side_convention"],
        "force_threshold_raw": float(label_details["force_threshold_raw"]),
        "baseline_threshold_raw": float(label_details["baseline_threshold_raw"]),
        "window_samples_10ms": int(label_details["window_samples_10ms"]),
        "estimated_sample_rate_hz": float(alignment["estimated_sample_rate_hz"]),
        "active_weight": float(run.active_weight),
        "side_weight": float(run.side_weight),
        "label_lag": int(run.label_lag),
    }
    np.savez_compressed(
        run.artifact_path,
        projection=projection,
        weights=weights,
        trace_mean=trace_mean,
        trace_std=trace_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        context=np.asarray(run.context, dtype=np.int32),
        feature_rows=feature_rows,
        target_rows=target_rows,
        block_ranges=np.asarray(block_ranges, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        thresholds_json=np.asarray(json.dumps(thresholds, sort_keys=True)),
    )

    label_path = run.output_dir / "direct_ephys_labels.npz"
    np.savez_compressed(
        label_path,
        rows=np.arange(frame_count, dtype=np.int32),
        feature_rows=feature_rows,
        target_rows=target_rows,
        left_power=label_details["left_power"],
        right_power=label_details["right_power"],
        total_power=label_details["total_power"],
        kick=labels_all["kick"],
        side=labels_all["side"],
        force=labels_all["force"],
        side_class=labels_all["side_class"],
        starts=starts,
        ends=ends,
    )
    metrics_path = run.output_dir / "direct_ephys_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = run.output_dir / "direct_ephys_report.md"
    _write_report(
        run=run,
        report_path=report_path,
        metrics=metrics,
        channel_stats=channel_stats,
        alignment=alignment,
        label_details=label_details,
        feature_rows=feature_rows,
        target_rows=target_rows,
        block_ranges=block_ranges,
    )
    return {
        "artifact_path": str(run.artifact_path),
        "label_path": str(label_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "metrics": metrics,
        "left_channel": int(left_channel),
        "right_channel": int(right_channel),
        "aligned_frames": int(frame_count),
        "selected_neuron_blocks": block_ranges,
        "label_lag": int(run.label_lag),
    }


def parse_args() -> EphysDecoderRun:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=SCRIPT_DIR / "cache" / "zapbench" / "stimuli_and_ephys.10chFlt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "out" / "zapbench_ephys_action_decoder",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=ACTIVE_INFERENCE
        / "simulations"
        / "zebrafish"
        / "data"
        / "zapbench_ephys_tail_action_decoder.npz",
    )
    parser.add_argument("--neurons-per-block", type=int, default=512)
    parser.add_argument("--neuron-blocks", type=int, default=8)
    parser.add_argument("--projection-components", type=int, default=128)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--split",
        choices=("stratified_random", "chronological"),
        default="chronological",
    )
    parser.add_argument("--left-channel", type=int, default=None)
    parser.add_argument("--right-channel", type=int, default=None)
    parser.add_argument("--active-weight", type=float, default=2.0)
    parser.add_argument("--side-weight", type=float, default=5.0)
    parser.add_argument(
        "--label-lag",
        type=int,
        default=-8,
        help="Target row offset from feature row. Negative values account for calcium lag.",
    )
    parser.add_argument(
        "--force-threshold-quantile",
        type=float,
        default=None,
        help="Override direct ephys kick threshold with a raw total-power quantile.",
    )
    args = parser.parse_args()
    return EphysDecoderRun(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        artifact_path=args.artifact_path,
        neurons_per_block=args.neurons_per_block,
        neuron_blocks=args.neuron_blocks,
        projection_components=args.projection_components,
        context=args.context,
        ridge_lambda=args.ridge_lambda,
        seed=args.seed,
        split=args.split,
        left_channel=args.left_channel,
        right_channel=args.right_channel,
        force_threshold_quantile=args.force_threshold_quantile,
        active_weight=args.active_weight,
        side_weight=args.side_weight,
        label_lag=args.label_lag,
    )


def main() -> None:
    result = train_direct_ephys_decoder(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
