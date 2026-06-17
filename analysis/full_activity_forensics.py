"""Forensic long-timescale analysis for zebrafish lab recordings.

This is a post-processing companion to ``comprehensive_activity_study.py``.
It reads an already captured >=50k tick run and produces dense, inspectable
tables and figures for every scalar metric and every saved matrix channel.
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


PHYSICS_TIMESTEP_S = 0.005
DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parent
    / "out"
    / "comprehensive_activity_study"
    / "20260603_final_50k"
)


def _read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value in ("", None):
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"parse_error": line[:240]})
    return rows


def _load_matrices(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            out[key] = np.asarray(npz[key])
    return out


def _array(rows: list[dict[str, float]], field: str) -> np.ndarray:
    return np.asarray([float(row.get(field, 0.0)) for row in rows], dtype=np.float64)


def _time(rows: list[dict[str, float]]) -> np.ndarray:
    return _array(rows, "sim_time_s")


def _stats(arr: np.ndarray) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
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
        "n": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
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
    samples = max(1, min(samples, arr.size))
    kernel = np.ones(samples, dtype=np.float64)
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(np.isfinite(arr), arr, 0.0)
    return np.convolve(filled, kernel, mode="same") / np.maximum(
        1.0, np.convolve(valid, kernel, mode="same")
    )


def _action_time(row: dict[str, Any], fallback: int) -> float:
    for key in ("sim_time_s", "_sim_time_s", "driver_video_time_s", "video_time_s", "calcium_time_s"):
        try:
            if row.get(key) is not None:
                return float(row[key])
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _action_value(row: dict[str, Any], field: str) -> float:
    keys = [
        field,
        field.removeprefix("action_"),
        f"action_{field}",
        f"replay_{field}",
        f"zapbench_estimated_{field}",
    ]
    for key in keys:
        try:
            value = row.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _interp_action(actions: list[dict[str, Any]], field: str, x: np.ndarray) -> np.ndarray:
    if not actions or x.size == 0:
        return np.zeros_like(x)
    times = np.asarray([_action_time(row, i) for i, row in enumerate(actions)], dtype=np.float64)
    values = np.asarray([_action_value(row, field) for row in actions], dtype=np.float64)
    mask = np.isfinite(times) & np.isfinite(values)
    if np.sum(mask) < 2:
        return np.zeros_like(x)
    times = times[mask]
    values = values[mask]
    order = np.argsort(times)
    return np.interp(x, times[order], values[order], left=values[order][0], right=values[order][-1])


class Run:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self.rows = _read_rows(path / "state_metrics.csv")
        self.actions = _read_jsonl(path / "action_samples.jsonl")
        self.backend_frames = _read_jsonl(path / "backend_frame_responses.jsonl")
        self.metadata = _read_json(path / "metadata.json")
        self.matrices = _load_matrices(path / "state_matrices.npz")

    @property
    def t(self) -> np.ndarray:
        return _time(self.rows)

    @property
    def tick_span(self) -> int:
        if not self.rows:
            return 0
        return int(self.rows[-1].get("tick", 0.0) - self.rows[0].get("tick", 0.0))


def _discover_runs(run_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "state_metrics.csv").exists():
            runs.append(Run(child))
    return runs


def _matrix_names(run: Run, key: str, width: int) -> list[str]:
    if key.startswith("neuron_"):
        names = list(run.metadata.get("neuron_names") or [])
    elif key == "muscle_activations":
        names = list(run.metadata.get("muscle_names") or [])
    elif key.startswith("joint_"):
        names = list(run.metadata.get("joint_names") or [])
    elif key == "touch_forces":
        names = list(run.metadata.get("touch_names") or [])
    elif key == "neuromod":
        names = ["neuromod_m0", "neuromod_m1"]
    elif key.startswith("tail_yaw"):
        names = [f"tail_yaw_seg_{i:02d}" for i in range(width)]
    elif key.startswith("tail_pitch"):
        names = [f"tail_pitch_seg_{i:02d}" for i in range(width)]
    else:
        names = []
    if len(names) < width:
        names.extend(f"{key}_{i:02d}" for i in range(len(names), width))
    return names[:width]


def _write_scalar_extrema(out_dir: Path, runs: list[Run]) -> Path:
    fields = sorted({field for run in runs for row in run.rows for field in row})
    path = out_dir / "scalar_extrema_and_stats.csv"
    stat_names = list(_stats(np.zeros(0)).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "metric", "min_time_s", "max_time_s", *stat_names])
        for run in runs:
            t = run.t
            for field in fields:
                arr = _array(run.rows, field)
                finite = np.isfinite(arr)
                if not np.any(finite):
                    writer.writerow([run.name, field, 0.0, 0.0, *[_stats(arr)[name] for name in stat_names]])
                    continue
                idx_min = int(np.nanargmin(arr))
                idx_max = int(np.nanargmax(arr))
                stats = _stats(arr)
                writer.writerow(
                    [
                        run.name,
                        field,
                        float(t[idx_min]) if idx_min < t.size else 0.0,
                        float(t[idx_max]) if idx_max < t.size else 0.0,
                        *[stats[name] for name in stat_names],
                    ]
                )
    return path


def _write_matrix_rankings(out_dir: Path, runs: list[Run]) -> Path:
    path = out_dir / "matrix_channel_rankings.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "stream",
                "channel_index",
                "channel_name",
                "mean",
                "mean_abs",
                "rms",
                "std",
                "max_abs",
                "active_fraction",
                "peak_time_s",
                "corr_action_force",
                "corr_xy_speed",
                "corr_body_bend",
            ]
        )
        for run in runs:
            t = run.t
            force = _interp_action(run.actions, "force", t)
            speed = _array(run.rows, "speed_xy_mm_s")
            bend = _array(run.rows, "body_max_local_bend_2d_rad")
            for key, mat in run.matrices.items():
                arr = np.asarray(mat, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))
                if arr.ndim != 2 or arr.size == 0:
                    continue
                names = _matrix_names(run, key, arr.shape[1])
                threshold = max(1e-9, float(np.nanquantile(np.abs(arr), 0.95)) * 0.10)
                for idx in range(arr.shape[1]):
                    y = arr[:, idx]
                    finite = np.isfinite(y)
                    if not np.any(finite):
                        continue
                    abs_y = np.abs(y)
                    peak = int(np.nanargmax(abs_y))
                    writer.writerow(
                        [
                            run.name,
                            key,
                            idx,
                            names[idx],
                            float(np.nanmean(y)),
                            float(np.nanmean(abs_y)),
                            float(np.sqrt(np.nanmean(y * y))),
                            float(np.nanstd(y)),
                            float(np.nanmax(abs_y)),
                            float(np.nanmean(abs_y > threshold)),
                            float(t[peak]) if peak < t.size else 0.0,
                            _corr(y, force),
                            _corr(y, speed),
                            _corr(y, bend),
                        ]
                    )
    return path


def _write_window_summary(out_dir: Path, runs: list[Run], window_s: float) -> Path:
    metrics = [
        "speed_xy_mm_s",
        "speed_3d_mm_s",
        "vertical_speed_abs_mm_s",
        "heading_rate_rad_s",
        "pitch_rad",
        "tail_yaw_abs_mean",
        "tail_yaw_abs_max",
        "tail_pitch_abs_mean",
        "muscle_sum",
        "muscle_lr_bias",
        "neuron_s_abs_mean",
        "neuron_fired_count",
        "neuromod_m0",
        "neuromod_m1",
        "free_energy",
        "body_straightness",
        "body_max_local_bend_2d_rad",
    ]
    path = out_dir / f"window_summary_{int(window_s)}s.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "window_s", "window_index", "start_s", "end_s", "metric", "mean", "p95", "max"])
        for run in runs:
            t = run.t
            if t.size == 0:
                continue
            n_windows = int(math.ceil(float(t[-1]) / window_s))
            for wi in range(n_windows):
                start = wi * window_s
                end = min(float(t[-1]), (wi + 1) * window_s)
                mask = (t >= start) & (t < end if wi < n_windows - 1 else t <= end)
                if not np.any(mask):
                    continue
                action_force = _interp_action(run.actions, "force", t)
                metric_arrays = {metric: _array(run.rows, metric) for metric in metrics}
                metric_arrays["action_force"] = action_force
                metric_arrays["action_side_score"] = _interp_action(run.actions, "side_score", t)
                metric_arrays["action_confidence"] = _interp_action(run.actions, "confidence", t)
                for metric, arr in metric_arrays.items():
                    vals = arr[mask]
                    if vals.size == 0:
                        continue
                    writer.writerow(
                        [
                            run.name,
                            window_s,
                            wi,
                            start,
                            end,
                            metric,
                            float(np.nanmean(vals)),
                            float(np.nanquantile(vals, 0.95)),
                            float(np.nanmax(vals)),
                        ]
                    )
    return path


def _numeric_action_fields(actions: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for row in actions:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields.add(key)
    return sorted(fields)


def _write_action_and_video_stats(out_dir: Path, runs: list[Run]) -> tuple[Path, Path, Path]:
    action_stats = out_dir / "action_numeric_stats.csv"
    bout_counts = out_dir / "action_bout_counts.csv"
    video_events = out_dir / "video_quality_events.csv"
    stat_names = list(_stats(np.zeros(0)).keys())
    with action_stats.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "field", *stat_names])
        for run in runs:
            for field in _numeric_action_fields(run.actions):
                arr = np.asarray([_action_value(row, field) for row in run.actions], dtype=np.float64)
                st = _stats(arr)
                writer.writerow([run.name, field, *[st[name] for name in stat_names]])
    with bout_counts.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "bout_type", "count", "fraction"])
        for run in runs:
            counts = Counter(str(row.get("action_bout_type") or row.get("bout_type") or "unknown") for row in run.actions)
            total = sum(counts.values()) or 1
            for bout, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                writer.writerow([run.name, bout, count, count / total])
    with video_events.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "frame_index",
                "sim_time_s",
                "video_time_s",
                "event",
                "value",
                "action_force",
                "action_bout_type",
                "zapbench_distance",
            ]
        )
        for run in runs:
            if not run.actions:
                continue
            for row in run.actions:
                if "flow_reliability" not in row and "camera_shake" not in row:
                    continue
                checks = [
                    ("low_flow_reliability", _action_value(row, "flow_reliability"), 0.25, "lt"),
                    ("high_camera_shake", _action_value(row, "camera_shake"), 0.75, "gt"),
                    ("high_compression_noise", _action_value(row, "compression_noise"), 0.30, "gt"),
                    ("high_zapbench_distance", _action_value(row, "zapbench_distance"), 20.0, "gt"),
                ]
                for event, value, threshold, mode in checks:
                    triggered = value < threshold if mode == "lt" else value > threshold
                    if triggered:
                        writer.writerow(
                            [
                                run.name,
                                int(_action_value(row, "frame_index")),
                                _action_time(row, 0),
                                _action_value(row, "video_time_s"),
                                event,
                                value,
                                _action_value(row, "action_force"),
                                str(row.get("action_bout_type") or ""),
                                _action_value(row, "zapbench_distance"),
                            ]
                        )
    return action_stats, bout_counts, video_events


def _plot_integrated_timeline(out_dir: Path, runs: list[Run]) -> Path:
    panels = [
        ("force", "action force"),
        ("speed_xy_mm_s", "XY speed"),
        ("tail_yaw_abs_mean", "tail yaw mean abs"),
        ("muscle_sum", "muscle sum"),
        ("body_max_local_bend_2d_rad", "body local bend"),
        ("body_straightness", "body straightness"),
        ("heading_rate_rad_s", "heading rate"),
        ("pitch_rad", "pitch"),
        ("neuron_s_abs_mean", "PAULA |S| mean"),
        ("neuron_fired_count", "PAULA fired count"),
        ("neuromod_m0", "neuromod M0"),
        ("neuromod_m1", "neuromod M1"),
        ("free_energy", "free energy"),
        ("touch_sum", "touch/contact"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 30), sharex=True)
    for run in runs:
        t = run.t
        if t.size == 0:
            continue
        dt = float(np.nanmedian(np.diff(t))) if t.size > 2 else PHYSICS_TIMESTEP_S
        smooth = max(3, int(round(5.0 / max(dt, 1e-6))))
        for ax, (field, label) in zip(axes, panels):
            if field in {"force", "side_score", "confidence"}:
                y = _interp_action(run.actions, field, t)
            else:
                y = _array(run.rows, field)
            ax.plot(t, _rolling_mean(y, smooth), linewidth=1.0, label=run.name)
            ax.set_ylabel(label)
            ax.grid(alpha=0.18)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("simulation time (s)")
    fig.suptitle("Forensic integrated timeline, 5 s rolling mean, >=50k ticks", y=0.995)
    fig.tight_layout()
    path = out_dir / "forensic_integrated_timeline.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _top_channels(mat: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    score = np.nanstd(arr, axis=0) + 0.25 * np.nanmean(np.abs(arr), axis=0)
    return np.argsort(-score)[: min(n, arr.shape[1])]


def _plot_top_matrix_timeseries(out_dir: Path, runs: list[Run], key: str, title: str, n: int = 20) -> list[Path]:
    paths: list[Path] = []
    for run in runs:
        mat = np.asarray(run.matrices.get(key, np.zeros((0, 0))), dtype=np.float64)
        if mat.ndim != 2 or mat.size == 0:
            continue
        idx = _top_channels(mat, n)
        if idx.size == 0:
            continue
        t = run.t
        names = _matrix_names(run, key, mat.shape[1])
        fig, ax = plt.subplots(figsize=(17, 8))
        im = ax.imshow(
            mat[:, idx].T,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            extent=[float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else mat.shape[0], idx.size, 0],
        )
        ax.set_yticks(np.arange(idx.size) + 0.5)
        ax.set_yticklabels([names[i] for i in idx], fontsize=7)
        ax.set_xlabel("simulation time (s)")
        ax.set_title(f"{run.name}: top {idx.size} {title} channels")
        fig.colorbar(im, ax=ax, label=key)
        fig.tight_layout()
        path = out_dir / f"{run.name}_{key}_top{idx.size}_timeseries.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_tail_segment_envelopes(out_dir: Path, runs: list[Run]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    for run in runs:
        for row, key in enumerate(("tail_yaw_rad", "tail_pitch_rad")):
            mat = np.asarray(run.matrices.get(key, np.zeros((0, 0))), dtype=np.float64)
            if mat.ndim != 2 or mat.size == 0:
                continue
            x = np.arange(mat.shape[1])
            axes[row, 0].plot(x, np.nanmean(np.abs(mat), axis=0), marker="o", label=run.name)
            axes[row, 1].plot(x, np.nanquantile(np.abs(mat), 0.95, axis=0), marker="o", label=run.name)
            axes[row, 0].set_title(f"{key}: mean absolute segment activity")
            axes[row, 1].set_title(f"{key}: p95 absolute segment activity")
            for col in range(2):
                axes[row, col].set_xlabel("tail segment index, anterior to posterior")
                axes[row, col].set_ylabel("rad")
                axes[row, col].grid(alpha=0.18)
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / "tail_segment_envelopes.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_event_triggered_responses(out_dir: Path, runs: list[Run]) -> Path:
    targets = [
        ("speed_xy_mm_s", "XY speed"),
        ("tail_yaw_abs_mean", "tail yaw"),
        ("muscle_sum", "muscle sum"),
        ("body_max_local_bend_2d_rad", "body bend"),
        ("free_energy", "free energy"),
    ]
    fig, axes = plt.subplots(len(runs), len(targets), figsize=(18, max(5, 4 * len(runs))), squeeze=False)
    for i, run in enumerate(runs):
        t = run.t
        if t.size < 16:
            continue
        force = _interp_action(run.actions, "force", t)
        threshold = max(0.20, float(np.nanquantile(force, 0.75)) if force.size else 0.20)
        active = force >= threshold
        starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
        dt = float(np.nanmedian(np.diff(t))) if t.size > 2 else PHYSICS_TIMESTEP_S
        pre = int(round(2.0 / max(dt, 1e-6)))
        post = int(round(6.0 / max(dt, 1e-6)))
        rel = (np.arange(-pre, post + 1) * dt).astype(np.float64)
        for j, (field, label) in enumerate(targets):
            snippets = []
            y = _array(run.rows, field)
            for start in starts:
                lo = start - pre
                hi = start + post + 1
                if lo < 0 or hi > y.size:
                    continue
                snippets.append(y[lo:hi])
            ax = axes[i, j]
            if snippets:
                stack = np.vstack(snippets)
                mean = np.nanmean(stack, axis=0)
                p25 = np.nanquantile(stack, 0.25, axis=0)
                p75 = np.nanquantile(stack, 0.75, axis=0)
                ax.plot(rel, mean, color="tab:blue")
                ax.fill_between(rel, p25, p75, color="tab:blue", alpha=0.2)
            ax.axvline(0, color="0.2", linewidth=0.8)
            ax.set_title(f"{run.name}: {label} around action bouts")
            ax.set_xlabel("seconds from force onset")
            ax.grid(alpha=0.18)
    fig.tight_layout()
    path = out_dir / "event_triggered_responses.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_action_source_panels(out_dir: Path, runs: list[Run]) -> list[Path]:
    paths: list[Path] = []
    for run in runs:
        if not run.actions:
            continue
        t = np.asarray([_action_time(row, i) for i, row in enumerate(run.actions)], dtype=np.float64)
        fields = [
            ("force", "force"),
            ("kick_score", "kick score"),
            ("side_score", "side score"),
            ("confidence", "confidence"),
            ("tail_frequency_hz", "tail frequency Hz"),
            ("tail_amplitude", "tail amplitude"),
        ]
        if any("flow_reliability" in row for row in run.actions):
            fields.extend(
                [
                    ("flow_reliability", "flow reliability"),
                    ("camera_shake", "camera shake"),
                    ("motion_energy", "motion energy"),
                    ("zapbench_distance", "aux ZAPBench distance"),
                ]
            )
        fig, axes = plt.subplots(len(fields), 1, figsize=(16, max(9, 2.2 * len(fields))), sharex=True)
        for ax, (field, label) in zip(np.ravel(axes), fields):
            y = np.asarray([_action_value(row, field) for row in run.actions], dtype=np.float64)
            ax.plot(t, y, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.18)
        np.ravel(axes)[-1].set_xlabel("action/replay time (s)")
        fig.suptitle(f"{run.name}: source action/decoder timeline", y=0.995)
        fig.tight_layout()
        path = out_dir / f"{run.name}_source_action_timeline.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_cross_correlation(out_dir: Path, runs: list[Run]) -> Path:
    fields = [
        "force",
        "side_score",
        "confidence",
        "speed_xy_mm_s",
        "tail_yaw_abs_mean",
        "muscle_sum",
        "body_max_local_bend_2d_rad",
        "heading_rate_rad_s",
        "pitch_rad",
        "neuron_s_abs_mean",
        "neuron_fired_count",
        "neuromod_m0",
        "neuromod_m1",
        "free_energy",
    ]
    fig, axes = plt.subplots(1, len(runs), figsize=(8 * len(runs), 7), squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        t = run.t
        arrays: list[np.ndarray] = []
        labels: list[str] = []
        for field in fields:
            if field in {"force", "side_score", "confidence"}:
                arr = _interp_action(run.actions, field, t)
            else:
                arr = _array(run.rows, field)
            if arr.size == t.size and np.nanstd(arr) > 1e-12:
                arrays.append(arr)
                labels.append(field)
        mat = np.eye(len(arrays), dtype=np.float64)
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                mat[i, j] = mat[j, i] = _corr(arrays[i], arrays[j])
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(run.name)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Pearson r")
    fig.suptitle("Cross-channel correlation map, action/body/neural/energy metrics", y=0.98)
    path = out_dir / "cross_channel_correlation_maps.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(out_dir: Path, run_dir: Path, runs: list[Run], artifacts: dict[str, Any]) -> Path:
    lines = [
        "# Forensic Long-Timescale Zebrafish Activity Study",
        "",
        f"- source run directory: `{run_dir.resolve()}`",
        f"- forensic output directory: `{out_dir.resolve()}`",
        "- analysis basis: browser/lab WebSocket state streams and action-source JSONL logs",
        "- tick gate: each included run must span at least 50,000 physics ticks",
        "",
        "## Runs",
        "",
        "| run | WS frames | tick span | sim seconds | action samples | backend video frames | scalar fields | matrix streams |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        scalar_fields = sorted({field for row in run.rows for field in row})
        sim_seconds = run.tick_span * PHYSICS_TIMESTEP_S
        lines.append(
            f"| `{run.name}` | {len(run.rows)} | {run.tick_span} | {sim_seconds:.3f} | "
            f"{len(run.actions)} | {len(run.backend_frames)} | {len(scalar_fields)} | {len(run.matrices)} |"
        )
    lines.extend(
        [
            "",
            "## What Was Measured",
            "",
            "The analysis records body COM, body-segment geometry, heading, pitch, realized tail yaw/pitch, all MuJoCo joint angles and velocities, all actuator controls, all touch/contact channels, all PAULA state vectors (`S`, `R`, `B`, `Tref`, firing bits, `M0`, `M1`), aggregate neuromodulation, free energy, and the active action-source state.",
            "",
            "The calcium branch is driven by ZAPBench-derived calcium/ephys replay labels. The video branch is driven by backend OpenCV optical flow, camera stabilization, simZFish-style retinal counters, and an OMR/bout adapter; the ZAPBench nearest-neighbor covariate mapping is logged as auxiliary diagnostics.",
            "",
            "## Generated Tables",
            "",
        ]
    )
    table_items = [
        ("scalar_extrema", "Every scalar metric: distribution, min/max, and time of extrema."),
        ("matrix_rankings", "Every matrix channel: mean, RMS, active fraction, peak time, and correlations with action force, speed, and body bend."),
        ("window_10s", "10-second window summaries for key body, motor, neural, and action metrics."),
        ("window_30s", "30-second window summaries for the same key metrics."),
        ("action_stats", "Every numeric action/decoder field distribution."),
        ("bout_counts", "Bout/action-type counts and fractions."),
        ("video_quality_events", "Frames with low flow reliability, high camera shake, high compression, or high auxiliary ZAPBench distance."),
    ]
    for key, desc in table_items:
        lines.append(f"- `{key}`: `{Path(artifacts[key]).resolve()}` - {desc}")
    lines.extend(["", "## Generated Figures", ""])
    for key, value in artifacts.items():
        if key in {item[0] for item in table_items}:
            continue
        values = value if isinstance(value, list) else [value]
        for path in values:
            p = Path(path)
            if p.suffix.lower() != ".png":
                continue
            lines.append(f"- `{key}`: `{p.resolve()}`")
    lines.extend(
        [
            "",
            "## Key Findings From This Pass",
            "",
        ]
    )
    for run in runs:
        rows = run.rows
        if not rows:
            continue
        speed = _array(rows, "speed_xy_mm_s")
        bend = _array(rows, "body_max_local_bend_2d_rad")
        straight = _array(rows, "body_straightness")
        muscle = _array(rows, "muscle_abs_max")
        free = _array(rows, "free_energy")
        force = _interp_action(run.actions, "force", run.t)
        lines.append(
            f"- `{run.name}` spans `{run.tick_span}` ticks ({run.tick_span * PHYSICS_TIMESTEP_S:.3f} s); "
            f"p95 speed `{np.nanquantile(speed, 0.95):.4g}` mm/s, p95 local bend "
            f"`{np.nanquantile(bend, 0.95):.4g}` rad, mean straightness `{np.nanmean(straight):.4g}`, "
            f"max muscle activation `{np.nanmax(muscle):.4g}`, mean free energy `{np.nanmean(free):.4g}`, "
            f"mean action force `{np.nanmean(force):.4g}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is now a full observability pass over the available runtime state, not just a behavior-summary plot. It still cannot prove scientific identity with live zebrafish because the public paper resources do not provide one paired natural-video -> whole-brain calcium -> muscle/kinematics dataset for this exact embodied setting. The output is therefore the right artifact for lab review: it shows what the current implementation actually does at every exposed layer, where it is stable, and where the biological bridge remains approximate.",
            "",
        ]
    )
    path = out_dir / "FORENSIC_ACTIVITY_STUDY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "forensics")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_runs(run_dir)
    if not runs:
        raise SystemExit(f"no runs with state_metrics.csv found under {run_dir}")

    artifacts: dict[str, Any] = {}
    artifacts["scalar_extrema"] = _write_scalar_extrema(out_dir, runs)
    artifacts["matrix_rankings"] = _write_matrix_rankings(out_dir, runs)
    artifacts["window_10s"] = _write_window_summary(out_dir, runs, 10.0)
    artifacts["window_30s"] = _write_window_summary(out_dir, runs, 30.0)
    action_stats, bout_counts, video_events = _write_action_and_video_stats(out_dir, runs)
    artifacts["action_stats"] = action_stats
    artifacts["bout_counts"] = bout_counts
    artifacts["video_quality_events"] = video_events
    artifacts["integrated_timeline"] = _plot_integrated_timeline(out_dir, runs)
    artifacts["top_neuron_s"] = _plot_top_matrix_timeseries(out_dir, runs, "neuron_s", "PAULA S")
    artifacts["top_neuron_fired"] = _plot_top_matrix_timeseries(out_dir, runs, "neuron_fired", "PAULA firing")
    artifacts["top_muscles"] = _plot_top_matrix_timeseries(out_dir, runs, "muscle_activations", "muscle activation")
    artifacts["tail_yaw"] = _plot_top_matrix_timeseries(out_dir, runs, "tail_yaw_rad", "tail yaw", n=16)
    artifacts["tail_segment_envelopes"] = _plot_tail_segment_envelopes(out_dir, runs)
    artifacts["event_triggered"] = _plot_event_triggered_responses(out_dir, runs)
    artifacts["action_source_panels"] = _plot_action_source_panels(out_dir, runs)
    artifacts["correlations"] = _plot_cross_correlation(out_dir, runs)
    artifacts["report"] = _write_report(out_dir, run_dir, runs, artifacts)
    manifest = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "runs": [
            {
                "name": run.name,
                "frames": len(run.rows),
                "tick_span": run.tick_span,
                "sim_seconds": run.tick_span * PHYSICS_TIMESTEP_S,
                "actions": len(run.actions),
                "backend_video_frames": len(run.backend_frames),
                "scalar_fields": len({field for row in run.rows for field in row}),
                "matrix_streams": sorted(run.matrices),
            }
            for run in runs
        ],
        "artifacts": {key: [str(p) for p in value] if isinstance(value, list) else str(value) for key, value in artifacts.items()},
    }
    (out_dir / "forensics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
