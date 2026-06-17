"""Synthesize a full activity atlas from a zebrafish comprehensive run.

This script is deliberately post-hoc: it reads the raw REST/WS artifacts
produced by ``comprehensive_activity_study.py`` and generates a consolidated
report with anomaly checks, lag analyses, all-metric stats, and visual summary
figures.  It does not touch the live backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHYSICS_TIMESTEP_S = 0.005
SOURCE_LINKS = {
    "ZAPBench landing": "https://zapbench-release.storage.googleapis.com/landing.html",
    "ZAPBench arXiv": "https://arxiv.org/abs/2503.02618",
    "ZAPBench GitHub": "https://github.com/google-research/zapbench",
    "ZAPBench datasets": "https://zapbench-release.storage.googleapis.com/volumes/README.html",
    "Google Research ZAPBench blog": "https://research.google/blog/improving-brain-models-with-zapbench/",
    "simZFish repository": "https://ponyo.epfl.ch/proj/zebrafish/simzfish",
    "DANDI 001076": "https://dandiarchive.org/dandiset/001076",
}


@dataclass
class RunData:
    name: str
    path: Path
    rows: list[dict[str, float]]
    actions: list[dict[str, Any]]
    metadata: dict[str, Any]
    matrices: dict[str, np.ndarray]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value in ("", None):
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"parse_error": line[:200]})
    return rows


def _load_matrices(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            out[key] = np.asarray(npz[key])
    return out


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
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


def _metric_fields(runs: list[RunData]) -> list[str]:
    fields: set[str] = set()
    for run in runs:
        for row in run.rows:
            fields.update(row)
    return sorted(fields)


def _array(rows: list[dict[str, float]], field: str) -> np.ndarray:
    return np.asarray([float(row.get(field, 0.0)) for row in rows], dtype=np.float64)


def _time(rows: list[dict[str, float]]) -> np.ndarray:
    return _array(rows, "sim_time_s")


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


def _action_time(row: dict[str, Any], fallback: int) -> float:
    for key in (
        "sim_time_s",
        "_sim_time_s",
        "driver_video_time_s",
        "video_time_s",
        "calcium_time_s",
    ):
        try:
            value = row.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _action_value(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = row.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _interp_action(run: RunData, field: str, x: np.ndarray) -> np.ndarray:
    if not run.actions or x.size == 0:
        return np.zeros_like(x)
    times = np.asarray([_action_time(row, i) for i, row in enumerate(run.actions)], dtype=np.float64)
    values = np.asarray(
        [
            _action_value(
                row,
                field,
                field.removeprefix("action_"),
                f"replay_{field}",
            )
            for row in run.actions
        ],
        dtype=np.float64,
    )
    keep = np.isfinite(times) & np.isfinite(values)
    if np.sum(keep) < 2:
        return np.zeros_like(x)
    times = times[keep]
    values = values[keep]
    order = np.argsort(times)
    return np.interp(x, times[order], values[order], left=values[order][0], right=values[order][-1])


def _discover_runs(run_dir: Path) -> list[RunData]:
    runs: list[RunData] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "state_metrics.csv").exists():
            continue
        runs.append(
            RunData(
                name=child.name,
                path=child,
                rows=_read_rows(child / "state_metrics.csv"),
                actions=_read_jsonl(child / "action_samples.jsonl"),
                metadata=_load_json(child / "metadata.json"),
                matrices=_load_matrices(child / "state_matrices.npz"),
            )
        )
    return runs


def _write_metric_stats(out_dir: Path, runs: list[RunData]) -> Path:
    path = out_dir / "all_metric_stats.csv"
    fields = _metric_fields(runs)
    stat_names = list(_stats([]).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "metric", *stat_names])
        for run in runs:
            for field in fields:
                st = _stats(_array(run.rows, field))
                writer.writerow([run.name, field, *[st[name] for name in stat_names]])
    return path


def _write_event_bouts(out_dir: Path, runs: list[RunData]) -> Path:
    path = out_dir / "event_bouts.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "event_index",
                "start_s",
                "end_s",
                "duration_s",
                "samples",
                "peak_force",
                "mean_force",
                "mean_side_score",
                "mean_confidence",
                "mean_speed_xy_mm_s",
                "mean_tail_yaw_abs",
                "mean_muscle_sum",
            ]
        )
        for run in runs:
            if not run.rows or not run.actions:
                continue
            action_t = np.asarray([_action_time(row, i) for i, row in enumerate(run.actions)], dtype=np.float64)
            force = np.asarray(
                [_action_value(row, "action_force", "force") for row in run.actions],
                dtype=np.float64,
            )
            kick = np.asarray(
                [_action_value(row, "action_kick", "kick") for row in run.actions],
                dtype=np.float64,
            )
            side = np.asarray(
                [_action_value(row, "action_side_score", "side_score") for row in run.actions],
                dtype=np.float64,
            )
            confidence = np.asarray(
                [_action_value(row, "action_confidence", "confidence") for row in run.actions],
                dtype=np.float64,
            )
            threshold = max(0.12, float(np.nanquantile(force, 0.70)) if force.size else 0.12)
            active = (kick >= 0.5) | (force >= threshold)
            x = _time(run.rows)
            speed = _array(run.rows, "speed_xy_mm_s")
            tail = _array(run.rows, "tail_yaw_abs_mean")
            muscle = _array(run.rows, "muscle_sum")
            idx = 0
            event_index = 0
            while idx < active.size:
                if not active[idx]:
                    idx += 1
                    continue
                start = idx
                while idx + 1 < active.size and active[idx + 1]:
                    idx += 1
                end = idx
                t0 = float(action_t[start])
                t1 = float(action_t[end])
                mask = (x >= t0) & (x <= max(t1, t0 + PHYSICS_TIMESTEP_S))
                event_index += 1
                writer.writerow(
                    [
                        run.name,
                        event_index,
                        t0,
                        t1,
                        max(0.0, t1 - t0),
                        end - start + 1,
                        float(np.nanmax(force[start : end + 1])),
                        float(np.nanmean(force[start : end + 1])),
                        float(np.nanmean(side[start : end + 1])),
                        float(np.nanmean(confidence[start : end + 1])),
                        float(np.nanmean(speed[mask])) if np.any(mask) else 0.0,
                        float(np.nanmean(tail[mask])) if np.any(mask) else 0.0,
                        float(np.nanmean(muscle[mask])) if np.any(mask) else 0.0,
                    ]
                )
                idx += 1
    return path


def _anomaly_checks(run: RunData, min_ticks: int) -> list[dict[str, Any]]:
    rows = run.rows
    if not rows:
        return [{"run": run.name, "check": "capture_presence", "level": "fail", "value": 0, "detail": "no rows captured"}]
    tick_span = int(rows[-1].get("tick", 0) - rows[0].get("tick", 0))
    checks: list[dict[str, Any]] = []

    def add(check: str, level: str, value: float | int, detail: str) -> None:
        checks.append({"run": run.name, "check": check, "level": level, "value": value, "detail": detail})

    add(
        "tick_span",
        "pass" if tick_span >= min_ticks else "fail",
        tick_span,
        f"target >= {min_ticks} ticks",
    )
    fields = _metric_fields([run])
    nonfinite = 0
    for field in fields:
        arr = _array(rows, field)
        nonfinite += int(np.sum(~np.isfinite(arr)))
    add("finite_values", "pass" if nonfinite == 0 else "warn", nonfinite, "non-finite scalar values")
    speed = np.abs(_array(rows, "speed_xy_mm_s"))
    speed_p99 = float(np.nanquantile(speed, 0.99)) if speed.size else 0.0
    add(
        "extreme_xy_speed",
        "warn" if speed_p99 > 25.0 else "pass",
        speed_p99,
        "p99 XY speed; high values can indicate unstable dynamics",
    )
    heading_rate = np.abs(_array(rows, "heading_rate_rad_s"))
    heading_p95 = float(np.nanquantile(heading_rate, 0.95)) if heading_rate.size else 0.0
    travel = float(rows[-1].get("xy_displacement_mm", 0.0))
    add(
        "spin_in_place",
        "warn" if heading_p95 > 4.0 and travel < 2.0 else "pass",
        heading_p95,
        f"p95 abs heading rate; travel_xy={travel:.3f} mm",
    )
    pitch = np.abs(_array(rows, "pitch_rad"))
    pitch_p95 = float(np.nanquantile(pitch, 0.95)) if pitch.size else 0.0
    add(
        "vertical_collapse",
        "warn" if pitch_p95 > 1.20 else "pass",
        pitch_p95,
        "p95 abs pitch; sustained near-vertical body would exceed this",
    )
    curvature = _array(rows, "body_max_local_bend_2d_rad")
    curv_p95 = float(np.nanquantile(curvature, 0.95)) if curvature.size else 0.0
    straight = _array(rows, "body_straightness")
    straight_mean = float(np.nanmean(straight)) if straight.size else 0.0
    add(
        "sustained_curl",
        "warn" if curv_p95 > 1.25 or straight_mean < 0.35 else "pass",
        curv_p95,
        f"p95 local bend; mean straightness={straight_mean:.3f}",
    )
    action_force = _interp_action(run, "action_force", _time(rows))
    tail = _array(rows, "tail_yaw_abs_mean")
    if action_force.size and tail.size:
        high_force = action_force > max(0.20, float(np.nanquantile(action_force, 0.75)))
        tail_when_force = float(np.nanmean(tail[high_force])) if np.any(high_force) else 0.0
    else:
        tail_when_force = 0.0
    add(
        "frozen_body_under_drive",
        "warn" if tail_when_force < 0.015 and float(np.nanmean(action_force)) > 0.15 else "pass",
        tail_when_force,
        "mean tail yaw during high-force action samples",
    )
    muscle_abs = _array(rows, "muscle_abs_max")
    muscle_sat = float(np.nanmean(muscle_abs > 0.95)) if muscle_abs.size else 0.0
    add(
        "muscle_saturation",
        "warn" if muscle_sat > 0.05 else "pass",
        muscle_sat,
        "fraction of samples with max normalized muscle activation > 0.95",
    )
    touch = _array(rows, "touch_sum")
    touch_p95 = float(np.nanquantile(touch, 0.95)) if touch.size else 0.0
    add("contact_load", "warn" if touch_p95 > 1e-4 else "pass", touch_p95, "p95 touch/contact sum")
    return checks


def _write_anomalies(out_dir: Path, runs: list[RunData], min_ticks: int) -> tuple[Path, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    for run in runs:
        checks.extend(_anomaly_checks(run, min_ticks))
    path = out_dir / "anomaly_checks.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "check", "level", "value", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    return path, checks


def _plot_master_timeline(out_dir: Path, runs: list[RunData]) -> Path:
    fields = [
        ("action_force", "action force", "action"),
        ("speed_xy_mm_s", "XY speed mm/s", "body"),
        ("vertical_speed_abs_mm_s", "vertical speed abs", "body"),
        ("tail_yaw_abs_mean", "tail yaw abs mean", "tail"),
        ("muscle_sum", "muscle sum", "muscle"),
        ("muscle_lr_bias", "muscle L/R bias", "muscle"),
        ("neuron_s_mean", "mean PAULA S", "neural"),
        ("neuron_fired_count", "fired neurons", "neural"),
        ("neuromod_m0", "neuromod M0", "neural"),
        ("neuromod_m1", "neuromod M1", "neural"),
        ("free_energy", "free energy", "energy"),
        ("body_max_local_bend_2d_rad", "max local bend", "shape"),
    ]
    fig, axes = plt.subplots(len(fields), 1, figsize=(16, 26), sharex=True)
    for run in runs:
        x = _time(run.rows)
        if x.size == 0:
            continue
        dt = float(np.nanmedian(np.diff(x))) if x.size > 2 else PHYSICS_TIMESTEP_S
        smooth = max(3, int(round(5.0 / max(dt, 1e-6))))
        for ax, (field, label, _kind) in zip(axes, fields):
            if field.startswith("action_"):
                y = _interp_action(run, field, x)
            else:
                y = _array(run.rows, field)
            ax.plot(x, _rolling_mean(y, smooth), label=run.name, linewidth=1.1)
            ax.set_ylabel(label)
            ax.grid(alpha=0.18)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("simulation time (s)")
    fig.suptitle("Full-run activity timeline: 5 s rolling means over >=50k ticks", y=0.995)
    fig.tight_layout()
    path = out_dir / "master_timeline.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_metric_fingerprint(out_dir: Path, runs: list[RunData]) -> Path:
    fields = [
        field
        for field in _metric_fields(runs)
        if field not in {"tick", "sim_time_s", "wall_s"}
        and not field.startswith("com_")
        and "rad" not in field
    ]
    dynamic: list[tuple[str, float]] = []
    for field in fields:
        vals = np.concatenate([_array(run.rows, field) for run in runs if run.rows])
        if vals.size:
            dynamic.append((field, float(np.nanstd(vals))))
    dynamic = sorted(dynamic, key=lambda x: x[1], reverse=True)[:36]
    fields = [field for field, _ in dynamic]
    data = []
    labels = []
    for run in runs:
        row = []
        for field in fields:
            st = _stats(_array(run.rows, field))
            row.append(st["mean"])
        data.append(row)
        labels.append(run.name)
    mat = np.asarray(data, dtype=np.float64)
    norm = np.nanmax(np.abs(mat), axis=0)
    matn = mat / np.maximum(norm, 1e-12)
    fig, ax = plt.subplots(figsize=(18, max(4, 1.2 + len(labels))))
    im = ax.imshow(matn, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=90, fontsize=7)
    ax.set_title("Normalized mean metric fingerprint across dynamic scalar metrics")
    fig.colorbar(im, ax=ax, label="mean / max_abs_mean")
    fig.tight_layout()
    path = out_dir / "metric_fingerprint.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_anomaly_dashboard(out_dir: Path, checks: list[dict[str, Any]]) -> Path:
    runs = sorted({str(c["run"]) for c in checks})
    names = sorted({str(c["check"]) for c in checks})
    score = {"pass": 0.0, "warn": 1.0, "fail": 2.0}
    mat = np.zeros((len(runs), len(names)), dtype=np.float64)
    for check in checks:
        i = runs.index(str(check["run"]))
        j = names.index(str(check["check"]))
        mat[i, j] = score.get(str(check["level"]), 1.0)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(mat, vmin=0.0, vmax=2.0, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels(runs)
    ax.set_title("Anomaly dashboard: pass/warn/fail checks")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["pass", "warn", "fail"])
    fig.tight_layout()
    path = out_dir / "anomaly_dashboard.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _lag_correlations(action: np.ndarray, target: np.ndarray, dt: float, max_lag_s: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(dt) or dt <= 1e-9:
        dt = PHYSICS_TIMESTEP_S
    lags = np.arange(-max_lag_s, max_lag_s + dt, dt)
    corrs = np.zeros_like(lags)
    a = np.asarray(action, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.size < 8 or b.size != a.size or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return lags, corrs
    n = a.size
    for i, lag in enumerate(lags):
        shift = int(round(lag / max(dt, 1e-9)))
        if shift > 0:
            aa = a[:-shift]
            bb = b[shift:]
        elif shift < 0:
            aa = a[-shift:]
            bb = b[:shift]
        else:
            aa = a
            bb = b
        if aa.size < 8:
            corrs[i] = 0.0
            continue
        aa = aa - np.nanmean(aa)
        bb = bb - np.nanmean(bb)
        denom = float(np.sqrt(np.nansum(aa * aa) * np.nansum(bb * bb)))
        corrs[i] = float(np.nansum(aa * bb) / denom) if denom > 1e-12 and aa.size <= n else 0.0
    return lags, corrs


def _plot_action_lag_scan(out_dir: Path, runs: list[RunData]) -> Path:
    targets = [
        ("tail_yaw_abs_mean", "tail yaw"),
        ("speed_xy_mm_s", "xy speed"),
        ("muscle_sum", "muscle sum"),
        ("free_energy", "free energy"),
    ]
    fig, axes = plt.subplots(len(runs), len(targets), figsize=(18, max(5, 4.2 * len(runs))), squeeze=False)
    for i, run in enumerate(runs):
        x = _time(run.rows)
        if x.size < 8:
            continue
        diffs = np.diff(x)
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
        dt = float(np.nanmedian(diffs)) if diffs.size else PHYSICS_TIMESTEP_S
        force = _interp_action(run, "action_force", x)
        for j, (target_field, label) in enumerate(targets):
            target = _array(run.rows, target_field)
            lags, corrs = _lag_correlations(force, target, dt, max_lag_s=8.0)
            axes[i, j].plot(lags, corrs)
            axes[i, j].axvline(0, color="0.2", linewidth=0.8, alpha=0.5)
            axes[i, j].set_title(f"{run.name}: action force -> {label}")
            axes[i, j].set_xlabel("lag s; positive = body after action")
            axes[i, j].set_ylabel("corr")
            axes[i, j].grid(alpha=0.18)
    fig.tight_layout()
    path = out_dir / "action_lag_scan.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_phase_space(out_dir: Path, runs: list[RunData]) -> Path:
    pairs = [
        ("action_force", "tail_yaw_abs_mean", "force", "tail yaw"),
        ("action_side_score", "muscle_lr_bias", "side score", "muscle L/R bias"),
        ("speed_xy_mm_s", "body_max_local_bend_2d_rad", "speed", "body bend"),
        ("neuron_s_mean", "free_energy", "mean PAULA S", "free energy"),
    ]
    fig, axes = plt.subplots(len(runs), len(pairs), figsize=(18, max(5, 4.2 * len(runs))), squeeze=False)
    for i, run in enumerate(runs):
        x = _time(run.rows)
        t = x if x.size else np.arange(len(run.rows), dtype=np.float64)
        for j, (xf, yf, xl, yl) in enumerate(pairs):
            xx = _interp_action(run, xf, x) if xf.startswith("action_") else _array(run.rows, xf)
            yy = _interp_action(run, yf, x) if yf.startswith("action_") else _array(run.rows, yf)
            sc = axes[i, j].scatter(xx, yy, c=t, s=6, alpha=0.45, cmap="viridis")
            axes[i, j].set_xlabel(xl)
            axes[i, j].set_ylabel(yl)
            axes[i, j].set_title(run.name)
            axes[i, j].grid(alpha=0.14)
        fig.colorbar(sc, ax=axes[i, -1], label="sim time s")
    fig.tight_layout()
    path = out_dir / "phase_space_panels.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_matrix_energy(out_dir: Path, runs: list[RunData]) -> Path:
    keys = [
        "tail_yaw_rad",
        "tail_pitch_rad",
        "joint_angles_rad",
        "joint_velocities_rad_s",
        "touch_forces",
        "muscle_activations",
        "neuron_s",
        "neuron_r",
        "neuron_b",
        "neuron_tref",
        "neuron_fired",
        "neuron_m0",
        "neuron_m1",
    ]
    fig, axes = plt.subplots(len(runs), 1, figsize=(16, max(5, 4.5 * len(runs))), squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        means = []
        stds = []
        labels = []
        for key in keys:
            mat = np.asarray(run.matrices.get(key, np.zeros((0, 0))), dtype=np.float64)
            if mat.size == 0:
                continue
            means.append(float(np.nanmean(np.abs(mat))))
            stds.append(float(np.nanstd(mat)))
            labels.append(key)
        x = np.arange(len(labels))
        ax.bar(x - 0.18, means, width=0.36, label="mean abs")
        ax.bar(x + 0.18, stds, width=0.36, label="std")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{run.name}: matrix channel energy summary")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    path = out_dir / "matrix_energy_summary.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_data_dictionary(out_dir: Path, runs: list[RunData]) -> Path:
    categories = {
        "body geometry": ("body_", "com_", "xy_", "xyz_"),
        "orientation": ("heading", "pitch", "vertical_"),
        "tail and joints": ("tail_", "joint_"),
        "muscle": ("muscle_"),
        "touch/contact": ("touch_"),
        "PAULA/neural": ("neuron_", "neuromod_"),
        "free energy": ("free_energy",),
        "time": ("tick", "sim_time_s", "wall_s"),
    }
    fields = _metric_fields(runs)
    lines = [
        "# Activity Metric Dictionary",
        "",
        "Every scalar field recorded in `state_metrics.csv` is listed below. Matrix-valued streams are in `state_matrices.npz`.",
        "",
    ]
    assigned: set[str] = set()
    for category, prefixes in categories.items():
        subset = [field for field in fields if field.startswith(prefixes)]
        if not subset:
            continue
        assigned.update(subset)
        lines.extend([f"## {category.title()}", ""])
        for field in subset:
            lines.append(f"- `{field}`")
        lines.append("")
    rest = [field for field in fields if field not in assigned]
    if rest:
        lines.extend(["## Other", ""])
        for field in rest:
            lines.append(f"- `{field}`")
        lines.append("")
    lines.extend(
        [
            "## Matrix Streams",
            "",
            "- `tail_yaw_rad`, `tail_pitch_rad`: per-tail-segment realized angles over time.",
            "- `joint_angles_rad`, `joint_velocities_rad_s`: MuJoCo joint state vectors.",
            "- `touch_forces`: touch/contact force vector.",
            "- `muscle_activations`: normalized actuator control values.",
            "- `neuron_s`, `neuron_r`, `neuron_b`, `neuron_tref`, `neuron_fired`, `neuron_m0`, `neuron_m1`: PAULA neuron state vectors in lab PAULA order.",
            "- `neuromod`: aggregate M0/M1 proxies.",
            "- `free_energy`: active-inference prediction-error proxy.",
            "",
        ]
    )
    path = out_dir / "DATA_DICTIONARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _artifact_link(path: Path) -> str:
    return f"`{path.resolve()}`"


def _write_report(
    run_dir: Path,
    out_dir: Path,
    summary: dict[str, Any],
    runs: list[RunData],
    artifacts: dict[str, Path],
    checks: list[dict[str, Any]],
    min_ticks: int,
) -> Path:
    lines: list[str] = [
        "# Full Zebrafish Activity Atlas",
        "",
        f"- source run: {_artifact_link(run_dir)}",
        f"- synthesis output: {_artifact_link(out_dir)}",
        f"- minimum tick target: `{min_ticks}`",
        f"- source summary: {_artifact_link(run_dir / 'summary.json')}",
        "",
        "## Primary Scientific Sources Rechecked",
        "",
    ]
    for label, url in SOURCE_LINKS.items():
        lines.append(f"- [{label}]({url})")
    lines.extend(
        [
            "",
            "## Runtime Pipeline Under Study",
            "",
            "```mermaid",
            "flowchart LR",
            "  A[ZAPBench calcium/ephys artifact] --> B[ZebrafishActionLatent: zapbench_calcium_ephys]",
            "  C[Natural/sample video] --> D[OpenCV flow + camera stabilization]",
            "  D --> E[simZFish-inspired retina/OMR adapter]",
            "  E --> F[ZebrafishActionLatent: video_simzfish_omr]",
            "  B --> G[environment action extras]",
            "  F --> G",
            "  G --> H[ZebrafishSensorEncoder ACTION channels]",
            "  H --> I[PAULA-backed zebrafish nervous wrapper]",
            "  I --> J[tail CPG + direct segment-target blend]",
            "  J --> K[MuJoCo muscles/body]",
            "  K --> L[REST/WebSocket lab state stream]",
            "```",
            "",
            "The calcium branch is the directly ZAPBench-grounded branch: decoded calcium/ephys labels become fictive tail-action commands. The video branch is simZFish/Z-Robot-inspired and uses backend optical flow plus retinal OMR logic; the ZAPBench covariate mapping is retained as diagnostics, not the primary action driver.",
            "",
            "## Capture Coverage",
            "",
            "| run | frames | tick span | sim seconds | action samples | raw state CSV | matrix NPZ |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for run in runs:
        tick_span = int(run.rows[-1].get("tick", 0) - run.rows[0].get("tick", 0)) if run.rows else 0
        sim_seconds = float(run.rows[-1].get("sim_time_s", 0.0)) if run.rows else 0.0
        lines.append(
            f"| `{run.name}` | {len(run.rows)} | {tick_span} | {sim_seconds:.3f} | {len(run.actions)} | "
            f"{_artifact_link(run.path / 'state_metrics.csv')} | {_artifact_link(run.path / 'state_matrices.npz')} |"
        )
    lines.extend(["", "## Generated Visuals", ""])
    for label, path in artifacts.items():
        if path.suffix.lower() in {".png", ".csv", ".md"}:
            lines.append(f"- `{label}`: {_artifact_link(path)}")
    lines.extend(["", "## Anomaly Findings", ""])
    lines.append("| run | check | level | value | detail |")
    lines.append("|---|---|---|---:|---|")
    for check in checks:
        lines.append(
            f"| `{check['run']}` | `{check['check']}` | `{check['level']}` | "
            f"{float(check['value']):.6g} | {check['detail']} |"
        )
    lines.extend(
        [
            "",
            "## What Is Happening In The Simulation",
            "",
            "1. The lab backend advances MuJoCo at `0.005 s` per physics tick. The WebSocket captures downsampled state frames while the actual simulation can advance several physics ticks between browser-visible frames.",
            "2. Calcium replay advances as a replay clock against ZAPBench-derived rows. Each calcium/ephys action is pulsed into the environment, converted into shared tail targets, and then blended into the PAULA-backed nervous wrapper.",
            "3. Video replay advances frame-by-frame through backend decoding. Each selected video frame is decoded with OpenCV, camera motion is estimated and subtracted, simZFish-style retinal counters are computed, and an OMR action latent is emitted.",
            "4. PAULA state variables are live and measured, but in these two replay regimes PAULA is a local dynamical substrate/diagnostic layer rather than a fully paper-validated zebrafish connectome-to-muscle controller.",
            "5. The recorded body stream includes COM motion, body-segment geometry, realized tail and joint angles, touch/contact, actuator controls, neural state vectors, neuromodulator proxies, and free energy. The synthesis treats every scalar metric and every saved matrix stream as evidence.",
            "",
            "## Interpretation Boundaries",
            "",
            "- ZAPBench is an activity-prediction benchmark for cellular-resolution neural activity, not a natural-video locomotion dataset. Its public release includes traces, volumes, segmentation, stimulus covariates, and raw stimulus/ephys resources, but not the exact projected video frames.",
            "- The ZAPBench ephys-derived calcium replay is suitable as a data-grounded fictive motor replay proxy. It is not a validated free-swimming tail-kinematics or muscle-force reproduction.",
            "- The simZFish/Z-Robot branch provides a mechanistic visual-to-OMR controller and public calcium/behavior resources, but it is not automatically aligned to ZAPBench whole-brain calcium traces or arbitrary underwater POV videos.",
            "- Therefore the correct scientific claim for this run is: the lab currently demonstrates two paper-grounded action-input branches sharing one embodied MuJoCo tail-action interface, with measured long-run dynamics and explicit uncertainty boundaries.",
            "",
            "## Existing Run Summary Excerpt",
            "",
            f"```json\n{json.dumps({k: summary.get(k) for k in ['generated_at', 'health_start', 'health_end', 'long_timescale_checks']}, indent=2, default=str)}\n```",
            "",
        ]
    )
    path = out_dir / "FULL_ACTIVITY_ATLAS_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def synthesize(run_dir: Path, min_ticks: int) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    out_dir = run_dir / "synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_json(run_dir / "summary.json")
    runs = _discover_runs(run_dir)
    if not runs:
        raise RuntimeError(f"no per-run state_metrics.csv files found under {run_dir}")
    artifacts: dict[str, Path] = {}
    artifacts["all_metric_stats"] = _write_metric_stats(out_dir, runs)
    artifacts["event_bouts"] = _write_event_bouts(out_dir, runs)
    artifacts["data_dictionary"] = _write_data_dictionary(out_dir, runs)
    anomaly_path, checks = _write_anomalies(out_dir, runs, min_ticks)
    artifacts["anomaly_checks"] = anomaly_path
    artifacts["master_timeline"] = _plot_master_timeline(out_dir, runs)
    artifacts["metric_fingerprint"] = _plot_metric_fingerprint(out_dir, runs)
    artifacts["anomaly_dashboard"] = _plot_anomaly_dashboard(out_dir, checks)
    artifacts["action_lag_scan"] = _plot_action_lag_scan(out_dir, runs)
    artifacts["phase_space_panels"] = _plot_phase_space(out_dir, runs)
    artifacts["matrix_energy_summary"] = _plot_matrix_energy(out_dir, runs)
    report = _write_report(run_dir, out_dir, summary, runs, artifacts, checks, min_ticks)
    artifacts["report"] = report
    manifest = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "min_ticks": int(min_ticks),
        "runs": [
            {
                "name": run.name,
                "frames": len(run.rows),
                "actions": len(run.actions),
                "tick_span": int(run.rows[-1].get("tick", 0) - run.rows[0].get("tick", 0)) if run.rows else 0,
                "sim_seconds": float(run.rows[-1].get("sim_time_s", 0.0)) if run.rows else 0.0,
            }
            for run in runs
        ],
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "checks": checks,
    }
    (out_dir / "synthesis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-ticks", type=int, default=50000)
    args = parser.parse_args()
    manifest = synthesize(args.run_dir, args.min_ticks)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
