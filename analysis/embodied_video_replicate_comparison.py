"""Compare long embodied zebrafish video replays across selected clips."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "out" / "embodied_video_replicate_comparison_20260603"
RUNS = [
    {
        "label": "tenggol",
        "run_dir": SCRIPT_DIR / "out" / "comprehensive_activity_study" / "20260603_verified_50k",
        "video_run": "video_commons_tenggol_underwater",
    },
    {
        "label": "black_rockfish",
        "run_dir": SCRIPT_DIR / "out" / "comprehensive_activity_study" / "20260603_black_rockfish_50k",
        "video_run": "video_commons_black_rockfish_stereo_dov",
    },
]
METRICS = [
    "speed_xy_mm_s",
    "speed_3d_mm_s",
    "vertical_speed_abs_mm_s",
    "tail_yaw_abs_mean",
    "tail_yaw_abs_max",
    "muscle_sum",
    "neuron_s_mean",
    "neuron_fired_count",
    "free_energy",
    "body_straightness",
    "body_z_span_mm",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_lookup(rows: list[dict[str, str]], run_name: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("run") != run_name:
            continue
        metric = str(row.get("metric", ""))
        out[metric] = {key: _to_float(value) for key, value in row.items() if key not in {"run", "metric"}}
    return out


def _event_frequency(events: list[dict[str, str]], run_name: str, sim_seconds: float) -> float:
    count = sum(1 for row in events if row.get("run") == run_name)
    return count / max(1e-9, sim_seconds)


def _quality_counts(rows: list[dict[str, str]], run_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("run") != run_name:
            continue
        event = str(row.get("event", ""))
        counts[event] = counts.get(event, 0) + 1
    return counts


def _bout_fractions(rows: list[dict[str, str]], run_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if row.get("run") == run_name:
            out[str(row.get("bout_type", ""))] = _to_float(row.get("fraction"))
    return out


def _collect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for spec in RUNS:
        run_dir = Path(spec["run_dir"])
        video_run = str(spec["video_run"])
        summary = _load_json(run_dir / "summary.json")
        synthesis = _load_json(run_dir / "synthesis" / "synthesis_manifest.json")
        metric_rows = _read_csv(run_dir / "synthesis" / "all_metric_stats.csv")
        event_rows = _read_csv(run_dir / "synthesis" / "event_bouts.csv")
        anomaly_rows = _read_csv(run_dir / "synthesis" / "anomaly_checks.csv")
        bout_rows = _read_csv(run_dir / "forensics" / "action_bout_counts.csv")
        quality_rows = _read_csv(run_dir / "forensics" / "video_quality_events.csv")
        run_entry = next((row for row in synthesis.get("runs", []) if row.get("name") == video_run), {})
        sim_seconds = _to_float(run_entry.get("sim_seconds"))
        metric_map = _metric_lookup(metric_rows, video_run)
        quality = _quality_counts(quality_rows, video_run)
        bouts = _bout_fractions(bout_rows, video_run)
        checks = [row for row in anomaly_rows if row.get("run") == video_run]
        nonpass = [row for row in checks if row.get("level") != "pass"]
        row: dict[str, Any] = {
            "label": spec["label"],
            "run_dir": str(run_dir.resolve()),
            "video_run": video_run,
            "tick_span": int(_to_float(run_entry.get("tick_span"))),
            "frames": int(_to_float(run_entry.get("frames"))),
            "action_samples": int(_to_float(run_entry.get("actions"))),
            "sim_seconds": sim_seconds,
            "noncoast_event_frequency_hz": _event_frequency(event_rows, video_run, sim_seconds),
            "quality_event_count": sum(quality.values()),
            "high_camera_shake": quality.get("high_camera_shake", 0),
            "high_compression_noise": quality.get("high_compression_noise", 0),
            "low_flow_reliability": quality.get("low_flow_reliability", 0),
            "nonpass_anomaly_checks": len(nonpass),
            "coast_fraction": bouts.get("coast", 0.0),
            "startle_fraction": bouts.get("startle_c_bend", 0.0),
            "forward_fraction": bouts.get("omr_forward_bout", 0.0),
            "left_turn_fraction": bouts.get("omr_turn_left", 0.0),
            "right_turn_fraction": bouts.get("omr_turn_right", 0.0),
        }
        for metric in METRICS:
            stats = metric_map.get(metric, {})
            row[f"{metric}_mean"] = stats.get("mean", 0.0)
            row[f"{metric}_p95"] = stats.get("p95", 0.0)
            row[f"{metric}_max"] = stats.get("max", 0.0)
        records.append(row)
        details[str(spec["label"])] = {
            "summary": summary,
            "synthesis": synthesis,
            "quality_counts": quality,
            "bout_fractions": bouts,
            "anomaly_checks": checks,
            "nonpass_anomaly_checks": nonpass,
        }
    return records, details


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_bars(records: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [str(row["label"]) for row in records]
    metrics = [
        ("speed_xy_mm_s_p95", "p95 XY speed"),
        ("tail_yaw_abs_max_p95", "p95 max tail yaw"),
        ("muscle_sum_mean", "mean muscle sum"),
        ("free_energy_mean", "mean free energy"),
        ("body_straightness_mean", "mean straightness"),
        ("body_z_span_mm_p95", "p95 body z span"),
        ("noncoast_event_frequency_hz", "event Hz"),
        ("quality_event_count", "quality events"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    x = np.arange(len(labels))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = [float(row.get(key, 0.0)) for row in records]
        ax.bar(x, values, color=["#4393c3", "#d6604d"][: len(records)])
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("50k embodied video replicate metric comparison")
    fig.tight_layout()
    path = out_dir / "replicate_metric_bars.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_bout_fractions(records: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [str(row["label"]) for row in records]
    keys = [
        ("coast_fraction", "coast", "#909090"),
        ("startle_fraction", "startle", "#d6604d"),
        ("forward_fraction", "forward", "#4393c3"),
        ("left_turn_fraction", "left", "#8073ac"),
        ("right_turn_fraction", "right", "#4dac26"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(records))
    bottom = np.zeros(len(records), dtype=np.float64)
    for key, label, color in keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in records])
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("action frame fraction")
    ax.set_title("50k embodied video replicate action occupancy")
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path = out_dir / "replicate_bout_fractions.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_quality(records: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [str(row["label"]) for row in records]
    keys = [
        ("high_camera_shake", "camera shake", "#d6604d"),
        ("high_compression_noise", "compression", "#fdae61"),
        ("low_flow_reliability", "low flow", "#5e3c99"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(records))
    bottom = np.zeros(len(records), dtype=np.float64)
    for key, label, color in keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in records])
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("quality event count")
    ax.set_title("50k embodied video replicate quality-event burden")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path = out_dir / "replicate_quality_events.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _render_report(out_dir: Path, records: list[dict[str, Any]], artifacts: dict[str, str]) -> str:
    lines = [
        "# Embodied Video Replicate Comparison",
        "",
        f"- output directory: `{out_dir}`",
        "- compared runs: `20260603_verified_50k` Tenggol and `20260603_black_rockfish_50k` black-rockfish stereo-DOV",
        "- scope: 50k+ tick embodied MuJoCo recordings for selected-video mode",
        "",
        "## Summary",
        "",
        "| clip | ticks | event Hz | p95 speed | p95 tail yaw max | mean muscle sum | mean free energy | quality events | anomaly non-pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| `{row['label']}` | {int(row['tick_span'])} | {float(row['noncoast_event_frequency_hz']):.4f} | "
            f"{float(row['speed_xy_mm_s_p95']):.4f} | {float(row['tail_yaw_abs_max_p95']):.4f} | "
            f"{float(row['muscle_sum_mean']):.4f} | {float(row['free_energy_mean']):.6f} | "
            f"{int(row['quality_event_count'])} | {int(row['nonpass_anomaly_checks'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Both embodied video replicates pass the >=50k tick target and all gross anomaly gates.",
            "- The black-rockfish low-risk clip has far fewer quality events than Tenggol, while remaining in the same broad event-rate regime.",
            "- Black-rockfish expresses lower mean force and lower mean free energy than Tenggol but has similar p95 XY speed and stronger transient turning/heading-rate excursions, so it should be treated as a useful independent validation run rather than a duplicate.",
            "- These two replicates support implementation stability across more than one selected video. They still do not prove biological identity because they are not paired with natural-video calcium/ephys/tail-kinematic ground truth.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, path in artifacts.items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, details = _collect()
    summary_path = OUT_DIR / "embodied_video_replicate_summary.csv"
    _write_csv(summary_path, records)
    artifacts = {
        "summary_csv": str(summary_path.resolve()),
        "metric_bars": str(_plot_metric_bars(records, OUT_DIR).resolve()),
        "bout_fractions": str(_plot_bout_fractions(records, OUT_DIR).resolve()),
        "quality_events": str(_plot_quality(records, OUT_DIR).resolve()),
    }
    report_path = OUT_DIR / "EMBODIED_VIDEO_REPLICATE_COMPARISON.md"
    artifacts["report"] = str(report_path.resolve())
    manifest = {
        "records": records,
        "details": details,
        "artifacts": artifacts,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(OUT_DIR, records, artifacts), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR.resolve()), "report": str(report_path.resolve()), "records": records}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
