"""Bootstrap robustness audit for simZFish/Z-Robot behavior alignment.

The first behavior-target audit reports point estimates.  This script adds
uncertainty estimates around those estimates using contiguous block
bootstrapping, which is more appropriate for temporally autocorrelated video
and action replay rows than independent frame resampling.
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

from simzfish_behavior_target_alignment_audit import (
    PROFILE_FIELDS,
    _profile_from_frame_rows,
    _read_csv,
    _safe_float,
    _score_profile,
    _target_for_dandi_stimulus,
    _write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PROFILES = (
    ROOT
    / "analysis"
    / "out"
    / "simzfish_behavior_target_alignment_audit_20260603"
    / "published_behavior_condition_profiles.csv"
)
DEFAULT_VIDEO_FRAMES = (
    ROOT / "analysis" / "out" / "backend_video_robustness" / "20260603_all_selected_60s" / "backend_video_frame_metrics.csv"
)
DEFAULT_DANDI_FRAMES = (
    ROOT / "analysis" / "out" / "dandi_omr_motor_response_audit_20260603" / "stimulus_motor_response_frames.csv"
)
DEFAULT_LONG_RUN_ROOT = ROOT / "analysis" / "out" / "comprehensive_activity_study"
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "simzfish_behavior_uncertainty_audit_20260603"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_target_profiles(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        rows.append({key: _safe_float(value, value) for key, value in row.items()})
        rows[-1]["condition"] = row.get("condition", "")
        rows[-1]["groups"] = row.get("groups", "")
    return rows, {str(row["condition"]): row for row in rows}


def _duration_from_rows(rows: list[dict[str, Any]]) -> float:
    times = [_safe_float(row.get("video_time_s", row.get("time_s", 0.0)), float("nan")) for row in rows]
    times = [t for t in times if math.isfinite(t)]
    if len(times) < 2:
        return 0.0
    unique = sorted(set(times))
    dt = float(np.median(np.diff(unique))) if len(unique) > 1 else 0.0
    return max(0.0, max(times) - min(times) + dt)


def _read_action_jsonl(path: Path) -> tuple[list[dict[str, Any]], float, str]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows, 0.0, "missing"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            state = obj.get("state", obj)
            rows.append(
                {
                    "action_bout_type": state.get("action_bout_type") or state.get("bout_type") or "coast",
                    "action_kick": state.get("action_kick", state.get("kick", 0.0)),
                    "action_force": state.get("action_force", state.get("force", 0.0)),
                    "action_side_score": state.get("action_side_score", state.get("side_score", 0.0)),
                    "time_s": state.get(
                        "sim_time_s",
                        state.get("_sim_time_s", state.get("source_time_s", state.get("time_s", len(rows)))),
                    ),
                }
            )
    duration_s = 0.0
    summary_path = path.parents[1] / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run = summary.get("runs", {}).get(path.parent.name, {})
            if isinstance(run, dict):
                run = run.get("summary", run)
            duration_s = _safe_float(run.get("sim_seconds"), 0.0) if isinstance(run, dict) else 0.0
        except Exception:  # noqa: BLE001
            duration_s = 0.0
    if duration_s <= 0.0:
        duration_s = _duration_from_rows(rows)
    note = "driver action samples; video is backend-frame sampled, calcium is replay-state REST sampled"
    return rows, duration_s, note


def _group_selected_video(path: Path) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(path):
        by_clip[row.get("clip", "")].append(row)
    return [
        {
            "label": clip,
            "source_kind": "selected_video_backend",
            "rows": members,
            "duration_s": _duration_from_rows(members),
            "sample_note": "dense 10 Hz backend-frame rows",
            "target_condition": "",
            "target_mode": "nearest_published_condition",
        }
        for clip, members in sorted(by_clip.items())
    ]


def _group_dandi(path: Path) -> list[dict[str, Any]]:
    by_stim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(path):
        by_stim[row.get("stimulus", "")].append(row)
    return [
        {
            "label": stim,
            "source_kind": "dandi_synthetic_motor",
            "rows": members,
            "duration_s": _duration_from_rows(members),
            "sample_note": "dense synthetic projector-like 10 Hz rows",
            "target_condition": _target_for_dandi_stimulus(stim),
            "target_mode": "mapped_expected_condition",
        }
        for stim, members in sorted(by_stim.items())
    ]


def _group_long_runs(root: Path) -> list[dict[str, Any]]:
    candidates = [
        root / "20260603_verified_50k" / "calcium_all" / "action_samples.jsonl",
        root / "20260603_verified_50k" / "video_commons_tenggol_underwater" / "action_samples.jsonl",
        root / "20260603_black_rockfish_50k" / "video_commons_black_rockfish_stereo_dov" / "action_samples.jsonl",
    ]
    out: list[dict[str, Any]] = []
    for path in candidates:
        rows, duration_s, note = _read_action_jsonl(path)
        if not rows:
            continue
        out.append(
            {
                "label": f"{path.parents[1].name}/{path.parent.name}",
                "source_kind": "long_run_action_samples",
                "rows": rows,
                "duration_s": duration_s,
                "sample_note": note,
                "target_condition": "",
                "target_mode": "nearest_published_condition",
            }
        )
    return out


def _block_len(source_kind: str, n: int) -> int:
    if n <= 1:
        return 1
    if source_kind == "long_run_action_samples":
        return min(n, max(12, int(round(n / 26))))  # about 10 s for 10 Hz video, smaller for calcium polls
    return min(n, 30)  # about 3 s at 10 Hz


def _block_resample(rows: list[dict[str, Any]], block_len: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    n = len(rows)
    if n == 0:
        return []
    out: list[dict[str, Any]] = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        for offset in range(block_len):
            out.append(rows[(start + offset) % n])
            if len(out) >= n:
                break
    return out


def _nearest_target(profile: dict[str, Any], targets: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    scored = [(str(target["condition"]), _score_profile(profile, target)) for target in targets]
    return min(scored, key=lambda item: item[1]["alignment_distance"])


def _q(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _summarize_distribution(prefix: str, values: list[float]) -> dict[str, Any]:
    return {
        f"{prefix}_mean": float(np.nanmean(values)) if values else float("nan"),
        f"{prefix}_p025": _q(values, 0.025),
        f"{prefix}_p50": _q(values, 0.50),
        f"{prefix}_p975": _q(values, 0.975),
    }


def _source_stability_class(row: dict[str, Any]) -> str:
    mean_inside = _safe_float(row.get("mean_metric_inside_probability"))
    dist_p50 = _safe_float(row.get("alignment_distance_p50"))
    dist_p975 = _safe_float(row.get("alignment_distance_p975"))
    startle_p975 = _safe_float(row.get("startle_fraction_p975"))
    target_stability = _safe_float(row.get("nearest_target_stability_fraction"))
    if mean_inside >= 0.75 and dist_p975 <= 2.5 and startle_p975 <= 0.25 and target_stability >= 0.75:
        return "supported_with_bootstrap"
    if mean_inside >= 0.25 or dist_p50 <= 3.0:
        return "suggestive_only"
    return "not_aligned"


def _bootstrap_profile(
    item: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
    target_by_condition: dict[str, dict[str, Any]],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(item["rows"])
    duration_s = _safe_float(item.get("duration_s"), 0.0)
    if duration_s <= 0.0:
        duration_s = _duration_from_rows(rows)
    point_profile = _profile_from_frame_rows(
        rows,
        label=str(item["label"]),
        source_kind=str(item["source_kind"]),
        duration_s=duration_s,
    )
    fixed_target_condition = str(item.get("target_condition") or "")
    if fixed_target_condition and fixed_target_condition in target_by_condition:
        target_condition = fixed_target_condition
        target = target_by_condition[target_condition]
    else:
        target_condition, point_score = _nearest_target(point_profile, targets)
        target = target_by_condition[target_condition]
    point_score = _score_profile(point_profile, target)
    block_len = _block_len(str(item["source_kind"]), len(rows))
    distributions: dict[str, list[float]] = {
        "bout_frequency_hz": [],
        "left_fraction": [],
        "forward_fraction": [],
        "right_fraction": [],
        "startle_fraction": [],
        "alignment_distance": [],
        "within_p05_p95_fraction": [],
    }
    inside: dict[str, list[float]] = {field: [] for field in PROFILE_FIELDS}
    nearest_counts: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    for i in range(iterations):
        boot_rows = _block_resample(rows, block_len, rng)
        profile = _profile_from_frame_rows(
            boot_rows,
            label=str(item["label"]),
            source_kind=str(item["source_kind"]),
            duration_s=duration_s,
        )
        score = _score_profile(profile, target)
        nearest_condition, nearest_score = _nearest_target(profile, targets)
        nearest_counts[nearest_condition] += 1
        for field in ["bout_frequency_hz", "left_fraction", "forward_fraction", "right_fraction", "startle_fraction"]:
            distributions[field].append(_safe_float(profile.get(field)))
        distributions["alignment_distance"].append(_safe_float(score.get("alignment_distance")))
        distributions["within_p05_p95_fraction"].append(_safe_float(score.get("within_p05_p95_fraction")))
        for field in PROFILE_FIELDS:
            inside[field].append(_safe_float(score.get(f"{field}_inside_p05_p95")))
        sample_rows.append(
            {
                "label": item["label"],
                "source_kind": item["source_kind"],
                "iteration": i,
                "target_condition": target_condition,
                "nearest_target_condition": nearest_condition,
                "bout_frequency_hz": profile["bout_frequency_hz"],
                "left_fraction": profile["left_fraction"],
                "forward_fraction": profile["forward_fraction"],
                "right_fraction": profile["right_fraction"],
                "startle_fraction": profile["startle_fraction"],
                "alignment_distance": score["alignment_distance"],
                "nearest_alignment_distance": nearest_score["alignment_distance"],
                "within_p05_p95_fraction": score["within_p05_p95_fraction"],
            }
        )
    nearest_top = [
        {"condition": condition, "count": count, "fraction": count / max(1, iterations)}
        for condition, count in nearest_counts.most_common(5)
    ]
    summary: dict[str, Any] = {
        "label": item["label"],
        "source_kind": item["source_kind"],
        "target_mode": item.get("target_mode", ""),
        "target_condition": target_condition,
        "target_published_rows": target.get("published_rows", 0),
        "sample_count": len(rows),
        "duration_s": duration_s,
        "block_len_rows": block_len,
        "iterations": iterations,
        "sample_note": item.get("sample_note", ""),
        "point_bout_frequency_hz": point_profile["bout_frequency_hz"],
        "point_left_fraction": point_profile["left_fraction"],
        "point_forward_fraction": point_profile["forward_fraction"],
        "point_right_fraction": point_profile["right_fraction"],
        "point_startle_fraction": point_profile["startle_fraction"],
        "point_alignment_distance": point_score["alignment_distance"],
        "point_within_p05_p95_fraction": point_score["within_p05_p95_fraction"],
        "nearest_target_stability_fraction": nearest_counts[target_condition] / max(1, iterations),
        "nearest_target_top5": json.dumps(nearest_top, sort_keys=True),
    }
    for field, values in distributions.items():
        summary.update(_summarize_distribution(field, values))
    for field, values in inside.items():
        summary[f"{field}_inside_probability"] = float(np.mean(values)) if values else float("nan")
    summary["mean_metric_inside_probability"] = float(
        np.mean([summary[f"{field}_inside_probability"] for field in PROFILE_FIELDS])
    )
    summary["stability_class"] = _source_stability_class(summary)
    return summary, sample_rows


def _plot_distance_forest(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "bootstrap_alignment_distance_forest.png"
    ordered = sorted(rows, key=lambda row: (_safe_float(row.get("alignment_distance_p50")), row.get("source_kind", "")))
    labels = [f"{row['source_kind']}\n{row['label']}" for row in ordered]
    med = np.asarray([_safe_float(row.get("alignment_distance_p50")) for row in ordered], dtype=np.float64)
    lo = np.asarray([_safe_float(row.get("alignment_distance_p025")) for row in ordered], dtype=np.float64)
    hi = np.asarray([_safe_float(row.get("alignment_distance_p975")) for row in ordered], dtype=np.float64)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(13, max(8, len(ordered) * 0.34)))
    ax.errorbar(
        med,
        y,
        xerr=[np.maximum(0.0, med - lo), np.maximum(0.0, hi - med)],
        fmt="o",
        color="#2166ac",
        ecolor="#999999",
        elinewidth=1.0,
        capsize=2,
    )
    ax.set_xscale("symlog", linthresh=2.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(2.5, color="#1b7837", linestyle="--", linewidth=1, label="strict heuristic distance 2.5")
    ax.axvline(5.0, color="#b2182b", linestyle="--", linewidth=1, label="warning distance 5.0")
    ax.set_xlabel("alignment distance bootstrap median and 95% CI (symlog)")
    ax.set_title("Behavior-target alignment uncertainty")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_inside_heatmap(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "bootstrap_inside_probability_heatmap.png"
    ordered = sorted(rows, key=lambda row: (row.get("source_kind", ""), row.get("label", "")))
    fields = [
        "bout_frequency_hz_inside_probability",
        "left_fraction_inside_probability",
        "forward_fraction_inside_probability",
        "right_fraction_inside_probability",
        "mean_metric_inside_probability",
        "nearest_target_stability_fraction",
    ]
    mat = np.asarray([[_safe_float(row.get(field)) for field in fields] for row in ordered], dtype=np.float64)
    labels = [f"{row['source_kind']} | {row['label']}" for row in ordered]
    fig, ax = plt.subplots(figsize=(12, max(8, len(ordered) * 0.28)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Bootstrap probability of falling inside published behavior targets")
    fig.colorbar(im, ax=ax, label="probability")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_video_target_stability(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "selected_video_target_stability.png"
    video = sorted(
        [row for row in rows if row.get("source_kind") == "selected_video_backend"],
        key=lambda row: _safe_float(row.get("nearest_target_stability_fraction")),
        reverse=True,
    )
    labels = [str(row["label"]) for row in video]
    stability = np.asarray([_safe_float(row.get("nearest_target_stability_fraction")) for row in video], dtype=np.float64)
    dist = np.asarray([_safe_float(row.get("alignment_distance_p50")) for row in video], dtype=np.float64)
    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(video))
    ax1.bar(x - 0.18, stability, width=0.36, color="#1b7837", label="nearest-target stability")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("bootstrap target stability")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, dist, width=0.36, color="#b2182b", alpha=0.75, label="distance p50")
    ax2.set_ylabel("alignment distance p50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Selected videos: target assignment stability versus distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_long_run_event_frequency(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "long_run_event_frequency_ci.png"
    long_rows = [row for row in rows if row.get("source_kind") == "long_run_action_samples"]
    labels = [str(row["label"]) for row in long_rows]
    med = np.asarray([_safe_float(row.get("bout_frequency_hz_p50")) for row in long_rows], dtype=np.float64)
    lo = np.asarray([_safe_float(row.get("bout_frequency_hz_p025")) for row in long_rows], dtype=np.float64)
    hi = np.asarray([_safe_float(row.get("bout_frequency_hz_p975")) for row in long_rows], dtype=np.float64)
    x = np.arange(len(long_rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(
        x,
        med,
        yerr=[np.maximum(0.0, med - lo), np.maximum(0.0, hi - med)],
        fmt="o",
        color="#2166ac",
        ecolor="#999999",
        capsize=3,
    )
    for i, row in enumerate(long_rows):
        ax.hlines(
            _safe_float(row.get("point_bout_frequency_hz")),
            i - 0.25,
            i + 0.25,
            color="#b2182b",
            linewidth=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("bout/event frequency Hz")
    ax.set_title(">=50k long-run action event frequency uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(path: Path, *, summary_rows: list[dict[str, Any]], plots: list[Path], manifest: dict[str, Any]) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_source[str(row["source_kind"])].append(row)
    classes = Counter(str(row.get("stability_class")) for row in summary_rows)
    best_video = min(
        by_source.get("selected_video_backend", []),
        key=lambda row: _safe_float(row.get("alignment_distance_p50"), 1e99),
        default={},
    )
    worst_dandi = max(
        by_source.get("dandi_synthetic_motor", []),
        key=lambda row: _safe_float(row.get("alignment_distance_p50"), -1e99),
        default={},
    )
    lines = [
        "# simZFish/Z-Robot Behavior Uncertainty Audit",
        "",
        "## Scope",
        "",
        "This audit adds contiguous block-bootstrap uncertainty estimates to the behavior-target alignment layer. It tests whether point-estimate conclusions are robust under temporal resampling of backend-frame, synthetic-stimulus, and long-run action rows.",
        "",
        "## Main Findings",
        "",
        f"- Profiles audited: `{len(summary_rows)}`.",
        f"- Bootstrap iterations per profile: `{manifest['iterations']}`.",
        f"- Stability classes: `{json.dumps(dict(classes), sort_keys=True)}`.",
        f"- Best selected video by median distance: `{best_video.get('label', '')}` -> `{best_video.get('target_condition', '')}` distance p50 `{_safe_float(best_video.get('alignment_distance_p50')):.4f}` with target stability `{_safe_float(best_video.get('nearest_target_stability_fraction')):.4f}`.",
        f"- Worst DANDI mapped stimulus by median distance: `{worst_dandi.get('label', '')}` -> `{worst_dandi.get('target_condition', '')}` distance p50 `{_safe_float(worst_dandi.get('alignment_distance_p50')):.4f}`.",
        "",
        "## Selected Video Robustness",
        "",
        "| clip | target | distance p50 [p025, p975] | mean inside prob | target stability | class |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in sorted(by_source.get("selected_video_backend", []), key=lambda item: _safe_float(item.get("alignment_distance_p50"))):
        lines.append(
            f"| `{row['label']}` | `{row['target_condition']}` | "
            f"{_safe_float(row.get('alignment_distance_p50')):.4f} "
            f"[{_safe_float(row.get('alignment_distance_p025')):.4f}, {_safe_float(row.get('alignment_distance_p975')):.4f}] | "
            f"{_safe_float(row.get('mean_metric_inside_probability')):.4f} | "
            f"{_safe_float(row.get('nearest_target_stability_fraction')):.4f} | `{row.get('stability_class')}` |"
        )
    lines.extend(
        [
            "",
            "## >=50k Long-Run Robustness",
            "",
            "| run | target | event Hz p50 [p025, p975] | distance p50 [p025, p975] | mean inside prob | note |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(by_source.get("long_run_action_samples", []), key=lambda item: _safe_float(item.get("alignment_distance_p50"))):
        lines.append(
            f"| `{row['label']}` | `{row['target_condition']}` | "
            f"{_safe_float(row.get('bout_frequency_hz_p50')):.4f} "
            f"[{_safe_float(row.get('bout_frequency_hz_p025')):.4f}, {_safe_float(row.get('bout_frequency_hz_p975')):.4f}] | "
            f"{_safe_float(row.get('alignment_distance_p50')):.4f} "
            f"[{_safe_float(row.get('alignment_distance_p025')):.4f}, {_safe_float(row.get('alignment_distance_p975')):.4f}] | "
            f"{_safe_float(row.get('mean_metric_inside_probability')):.4f} | {row.get('sample_note', '')} |"
        )
    lines.extend(
        [
            "",
            "## DANDI Synthetic Motor Robustness",
            "",
            "| stimulus | expected target | distance p50 [p025, p975] | mean inside prob | nearest target stability | class |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(by_source.get("dandi_synthetic_motor", []), key=lambda item: _safe_float(item.get("alignment_distance_p50")), reverse=True):
        lines.append(
            f"| `{row['label']}` | `{row['target_condition']}` | "
            f"{_safe_float(row.get('alignment_distance_p50')):.4f} "
            f"[{_safe_float(row.get('alignment_distance_p025')):.4f}, {_safe_float(row.get('alignment_distance_p975')):.4f}] | "
            f"{_safe_float(row.get('mean_metric_inside_probability')):.4f} | "
            f"{_safe_float(row.get('nearest_target_stability_fraction')):.4f} | `{row.get('stability_class')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Supported: selected-video and long-run point estimates are now accompanied by temporal-resampling confidence intervals.",
            "- Supported: the behavior-target gap is not a single-point-estimate artifact; many DANDI mapped classes remain outside target envelopes across bootstrap resamples.",
            "- Suggestive only: some selected videos remain near a published nearest-condition envelope, but this is not identity validation for arbitrary underwater video.",
            "- Missing: this still does not replace parameter fitting against the published simZFish/Z-Robot behavior targets or a paired natural-video/calcium/kinematics dataset.",
            "",
            "## Generated Visualizations",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot.resolve()}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-profiles", type=Path, default=DEFAULT_TARGET_PROFILES)
    parser.add_argument("--video-frames", type=Path, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument("--dandi-frames", type=Path, default=DEFAULT_DANDI_FRAMES)
    parser.add_argument("--long-run-root", type=Path, default=DEFAULT_LONG_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    targets, target_by_condition = _load_target_profiles(args.target_profiles)
    items = _group_dandi(args.dandi_frames) + _group_selected_video(args.video_frames) + _group_long_runs(args.long_run_root)
    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for item in items:
        if not item["rows"]:
            continue
        summary, samples = _bootstrap_profile(
            item,
            targets=targets,
            target_by_condition=target_by_condition,
            iterations=max(1, int(args.iterations)),
            rng=rng,
        )
        summary_rows.append(summary)
        sample_rows.extend(samples)

    summary_path = out_dir / "bootstrap_behavior_alignment_summary.csv"
    samples_path = out_dir / "bootstrap_behavior_alignment_samples.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(samples_path, sample_rows)
    plots = [
        _plot_distance_forest(out_dir, summary_rows),
        _plot_inside_heatmap(out_dir, summary_rows),
        _plot_video_target_stability(out_dir, summary_rows),
        _plot_long_run_event_frequency(out_dir, summary_rows),
    ]
    report_path = out_dir / "SIMZFISH_BEHAVIOR_UNCERTAINTY_AUDIT.md"
    class_counts = Counter(str(row.get("stability_class")) for row in summary_rows)
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "target_profiles": str(args.target_profiles.resolve()),
        "video_frames": str(args.video_frames.resolve()),
        "dandi_frames": str(args.dandi_frames.resolve()),
        "long_run_root": str(args.long_run_root.resolve()),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "profile_count": len(summary_rows),
        "sample_count": len(sample_rows),
        "stability_class_counts": dict(class_counts),
        "summary_csv": str(summary_path.resolve()),
        "samples_csv": str(samples_path.resolve()),
        "plots": [str(path.resolve()) for path in plots],
    }
    _write_report(report_path, summary_rows=summary_rows, plots=plots, manifest=manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
