"""Audit video-to-motion behavior against published simZFish/Z-Robot targets.

Long-run stability, DANDI class-level calcium plausibility, and simZFish source
fidelity are separate evidence lanes.  This script audits the behavior lane:
does the current lab's video-to-motion output fall near the published
simZFish/Z-Robot bout-frequency and left/forward/right bout-distribution
targets extracted from Data_Liu_simZFish_2025?

The audit uses transition-based bout events wherever frame-level outputs are
available.  Raw noncoast occupancy is reported but not treated as bout
frequency, because the published spreadsheets summarize bout/action events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLISHED_TARGETS = (
    ROOT / "analysis" / "out" / "simzfish_calibration_targets_20260603" / "published_locomotion_targets.csv"
)
DEFAULT_VIDEO_FRAMES = (
    ROOT / "analysis" / "out" / "backend_video_robustness" / "20260603_all_selected_60s" / "backend_video_frame_metrics.csv"
)
DEFAULT_DANDI_MOTOR_FRAMES = (
    ROOT / "analysis" / "out" / "dandi_omr_motor_response_audit_20260603" / "stimulus_motor_response_frames.csv"
)
DEFAULT_LONG_RUN_ROOT = ROOT / "analysis" / "out" / "comprehensive_activity_study"
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "simzfish_behavior_target_alignment_audit_20260603"

PROFILE_FIELDS = ["bout_frequency_hz", "left_fraction", "forward_fraction", "right_fraction"]


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
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _condition(sheet: str) -> str:
    text = str(sheet).strip()
    text = re.sub(r"\s+\d+$", "", text)
    return text


def _mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _quantile(values: list[float], q: float) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _load_target_profiles(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _read_csv(path)
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cond = _condition(row.get("sheet", ""))
        if cond:
            by_condition[cond].append(row)

    profiles: list[dict[str, Any]] = []
    for cond, members in sorted(by_condition.items()):
        profile: dict[str, Any] = {
            "condition": cond,
            "published_rows": len(members),
            "groups": ";".join(sorted({row.get("group", "") for row in members})),
        }
        for field in PROFILE_FIELDS:
            values = [_safe_float(row.get(field), float("nan")) for row in members]
            profile[f"{field}_mean"] = _mean(values)
            profile[f"{field}_std"] = _std(values)
            profile[f"{field}_p05"] = _quantile(values, 0.05)
            profile[f"{field}_p95"] = _quantile(values, 0.95)
        profiles.append(profile)

    global_profile: dict[str, Any] = {
        "condition": "global_all_published",
        "published_rows": len(rows),
        "groups": ";".join(sorted({row.get("group", "") for row in rows})),
    }
    for field in PROFILE_FIELDS:
        values = [_safe_float(row.get(field), float("nan")) for row in rows]
        global_profile[f"{field}_mean"] = _mean(values)
        global_profile[f"{field}_std"] = _std(values)
        global_profile[f"{field}_p05"] = _quantile(values, 0.05)
        global_profile[f"{field}_p95"] = _quantile(values, 0.95)
    by_name = {row["condition"]: row for row in profiles}
    return profiles, by_name, global_profile


def _is_noncoast(row: dict[str, Any]) -> bool:
    bout_type = str(row.get("action_bout_type") or row.get("bout_type") or row.get("side") or "coast")
    kick = _safe_float(row.get("action_kick", row.get("kick", 0.0)))
    force = _safe_float(row.get("action_force", row.get("force", 0.0)))
    return bout_type not in {"coast", "none", ""} and (kick >= 0.5 or force > 0.02)


def _event_direction(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("action_bout_type") or row.get("bout_type") or "") for row in rows)
    bout_type = counts.most_common(1)[0][0] if counts else ""
    if "startle" in bout_type:
        return "startle"
    if "left" in bout_type:
        return "left"
    if "right" in bout_type:
        return "right"
    side_values = [
        _safe_float(row.get("action_side_score", row.get("side_score", 0.0)))
        for row in rows
        if row.get("action_side_score", row.get("side_score", "")) not in ("", None)
    ]
    side = _mean(side_values)
    if abs(side) >= 0.18:
        # simZFish adapter uses negative side_score for left bouts.
        return "left" if side < 0.0 else "right"
    return "forward"


def _profile_from_frame_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    source_kind: str,
    duration_s: float | None = None,
) -> dict[str, Any]:
    if duration_s is None:
        times = [_safe_float(row.get("video_time_s", row.get("time_s", 0.0)), float("nan")) for row in rows]
        times = [t for t in times if math.isfinite(t)]
        if len(times) >= 2:
            duration_s = max(times) - min(times) + float(np.median(np.diff(sorted(set(times))))) if len(set(times)) > 1 else max(times)
        else:
            duration_s = 0.0
    duration_s = max(1e-9, float(duration_s or 0.0))
    noncoast_flags = [_is_noncoast(row) for row in rows]
    groups: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] = []
    for row, flag in zip(rows, noncoast_flags, strict=True):
        if flag:
            active.append(row)
        elif active:
            groups.append(active)
            active = []
    if active:
        groups.append(active)
    event_dirs = [_event_direction(group) for group in groups]
    counts = Counter(event_dirs)
    total = max(1, len(groups))
    lfr_total = max(1, counts["left"] + counts["forward"] + counts["right"])
    return {
        "label": label,
        "source_kind": source_kind,
        "frames": len(rows),
        "duration_s": duration_s,
        "event_count": len(groups),
        "bout_frequency_hz": len(groups) / duration_s,
        "noncoast_frame_fraction": float(np.mean(noncoast_flags)) if noncoast_flags else 0.0,
        "left_fraction": counts["left"] / total,
        "forward_fraction": counts["forward"] / total,
        "right_fraction": counts["right"] / total,
        "startle_fraction": counts["startle"] / total,
        "left_lfr_fraction": counts["left"] / lfr_total,
        "forward_lfr_fraction": counts["forward"] / lfr_total,
        "right_lfr_fraction": counts["right"] / lfr_total,
        "event_direction_counts": json.dumps(dict(counts), sort_keys=True),
    }


def _profile_from_action_jsonl(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
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
    summary_path = path.parents[1] / "summary.json"
    duration_s = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runs = summary.get("runs", {})
            run_name = path.parent.name
            if isinstance(runs, dict):
                run = runs.get(run_name, {})
                if isinstance(run, dict):
                    run = run.get("summary", run)
            else:
                run = next((row for row in runs if row.get("name") == run_name), {})
            duration_s = _safe_float(run.get("sim_seconds"), 0.0) or None if isinstance(run, dict) else None
        except Exception:  # noqa: BLE001
            duration_s = None
    profile = _profile_from_frame_rows(rows, label=label, source_kind="long_run_action_samples", duration_s=duration_s)
    profile["sample_note"] = (
        "driver action samples; video is backend-frame sampled, calcium is replay-state REST sampled"
    )
    return profile


def _profiles_from_selected_video(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[row.get("clip", "")].append(row)
    return [
        _profile_from_frame_rows(members, label=clip, source_kind="selected_video_backend")
        for clip, members in sorted(by_clip.items())
    ]


def _profiles_from_dandi_frames(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    by_stim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stim[row.get("stimulus", "")].append(row)
    return [
        _profile_from_frame_rows(members, label=stim, source_kind="dandi_synthetic_motor")
        for stim, members in sorted(by_stim.items())
    ]


def _long_run_profiles(root: Path) -> list[dict[str, Any]]:
    candidates = [
        root / "20260603_verified_50k" / "calcium_all" / "action_samples.jsonl",
        root / "20260603_verified_50k" / "video_commons_tenggol_underwater" / "action_samples.jsonl",
        root / "20260603_black_rockfish_50k" / "video_commons_black_rockfish_stereo_dov" / "action_samples.jsonl",
    ]
    out: list[dict[str, Any]] = []
    for path in candidates:
        profile = _profile_from_action_jsonl(path, label=f"{path.parents[1].name}/{path.parent.name}")
        if profile is not None:
            out.append(profile)
    return out


def _target_for_dandi_stimulus(stimulus: str) -> str:
    mapping = {
        "converging": "In",
        "diverging": "Out",
        "forward": "F-F",
        "backward": "B-B",
        "left": "L-L",
        "right": "R-R",
        "x_forward": "N-F",
        "forward_x": "F-N",
        "x_backward": "N-B",
        "backward_x": "B-N",
        "forward_left": "F-N",
        "forward_right": "N-F",
        "backward_left": "B-N",
        "backward_right": "N-B",
        "backward_forward": "B-F",
        "forward_backward": "F-B",
        # Eye-relative medial/lateral labels mapped through expected monocular
        # horizontal direction under the first-left hypothesis.
        "medial_left": "R-N",
        "lateral_left": "L-N",
        "medial_right": "N-L",
        "lateral_right": "N-R",
    }
    return mapping.get(stimulus, "")


def _z(value: float, mean: float, std: float, p05: float, p95: float) -> float:
    scale = float(std) if math.isfinite(std) and std > 1e-9 else abs(float(p95) - float(p05)) / 3.29
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 0.1
    return (float(value) - float(mean)) / scale


def _score_profile(current: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    values = {}
    zsq = []
    within = 0
    for field in PROFILE_FIELDS:
        value = _safe_float(current.get(field))
        mean = _safe_float(target.get(f"{field}_mean"), float("nan"))
        std = _safe_float(target.get(f"{field}_std"), 0.0)
        p05 = _safe_float(target.get(f"{field}_p05"), float("nan"))
        p95 = _safe_float(target.get(f"{field}_p95"), float("nan"))
        z = _z(value, mean, std, p05, p95)
        zsq.append(z * z)
        inside = math.isfinite(p05) and math.isfinite(p95) and p05 <= value <= p95
        within += int(inside)
        values[f"{field}_target_mean"] = mean
        values[f"{field}_target_p05"] = p05
        values[f"{field}_target_p95"] = p95
        values[f"{field}_z"] = z
        values[f"{field}_inside_p05_p95"] = float(inside)
    lfr_error = abs(_safe_float(current.get("left_fraction")) - _safe_float(target.get("left_fraction_mean")))
    lfr_error += abs(_safe_float(current.get("forward_fraction")) - _safe_float(target.get("forward_fraction_mean")))
    lfr_error += abs(_safe_float(current.get("right_fraction")) - _safe_float(target.get("right_fraction_mean")))
    startle_penalty = 1.5 * _safe_float(current.get("startle_fraction"))
    distance = float(math.sqrt(np.mean(zsq)) + 0.75 * lfr_error + startle_penalty)
    return {
        **values,
        "target_condition": target["condition"],
        "target_published_rows": target.get("published_rows", 0),
        "alignment_distance": distance,
        "within_p05_p95_count": within,
        "within_p05_p95_fraction": within / len(PROFILE_FIELDS),
        "startle_penalty": startle_penalty,
    }


def _alignment_rows(
    current_profiles: list[dict[str, Any]],
    target_profiles: list[dict[str, Any]],
    target_by_condition: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in current_profiles:
        expected = ""
        if profile["source_kind"] == "dandi_synthetic_motor":
            expected = _target_for_dandi_stimulus(str(profile["label"]))
        candidates = [target_by_condition[expected]] if expected in target_by_condition else target_profiles
        scored = [{**_score_profile(profile, target), "expected_condition": expected} for target in candidates]
        best = min(scored, key=lambda row: row["alignment_distance"])
        rows.append({**profile, **best})
    return sorted(rows, key=lambda row: (row["source_kind"], row["alignment_distance"]))


def _nearest_rows(
    profiles: list[dict[str, Any]],
    target_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for profile in profiles:
        scored = [_score_profile(profile, target) for target in target_profiles]
        best = min(scored, key=lambda row: row["alignment_distance"])
        out.append({**profile, **best, "expected_condition": ""})
    return sorted(out, key=lambda row: row["alignment_distance"])


def _save_target_envelope_plot(out_dir: Path, targets: list[dict[str, Any]]) -> Path:
    path = out_dir / "published_behavior_target_envelopes.png"
    labels = [row["condition"] for row in targets]
    freq = np.asarray([row["bout_frequency_hz_mean"] for row in targets], dtype=np.float64)
    left = np.asarray([row["left_fraction_mean"] for row in targets], dtype=np.float64)
    forward = np.asarray([row["forward_fraction_mean"] for row in targets], dtype=np.float64)
    right = np.asarray([row["right_fraction_mean"] for row in targets], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    x = np.arange(len(labels))
    p05 = np.asarray([row["bout_frequency_hz_p05"] for row in targets], dtype=np.float64)
    p95 = np.asarray([row["bout_frequency_hz_p95"] for row in targets], dtype=np.float64)
    axes[0].bar(x, freq, color="#2b8cbe")
    axes[0].errorbar(x, freq, yerr=[np.maximum(0.0, freq - p05), np.maximum(0.0, p95 - freq)], fmt="none", ecolor="black", linewidth=0.8)
    axes[0].set_ylabel("bout frequency Hz")
    axes[0].set_title("Published simZFish/Z-Robot behavior target envelopes")
    width = 0.25
    axes[1].bar(x - width, left, width=width, label="left", color="#ef8a62")
    axes[1].bar(x, forward, width=width, label="forward", color="#67a9cf")
    axes[1].bar(x + width, right, width=width, label="right", color="#1b7837")
    axes[1].set_ylabel("bout fraction")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_alignment_distance_plot(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "behavior_alignment_distance_rank.png"
    selected = sorted(rows, key=lambda row: row["alignment_distance"])
    labels = [f"{row['source_kind']}\\n{row['label']}" for row in selected]
    dist = np.asarray([row["alignment_distance"] for row in selected], dtype=np.float64)
    inside = np.asarray([row["within_p05_p95_fraction"] for row in selected], dtype=np.float64)
    fig, ax1 = plt.subplots(figsize=(16, 7))
    x = np.arange(len(selected))
    ax1.bar(x, dist, color="#d73027", alpha=0.78, label="alignment distance")
    ax1.set_ylabel("target distance (lower is better)")
    ax2 = ax1.twinx()
    ax2.plot(x, inside, color="#2166ac", marker="o", linewidth=1.2, label="within p05-p95 fraction")
    ax2.set_ylabel("fraction of metrics inside target p05-p95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax1.set_title("Current behavior profiles versus published simZFish/Z-Robot targets")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_dandi_expected_heatmap(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "dandi_expected_condition_error_heatmap.png"
    rows = [row for row in rows if row["source_kind"] == "dandi_synthetic_motor"]
    fields = [
        "bout_frequency_hz_z",
        "left_fraction_z",
        "forward_fraction_z",
        "right_fraction_z",
        "startle_fraction",
        "alignment_distance",
    ]
    labels = [str(row["label"]) for row in rows]
    mat = np.asarray([[float(row.get(field, 0.0)) for field in fields] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-3.0, vmax=3.0)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("DANDI synthetic motor output error versus mapped published behavior targets")
    fig.colorbar(im, ax=ax, label="z/error scale")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_video_nearest_target_plot(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "selected_video_nearest_behavior_target.png"
    rows = [row for row in rows if row["source_kind"] == "selected_video_backend"]
    labels = [str(row["label"]) for row in rows]
    targets = [str(row["target_condition"]) for row in rows]
    dist = np.asarray([row["alignment_distance"] for row in rows], dtype=np.float64)
    startle = np.asarray([row["startle_fraction"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, dist, width=0.36, label="nearest-target distance", color="#b2182b")
    ax.bar(x + 0.18, startle, width=0.36, label="startle fraction", color="#fdae61")
    for i, target in enumerate(targets):
        ax.text(i, dist[i] + 0.05, target, rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("score / fraction")
    ax.set_title("Selected videos: nearest published behavior condition and startle burden")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    target_profiles: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    nearest_rows: list[dict[str, Any]],
    plots: list[Path],
) -> None:
    dandi_rows = [row for row in alignment_rows if row["source_kind"] == "dandi_synthetic_motor"]
    video_rows = [row for row in alignment_rows if row["source_kind"] == "selected_video_backend"]
    long_rows = [row for row in alignment_rows if row["source_kind"] == "long_run_action_samples"]
    dandi_inside = [row["within_p05_p95_fraction"] for row in dandi_rows]
    video_inside = [row["within_p05_p95_fraction"] for row in video_rows]
    top_video = min(video_rows, key=lambda row: row["alignment_distance"], default={})
    worst_dandi = max(dandi_rows, key=lambda row: row["alignment_distance"], default={})
    lines = [
        "# simZFish/Z-Robot Behavior Target Alignment Audit",
        "",
        "## Scope",
        "",
        "This audit compares current zebrafish lab motor outputs against published simZFish/Z-Robot locomotion target tables extracted from `Data_Liu_simZFish_2025`. It uses transition-based bout events, not raw noncoast frame occupancy, as the primary bout-frequency evidence.",
        "",
        "## Main Findings",
        "",
        f"- Published target conditions: `{len(target_profiles)}` condition profiles.",
        f"- DANDI synthetic-stimulus profiles: `{len(dandi_rows)}`; mean within-target fraction `{_mean(dandi_inside):.4f}`.",
        f"- Selected-video backend profiles: `{len(video_rows)}`; mean within-target fraction `{_mean(video_inside):.4f}`.",
        f"- Best selected-video nearest target: `{top_video.get('label', '')}` -> `{top_video.get('target_condition', '')}` distance `{_safe_float(top_video.get('alignment_distance')):.4f}`.",
        f"- Worst mapped DANDI synthetic target: `{worst_dandi.get('label', '')}` expected `{worst_dandi.get('expected_condition', '')}` distance `{_safe_float(worst_dandi.get('alignment_distance')):.4f}`.",
        "",
        "## DANDI Synthetic Stimuli Versus Expected Published Conditions",
        "",
        "| stimulus | expected condition | event Hz | target Hz | L/F/R/startle | distance | inside target |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in sorted(dandi_rows, key=lambda item: item["alignment_distance"], reverse=True):
        lines.append(
            f"| `{row['label']}` | `{row.get('expected_condition', '')}` | {row['bout_frequency_hz']:.4f} | "
            f"{row['bout_frequency_hz_target_mean']:.4f} | "
            f"{row['left_fraction']:.2f}/{row['forward_fraction']:.2f}/{row['right_fraction']:.2f}/{row['startle_fraction']:.2f} | "
            f"{row['alignment_distance']:.4f} | {row['within_p05_p95_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Videos Versus Nearest Published Conditions",
            "",
            "| clip | nearest condition | event Hz | target Hz | L/F/R/startle | distance | inside target |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in sorted(video_rows, key=lambda item: item["alignment_distance"]):
        lines.append(
            f"| `{row['label']}` | `{row['target_condition']}` | {row['bout_frequency_hz']:.4f} | "
            f"{row['bout_frequency_hz_target_mean']:.4f} | "
            f"{row['left_fraction']:.2f}/{row['forward_fraction']:.2f}/{row['right_fraction']:.2f}/{row['startle_fraction']:.2f} | "
            f"{row['alignment_distance']:.4f} | {row['within_p05_p95_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 50k Long-Run Profiles",
            "",
            "| run | nearest condition | event Hz | target Hz | L/F/R/startle | distance | inside target |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in sorted(long_rows, key=lambda item: item["alignment_distance"]):
        lines.append(
            f"| `{row['label']}` | `{row['target_condition']}` | {row['bout_frequency_hz']:.4f} | "
            f"{row['bout_frequency_hz_target_mean']:.4f} | "
            f"{row['left_fraction']:.2f}/{row['forward_fraction']:.2f}/{row['right_fraction']:.2f}/{row['startle_fraction']:.2f} | "
            f"{row['alignment_distance']:.4f} | {row['within_p05_p95_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Supported: the current selected-video and synthetic-stimulus motor outputs can now be compared directly to published simZFish/Z-Robot behavior target envelopes using transition-based event rates and bout fractions.",
            "- Supported: selected-video event frequencies often fall near some published behavior condition, but nearest-condition matching is only a plausibility diagnostic for arbitrary videos.",
            "- Not supported: DANDI synthetic stimuli are not yet reliably aligned to their mapped published behavior conditions; this indicates a motor-adapter fitting gap, not just a neural-label gap.",
            "- Not supported: startle-heavy selected-video outputs should not be treated as clean OMR validation, even when event frequency lies inside a published envelope.",
            "- Required next step: fit or replace the live adapter against the published condition profiles and compiled simZFish controller surfaces, then repeat >=50k embodied validation.",
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
    parser.add_argument("--published-targets", type=Path, default=DEFAULT_PUBLISHED_TARGETS)
    parser.add_argument("--video-frames", type=Path, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument("--dandi-motor-frames", type=Path, default=DEFAULT_DANDI_MOTOR_FRAMES)
    parser.add_argument("--long-run-root", type=Path, default=DEFAULT_LONG_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    target_profiles, target_by_condition, global_profile = _load_target_profiles(args.published_targets)
    video_profiles = _profiles_from_selected_video(args.video_frames)
    dandi_profiles = _profiles_from_dandi_frames(args.dandi_motor_frames)
    long_profiles = _long_run_profiles(args.long_run_root)
    current_profiles = dandi_profiles + video_profiles + long_profiles
    alignment_rows = _alignment_rows(current_profiles, target_profiles, target_by_condition)
    nearest_all_rows = _nearest_rows(current_profiles, target_profiles)

    target_path = out_dir / "published_behavior_condition_profiles.csv"
    current_path = out_dir / "current_behavior_profiles.csv"
    alignment_path = out_dir / "behavior_target_alignment_scores.csv"
    nearest_path = out_dir / "behavior_nearest_target_scores.csv"
    _write_csv(target_path, target_profiles)
    _write_csv(current_path, current_profiles)
    _write_csv(alignment_path, alignment_rows)
    _write_csv(nearest_path, nearest_all_rows)

    plots = [
        _save_target_envelope_plot(out_dir, target_profiles),
        _save_alignment_distance_plot(out_dir, alignment_rows),
        _save_dandi_expected_heatmap(out_dir, alignment_rows),
        _save_video_nearest_target_plot(out_dir, nearest_all_rows),
    ]
    report_path = out_dir / "SIMZFISH_BEHAVIOR_TARGET_ALIGNMENT_AUDIT.md"
    _write_report(
        report_path,
        target_profiles=target_profiles,
        alignment_rows=alignment_rows,
        nearest_rows=nearest_all_rows,
        plots=plots,
    )

    dandi_rows = [row for row in alignment_rows if row["source_kind"] == "dandi_synthetic_motor"]
    video_rows = [row for row in alignment_rows if row["source_kind"] == "selected_video_backend"]
    long_rows = [row for row in alignment_rows if row["source_kind"] == "long_run_action_samples"]
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "published_targets": str(args.published_targets.resolve()),
        "video_frames": str(args.video_frames.resolve()),
        "dandi_motor_frames": str(args.dandi_motor_frames.resolve()),
        "target_condition_count": len(target_profiles),
        "dandi_profile_count": len(dandi_rows),
        "video_profile_count": len(video_rows),
        "long_run_profile_count": len(long_rows),
        "mean_dandi_within_target_fraction": _mean([row["within_p05_p95_fraction"] for row in dandi_rows]),
        "mean_video_within_target_fraction": _mean([row["within_p05_p95_fraction"] for row in video_rows]),
        "mean_long_run_within_target_fraction": _mean([row["within_p05_p95_fraction"] for row in long_rows]),
        "best_video_label": min(video_rows, key=lambda row: row["alignment_distance"])["label"] if video_rows else "",
        "best_video_distance": min([row["alignment_distance"] for row in video_rows]) if video_rows else 0.0,
        "worst_dandi_label": max(dandi_rows, key=lambda row: row["alignment_distance"])["label"] if dandi_rows else "",
        "worst_dandi_distance": max([row["alignment_distance"] for row in dandi_rows]) if dandi_rows else 0.0,
        "global_published_profile": global_profile,
        "csv": {
            "published_behavior_condition_profiles": str(target_path.resolve()),
            "current_behavior_profiles": str(current_path.resolve()),
            "behavior_target_alignment_scores": str(alignment_path.resolve()),
            "behavior_nearest_target_scores": str(nearest_path.resolve()),
        },
        "plots": [str(path.resolve()) for path in plots],
        "limitations": [
            "Published targets are extracted from spreadsheets and grouped by sheet condition labels.",
            "Selected videos are arbitrary natural/underwater clips, so nearest published condition is plausibility only.",
            "DANDI labels are mapped to published behavior condition labels by a documented first-left eye-order hypothesis.",
            "The audit does not fit parameters; it measures the current implementation.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
