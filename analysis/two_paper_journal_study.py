"""Build a journal-style validation package for the two-paper zebrafish pipeline.

The generated files are intentionally claim-centric.  Each claim is tied to
paper evidence, implementation evidence, quantitative artifacts, and an
explicit status so the current lab can be presented honestly to zebrafish
researchers.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_INFERENCE = ROOT.parent / "active-inference"
OUT_DIR = ROOT / "analysis" / "out" / "two_paper_journal_study_20260603"
FORENSICS_DIR = (
    ROOT
    / "analysis"
    / "out"
    / "comprehensive_activity_study"
    / "20260603_verified_50k"
    / "forensics"
)
SYNTHESIS_DIR = (
    ROOT
    / "analysis"
    / "out"
    / "comprehensive_activity_study"
    / "20260603_verified_50k"
    / "synthesis"
)
SIMZFISH_FIDELITY_DIR = ROOT / "analysis" / "out" / "simzfish_fidelity_audit_20260603"
SIMZFISH_CALIBRATION_DIR = ROOT / "analysis" / "out" / "simzfish_calibration_targets_20260603"
SIMZFISH_CONTROLLER_DIR = ROOT / "analysis" / "out" / "simzfish_controller_regression_20260603"
BACKEND_VIDEO_ROBUSTNESS_DIR = ROOT / "analysis" / "out" / "backend_video_robustness" / "20260603_all_selected_60s"
EMBODIED_VIDEO_REPLICATE_DIR = ROOT / "analysis" / "out" / "embodied_video_replicate_comparison_20260603"
DANDI_NEURAL_VALIDATION_DIR = ROOT / "analysis" / "out" / "dandi_omr_neural_validation_20260603"
DANDI_VIDEO_ALIGNMENT_DIR = ROOT / "analysis" / "out" / "dandi_video_omr_alignment_20260603"
DANDI_MOTOR_RESPONSE_DIR = ROOT / "analysis" / "out" / "dandi_omr_motor_response_audit_20260603"
DANDI_IDENTIFIABILITY_DIR = ROOT / "analysis" / "out" / "dandi_stimulus_identifiability_audit_20260603"
DANDI_PROJECTOR_SEMANTIC_DIR = ROOT / "analysis" / "out" / "dandi_projector_semantic_bridge_audit_20260603"
SIMZFISH_BEHAVIOR_ALIGNMENT_DIR = ROOT / "analysis" / "out" / "simzfish_behavior_target_alignment_audit_20260603"
SIMZFISH_BEHAVIOR_UNCERTAINTY_DIR = ROOT / "analysis" / "out" / "simzfish_behavior_uncertainty_audit_20260603"
REALIZED_KINEMATICS_DIR = ROOT / "analysis" / "out" / "realized_kinematics_transfer_audit_20260603"
EXTERNAL_KINEMATICS_DIR = ROOT / "analysis" / "out" / "external_kinematics_validation_audit_20260603"
HIGH_RATE_TAIL_DIR = ROOT / "analysis" / "out" / "high_rate_tail_validation_20260603"
HIGH_RATE_REPLAY_DIR = ROOT / "analysis" / "out" / "high_rate_replay_validation_20260603"


SOURCES = [
    {
        "id": "zapbench_openreview",
        "title": "ZAPBench: A Benchmark for Whole-Brain Activity Prediction in Zebrafish",
        "url": "https://openreview.net/forum?id=oCHsDpyawq",
        "role": "primary paper page",
        "key_facts": [
            "ICLR 2025 Spotlight benchmark paper.",
            "Forecasting target is future whole-brain neural activity.",
            "Dataset contains 4D light-sheet recordings of over 70,000 neurons.",
        ],
    },
    {
        "id": "zapbench_landing",
        "title": "ZAPBench landing and interactive results",
        "url": "https://zapbench-release.storage.googleapis.com/landing.html",
        "role": "official project/data portal",
        "key_facts": [
            "Describes ZAPBench as cellular-resolution whole-brain activity prediction.",
            "Links traces, segmentation, volumetric activity, code, tutorials, and conditions.",
        ],
    },
    {
        "id": "zapbench_datasets",
        "title": "ZAPBench datasets README",
        "url": "https://zapbench-release.storage.googleapis.com/volumes/README.html",
        "role": "official dataset manifest",
        "key_facts": [
            "Lists raw, anatomy, aligned, df_over_f, segmentation, and trace buckets.",
            "States datasets are hosted in the zapbench-release Google Cloud bucket.",
            "States datasets are CC-BY 4.0.",
        ],
    },
    {
        "id": "zapbench_github",
        "title": "google-research/zapbench",
        "url": "https://github.com/google-research/zapbench",
        "role": "official code",
        "key_facts": [
            "Repository contains forecasting models, processing notebooks, and visualization code.",
            "Repository documents access/tutorial notebooks for datasets, training, metrics, and interactive forecasting.",
        ],
    },
    {
        "id": "google_research_blog",
        "title": "Improving brain models with ZAPBench",
        "url": "https://research.google/blog/improving-brain-models-with-zapbench/",
        "role": "official explanatory article",
        "key_facts": [
            "Confirms roughly 70,000 neurons, two hours of brain activity, VR stimuli, tail electrodes, and overhead video.",
            "States the benchmark asks how accurately subsequent brain activity can be predicted from recorded brain activity.",
        ],
    },
    {
        "id": "zrobot_science",
        "title": "Artificial embodied circuits uncover neural architectures of vertebrate visuomotor behaviors",
        "url": "https://www.science.org/doi/10.1126/scirobotics.adv4408",
        "role": "primary paper page/local PDF",
        "key_facts": [
            "Science Robotics 10:eAdv4408, 2025.",
            "Introduces simZFish and ZBot for embodied zebrafish OMR/rheotaxis.",
            "Local PDF is docs/Z-Robot.pdf because web access can be gated.",
        ],
    },
    {
        "id": "zrobot_pmc",
        "title": "PMC manuscript page for Z-Robot paper",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13077711/",
        "role": "public article mirror",
        "key_facts": [
            "Search result exposes abstract and methods, but direct page access in this environment is reCAPTCHA-gated.",
        ],
    },
    {
        "id": "simzfish_repo",
        "title": "simZFish public repository",
        "url": "https://ponyo.epfl.ch/proj/zebrafish/simzfish",
        "role": "official code and data repository",
        "key_facts": [
            "Public GitLab repository, Apache 2.0.",
            "Local cache includes repository tree, controller resources, user guide, and data bundle.",
        ],
    },
    {
        "id": "dandi_001076",
        "title": "DANDI 001076: OMR Robot CaImaging",
        "url": "https://dandiarchive.org/dandiset/001076",
        "role": "public calcium imaging data",
        "key_facts": [
            "Open DANDI dataset for OMR Robot CaImaging.",
            "DANDI page lists 48 files, 629.7 MiB, CC-BY-4.0, Danio rerio.",
            "Local manifest has 48 assets and one downloaded NWB sample.",
        ],
    },
]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _fetch_status(url: str, timeout_s: float = 12.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "zebrafish-validation-study/1.0"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read(4096)
            return {
                "ok": 200 <= int(resp.status) < 400,
                "status": int(resp.status),
                "elapsed_s": time.perf_counter() - start,
                "content_type": resp.headers.get("Content-Type", ""),
                "sample_bytes": len(body),
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": int(exc.code),
            "elapsed_s": time.perf_counter() - start,
            "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": None,
            "elapsed_s": time.perf_counter() - start,
            "content_type": "",
            "error": repr(exc),
        }


def _read_json(path: Path) -> Any:
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


def _npz_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    with np.load(path, allow_pickle=False) as npz:
        out["keys"] = {}
        for key in npz.files:
            arr = np.asarray(npz[key])
            out["keys"][key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            if arr.size and np.issubdtype(arr.dtype, np.number):
                out["keys"][key].update(
                    {
                        "mean": float(np.nanmean(arr)),
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                    }
                )
    return out


def _load_metrics() -> dict[str, Any]:
    direct_metrics = _read_json(ROOT / "analysis" / "out" / "zapbench_ephys_action_decoder" / "direct_ephys_metrics.json")
    synth_manifest = _read_json(SYNTHESIS_DIR / "synthesis_manifest.json")
    forensic_manifest = _read_json(FORENSICS_DIR / "forensics_manifest.json")
    action_stats = _read_csv(FORENSICS_DIR / "action_numeric_stats.csv")
    scalar_stats = _read_csv(FORENSICS_DIR / "scalar_extrema_and_stats.csv")
    bout_counts = _read_csv(FORENSICS_DIR / "action_bout_counts.csv")
    video_quality = _read_csv(FORENSICS_DIR / "video_quality_events.csv")
    video_event_counts: dict[str, int] = {}
    for row in video_quality:
        video_event_counts[row["event"]] = video_event_counts.get(row["event"], 0) + 1
    simzfish_manifest = _read_json(ROOT / "analysis" / "cache" / "z_robot" / "simzfish_data_manifest.json")
    dandi_manifest = _read_json(ROOT / "analysis" / "cache" / "z_robot" / "dandi_001076_assets_manifest.json")
    repo_tree = _read_json(ROOT / "analysis" / "cache" / "z_robot" / "simzfish_repository_tree.json")
    label_npz = _npz_summary(ROOT / "analysis" / "out" / "zapbench_ephys_action_decoder" / "direct_ephys_labels.npz")
    replay_npz = _npz_summary(ROOT / "analysis" / "cache" / "zapbench" / "zapbench_calcium_action_replay.npz")
    stimulus_npz = _npz_summary(ROOT / "analysis" / "cache" / "zapbench" / "zapbench_stimulus_features.npz")
    simzfish_fidelity_manifest = _read_json(SIMZFISH_FIDELITY_DIR / "manifest.json")
    simzfish_fidelity_data = _read_json(SIMZFISH_FIDELITY_DIR / "published_data_inventory.json")
    simzfish_calibration_manifest = _read_json(SIMZFISH_CALIBRATION_DIR / "manifest.json")
    simzfish_calibration_comparison = _read_csv(SIMZFISH_CALIBRATION_DIR / "current_vs_published_behavior.csv")
    simzfish_controller_manifest = _read_json(SIMZFISH_CONTROLLER_DIR / "manifest.json")
    simzfish_controller_robot = _read_csv(SIMZFISH_CONTROLLER_DIR / "robot_c_motor_regression.csv")
    simzfish_controller_retina = _read_csv(SIMZFISH_CONTROLLER_DIR / "image_c_retina_regression.csv")
    simzfish_controller_omr = _read_csv(SIMZFISH_CONTROLLER_DIR / "omr_c_compiled_regression.csv")
    simzfish_controller_leaky = _read_csv(SIMZFISH_CONTROLLER_DIR / "leaky_integrator_c_compiled_regression.csv")
    backend_video_manifest = _read_json(BACKEND_VIDEO_ROBUSTNESS_DIR / "manifest.json")
    backend_video_summary = _read_csv(BACKEND_VIDEO_ROBUSTNESS_DIR / "backend_video_robustness_summary.csv")
    embodied_video_replicate_manifest = _read_json(EMBODIED_VIDEO_REPLICATE_DIR / "manifest.json")
    embodied_video_replicate_summary = _read_csv(EMBODIED_VIDEO_REPLICATE_DIR / "embodied_video_replicate_summary.csv")
    dandi_neural_manifest = _read_json(DANDI_NEURAL_VALIDATION_DIR / "manifest.json")
    dandi_neural_stimulus_summary = _read_csv(DANDI_NEURAL_VALIDATION_DIR / "stimulus_response_summary.csv")
    dandi_neural_file_summary = _read_csv(DANDI_NEURAL_VALIDATION_DIR / "nwb_file_summary.csv")
    dandi_video_alignment_manifest = _read_json(DANDI_VIDEO_ALIGNMENT_DIR / "manifest.json")
    dandi_video_alignment_summary = _read_csv(DANDI_VIDEO_ALIGNMENT_DIR / "clip_neural_alignment_summary.csv")
    dandi_motor_response_manifest = _read_json(DANDI_MOTOR_RESPONSE_DIR / "manifest.json")
    dandi_motor_response_summary = _read_csv(DANDI_MOTOR_RESPONSE_DIR / "stimulus_motor_response_summary.csv")
    dandi_identifiability_manifest = _read_json(DANDI_IDENTIFIABILITY_DIR / "manifest.json")
    dandi_identifiability_collisions = _read_csv(DANDI_IDENTIFIABILITY_DIR / "coarse_axis_collision_groups.csv")
    dandi_identifiability_models = _read_csv(DANDI_IDENTIFIABILITY_DIR / "stimulus_feature_model_scores.csv")
    dandi_identifiability_signs = _read_csv(DANDI_IDENTIFIABILITY_DIR / "axis_sign_sensitivity.csv")
    dandi_projector_manifest = _read_json(DANDI_PROJECTOR_SEMANTIC_DIR / "manifest.json")
    dandi_projector_models = _read_csv(DANDI_PROJECTOR_SEMANTIC_DIR / "semantic_feature_model_scores.csv")
    dandi_projector_collisions = _read_csv(DANDI_PROJECTOR_SEMANTIC_DIR / "semantic_feature_collision_groups.csv")
    dandi_projector_clip_summary = _read_csv(DANDI_PROJECTOR_SEMANTIC_DIR / "rich_video_clip_alignment_summary.csv")
    dandi_projector_permutations = _read_csv(DANDI_PROJECTOR_SEMANTIC_DIR / "semantic_model_permutation_scores.csv")
    simzfish_behavior_manifest = _read_json(SIMZFISH_BEHAVIOR_ALIGNMENT_DIR / "manifest.json")
    simzfish_behavior_targets = _read_csv(SIMZFISH_BEHAVIOR_ALIGNMENT_DIR / "published_behavior_condition_profiles.csv")
    simzfish_behavior_profiles = _read_csv(SIMZFISH_BEHAVIOR_ALIGNMENT_DIR / "current_behavior_profiles.csv")
    simzfish_behavior_scores = _read_csv(SIMZFISH_BEHAVIOR_ALIGNMENT_DIR / "behavior_target_alignment_scores.csv")
    simzfish_behavior_nearest = _read_csv(SIMZFISH_BEHAVIOR_ALIGNMENT_DIR / "behavior_nearest_target_scores.csv")
    simzfish_behavior_uncertainty_manifest = _read_json(SIMZFISH_BEHAVIOR_UNCERTAINTY_DIR / "manifest.json")
    simzfish_behavior_uncertainty_summary = _read_csv(
        SIMZFISH_BEHAVIOR_UNCERTAINTY_DIR / "bootstrap_behavior_alignment_summary.csv"
    )
    realized_kinematics_manifest = _read_json(REALIZED_KINEMATICS_DIR / "manifest.json")
    realized_kinematics_summary = _read_csv(REALIZED_KINEMATICS_DIR / "realized_kinematics_transfer_summary.csv")
    external_kinematics_manifest = _read_json(EXTERNAL_KINEMATICS_DIR / "manifest.json")
    external_kinematics_summary = _read_csv(EXTERNAL_KINEMATICS_DIR / "external_kinematic_run_summary.csv")
    external_kinematics_comparison = _read_csv(EXTERNAL_KINEMATICS_DIR / "external_kinematic_target_comparison.csv")
    high_rate_tail_manifest = _read_json(HIGH_RATE_TAIL_DIR / "manifest.json")
    high_rate_tail_summary = _read_csv(HIGH_RATE_TAIL_DIR / "high_rate_run_summary.csv")
    high_rate_tail_comparison = _read_csv(HIGH_RATE_TAIL_DIR / "high_rate_external_comparison.csv")
    high_rate_tail_anomalies = _read_csv(HIGH_RATE_TAIL_DIR / "high_rate_anomaly_checks.csv")
    high_rate_replay_manifest = _read_json(HIGH_RATE_REPLAY_DIR / "manifest.json")
    high_rate_replay_summary = _read_csv(HIGH_RATE_REPLAY_DIR / "high_rate_run_summary.csv")
    high_rate_replay_comparison = _read_csv(HIGH_RATE_REPLAY_DIR / "high_rate_external_comparison.csv")
    high_rate_replay_anomalies = _read_csv(HIGH_RATE_REPLAY_DIR / "high_rate_anomaly_checks.csv")
    return {
        "direct_metrics": direct_metrics,
        "synthesis": synth_manifest,
        "forensics": forensic_manifest,
        "action_stats": action_stats,
        "scalar_stats": scalar_stats,
        "bout_counts": bout_counts,
        "video_quality_event_counts": video_event_counts,
        "simzfish_data_files": len(simzfish_manifest) if isinstance(simzfish_manifest, list) else 0,
        "simzfish_data_bytes": sum(int(x.get("size", 0)) for x in simzfish_manifest if isinstance(x, dict))
        if isinstance(simzfish_manifest, list)
        else 0,
        "dandi_asset_files": len(dandi_manifest) if isinstance(dandi_manifest, list) else 0,
        "dandi_asset_bytes": sum(int(x.get("size", 0)) for x in dandi_manifest if isinstance(x, dict))
        if isinstance(dandi_manifest, list)
        else 0,
        "simzfish_repo_items": len(repo_tree) if isinstance(repo_tree, list) else 0,
        "label_npz": label_npz,
        "replay_npz": replay_npz,
        "stimulus_npz": stimulus_npz,
        "simzfish_fidelity_manifest": simzfish_fidelity_manifest,
        "simzfish_fidelity_data": simzfish_fidelity_data,
        "simzfish_calibration_manifest": simzfish_calibration_manifest,
        "simzfish_calibration_comparison": simzfish_calibration_comparison,
        "simzfish_controller_manifest": simzfish_controller_manifest,
        "simzfish_controller_robot": simzfish_controller_robot,
        "simzfish_controller_retina": simzfish_controller_retina,
        "simzfish_controller_omr": simzfish_controller_omr,
        "simzfish_controller_leaky": simzfish_controller_leaky,
        "backend_video_robustness_manifest": backend_video_manifest,
        "backend_video_robustness_summary": backend_video_summary,
        "embodied_video_replicate_manifest": embodied_video_replicate_manifest,
        "embodied_video_replicate_summary": embodied_video_replicate_summary,
        "dandi_neural_manifest": dandi_neural_manifest,
        "dandi_neural_stimulus_summary": dandi_neural_stimulus_summary,
        "dandi_neural_file_summary": dandi_neural_file_summary,
        "dandi_video_alignment_manifest": dandi_video_alignment_manifest,
        "dandi_video_alignment_summary": dandi_video_alignment_summary,
        "dandi_motor_response_manifest": dandi_motor_response_manifest,
        "dandi_motor_response_summary": dandi_motor_response_summary,
        "dandi_identifiability_manifest": dandi_identifiability_manifest,
        "dandi_identifiability_collisions": dandi_identifiability_collisions,
        "dandi_identifiability_models": dandi_identifiability_models,
        "dandi_identifiability_signs": dandi_identifiability_signs,
        "dandi_projector_manifest": dandi_projector_manifest,
        "dandi_projector_models": dandi_projector_models,
        "dandi_projector_collisions": dandi_projector_collisions,
        "dandi_projector_clip_summary": dandi_projector_clip_summary,
        "dandi_projector_permutations": dandi_projector_permutations,
        "simzfish_behavior_manifest": simzfish_behavior_manifest,
        "simzfish_behavior_targets": simzfish_behavior_targets,
        "simzfish_behavior_profiles": simzfish_behavior_profiles,
        "simzfish_behavior_scores": simzfish_behavior_scores,
        "simzfish_behavior_nearest": simzfish_behavior_nearest,
        "simzfish_behavior_uncertainty_manifest": simzfish_behavior_uncertainty_manifest,
        "simzfish_behavior_uncertainty_summary": simzfish_behavior_uncertainty_summary,
        "realized_kinematics_manifest": realized_kinematics_manifest,
        "realized_kinematics_summary": realized_kinematics_summary,
        "external_kinematics_manifest": external_kinematics_manifest,
        "external_kinematics_summary": external_kinematics_summary,
        "external_kinematics_comparison": external_kinematics_comparison,
        "high_rate_tail_manifest": high_rate_tail_manifest,
        "high_rate_tail_summary": high_rate_tail_summary,
        "high_rate_tail_comparison": high_rate_tail_comparison,
        "high_rate_tail_anomalies": high_rate_tail_anomalies,
        "high_rate_replay_manifest": high_rate_replay_manifest,
        "high_rate_replay_summary": high_rate_replay_summary,
        "high_rate_replay_comparison": high_rate_replay_comparison,
        "high_rate_replay_anomalies": high_rate_replay_anomalies,
    }


def _backend_video_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("backend_video_robustness_manifest", {})
    summary = manifest.get("summary") if isinstance(manifest, dict) else None
    if not isinstance(summary, list) or not summary:
        summary = metrics.get("backend_video_robustness_summary", [])
    if not isinstance(summary, list):
        summary = []
    clip_count = int(_to_float(manifest.get("clip_count") if isinstance(manifest, dict) else 0, len(summary)))
    frame_count = int(_to_float(manifest.get("frame_count") if isinstance(manifest, dict) else 0, 0.0))
    event_rates = [_to_float(row.get("noncoast_event_frequency_hz")) for row in summary]
    quality = [_to_float(row.get("quality_event_fraction")) for row in summary]
    leaks = [_to_float(row.get("coast_nonzero_force_fraction")) for row in summary]
    inside = sum(1 for row in summary if _to_float(row.get("event_frequency_inside_published_p05_p95")) > 0.5)
    high_quality_risk = sum(1 for row in summary if _to_float(row.get("quality_event_fraction")) > 0.50)
    published = manifest.get("published_targets", {}) if isinstance(manifest, dict) else {}
    return {
        "clip_count": clip_count,
        "frame_count": frame_count,
        "inside_published_band": inside,
        "high_quality_risk": high_quality_risk,
        "mean_event_hz": float(np.mean(event_rates)) if event_rates else 0.0,
        "min_event_hz": float(np.min(event_rates)) if event_rates else 0.0,
        "max_event_hz": float(np.max(event_rates)) if event_rates else 0.0,
        "mean_quality_fraction": float(np.mean(quality)) if quality else 0.0,
        "max_quality_fraction": float(np.max(quality)) if quality else 0.0,
        "max_coast_force_leak": float(np.max(leaks)) if leaks else 0.0,
        "published_mean_hz": _to_float(published.get("published_bout_frequency_hz_mean")),
        "published_p05_hz": _to_float(published.get("published_bout_frequency_hz_p05")),
        "published_p95_hz": _to_float(published.get("published_bout_frequency_hz_p95")),
        "report": manifest.get("artifacts", {}).get("report", "") if isinstance(manifest, dict) else "",
        "summary": summary,
    }


def _embodied_video_replicate_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("embodied_video_replicate_manifest", {})
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list) or not records:
        records = metrics.get("embodied_video_replicate_summary", [])
    if not isinstance(records, list):
        records = []
    tick_pass = sum(1 for row in records if _to_float(row.get("tick_span")) >= 50000)
    nonpass = sum(int(_to_float(row.get("nonpass_anomaly_checks"))) for row in records)
    quality_counts = [_to_float(row.get("quality_event_count")) for row in records]
    event_rates = [_to_float(row.get("noncoast_event_frequency_hz")) for row in records]
    return {
        "replicate_count": len(records),
        "tick_pass_count": tick_pass,
        "nonpass_anomaly_checks": nonpass,
        "mean_event_hz": float(np.mean(event_rates)) if event_rates else 0.0,
        "min_event_hz": float(np.min(event_rates)) if event_rates else 0.0,
        "max_event_hz": float(np.max(event_rates)) if event_rates else 0.0,
        "min_quality_events": int(np.min(quality_counts)) if quality_counts else 0,
        "max_quality_events": int(np.max(quality_counts)) if quality_counts else 0,
        "records": records,
        "report": manifest.get("artifacts", {}).get("report", "") if isinstance(manifest, dict) else "",
    }


def _dandi_neural_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("dandi_neural_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    stimulus_rows = metrics.get("dandi_neural_stimulus_summary", [])
    file_rows = metrics.get("dandi_neural_file_summary", [])
    if not isinstance(stimulus_rows, list):
        stimulus_rows = []
    if not isinstance(file_rows, list):
        file_rows = []
    rate_values = [_to_float(row.get("acquisition_rate_hz"), float("nan")) for row in file_rows]
    rate_values = [value for value in rate_values if np.isfinite(value)]
    accepted_values = [_to_float(row.get("accepted_rois"), float("nan")) for row in file_rows]
    accepted_values = [value for value in accepted_values if np.isfinite(value)]
    top_rows = sorted(
        stimulus_rows,
        key=lambda row: _to_float(row.get("population_response_mean"), -1e99),
        reverse=True,
    )
    return {
        "parsed_files": int(_to_float(manifest.get("parsed_files"), len(file_rows))),
        "requested_files": int(_to_float(manifest.get("requested_files"), len(file_rows))),
        "parse_errors": int(_to_float(manifest.get("parse_errors"), 0)),
        "trial_rows": int(_to_float(manifest.get("trial_rows"), 0)),
        "unique_stimuli": int(_to_float(manifest.get("unique_stimuli"), len(stimulus_rows))),
        "total_accepted_rois": int(_to_float(manifest.get("total_accepted_rois"), 0)),
        "roi_rows": int(_to_float(manifest.get("roi_rows"), 0)),
        "top_stimulus": str(manifest.get("top_stimulus", top_rows[0].get("stimulus", "") if top_rows else "")),
        "top_population_response_mean": _to_float(
            manifest.get("top_population_response_mean"),
            _to_float(top_rows[0].get("population_response_mean")) if top_rows else 0.0,
        ),
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "rate_min_hz": float(np.min(rate_values)) if rate_values else 0.0,
        "rate_max_hz": float(np.max(rate_values)) if rate_values else 0.0,
        "accepted_rois_min": int(np.min(accepted_values)) if accepted_values else 0,
        "accepted_rois_max": int(np.max(accepted_values)) if accepted_values else 0,
        "top_rows": top_rows[:8],
        "report": manifest.get("report", ""),
    }


def _dandi_video_alignment_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("dandi_video_alignment_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("dandi_video_alignment_summary", [])
    if not isinstance(rows, list):
        rows = []
    sorted_rows = sorted(
        rows,
        key=lambda row: _to_float(row.get("quality_weighted_neural_plausibility"), -1e99),
        reverse=True,
    )
    top = sorted_rows[0] if sorted_rows else {}
    quality_flagged = int(_to_float(manifest.get("quality_flagged_frames"), 0))
    frame_rows = int(_to_float(manifest.get("frame_rows"), 0))
    return {
        "frame_rows": frame_rows,
        "clip_count": int(_to_float(manifest.get("clip_count"), len(rows))),
        "stimulus_count": int(_to_float(manifest.get("stimulus_count"), 0)),
        "quality_flagged_frames": quality_flagged,
        "quality_flagged_fraction": float(quality_flagged / frame_rows) if frame_rows else 0.0,
        "mean_dandi_neural_plausibility": _to_float(manifest.get("mean_dandi_neural_plausibility")),
        "mean_quality_weighted_plausibility": _to_float(manifest.get("mean_quality_weighted_plausibility")),
        "top_clip": str(manifest.get("top_clip", top.get("clip", ""))),
        "top_clip_quality_weighted_plausibility": _to_float(
            manifest.get("top_clip_quality_weighted_plausibility"),
            _to_float(top.get("quality_weighted_neural_plausibility")),
        ),
        "top_clip_top_stimulus": str(top.get("top_stimulus", "")),
        "top_clip_top_axis": str(top.get("top_primary_axis", "")),
        "top_clip_quality_weight": _to_float(top.get("mean_quality_weight")),
        "top_clip_event_hz": _to_float(top.get("backend_event_frequency_hz")),
        "rows": sorted_rows,
        "report": manifest.get("report", ""),
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
    }


def _dandi_motor_response_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("dandi_motor_response_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("dandi_motor_response_summary", [])
    if not isinstance(rows, list):
        rows = []
    by_dandi = sorted(rows, key=lambda row: _to_float(row.get("dandi_rank"), 1e9))
    by_motor = sorted(rows, key=lambda row: _to_float(row.get("motor_drive_index"), -1e99), reverse=True)
    top_dandi = by_dandi[0] if by_dandi else {}
    top_motor = by_motor[0] if by_motor else {}
    return {
        "stimulus_count": int(_to_float(manifest.get("stimulus_count"), len(rows))),
        "frames": int(_to_float(manifest.get("frames"), 0)),
        "sample_hz": _to_float(manifest.get("sample_hz")),
        "seconds": _to_float(manifest.get("seconds")),
        "dandi_response_motor_drive_corr": _to_float(manifest.get("dandi_response_motor_drive_corr")),
        "dandi_rank_motor_rank_corr": _to_float(manifest.get("dandi_rank_motor_rank_corr")),
        "top_dandi_stimulus": str(manifest.get("top_dandi_stimulus", top_dandi.get("stimulus", ""))),
        "top_motor_stimulus": str(manifest.get("top_motor_stimulus", top_motor.get("stimulus", ""))),
        "top_dandi_motor_rank": int(_to_float(top_dandi.get("motor_drive_rank"), 0)),
        "top_dandi_motor_drive": _to_float(top_dandi.get("motor_drive_index")),
        "top_motor_dandi_rank": int(_to_float(top_motor.get("dandi_rank"), 0)),
        "top_motor_drive": _to_float(top_motor.get("motor_drive_index")),
        "top_motor_dandi_response": _to_float(top_motor.get("dandi_population_response_mean")),
        "rows_by_dandi": by_dandi,
        "rows_by_motor": by_motor,
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "report": manifest.get("report", ""),
    }


def _dandi_identifiability_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("dandi_identifiability_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    collisions = metrics.get("dandi_identifiability_collisions", [])
    models = metrics.get("dandi_identifiability_models", [])
    signs = metrics.get("dandi_identifiability_signs", [])
    if not isinstance(collisions, list):
        collisions = []
    if not isinstance(models, list):
        models = []
    if not isinstance(signs, list):
        signs = []
    collision_rows = sorted(
        collisions,
        key=lambda row: _to_float(row.get("response_range"), -1e99),
        reverse=True,
    )
    model_rows = sorted(
        models,
        key=lambda row: _to_float(row.get("loo_r2"), -1e99),
        reverse=True,
    )
    best_sign = max(signs, key=lambda row: _to_float(row.get("axis_train_r2"), -1e99), default={})
    axes_model = next((row for row in models if row.get("feature_set") == "axes"), {})
    tokens_model = next((row for row in models if row.get("feature_set") == "tokens"), {})
    return {
        "stimulus_count": int(_to_float(manifest.get("stimulus_count"), 0)),
        "axis_group_count": int(_to_float(manifest.get("axis_group_count"), len(collisions))),
        "collision_group_count": int(_to_float(manifest.get("collision_group_count"), 0)),
        "max_collision_response_range": _to_float(manifest.get("max_collision_response_range")),
        "max_collision_labels": str(manifest.get("max_collision_labels", "")),
        "axes_train_r2": _to_float(manifest.get("axes_train_r2"), _to_float(axes_model.get("train_r2"))),
        "axes_loo_r2": _to_float(manifest.get("axes_loo_r2"), _to_float(axes_model.get("loo_r2"))),
        "tokens_train_r2": _to_float(manifest.get("tokens_train_r2"), _to_float(tokens_model.get("train_r2"))),
        "tokens_loo_r2": _to_float(manifest.get("tokens_loo_r2"), _to_float(tokens_model.get("loo_r2"))),
        "best_sign_axis_train_r2": _to_float(
            manifest.get("best_sign_axis_train_r2"),
            _to_float(best_sign.get("axis_train_r2")),
        ),
        "best_signs": (
            int(_to_float(best_sign.get("forward_sign"), 0)),
            int(_to_float(best_sign.get("side_sign"), 0)),
            int(_to_float(best_sign.get("expansion_sign"), 0)),
        ),
        "collision_rows": collision_rows,
        "model_rows": model_rows,
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "report": manifest.get("report", ""),
    }


def _dandi_projector_semantic_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("dandi_projector_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    models = metrics.get("dandi_projector_models", [])
    collisions = metrics.get("dandi_projector_collisions", [])
    clips = metrics.get("dandi_projector_clip_summary", [])
    permutations = metrics.get("dandi_projector_permutations", [])
    if not isinstance(models, list):
        models = []
    if not isinstance(collisions, list):
        collisions = []
    if not isinstance(clips, list):
        clips = []
    if not isinstance(permutations, list):
        permutations = []
    model_rows = sorted(models, key=lambda row: _to_float(row.get("loo_r2"), -1e99), reverse=True)
    best = model_rows[0] if model_rows else {}
    best_order = str(manifest.get("best_order_hypothesis", best.get("order_hypothesis", "")))
    baseline = next(
        (
            row
            for row in model_rows
            if row.get("order_hypothesis") == best_order and row.get("feature_set") == "coarse_axes"
        ),
        {},
    )
    zero_collision = [
        row
        for row in model_rows
        if row.get("order_hypothesis") == best_order and int(_to_float(row.get("collision_group_count"))) == 0
    ]
    best_zero = max(zero_collision, key=lambda row: _to_float(row.get("loo_r2"), -1e99), default={})
    clip_rows = sorted(
        clips,
        key=lambda row: _to_float(row.get("quality_weighted_neural_plausibility"), -1e99),
        reverse=True,
    )
    top_clip = clip_rows[0] if clip_rows else {}
    perm = next(
        (
            row
            for row in permutations
            if row.get("order_hypothesis") == best.get("order_hypothesis")
            and row.get("feature_set") == best.get("feature_set")
        ),
        {},
    )
    return {
        "stimulus_count": int(_to_float(manifest.get("stimulus_count"), 0)),
        "video_frame_count": int(_to_float(manifest.get("video_frame_count"), 0)),
        "clip_count": int(_to_float(manifest.get("clip_count"), len(clip_rows))),
        "best_order_hypothesis": best_order,
        "best_feature_set": str(manifest.get("best_feature_set", best.get("feature_set", ""))),
        "best_loo_r2": _to_float(manifest.get("best_loo_r2"), _to_float(best.get("loo_r2"))),
        "baseline_coarse_loo_r2": _to_float(
            manifest.get("baseline_coarse_loo_r2"),
            _to_float(baseline.get("loo_r2")),
        ),
        "best_collision_group_count": int(
            _to_float(manifest.get("best_collision_group_count"), _to_float(best.get("collision_group_count")))
        ),
        "baseline_collision_group_count": int(
            _to_float(
                manifest.get("baseline_collision_group_count"),
                _to_float(baseline.get("collision_group_count")),
            )
        ),
        "best_permutation_p_ge_observed": _to_float(
            manifest.get("best_permutation_p_ge_observed"),
            _to_float(perm.get("permutation_p_ge_observed")),
        ),
        "best_zero_collision_feature_set": str(best_zero.get("feature_set", "")),
        "best_zero_collision_loo_r2": _to_float(best_zero.get("loo_r2")),
        "top_clip": str(manifest.get("top_clip", top_clip.get("clip", ""))),
        "top_clip_rich_quality_weighted_plausibility": _to_float(
            manifest.get("top_clip_rich_quality_weighted_plausibility"),
            _to_float(top_clip.get("quality_weighted_neural_plausibility")),
        ),
        "top_clip_coarse_quality_weighted_plausibility": _to_float(
            manifest.get("top_clip_coarse_quality_weighted_plausibility"),
            _to_float(top_clip.get("coarse_quality_weighted_plausibility")),
        ),
        "mean_rich_quality_weighted_plausibility": _to_float(
            manifest.get("mean_rich_quality_weighted_plausibility")
        ),
        "mean_coarse_quality_weighted_plausibility": _to_float(
            manifest.get("mean_coarse_quality_weighted_plausibility")
        ),
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "model_rows": model_rows,
        "collision_rows": sorted(collisions, key=lambda row: _to_float(row.get("response_range"), -1e99), reverse=True),
        "clip_rows": clip_rows,
        "report": manifest.get("report", ""),
    }


def _simzfish_behavior_alignment_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("simzfish_behavior_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("simzfish_behavior_scores", [])
    if not isinstance(rows, list):
        rows = []
    targets = metrics.get("simzfish_behavior_targets", [])
    if not isinstance(targets, list):
        targets = []
    dandi_rows = sorted(
        [row for row in rows if row.get("source_kind") == "dandi_synthetic_motor"],
        key=lambda row: _to_float(row.get("alignment_distance"), -1e99),
        reverse=True,
    )
    video_rows = sorted(
        [row for row in rows if row.get("source_kind") == "selected_video_backend"],
        key=lambda row: _to_float(row.get("alignment_distance"), 1e99),
    )
    long_rows = sorted(
        [row for row in rows if row.get("source_kind") == "long_run_action_samples"],
        key=lambda row: _to_float(row.get("alignment_distance"), 1e99),
    )
    best_video = video_rows[0] if video_rows else {}
    best_long = long_rows[0] if long_rows else {}
    worst_dandi = dandi_rows[0] if dandi_rows else {}
    return {
        "target_condition_count": int(_to_float(manifest.get("target_condition_count"), len(targets))),
        "dandi_profile_count": int(_to_float(manifest.get("dandi_profile_count"), len(dandi_rows))),
        "video_profile_count": int(_to_float(manifest.get("video_profile_count"), len(video_rows))),
        "long_run_profile_count": int(_to_float(manifest.get("long_run_profile_count"), len(long_rows))),
        "mean_dandi_within_target_fraction": _to_float(manifest.get("mean_dandi_within_target_fraction")),
        "mean_video_within_target_fraction": _to_float(manifest.get("mean_video_within_target_fraction")),
        "mean_long_run_within_target_fraction": _to_float(manifest.get("mean_long_run_within_target_fraction")),
        "published_global_hz_mean": _to_float(
            (manifest.get("global_published_profile") or {}).get("bout_frequency_hz_mean")
        )
        if isinstance(manifest.get("global_published_profile"), dict)
        else 0.0,
        "published_global_hz_p05": _to_float(
            (manifest.get("global_published_profile") or {}).get("bout_frequency_hz_p05")
        )
        if isinstance(manifest.get("global_published_profile"), dict)
        else 0.0,
        "published_global_hz_p95": _to_float(
            (manifest.get("global_published_profile") or {}).get("bout_frequency_hz_p95")
        )
        if isinstance(manifest.get("global_published_profile"), dict)
        else 0.0,
        "best_video_label": str(manifest.get("best_video_label", best_video.get("label", ""))),
        "best_video_target": str(best_video.get("target_condition", "")),
        "best_video_distance": _to_float(manifest.get("best_video_distance"), _to_float(best_video.get("alignment_distance"))),
        "best_video_event_hz": _to_float(best_video.get("bout_frequency_hz")),
        "best_video_startle_fraction": _to_float(best_video.get("startle_fraction")),
        "best_long_label": str(best_long.get("label", "")),
        "best_long_target": str(best_long.get("target_condition", "")),
        "best_long_distance": _to_float(best_long.get("alignment_distance")),
        "best_long_event_hz": _to_float(best_long.get("bout_frequency_hz")),
        "worst_dandi_label": str(manifest.get("worst_dandi_label", worst_dandi.get("label", ""))),
        "worst_dandi_expected": str(worst_dandi.get("expected_condition", "")),
        "worst_dandi_distance": _to_float(
            manifest.get("worst_dandi_distance"), _to_float(worst_dandi.get("alignment_distance"))
        ),
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "dandi_rows": dandi_rows,
        "video_rows": video_rows,
        "long_rows": long_rows,
        "target_rows": targets,
        "report": manifest.get("report", ""),
    }


def _simzfish_behavior_uncertainty_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("simzfish_behavior_uncertainty_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("simzfish_behavior_uncertainty_summary", [])
    if not isinstance(rows, list):
        rows = []
    class_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("stability_class", ""))
        class_counts[key] = class_counts.get(key, 0) + 1
    selected = sorted(
        [row for row in rows if row.get("source_kind") == "selected_video_backend"],
        key=lambda row: _to_float(row.get("alignment_distance_p50"), 1e99),
    )
    long_rows = sorted(
        [row for row in rows if row.get("source_kind") == "long_run_action_samples"],
        key=lambda row: _to_float(row.get("alignment_distance_p50"), 1e99),
    )
    dandi = sorted(
        [row for row in rows if row.get("source_kind") == "dandi_synthetic_motor"],
        key=lambda row: _to_float(row.get("alignment_distance_p50"), -1e99),
        reverse=True,
    )
    best_selected = selected[0] if selected else {}
    best_long = long_rows[0] if long_rows else {}
    worst_dandi = dandi[0] if dandi else {}
    return {
        "profile_count": int(_to_float(manifest.get("profile_count"), len(rows))),
        "sample_count": int(_to_float(manifest.get("sample_count"), 0)),
        "iterations": int(_to_float(manifest.get("iterations"), 0)),
        "not_aligned_count": int(class_counts.get("not_aligned", 0)),
        "suggestive_count": int(class_counts.get("suggestive_only", 0)),
        "supported_count": int(class_counts.get("supported_with_bootstrap", 0)),
        "selected_count": len(selected),
        "dandi_count": len(dandi),
        "long_count": len(long_rows),
        "best_selected_label": str(best_selected.get("label", "")),
        "best_selected_target": str(best_selected.get("target_condition", "")),
        "best_selected_distance_p50": _to_float(best_selected.get("alignment_distance_p50")),
        "best_selected_distance_p025": _to_float(best_selected.get("alignment_distance_p025")),
        "best_selected_distance_p975": _to_float(best_selected.get("alignment_distance_p975")),
        "best_selected_inside_prob": _to_float(best_selected.get("mean_metric_inside_probability")),
        "best_selected_target_stability": _to_float(best_selected.get("nearest_target_stability_fraction")),
        "best_long_label": str(best_long.get("label", "")),
        "best_long_target": str(best_long.get("target_condition", "")),
        "best_long_event_hz_p50": _to_float(best_long.get("bout_frequency_hz_p50")),
        "best_long_event_hz_p025": _to_float(best_long.get("bout_frequency_hz_p025")),
        "best_long_event_hz_p975": _to_float(best_long.get("bout_frequency_hz_p975")),
        "best_long_distance_p50": _to_float(best_long.get("alignment_distance_p50")),
        "best_long_inside_prob": _to_float(best_long.get("mean_metric_inside_probability")),
        "worst_dandi_label": str(worst_dandi.get("label", "")),
        "worst_dandi_target": str(worst_dandi.get("target_condition", "")),
        "worst_dandi_distance_p50": _to_float(worst_dandi.get("alignment_distance_p50")),
        "worst_dandi_distance_p025": _to_float(worst_dandi.get("alignment_distance_p025")),
        "worst_dandi_distance_p975": _to_float(worst_dandi.get("alignment_distance_p975")),
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "selected_rows": selected,
        "long_rows": long_rows,
        "dandi_rows": dandi,
        "report": manifest.get("report", ""),
    }


def _realized_kinematics_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("realized_kinematics_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("realized_kinematics_summary", [])
    if not isinstance(rows, list):
        rows = []
    class_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("transfer_class", ""))
        class_counts[key] = class_counts.get(key, 0) + 1
    best_tail = max(
        rows,
        key=lambda row: abs(_to_float(row.get("tail_yaw_abs_mean_best_corr"), float("nan"))),
        default={},
    )
    best_muscle = max(
        rows,
        key=lambda row: abs(_to_float(row.get("muscle_sum_best_corr"), float("nan"))),
        default={},
    )
    best_direction = max(
        rows,
        key=lambda row: abs(_to_float(row.get("side_muscle_lr_bias_corr0"), float("nan"))),
        default={},
    )
    black_video = next((row for row in rows if row.get("run") == "black_rockfish_video"), {})
    verified_video = next((row for row in rows if row.get("run") == "verified_video_tenggol"), {})
    calcium_rows = [row for row in rows if "calcium" in str(row.get("run", ""))]
    video_rows = [row for row in rows if "video" in str(row.get("run", ""))]
    supported = int(class_counts.get("supported_command_to_body_transfer", 0))
    suggestive = int(class_counts.get("suggestive_transfer", 0))
    weak = int(class_counts.get("weak_or_sparse_transfer", 0))
    return {
        "run_count": int(_to_float(manifest.get("run_count"), len(rows))),
        "lag_scan_rows": int(_to_float(manifest.get("lag_scan_rows"), 0)),
        "event_response_rows": int(_to_float(manifest.get("event_response_rows"), 0)),
        "supported_count": supported,
        "suggestive_count": suggestive,
        "weak_count": weak,
        "best_tail_run": str(best_tail.get("run", "")),
        "best_tail_corr": _to_float(best_tail.get("tail_yaw_abs_mean_best_corr")),
        "best_tail_lag_s": _to_float(best_tail.get("tail_yaw_abs_mean_best_lag_s")),
        "best_muscle_run": str(best_muscle.get("run", "")),
        "best_muscle_corr": _to_float(best_muscle.get("muscle_sum_best_corr")),
        "best_muscle_lag_s": _to_float(best_muscle.get("muscle_sum_best_lag_s")),
        "best_direction_run": str(best_direction.get("run", "")),
        "best_direction_corr": _to_float(best_direction.get("side_muscle_lr_bias_corr0")),
        "black_video_tail_corr": _to_float(black_video.get("tail_yaw_abs_mean_best_corr")),
        "black_video_muscle_corr": _to_float(black_video.get("muscle_sum_best_corr")),
        "black_video_tail_event_delta": _to_float(black_video.get("tail_yaw_abs_mean_event_delta_mean")),
        "black_video_speed_mean": _to_float(black_video.get("speed_xy_mm_s_mean")),
        "black_video_z_span_p95": _to_float(black_video.get("body_z_span_mm_p95")),
        "verified_video_tail_corr": _to_float(verified_video.get("tail_yaw_abs_mean_best_corr")),
        "verified_video_muscle_corr": _to_float(verified_video.get("muscle_sum_best_corr")),
        "calcium_mean_event_count": float(np.mean([_to_float(row.get("event_count")) for row in calcium_rows]))
        if calcium_rows
        else 0.0,
        "video_mean_event_count": float(np.mean([_to_float(row.get("event_count")) for row in video_rows]))
        if video_rows
        else 0.0,
        "calcium_mean_tail_delta": float(
            np.mean([_to_float(row.get("tail_yaw_abs_mean_event_delta_mean")) for row in calcium_rows])
        )
        if calcium_rows
        else 0.0,
        "video_mean_tail_delta": float(
            np.mean([_to_float(row.get("tail_yaw_abs_mean_event_delta_mean")) for row in video_rows])
        )
        if video_rows
        else 0.0,
        "rows": rows,
        "plots": manifest.get("plots", []) if isinstance(manifest.get("plots"), list) else [],
        "report": manifest.get("report", ""),
    }


def _external_kinematics_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("external_kinematics_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("external_kinematics_summary", [])
    if not isinstance(rows, list):
        rows = []
    comparison = metrics.get("external_kinematics_comparison", [])
    if not isinstance(comparison, list):
        comparison = []
    status_counts: dict[str, int] = {}
    inside_counts: dict[str, int] = {}
    for row in comparison:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "inside_target":
            run = str(row.get("run", ""))
            inside_counts[run] = inside_counts.get(run, 0) + 1
    black_video = next((row for row in rows if row.get("run") == "black_rockfish_video"), {})
    tenggol_video = next((row for row in rows if row.get("run") == "verified_video_tenggol"), {})
    calcium_rows = [row for row in rows if "calcium" in str(row.get("run", ""))]
    video_rows = [row for row in rows if "video" in str(row.get("run", ""))]
    speed_below_count = sum(
        1
        for row in comparison
        if row.get("metric") == "speed_xy_mm_s_mean" and row.get("status") == "below_target"
    )
    speed_p95_below_count = sum(
        1
        for row in comparison
        if row.get("metric") == "speed_xy_mm_s_p95" and row.get("status") == "below_target"
    )
    event_inside_count = sum(
        1 for row in comparison if row.get("metric") == "event_frequency_hz" and row.get("status") == "inside_target"
    )
    command_tbf_inside_count = sum(
        1
        for row in comparison
        if row.get("metric") == "command_tail_frequency_hz_mean" and row.get("status") == "inside_target"
    )
    realized_tbf_unmeasurable_count = sum(
        1
        for row in comparison
        if row.get("metric") == "realized_tail_frequency_hz_observable"
        and row.get("status") == "not_measurable_telemetry_undersampled"
    )
    return {
        "run_count": int(_to_float(manifest.get("run_count"), len(rows))),
        "comparison_rows": int(_to_float(manifest.get("comparison_rows"), len(comparison))),
        "inside_target_count": int(status_counts.get("inside_target", 0)),
        "below_target_count": int(status_counts.get("below_target", 0)),
        "above_target_count": int(status_counts.get("above_target", 0)),
        "undersampled_count": int(status_counts.get("not_measurable_telemetry_undersampled", 0)),
        "event_inside_count": event_inside_count,
        "command_tbf_inside_count": command_tbf_inside_count,
        "speed_below_count": speed_below_count,
        "speed_p95_below_count": speed_p95_below_count,
        "realized_tbf_unmeasurable_count": realized_tbf_unmeasurable_count,
        "black_video_event_hz": _to_float(black_video.get("event_frequency_hz")),
        "black_video_bout_ms": _to_float(black_video.get("bout_duration_ms_p50")),
        "black_video_interbout_ms": _to_float(black_video.get("interbout_ms_p50")),
        "black_video_speed_mean": _to_float(black_video.get("speed_xy_mm_s_mean")),
        "black_video_speed_p95": _to_float(black_video.get("speed_xy_mm_s_p95")),
        "black_video_distance_event": _to_float(black_video.get("distance_per_event_mm_p50")),
        "black_video_heading_deg": _to_float(black_video.get("heading_delta_abs_deg_p50")),
        "black_video_tail_deg": _to_float(black_video.get("tail_yaw_abs_max_deg_p95")),
        "black_video_command_tbf": _to_float(black_video.get("command_tail_frequency_hz_mean")),
        "black_video_nyquist": _to_float(black_video.get("telemetry_nyquist_hz")),
        "tenggol_video_event_hz": _to_float(tenggol_video.get("event_frequency_hz")),
        "tenggol_video_speed_mean": _to_float(tenggol_video.get("speed_xy_mm_s_mean")),
        "video_mean_event_hz": float(np.mean([_to_float(row.get("event_frequency_hz")) for row in video_rows]))
        if video_rows
        else 0.0,
        "calcium_mean_event_hz": float(np.mean([_to_float(row.get("event_frequency_hz")) for row in calcium_rows]))
        if calcium_rows
        else 0.0,
        "max_telemetry_nyquist": float(np.max([_to_float(row.get("telemetry_nyquist_hz")) for row in rows]))
        if rows
        else 0.0,
        "rows": rows,
        "comparison": comparison,
        "plots": manifest.get("plots", []) if isinstance(manifest.get("plots"), list) else [],
        "report": manifest.get("report", ""),
    }


def _high_rate_tail_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("high_rate_tail_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("high_rate_tail_summary", [])
    if not isinstance(rows, list):
        rows = []
    comparison = metrics.get("high_rate_tail_comparison", [])
    if not isinstance(comparison, list):
        comparison = []
    anomalies = metrics.get("high_rate_tail_anomalies", [])
    if not isinstance(anomalies, list):
        anomalies = []
    status_counts: dict[str, int] = {}
    for row in comparison:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    failed_anomalies = [row for row in anomalies if str(row.get("status")) != "pass"]
    baseline = next((row for row in rows if str(row.get("run", "")).startswith("baseline")), {})
    sensorimotor = next((row for row in rows if str(row.get("run", "")).startswith("sensorimotor")), {})
    tbf_values = [_to_float(row.get("realized_tail_frequency_tbf_band_hz_median"), float("nan")) for row in rows]
    tbf_values = [value for value in tbf_values if np.isfinite(value)]
    sample_hz_values = [_to_float(row.get("sample_hz"), float("nan")) for row in rows]
    sample_hz_values = [value for value in sample_hz_values if np.isfinite(value)]
    speed_values = [_to_float(row.get("speed_xy_mm_s_mean"), float("nan")) for row in rows]
    speed_values = [value for value in speed_values if np.isfinite(value)]
    return {
        "run_count": int(_to_float(manifest.get("row_counts", {}).get("run_summary"), len(rows)))
        if isinstance(manifest.get("row_counts"), dict)
        else len(rows),
        "recording_count": len(manifest.get("recordings", [])) if isinstance(manifest.get("recordings"), list) else len(rows),
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "scalar_stats_rows": int(_to_float((manifest.get("row_counts") or {}).get("scalar_stats"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "matrix_channel_stats_rows": int(
            _to_float((manifest.get("row_counts") or {}).get("matrix_channel_stats"), 0)
        )
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "bout_events": int(_to_float((manifest.get("row_counts") or {}).get("bout_events"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "spectral_window_rows": int(_to_float((manifest.get("row_counts") or {}).get("spectral_window_rows"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "comparison_rows": len(comparison),
        "inside_target_count": int(status_counts.get("inside_target", 0)),
        "below_target_count": int(status_counts.get("below_target", 0)),
        "above_target_count": int(status_counts.get("above_target", 0)),
        "failed_anomaly_count": len(failed_anomalies),
        "sample_hz_min": float(np.min(sample_hz_values)) if sample_hz_values else 0.0,
        "sample_hz_max": float(np.max(sample_hz_values)) if sample_hz_values else 0.0,
        "tbf_median_min_hz": float(np.min(tbf_values)) if tbf_values else 0.0,
        "tbf_median_max_hz": float(np.max(tbf_values)) if tbf_values else 0.0,
        "mean_speed_min": float(np.min(speed_values)) if speed_values else 0.0,
        "mean_speed_max": float(np.max(speed_values)) if speed_values else 0.0,
        "baseline_tbf_hz": _to_float(baseline.get("realized_tail_frequency_tbf_band_hz_median")),
        "baseline_bouts": int(_to_float(baseline.get("bout_count"), 0)),
        "baseline_event_hz": _to_float(baseline.get("event_frequency_hz")),
        "sensorimotor_tbf_hz": _to_float(sensorimotor.get("realized_tail_frequency_tbf_band_hz_median")),
        "sensorimotor_bouts": int(_to_float(sensorimotor.get("bout_count"), 0)),
        "sensorimotor_event_hz": _to_float(sensorimotor.get("event_frequency_hz")),
        "rows": rows,
        "comparison": comparison,
        "anomalies": anomalies,
        "plots": manifest.get("plots", []) if isinstance(manifest.get("plots"), list) else [],
        "report": manifest.get("report", ""),
    }


def _high_rate_replay_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = metrics.get("high_rate_replay_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    rows = metrics.get("high_rate_replay_summary", [])
    if not isinstance(rows, list):
        rows = []
    comparison = metrics.get("high_rate_replay_comparison", [])
    if not isinstance(comparison, list):
        comparison = []
    anomalies = metrics.get("high_rate_replay_anomalies", [])
    if not isinstance(anomalies, list):
        anomalies = []
    status_counts: dict[str, int] = {}
    for row in comparison:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    failed_anomalies = [row for row in anomalies if str(row.get("status")) != "pass"]
    calcium = next((row for row in rows if str(row.get("run", "")).startswith("calcium")), {})
    video = next((row for row in rows if str(row.get("run", "")).startswith("video")), {})
    tbf_values = [_to_float(row.get("realized_tail_frequency_tbf_band_hz_median"), float("nan")) for row in rows]
    tbf_values = [value for value in tbf_values if np.isfinite(value)]
    sample_hz_values = [_to_float(row.get("sample_hz"), float("nan")) for row in rows]
    sample_hz_values = [value for value in sample_hz_values if np.isfinite(value)]
    action_forces = [_to_float(row.get("action_force_mean"), float("nan")) for row in rows]
    action_forces = [value for value in action_forces if np.isfinite(value)]
    summary_docs: dict[str, Any] = {}
    for row in rows:
        run = str(row.get("run", ""))
        path_text = str(row.get("summary_path", ""))
        path = Path(path_text) if path_text else Path("__missing__")
        summary_docs[run] = _read_json(path) if path.is_file() else {}
    calcium_doc = summary_docs.get(str(calcium.get("run", "")), {})
    video_doc = summary_docs.get(str(video.get("run", "")), {})
    calcium_vertical = int(
        _to_float(((calcium_doc.get("motifs") or {}).get("vertical_instability_events") or {}).get("count"), 0)
    )
    video_vertical = int(
        _to_float(((video_doc.get("motifs") or {}).get("vertical_instability_events") or {}).get("count"), 0)
    )
    return {
        "recording_count": len(manifest.get("recordings", [])) if isinstance(manifest.get("recordings"), list) else len(rows),
        "plot_count": len(manifest.get("plots", [])) if isinstance(manifest.get("plots"), list) else 0,
        "scalar_stats_rows": int(_to_float((manifest.get("row_counts") or {}).get("scalar_stats"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "matrix_channel_stats_rows": int(
            _to_float((manifest.get("row_counts") or {}).get("matrix_channel_stats"), 0)
        )
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "bout_events": int(_to_float((manifest.get("row_counts") or {}).get("bout_events"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "spectral_window_rows": int(_to_float((manifest.get("row_counts") or {}).get("spectral_window_rows"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "window_10s_rows": int(_to_float((manifest.get("row_counts") or {}).get("window_10s"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "window_30s_rows": int(_to_float((manifest.get("row_counts") or {}).get("window_30s"), 0))
        if isinstance(manifest.get("row_counts"), dict)
        else 0,
        "inside_target_count": int(status_counts.get("inside_target", 0)),
        "below_target_count": int(status_counts.get("below_target", 0)),
        "above_target_count": int(status_counts.get("above_target", 0)),
        "failed_anomaly_count": len(failed_anomalies),
        "sample_hz_min": float(np.min(sample_hz_values)) if sample_hz_values else 0.0,
        "sample_hz_max": float(np.max(sample_hz_values)) if sample_hz_values else 0.0,
        "tbf_median_min_hz": float(np.min(tbf_values)) if tbf_values else 0.0,
        "tbf_median_max_hz": float(np.max(tbf_values)) if tbf_values else 0.0,
        "action_force_mean_min": float(np.min(action_forces)) if action_forces else 0.0,
        "action_force_mean_max": float(np.max(action_forces)) if action_forces else 0.0,
        "calcium_tbf_hz": _to_float(calcium.get("realized_tail_frequency_tbf_band_hz_median")),
        "calcium_bouts": int(_to_float(calcium.get("bout_count"), 0)),
        "calcium_event_hz": _to_float(calcium.get("event_frequency_hz")),
        "calcium_action_force_mean": _to_float(calcium.get("action_force_mean")),
        "calcium_action_confidence_mean": _to_float(calcium.get("action_confidence_mean")),
        "calcium_source_frames": int(_to_float(calcium.get("source_frames_unique"), 0)),
        "calcium_vertical_instability_events": calcium_vertical,
        "video_tbf_hz": _to_float(video.get("realized_tail_frequency_tbf_band_hz_median")),
        "video_bouts": int(_to_float(video.get("bout_count"), 0)),
        "video_event_hz": _to_float(video.get("event_frequency_hz")),
        "video_action_force_mean": _to_float(video.get("action_force_mean")),
        "video_action_confidence_mean": _to_float(video.get("action_confidence_mean")),
        "video_source_frames": int(_to_float(video.get("source_frames_unique"), 0)),
        "video_source_loop_index_max": _to_float(video.get("source_loop_index_max")),
        "video_vertical_instability_events": video_vertical,
        "video_body_pitch_abs_p95_rad": _to_float(video.get("body_pitch_abs_p95_rad")),
        "video_z_span_p95_mm": _to_float(video.get("body_z_span_mm_p95")),
        "rows": rows,
        "comparison": comparison,
        "anomalies": anomalies,
        "summary_docs": summary_docs,
        "plots": manifest.get("plots", []) if isinstance(manifest.get("plots"), list) else [],
        "report": manifest.get("report", ""),
    }


def _artifact(path: Path) -> str:
    return str(path.resolve())


def _claim_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    direct = metrics["direct_metrics"]
    kick_acc = direct.get("kick_accuracy", 0.0)
    kick_f1 = direct.get("direct_extra", {}).get("kick_f1", 0.0)
    force_r2 = direct.get("force_r2", 0.0)
    side_active = direct.get("direct_extra", {}).get("side_active_accuracy", 0.0)
    video_events = metrics["video_quality_event_counts"]
    runs = {row["name"]: row for row in metrics["forensics"].get("runs", [])}
    calcium = runs.get("calcium_all", {})
    video = runs.get("video_commons_tenggol_underwater", {})
    checks = metrics["synthesis"].get("checks", [])
    video_checks = [c for c in checks if c.get("run") == "video_commons_tenggol_underwater"]
    video_nonpass = [c for c in video_checks if c.get("level") != "pass"]
    video_curl = next((c for c in video_checks if c.get("check") == "sustained_curl"), {})
    fidelity = metrics.get("simzfish_fidelity_manifest", {})
    fidelity_data = metrics.get("simzfish_fidelity_data", {})
    calibration = metrics.get("simzfish_calibration_manifest", {})
    comparison = metrics.get("simzfish_calibration_comparison", [])
    controller = metrics.get("simzfish_controller_manifest", {})
    robot_rows = metrics.get("simzfish_controller_robot", [])
    omr_rows = metrics.get("simzfish_controller_omr", [])
    leaky_rows = metrics.get("simzfish_controller_leaky", [])
    backend_video = _backend_video_rollup(metrics)
    embodied_replicates = _embodied_video_replicate_rollup(metrics)
    dandi_neural = _dandi_neural_rollup(metrics)
    dandi_video = _dandi_video_alignment_rollup(metrics)
    dandi_motor = _dandi_motor_response_rollup(metrics)
    dandi_ident = _dandi_identifiability_rollup(metrics)
    dandi_projector = _dandi_projector_semantic_rollup(metrics)
    behavior_alignment = _simzfish_behavior_alignment_rollup(metrics)
    behavior_uncertainty = _simzfish_behavior_uncertainty_rollup(metrics)
    realized = _realized_kinematics_rollup(metrics)
    external_kinematics = _external_kinematics_rollup(metrics)
    high_rate_tail = _high_rate_tail_rollup(metrics)
    high_rate_replay = _high_rate_replay_rollup(metrics)
    return [
        {
            "claim_id": "C01",
            "claim": "The calcium branch is grounded in public ZAPBench calcium traces and raw tail motor ephys labels.",
            "paper_or_dataset_support": "ZAPBench paper/landing/dataset README; raw 10-channel stimulus/ephys file; local direct_ephys_labels.npz.",
            "implementation_evidence": "analysis/zapbench_ephys_action_decoder.py; analysis/zapbench_calcium_replay.py; lab/sim_runtime.py:_advance_calcium_replay; action_latent.py:from_zapbench_ephys.",
            "quantitative_evidence": f"direct ephys decoder: kick accuracy {kick_acc:.4f}, kick F1 {kick_f1:.4f}, force R2 {force_r2:.4f}, side-active accuracy {side_active:.4f}.",
            "evidence_strength": "direct for kick/force; weak for turn side",
            "status": "pass_with_limits",
            "limitation": "Ephys is fictive motor nerve activity; this is not free-swimming kinematics or muscle force calibration.",
            "next_required_evidence": "Validate decoded motor commands against independently measured tail kinematics.",
        },
        {
            "claim_id": "C02",
            "claim": "The calcium replay produces stable embodied MuJoCo behavior over at least 50k ticks.",
            "paper_or_dataset_support": "Runtime validation, not a direct paper claim.",
            "implementation_evidence": "analysis/comprehensive_activity_study.py; analysis/activity_study_synthesis.py; analysis/full_activity_forensics.py.",
            "quantitative_evidence": f"{calcium.get('tick_span', 0)} ticks, {calcium.get('sim_seconds', 0):.3f}s, {calcium.get('frames', 0)} WS frames, all synthesis anomaly checks pass.",
            "evidence_strength": "direct runtime evidence",
            "status": "pass",
            "limitation": "Stable replay does not prove biological equivalence.",
            "next_required_evidence": "Repeat across all ZAPBench conditions and randomized seeds.",
        },
        {
            "claim_id": "C03",
            "claim": "The video branch uses backend optical flow, camera-motion compensation, and simZFish-style retinal OMR logic as the primary action path.",
            "paper_or_dataset_support": "Z-Robot/simZFish paper and public repository support retinal direction-selective OMR as a relevant mechanism.",
            "implementation_evidence": "lab/video_pipeline.py:55-134; simzfish_omr.py:70-268; diagnostics primary_action_model=simzfish_retina_omr.",
            "quantitative_evidence": f"{video.get('actions', 0)} backend-decoded frames; video quality events: {video_events}.",
            "evidence_strength": "partial: mechanism-inspired implementation, not a verified exact port",
            "status": "partial",
            "limitation": "Python adapter approximates simZFish motifs; deterministic and compiled-source reference tests now exist, but current action targets and side dynamics still differ from the original controller surfaces.",
            "next_required_evidence": "Replace or fit the runtime adapter against the compiled OMR.c/LeakyIntegrator.c reference and published behavior tables.",
        },
        {
            "claim_id": "C04",
            "claim": "The video branch is not ZAPBench-primary; ZAPBench mapping is an auxiliary diagnostic covariate adapter.",
            "paper_or_dataset_support": "ZAPBench provides controlled stimulus covariates, not arbitrary underwater POV videos.",
            "implementation_evidence": "lab/video_pipeline.py:109-133 sets zapbench_adapter_role=auxiliary_estimated_covariates_not_primary_action.",
            "quantitative_evidence": "Video source timeline includes auxiliary ZAPBench distances; mean distance approximately 11 in forensic action stats.",
            "evidence_strength": "direct implementation evidence",
            "status": "pass",
            "limitation": "Auxiliary nearest-neighbor distances should not be used as proof of ZAPBench visual stimulus equivalence.",
            "next_required_evidence": "Only claim video-to-ZAPBench alignment if exact stimulus movies or paired videos become available.",
        },
        {
            "claim_id": "C05",
            "claim": "Both paper branches converge through one common motor-action latent and the same MuJoCo tail-target pathway.",
            "paper_or_dataset_support": "Engineering integration claim.",
            "implementation_evidence": "action_latent.py:28-213; environment.py:333; sensors.py:53-56; neuron_mapping.py:354-372 and 530.",
            "quantitative_evidence": "Both forensics runs include tail_yaw_rad, joint, muscle, PAULA, free-energy, and action-source streams.",
            "evidence_strength": "direct code and runtime evidence",
            "status": "pass",
            "limitation": "Common latent is an engineering bridge, not a published biological latent.",
            "next_required_evidence": "Fit latent-to-tail parameters against measured larval tail kinematics.",
        },
        {
            "claim_id": "C06",
            "claim": "The current video replay is long-run stable at the gross-anomaly level over at least 50k ticks.",
            "paper_or_dataset_support": "Runtime validation, not a direct paper claim.",
            "implementation_evidence": "synthesis/anomaly_checks.csv; forensics/scalar_extrema_and_stats.csv; forensics/video_quality_events.csv.",
            "quantitative_evidence": (
                f"{video.get('tick_span', 0)} ticks; sustained_curl {video_curl.get('level', 'unknown')} "
                f"with p95 local bend {video_curl.get('value', 'n/a')}; {sum(video_events.values())} video quality events."
            ),
            "evidence_strength": "direct runtime evidence",
            "status": "pass" if not video_nonpass else "warn",
            "limitation": "High compression/noise/shake intervals can overdrive OMR/startle and curl the body.",
            "next_required_evidence": "Repeat across additional videos and fit/gate the video adapter against published simZFish/Z-Robot target surfaces.",
        },
        {
            "claim_id": "C07",
            "claim": "Public resources do not yet prove natural video -> whole-brain calcium -> free-swimming movement as one supervised chain.",
            "paper_or_dataset_support": "ZAPBench has VR covariates/calcium/ephys; Z-Robot has OMR sim/robot/calcium resources; no single paired chain exists in public cache.",
            "implementation_evidence": "two_pipeline_chain_handoff_20260516.md and z_robot_resource_audit_20260516.md preserve this boundary.",
            "quantitative_evidence": "No artifact with paired natural video, ZAPBench whole-brain calcium, and measured tail/body kinematics exists in current cache.",
            "evidence_strength": "missing evidence",
            "status": "not_supported",
            "limitation": "This is the primary barrier to a top-tier biological digital-twin claim.",
            "next_required_evidence": "Acquire or locate a paired dataset, or run new experiments with synchronized video, whole-brain calcium, ephys/tail kinematics.",
        },
        {
            "claim_id": "C08",
            "claim": "Z-Robot/simZFish data are available for future calibration of video-to-motion and neural response motifs.",
            "paper_or_dataset_support": "Z-Robot paper, simZFish repository, DANDI 001076.",
            "implementation_evidence": "analysis/cache/z_robot manifests and downloaded NWB sample.",
            "quantitative_evidence": f"simZFish repository items {metrics['simzfish_repo_items']}; data files {metrics['simzfish_data_files']} ({metrics['simzfish_data_bytes']} bytes); DANDI assets {metrics['dandi_asset_files']} ({metrics['dandi_asset_bytes']} bytes).",
            "evidence_strength": "direct resource availability; partial integration",
            "status": "partial",
            "limitation": "These resources are cached/audited but not yet used to fit current MuJoCo behavior parameters.",
            "next_required_evidence": "Fit current video branch against simZFish/ZBot OMR tables and DANDI calcium response classes.",
        },
        {
            "claim_id": "C09",
            "claim": "The current PAULA layer is observable but not a validated zebrafish connectome-to-muscle controller.",
            "paper_or_dataset_support": "ZAPBench connectome is future/unreleased; Z-Robot uses rate-coded artificial neurons, not PAULA.",
            "implementation_evidence": "forensics matrix streams record PAULA S/R/B/Tref/firing/M0/M1; neuron_mapping.py blends external tail targets.",
            "quantitative_evidence": "Forensics record PAULA vectors for 104 neurons in both long runs; neuron_m0/m1 vectors are zero while aggregate neuromod changes are logged.",
            "evidence_strength": "direct observability, missing biological validation",
            "status": "partial",
            "limitation": "PAULA dynamics are diagnostic/substrate here, not paper-validated zebrafish circuit dynamics.",
            "next_required_evidence": "Wire a validated zebrafish circuit/connectome or experimentally fitted PAULA network before claiming PAULA biological fidelity.",
        },
        {
            "claim_id": "C10",
            "claim": "The current video branch is simZFish-inspired but not an exact Z-Robot/simZFish controller port.",
            "paper_or_dataset_support": "Public simZFish C/Webots controller and Data_Liu_simZFish_2025 resources are cached and auditable.",
            "implementation_evidence": "analysis/simzfish_fidelity_audit.py; SIMZFISH_FIDELITY_AUDIT.md; component_fidelity_scorecard.csv.",
            "quantitative_evidence": f"mean component fidelity {float(fidelity.get('mean_score', 0.0)):.4f}; {fidelity_data.get('total_files', 0)} cached simZFish files; {fidelity_data.get('mat_nested_numeric_cells', 0)} nested numeric MAT behavior cells.",
            "evidence_strength": "direct implementation-fidelity audit",
            "status": "partial",
            "limitation": "Image.c direct-port tests and compiled OMR.c/LeakyIntegrator.c reference tests now exist, but the live Python/MuJoCo adapter is not equivalent and calibration fitting is still missing.",
            "next_required_evidence": "Fit or replace the live adapter against the compiled reference and published behavior/neural tables.",
        },
        {
            "claim_id": "C11",
            "claim": "The public simZFish/Z-Robot resources have now been converted into concrete calibration targets for the current lab.",
            "paper_or_dataset_support": "Data_Liu_simZFish_2025 spreadsheets/MAT files and ZBot statistics.",
            "implementation_evidence": "analysis/simzfish_calibration_target_atlas.py; SIMZFISH_CALIBRATION_TARGET_ATLAS.md.",
            "quantitative_evidence": f"{calibration.get('locomotion_targets', 0)} locomotion targets; {calibration.get('neural_targets', 0)} neural targets; {calibration.get('rheotaxis_targets', 0)} rheotaxis targets; {calibration.get('zbot_stats', 0)} ZBot stats; {calibration.get('mat_numeric_cells', 0)} MAT numeric cells; comparison rows {len(comparison)}.",
            "evidence_strength": "direct dataset extraction and current-run comparison",
            "status": "pass_for_target_extraction",
            "limitation": "This establishes calibration targets; it does not yet fit MuJoCo/video-branch parameters.",
            "next_required_evidence": "Run parameter fitting and repeat >=50k validation against these extracted targets.",
        },
        {
            "claim_id": "C12",
            "claim": "A deterministic simZFish controller-regression layer now separates exact direct ports from current-adapter mismatches.",
            "paper_or_dataset_support": "Public simZFish source files Image.c, Robot.c, OMR.c, LeakyIntegrator.c, and params.csv.",
            "implementation_evidence": "analysis/simzfish_controller_regression.py; SIMZFISH_CONTROLLER_REGRESSION.md; image_c_retina_regression.csv; robot_c_motor_regression.csv; omr_c_compiled_regression.csv; leaky_integrator_c_compiled_regression.csv.",
            "quantitative_evidence": (
                f"Image.c OFF exact {controller.get('image_c_off_exact')}; Image.c DSC exact {controller.get('image_c_dsc_exact')}; "
                f"retina winner matches L/R {controller.get('retina_left_winner_matches')}/{controller.get('retina_right_winner_matches')}; "
                f"Robot.c current-vs-direct mean corr {float(controller.get('robot_target_corr_mean', 0.0)):.4f}, "
                f"mean RMSE {float(controller.get('robot_target_rmse_mean', 0.0)):.4f}; robot cases {len(robot_rows)}; "
                f"compiled OMR rows {controller.get('omr_c_rows')} all ok {controller.get('omr_c_all_results_ok')}; "
                f"current-vs-compiled OMR side matches {controller.get('omr_current_side_sign_matches')}/{len(omr_rows)}; "
                f"compiled leaky rows {controller.get('leaky_c_rows')} with total bouts {controller.get('leaky_c_total_bouts')}."
            ),
            "evidence_strength": "direct deterministic regression plus compiled-source regression",
            "status": "pass_for_direct_ports_warn_for_current_adapter",
            "limitation": "The regression harness confirms source-level reference surfaces, but also shows the current Python/MuJoCo adapter is not equivalent to direct Robot.c targets or compiled OMR.c side dynamics.",
            "next_required_evidence": "Use this harness as a fitting/replacement target before claiming exact simZFish equivalence.",
        },
        {
            "claim_id": "C13",
            "claim": "The selected-video backend extractor is robust against the prior coast-force-leak failure across the cached public video set.",
            "paper_or_dataset_support": "Z-Robot/simZFish supports OMR-style video-to-bout motifs; this is an implementation robustness claim.",
            "implementation_evidence": "analysis/backend_video_robustness_audit.py; BACKEND_VIDEO_ROBUSTNESS_AUDIT.md; backend_video_robustness_summary.csv; backend_video_frame_metrics.csv.",
            "quantitative_evidence": (
                f"{backend_video['clip_count']} selected clips, {backend_video['frame_count']} backend-decoded frames, "
                f"mean event rate {backend_video['mean_event_hz']:.4f} Hz versus published mean "
                f"{backend_video['published_mean_hz']:.4f} Hz, {backend_video['inside_published_band']}/{backend_video['clip_count']} "
                f"clips inside the published p05-p95 event-rate band, max coast-force leak fraction "
                f"{backend_video['max_coast_force_leak']:.6f}, mean quality-event fraction {backend_video['mean_quality_fraction']:.4f}."
            ),
            "evidence_strength": "direct backend-extractor regression over all cached selected videos",
            "status": "pass_for_coast_force_regression_warn_for_video_quality",
            "limitation": "This isolates the video decoder/action layer and does not prove embodied MuJoCo stability for every clip; many clips are high-quality-risk stimuli.",
            "next_required_evidence": "Run >=50k embodied recordings for the low-risk clips and exclude or down-weight clips with high quality-event fractions.",
        },
        {
            "claim_id": "C14",
            "claim": "Selected-video embodied stability has now been replicated across two independent 50k+ MuJoCo recordings.",
            "paper_or_dataset_support": "Runtime validation against the implementation's shared MuJoCo action path; not a direct paper claim.",
            "implementation_evidence": "20260603_verified_50k; 20260603_black_rockfish_50k; analysis/embodied_video_replicate_comparison.py.",
            "quantitative_evidence": (
                f"{embodied_replicates['replicate_count']} embodied video replicates, "
                f"{embodied_replicates['tick_pass_count']}/{embodied_replicates['replicate_count']} pass >=50k ticks, "
                f"{embodied_replicates['nonpass_anomaly_checks']} non-pass anomaly checks, event-rate range "
                f"{embodied_replicates['min_event_hz']:.4f}-{embodied_replicates['max_event_hz']:.4f} Hz, "
                f"quality-event count range {embodied_replicates['min_quality_events']}-{embodied_replicates['max_quality_events']}."
            ),
            "evidence_strength": "direct replicated runtime evidence for stability across selected videos",
            "status": "pass_for_stability_replication_warn_for_biological_identity",
            "limitation": "Two stable selected-video runs are still not a paired natural-video/calcium/ephys/kinematics validation set.",
            "next_required_evidence": "Scale to all low-risk clips, run repeated seeds, and compare against measured zebrafish tail kinematics.",
        },
        {
            "claim_id": "C15",
            "claim": "Public Z-Robot/DANDI OMR calcium data have been converted into a class-level neural target atlas for the current lab.",
            "paper_or_dataset_support": "DANDI 001076 OMR Robot CaImaging; Z-Robot/simZFish OMR calcium-imaging resource.",
            "implementation_evidence": "analysis/dandi_omr_neural_validation.py; DANDI_OMR_NEURAL_VALIDATION.md; stimulus_response_summary.csv; roi_selectivity_summary.csv.",
            "quantitative_evidence": (
                f"{dandi_neural['parsed_files']}/{dandi_neural['requested_files']} NWB files parsed, "
                f"{dandi_neural['trial_rows']} trials, {dandi_neural['unique_stimuli']} stimulus classes, "
                f"{dandi_neural['total_accepted_rois']} accepted ROIs, top population-response stimulus "
                f"{dandi_neural['top_stimulus']} ({dandi_neural['top_population_response_mean']:.6g}), "
                f"{dandi_neural['plot_count']} validation plots."
            ),
            "evidence_strength": "direct neural-resource extraction; class-level target only",
            "status": "pass_for_neural_target_extraction",
            "limitation": "DANDI files provide OMR stimulus labels and calcium responses, not arbitrary selected-video frames, synchronized ephys/muscle labels, or free-swimming kinematics.",
            "next_required_evidence": "Compare current video-branch inferred OMR covariates and internal retinal state trajectories against this neural target atlas, then fit movement separately against tail/ephys/kinematics.",
        },
        {
            "claim_id": "C16",
            "claim": "The selected-video backend frames have been scored against the DANDI OMR calcium target atlas, exposing motif-level neural plausibility and mismatch.",
            "paper_or_dataset_support": "DANDI 001076 provides OMR stimulus-class calcium responses; selected videos provide arbitrary underwater POV frame features through the current backend extractor.",
            "implementation_evidence": "analysis/dandi_video_omr_alignment.py; DANDI_VIDEO_OMR_ALIGNMENT.md; video_frame_neural_alignment.csv; clip_neural_alignment_summary.csv.",
            "quantitative_evidence": (
                f"{dandi_video['frame_rows']} backend frames across {dandi_video['clip_count']} clips scored against "
                f"{dandi_video['stimulus_count']} DANDI stimuli; mean neural plausibility "
                f"{dandi_video['mean_dandi_neural_plausibility']:.4f}, quality-weighted "
                f"{dandi_video['mean_quality_weighted_plausibility']:.4f}; quality-flagged frames "
                f"{dandi_video['quality_flagged_frames']} ({dandi_video['quality_flagged_fraction']:.3f}); top clip "
                f"{dandi_video['top_clip']} with quality-weighted score "
                f"{dandi_video['top_clip_quality_weighted_plausibility']:.4f}, top mapped stimulus "
                f"{dandi_video['top_clip_top_stimulus']}."
            ),
            "evidence_strength": "direct audit of current selected-video frames against DANDI class-level neural motifs",
            "status": "pass_for_alignment_audit_warn_for_stimulus_identity",
            "limitation": "Most selected-video frames map to mixed OMR classes rather than the strongest DANDI calcium-response class; the projection is coarse and not a frame-exact calcium prediction.",
            "next_required_evidence": "Use projector-like stimuli or calibrated two-eye video transforms, then compare inferred OMR trajectories against DANDI trial-aligned calcium classes and simZFish controller outputs.",
        },
        {
            "claim_id": "C17",
            "claim": "The current simZFish-inspired motor adapter has been stress-tested against calibrated DANDI OMR stimulus classes, and its motor ranking does not yet align strongly with the DANDI calcium-response ranking.",
            "paper_or_dataset_support": "DANDI 001076 OMR stimulus classes and calcium-response atlas; current lab simZFish OMR adapter.",
            "implementation_evidence": "analysis/dandi_omr_motor_response_audit.py; DANDI_OMR_MOTOR_RESPONSE_AUDIT.md; stimulus_motor_response_summary.csv.",
            "quantitative_evidence": (
                f"{dandi_motor['stimulus_count']} stimulus classes, {dandi_motor['frames']} synthetic projector-like frames; "
                f"DANDI response versus motor-drive correlation {dandi_motor['dandi_response_motor_drive_corr']:.4f}, "
                f"rank correlation {dandi_motor['dandi_rank_motor_rank_corr']:.4f}; top DANDI stimulus "
                f"{dandi_motor['top_dandi_stimulus']} has motor rank {dandi_motor['top_dandi_motor_rank']}; "
                f"top motor stimulus {dandi_motor['top_motor_stimulus']} has DANDI rank {dandi_motor['top_motor_dandi_rank']}."
            ),
            "evidence_strength": "direct calibrated-stimulus adapter audit",
            "status": "warn_current_motor_adapter_not_dandi_neural_rank_aligned",
            "limitation": "Synthetic projector-like stimuli are not the actual DANDI movies and this audit does not include full MuJoCo body kinematics, but the weak rank/value agreement is a real implementation warning.",
            "next_required_evidence": "Fit or replace the current OMR-to-motor adapter against DANDI stimulus-class neural targets plus simZFish/ZBot motor targets, then rerun long MuJoCo validation.",
        },
        {
            "claim_id": "C18",
            "claim": "The current coarse video-to-OMR projection is not identifiable enough to support class-level DANDI calcium claims.",
            "paper_or_dataset_support": "DANDI 001076 provides richer OMR stimulus labels than the current forward/side/expansion projection preserves.",
            "implementation_evidence": "analysis/dandi_stimulus_identifiability_audit.py; DANDI_STIMULUS_IDENTIFIABILITY_AUDIT.md; coarse_axis_collision_groups.csv; stimulus_feature_model_scores.csv; axis_sign_sensitivity.csv.",
            "quantitative_evidence": (
                f"{dandi_ident['stimulus_count']} DANDI stimuli collapse to {dandi_ident['axis_group_count']} coarse-axis groups with "
                f"{dandi_ident['collision_group_count']} collision groups; worst collision `{dandi_ident['max_collision_labels']}` has "
                f"response range {dandi_ident['max_collision_response_range']:.4f}; axis-only train R2 "
                f"{dandi_ident['axes_train_r2']:.4f}, LOO R2 {dandi_ident['axes_loo_r2']:.4f}; best sign-flip axis R2 "
                f"{dandi_ident['best_sign_axis_train_r2']:.4f}."
            ),
            "evidence_strength": "direct negative validation of current stimulus representation",
            "status": "warn_current_video_omr_projection_not_identifiable",
            "limitation": "This audit uses labels and derived axes, not the original projector movies; however, it proves the current coarse representation discards stimulus semantics needed for class-level DANDI alignment.",
            "next_required_evidence": "Implement a richer two-eye/projector-like visual representation before fitting or judging video-to-calcium alignment against DANDI classes.",
        },
        {
            "claim_id": "C19",
            "claim": "A richer two-eye/projector semantic bridge improves anatomical traceability and selected-video scoring, but still does not validate class-level DANDI calcium prediction.",
            "paper_or_dataset_support": "Z-Robot/simZFish paper emphasizes eye specificity, lower-posterior retinal motion, monocular/binocular/conflicting stimuli, and DANDI 001076 OMR calcium labels.",
            "implementation_evidence": "analysis/dandi_projector_semantic_bridge_audit.py; DANDI_PROJECTOR_SEMANTIC_BRIDGE_AUDIT.md; semantic_feature_model_scores.csv; semantic_feature_collision_groups.csv; rich_video_clip_alignment_summary.csv.",
            "quantitative_evidence": (
                f"{dandi_projector['stimulus_count']} DANDI stimuli and {dandi_projector['video_frame_count']} video frames; "
                f"best semantic model {dandi_projector['best_order_hypothesis']}/{dandi_projector['best_feature_set']} "
                f"LOO R2 {dandi_projector['best_loo_r2']:.4f} versus coarse {dandi_projector['baseline_coarse_loo_r2']:.4f}, "
                f"permutation p {dandi_projector['best_permutation_p_ge_observed']:.4f}; rich mean video plausibility "
                f"{dandi_projector['mean_rich_quality_weighted_plausibility']:.4f} versus coarse "
                f"{dandi_projector['mean_coarse_quality_weighted_plausibility']:.4f}; top clip "
                f"{dandi_projector['top_clip']} rich {dandi_projector['top_clip_rich_quality_weighted_plausibility']:.4f} "
                f"versus coarse {dandi_projector['top_clip_coarse_quality_weighted_plausibility']:.4f}."
            ),
            "evidence_strength": "direct representation audit; negative predictive-validation result",
            "status": "warn_traceability_improved_prediction_not_validated",
            "limitation": "The original DANDI projector movies are not present in the NWB cache; ambiguous two-token labels require eye-order hypotheses; current selected-video matching uses backend retinal counters rather than calibrated projector geometry.",
            "next_required_evidence": "Obtain or reconstruct exact projector stimuli, fit the semantic/retinal bridge with independent validation, and then rerun DANDI calcium plus motor-adapter audits.",
        },
        {
            "claim_id": "C20",
            "claim": "Current motor outputs can now be scored against published simZFish/Z-Robot behavior target envelopes, and the result shows behavior-target fitting is still missing.",
            "paper_or_dataset_support": "Data_Liu_simZFish_2025 locomotion target tables extracted from the public simZFish/Z-Robot resource cache.",
            "implementation_evidence": "analysis/simzfish_behavior_target_alignment_audit.py; SIMZFISH_BEHAVIOR_TARGET_ALIGNMENT_AUDIT.md; behavior_target_alignment_scores.csv; behavior_nearest_target_scores.csv.",
            "quantitative_evidence": (
                f"{behavior_alignment['target_condition_count']} published behavior-condition profiles; "
                f"{behavior_alignment['dandi_profile_count']} DANDI synthetic motor profiles mean within-target fraction "
                f"{behavior_alignment['mean_dandi_within_target_fraction']:.4f}; "
                f"{behavior_alignment['video_profile_count']} selected-video profiles mean within-target fraction "
                f"{behavior_alignment['mean_video_within_target_fraction']:.4f}; "
                f"{behavior_alignment['long_run_profile_count']} >=50k long-run profiles mean within-target fraction "
                f"{behavior_alignment['mean_long_run_within_target_fraction']:.4f}; best selected video "
                f"{behavior_alignment['best_video_label']} -> {behavior_alignment['best_video_target']} distance "
                f"{behavior_alignment['best_video_distance']:.4f}; worst mapped DANDI stimulus "
                f"{behavior_alignment['worst_dandi_label']} -> {behavior_alignment['worst_dandi_expected']} distance "
                f"{behavior_alignment['worst_dandi_distance']:.4f}."
            ),
            "evidence_strength": "direct current-output versus published-target audit",
            "status": "warn_behavior_target_fitting_missing",
            "limitation": "Selected-video nearest-condition matching is only a plausibility diagnostic for arbitrary videos, and calcium long-run action rows are replay-state samples rather than dense motor recordings.",
            "next_required_evidence": "Fit the motor adapter against published condition profiles and compiled simZFish controller surfaces, then repeat >=50k embodied validation.",
        },
        {
            "claim_id": "C21",
            "claim": "Bootstrap uncertainty analysis confirms that the published behavior-target alignment gap is robust under temporal resampling.",
            "paper_or_dataset_support": "Same published simZFish/Z-Robot behavior-condition envelopes as C20; statistical robustness is an implementation-validation layer.",
            "implementation_evidence": "analysis/simzfish_behavior_uncertainty_audit.py; SIMZFISH_BEHAVIOR_UNCERTAINTY_AUDIT.md; bootstrap_behavior_alignment_summary.csv; bootstrap_behavior_alignment_samples.csv.",
            "quantitative_evidence": (
                f"{behavior_uncertainty['profile_count']} profiles, {behavior_uncertainty['iterations']} block-bootstrap iterations each, "
                f"{behavior_uncertainty['sample_count']} resampled profiles; stability classes "
                f"supported={behavior_uncertainty['supported_count']}, suggestive={behavior_uncertainty['suggestive_count']}, "
                f"not_aligned={behavior_uncertainty['not_aligned_count']}; best selected video "
                f"{behavior_uncertainty['best_selected_label']}->{behavior_uncertainty['best_selected_target']} distance p50 "
                f"{behavior_uncertainty['best_selected_distance_p50']:.4f} "
                f"[{behavior_uncertainty['best_selected_distance_p025']:.4f}, {behavior_uncertainty['best_selected_distance_p975']:.4f}], "
                f"target stability {behavior_uncertainty['best_selected_target_stability']:.4f}; worst DANDI "
                f"{behavior_uncertainty['worst_dandi_label']}->{behavior_uncertainty['worst_dandi_target']} distance p50 "
                f"{behavior_uncertainty['worst_dandi_distance_p50']:.4f}."
            ),
            "evidence_strength": "direct temporal block-bootstrap robustness audit",
            "status": "warn_gap_robust_under_bootstrap",
            "limitation": "The bootstrap quantifies uncertainty in existing generated rows; it does not create new biological ground truth or fit the adapter.",
            "next_required_evidence": "Use the bootstrap CI targets as objective functions for adapter fitting, then repeat the uncertainty audit on held-out runs.",
        },
        {
            "claim_id": "C22",
            "claim": "Decoded video/calcium action commands have now been audited against realized MuJoCo body, muscle, heading, and speed telemetry over the long recordings.",
            "paper_or_dataset_support": "Implementation-level physical transfer audit; biological tail-kinematics ground truth is still missing from the paired public datasets.",
            "implementation_evidence": "analysis/realized_kinematics_transfer_audit.py; REALIZED_KINEMATICS_TRANSFER_AUDIT.md; realized_kinematics_transfer_summary.csv; command_body_lag_scan.csv; event_triggered_kinematic_responses.csv.",
            "quantitative_evidence": (
                f"{realized['run_count']} audited long recordings; lag rows {realized['lag_scan_rows']}; "
                f"event-response rows {realized['event_response_rows']}; supported transfers "
                f"{realized['supported_count']}, suggestive {realized['suggestive_count']}, weak {realized['weak_count']}; "
                f"best force-to-tail transfer {realized['best_tail_run']} r={realized['best_tail_corr']:.4f} "
                f"at {realized['best_tail_lag_s']:.4f}s; best force-to-muscle transfer "
                f"{realized['best_muscle_run']} r={realized['best_muscle_corr']:.4f} at "
                f"{realized['best_muscle_lag_s']:.4f}s; video mean event-triggered tail delta "
                f"{realized['video_mean_tail_delta']:.6f} versus calcium {realized['calcium_mean_tail_delta']:.6f}."
            ),
            "evidence_strength": "direct realized-body telemetry audit; not biological ground truth",
            "status": "pass_for_physical_transfer_audit_warn_for_biological_validation",
            "limitation": "This proves command-to-body expression in the current MuJoCo implementation, not that the expressed kinematics match measured larval zebrafish tail trajectories.",
            "next_required_evidence": "Compare realized tail curvature, bout angle, beat frequency, and speed against measured zebrafish kinematic recordings under matched stimuli.",
        },
        {
            "claim_id": "C23",
            "claim": "External larval zebrafish kinematic targets now expose concrete biological scale gaps in the current long-run motion.",
            "paper_or_dataset_support": "ZebraZoom movement statistics, PLOS lexical bout timing, OMR/prey-capture kinematics, and Dryad larval swimming biomechanics targets.",
            "implementation_evidence": "analysis/external_kinematics_validation_audit.py; EXTERNAL_KINEMATICS_VALIDATION_AUDIT.md; external_kinematic_run_summary.csv; external_kinematic_target_comparison.csv.",
            "quantitative_evidence": (
                f"{external_kinematics['run_count']} runs, {external_kinematics['comparison_rows']} comparisons; "
                f"inside={external_kinematics['inside_target_count']}, below={external_kinematics['below_target_count']}, "
                f"above={external_kinematics['above_target_count']}, undersampled={external_kinematics['undersampled_count']}; "
                f"video event rates average {external_kinematics['video_mean_event_hz']:.4f} Hz versus calcium "
                f"{external_kinematics['calcium_mean_event_hz']:.4f} Hz; all {external_kinematics['speed_below_count']} "
                f"mean-speed comparisons are below target; realized TBF is undersampled in "
                f"{external_kinematics['realized_tbf_unmeasurable_count']} runs; max telemetry Nyquist "
                f"{external_kinematics['max_telemetry_nyquist']:.4f} Hz."
            ),
            "evidence_strength": "direct external-target audit over current long-run telemetry",
            "status": "warn_external_kinematics_not_yet_biologically_matched",
            "limitation": "Target bands mix spontaneous, OMR, prey-capture, and cyclic-swimming contexts; however, the speed/telemetry failures are broad enough to be real calibration blockers.",
            "next_required_evidence": "Increase high-speed telemetry, calibrate force/scale/duration against measured larval kinematics, and rerun the target comparison on held-out recordings.",
        },
        {
            "claim_id": "C24",
            "claim": "Fresh current-code high-rate MuJoCo recordings now resolve realized larval-range tail-beat frequencies over >=50k physics ticks.",
            "paper_or_dataset_support": "External larval TBF targets from ZebraZoom, OMR/prey-capture kinematic papers, and Dryad larval swimming biomechanics; implementation-level high-rate telemetry audit.",
            "implementation_evidence": "analysis/high_rate_tail_validation_audit.py; HIGH_RATE_TAIL_VALIDATION_AUDIT.md; high_rate_run_summary.csv; high_rate_tail_segment_spectra.csv; high_rate_tail_window_spectra.csv.",
            "quantitative_evidence": (
                f"{high_rate_tail['recording_count']} current-code recordings, {high_rate_tail['plot_count']} plots, "
                f"{high_rate_tail['scalar_stats_rows']} scalar-stat rows, {high_rate_tail['matrix_channel_stats_rows']} matrix-channel rows, "
                f"{high_rate_tail['bout_events']} bout rows, {high_rate_tail['spectral_window_rows']} spectral windows; "
                f"sample rate {high_rate_tail['sample_hz_min']:.1f}-{high_rate_tail['sample_hz_max']:.1f} Hz; "
                f"realized median TBF range {high_rate_tail['tbf_median_min_hz']:.3f}-{high_rate_tail['tbf_median_max_hz']:.3f} Hz; "
                f"external comparisons inside={high_rate_tail['inside_target_count']}, below={high_rate_tail['below_target_count']}, "
                f"above={high_rate_tail['above_target_count']}; anomaly failures={high_rate_tail['failed_anomaly_count']}."
            ),
            "evidence_strength": "direct high-rate physics telemetry over fresh >=50k current-code recordings",
            "status": "pass_for_tbf_observability_warn_for_speed_and_bout_scale",
            "limitation": "This proves high-rate observability and in-band tail-yaw spectral content for direct MuJoCo baseline/sensorimotor protocols, not that video/calcium replay is biologically exact.",
            "next_required_evidence": "Use high-rate replay capture for the exact video/calcium branches and fit force, drag, bout duration, distance, and tail amplitude against measured larval kinematics.",
        },
        {
            "claim_id": "C25",
            "claim": "Exact calcium and backend-video replay branches are now observable at every MuJoCo physics tick over >=50k ticks.",
            "paper_or_dataset_support": "Runtime validation over the ZAPBench calcium/ephys replay artifact and the backend simZFish-inspired video replay path; external larval kinematic target bands supply biological scale checks.",
            "implementation_evidence": "analysis/high_rate_replay_recording.py; analysis/high_rate_tail_validation_audit.py; high_rate_replay_validation_20260603/HIGH_RATE_TAIL_VALIDATION_AUDIT.md.",
            "quantitative_evidence": (
                f"{high_rate_replay['recording_count']} exact replay recordings, {high_rate_replay['plot_count']} plots, "
                f"{high_rate_replay['scalar_stats_rows']} scalar-stat rows, {high_rate_replay['matrix_channel_stats_rows']} matrix-channel rows, "
                f"{high_rate_replay['bout_events']} bout rows, {high_rate_replay['spectral_window_rows']} spectral windows; "
                f"calcium source frames={high_rate_replay['calcium_source_frames']}, event Hz={high_rate_replay['calcium_event_hz']:.3f}, "
                f"TBF={high_rate_replay['calcium_tbf_hz']:.3f} Hz, vertical-instability events={high_rate_replay['calcium_vertical_instability_events']}; "
                f"video source frames={high_rate_replay['video_source_frames']}, loop max={high_rate_replay['video_source_loop_index_max']:.0f}, "
                f"event Hz={high_rate_replay['video_event_hz']:.3f}, TBF={high_rate_replay['video_tbf_hz']:.3f} Hz, "
                f"vertical-instability events={high_rate_replay['video_vertical_instability_events']}; "
                f"external comparisons inside={high_rate_replay['inside_target_count']}, below={high_rate_replay['below_target_count']}, "
                f"above={high_rate_replay['above_target_count']}."
            ),
            "evidence_strength": "direct exact-branch physics telemetry over fresh >=50k recordings",
            "status": "pass_for_observability_warn_video_instability_and_motion_scale",
            "limitation": "The video branch can be analyzed at full temporal resolution but still shows high-turn/high-bend instability and below-target speed/distance; this is not a biological-fidelity pass.",
            "next_required_evidence": "Use the exact-replay telemetry to fit video action gain, pitch/turn damping, drag, and motor scaling against measured larval zebrafish kinematics.",
        },
    ]


def _traceability_rows() -> list[dict[str, str]]:
    return [
        {
            "subsystem": "Common action latent",
            "code_reference": "../active-inference/simulations/zebrafish/action_latent.py:28-213",
            "paper_anchor": "Integration layer; no direct paper object.",
            "runtime_artifact": "action_samples.jsonl in calcium and video 50k runs.",
            "status": "implemented",
        },
        {
            "subsystem": "ZAPBench calcium/ephys decoder",
            "code_reference": "analysis/zapbench_ephys_action_decoder.py; analysis/zapbench_calcium_replay.py",
            "paper_anchor": "ZAPBench public traces and raw tail ephys/stimulus resources.",
            "runtime_artifact": "analysis/cache/zapbench/zapbench_calcium_action_replay.npz",
            "status": "implemented_with_decoder_limits",
        },
        {
            "subsystem": "Calcium replay clock and manual propagation",
            "code_reference": "lab/sim_runtime.py:220-280, 433-493",
            "paper_anchor": "Replay engineering; not a ZAPBench benchmark method.",
            "runtime_artifact": "calcium_all 51,008 tick run",
            "status": "implemented",
        },
        {
            "subsystem": "Backend video extraction",
            "code_reference": "lab/video_pipeline.py:55-239",
            "paper_anchor": "simZFish/Z-Robot optic-flow/retinal OMR motifs; ZAPBench covariates auxiliary only.",
            "runtime_artifact": "backend_frame_responses.jsonl and video source-action timeline.",
            "status": "implemented_partial_fidelity",
        },
        {
            "subsystem": "simZFish OMR adapter",
            "code_reference": "../active-inference/simulations/zebrafish/simzfish_omr.py:70-268",
            "paper_anchor": "Z-Robot/simZFish retina, direction-selective counters, OMR, bouts, tail targets.",
            "runtime_artifact": "diagnostics simzfish_retinal_counters and action_bout_type.",
            "status": "approximate_port",
        },
        {
            "subsystem": "simZFish deterministic controller regression",
            "code_reference": "analysis/simzfish_controller_regression.py",
            "paper_anchor": "Public simZFish Image.c, Robot.c, OMR.c, LeakyIntegrator.c, and params.csv.",
            "runtime_artifact": "simzfish_controller_regression_20260603/*.csv and plots.",
            "status": "direct_and_compiled_references_pass_current_adapter_mismatch",
        },
        {
            "subsystem": "Sensor bridge to nervous/motor system",
            "code_reference": "../active-inference/simulations/zebrafish/sensors.py:53-56; neuron_mapping.py:354-372, 530",
            "paper_anchor": "Engineering bridge into PAULA/MuJoCo body.",
            "runtime_artifact": "tail_yaw_rad, joint_angles_rad, muscle_activations matrix streams.",
            "status": "implemented",
        },
        {
            "subsystem": "Long-run observability",
            "code_reference": "analysis/comprehensive_activity_study.py; analysis/activity_study_synthesis.py; analysis/full_activity_forensics.py",
            "paper_anchor": "Validation method for this lab implementation.",
            "runtime_artifact": "20260603_verified_50k/synthesis and /forensics.",
            "status": "implemented",
        },
        {
            "subsystem": "Selected-video backend robustness",
            "code_reference": "analysis/backend_video_robustness_audit.py; lab/video_pipeline.py; ../active-inference/simulations/zebrafish/simzfish_omr.py",
            "paper_anchor": "Z-Robot/simZFish OMR mechanism as source motif; ZAPBench covariate mapper auxiliary only.",
            "runtime_artifact": "backend_video_robustness/20260603_all_selected_60s/*.csv and plots.",
            "status": "implemented_action_extractor_regression_video_quality_warn",
        },
        {
            "subsystem": "Embodied selected-video replicate comparison",
            "code_reference": "analysis/embodied_video_replicate_comparison.py; analysis/comprehensive_activity_study.py; analysis/activity_study_synthesis.py; analysis/full_activity_forensics.py",
            "paper_anchor": "Implementation-level stability replication; not a paper-specific biological validation.",
            "runtime_artifact": "embodied_video_replicate_comparison_20260603/*.csv, plots, and report.",
            "status": "implemented_two_50k_video_replicates",
        },
        {
            "subsystem": "DANDI OMR calcium neural target atlas",
            "code_reference": "analysis/dandi_omr_neural_validation.py",
            "paper_anchor": "DANDI 001076 OMR Robot CaImaging and Z-Robot/simZFish OMR calcium resource.",
            "runtime_artifact": "dandi_omr_neural_validation_20260603/*.csv, plots, manifest.json, and DANDI_OMR_NEURAL_VALIDATION.md.",
            "status": "implemented_class_level_neural_target_extraction",
        },
        {
            "subsystem": "DANDI-grounded selected-video OMR alignment",
            "code_reference": "analysis/dandi_video_omr_alignment.py",
            "paper_anchor": "DANDI 001076 OMR calcium classes used as motif-level neural target surface for current selected-video backend frames.",
            "runtime_artifact": "dandi_video_omr_alignment_20260603/*.csv, plots, manifest.json, and DANDI_VIDEO_OMR_ALIGNMENT.md.",
            "status": "implemented_selected_video_neural_motif_alignment_audit",
        },
        {
            "subsystem": "DANDI calibrated OMR stimulus-to-motor audit",
            "code_reference": "analysis/dandi_omr_motor_response_audit.py; ../active-inference/simulations/zebrafish/simzfish_omr.py",
            "paper_anchor": "DANDI 001076 OMR stimulus classes and current simZFish-inspired motor adapter.",
            "runtime_artifact": "dandi_omr_motor_response_audit_20260603/*.csv, plots, manifest.json, and DANDI_OMR_MOTOR_RESPONSE_AUDIT.md.",
            "status": "implemented_adapter_alignment_warning",
        },
        {
            "subsystem": "DANDI stimulus-identifiability audit",
            "code_reference": "analysis/dandi_stimulus_identifiability_audit.py",
            "paper_anchor": "DANDI 001076 OMR stimulus labels versus current coarse forward/side/expansion video projection.",
            "runtime_artifact": "dandi_stimulus_identifiability_audit_20260603/*.csv, plots, manifest.json, and DANDI_STIMULUS_IDENTIFIABILITY_AUDIT.md.",
            "status": "implemented_representation_warning",
        },
        {
            "subsystem": "DANDI two-eye/projector semantic bridge audit",
            "code_reference": "analysis/dandi_projector_semantic_bridge_audit.py",
            "paper_anchor": "Z-Robot/simZFish eye-specific lower-posterior retinal motion semantics and DANDI 001076 OMR stimulus labels.",
            "runtime_artifact": "dandi_projector_semantic_bridge_audit_20260603/*.csv, plots, manifest.json, and DANDI_PROJECTOR_SEMANTIC_BRIDGE_AUDIT.md.",
            "status": "implemented_traceability_improved_prediction_warning",
        },
        {
            "subsystem": "simZFish/Z-Robot behavior target alignment audit",
            "code_reference": "analysis/simzfish_behavior_target_alignment_audit.py",
            "paper_anchor": "Data_Liu_simZFish_2025 published bout-frequency and left/forward/right behavior condition profiles.",
            "runtime_artifact": "simzfish_behavior_target_alignment_audit_20260603/*.csv, plots, manifest.json, and SIMZFISH_BEHAVIOR_TARGET_ALIGNMENT_AUDIT.md.",
            "status": "implemented_behavior_target_warning",
        },
        {
            "subsystem": "simZFish/Z-Robot behavior target uncertainty audit",
            "code_reference": "analysis/simzfish_behavior_uncertainty_audit.py",
            "paper_anchor": "Published behavior-condition profiles from Data_Liu_simZFish_2025 with temporal uncertainty estimates over current generated rows.",
            "runtime_artifact": "simzfish_behavior_uncertainty_audit_20260603/*.csv, plots, manifest.json, and SIMZFISH_BEHAVIOR_UNCERTAINTY_AUDIT.md.",
            "status": "implemented_bootstrap_behavior_warning",
        },
        {
            "subsystem": "Realized MuJoCo kinematics transfer audit",
            "code_reference": "analysis/realized_kinematics_transfer_audit.py",
            "paper_anchor": "Implementation-level bridge from decoded action commands to embodied MuJoCo motion; external zebrafish tail-kinematic ground truth remains required.",
            "runtime_artifact": "realized_kinematics_transfer_audit_20260603/*.csv, plots, manifest.json, and REALIZED_KINEMATICS_TRANSFER_AUDIT.md.",
            "status": "implemented_physical_transfer_audit_biological_validation_warning",
        },
        {
            "subsystem": "External larval zebrafish kinematics validation audit",
            "code_reference": "analysis/external_kinematics_validation_audit.py",
            "paper_anchor": "ZebraZoom global movement statistics, PLOS lexical bout timing, OMR/prey-capture TBF ranges, and Dryad larval swimming biomechanics.",
            "runtime_artifact": "external_kinematics_validation_audit_20260603/*.csv, plots, manifest.json, and EXTERNAL_KINEMATICS_VALIDATION_AUDIT.md.",
            "status": "implemented_external_kinematics_warning",
        },
        {
            "subsystem": "High-rate realized tail-beat and full activity audit",
            "code_reference": "analysis/zebrafish_long_recording.py; analysis/high_rate_tail_validation_audit.py",
            "paper_anchor": "External larval zebrafish TBF, bout timing, speed, distance, and tail-amplitude target bands from the public kinematics source register.",
            "runtime_artifact": "high_rate_tail_validation_20260603/*.csv, plots, manifest.json, and HIGH_RATE_TAIL_VALIDATION_AUDIT.md.",
            "status": "implemented_high_rate_tbf_observability_warning_for_scale",
        },
        {
            "subsystem": "Exact replay high-rate calcium/video activity audit",
            "code_reference": "analysis/high_rate_replay_recording.py; analysis/high_rate_tail_validation_audit.py",
            "paper_anchor": "ZAPBench calcium/ephys replay artifact, backend simZFish-inspired video action path, and external larval zebrafish kinematic target bands.",
            "runtime_artifact": "high_rate_replay_validation_20260603/*.csv, plots, manifest.json, recordings/*.npz, and HIGH_RATE_TAIL_VALIDATION_AUDIT.md.",
            "status": "implemented_exact_replay_observability_warning_video_instability",
        },
    ]


def _gate_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    checks = metrics["synthesis"].get("checks", [])
    video_checks = [c for c in checks if c.get("run") == "video_commons_tenggol_underwater"]
    video_nonpass = [c for c in video_checks if c.get("level") != "pass"]
    video_curl = next(
        (c for c in video_checks if c.get("check") == "sustained_curl"),
        {},
    )
    backend_video = _backend_video_rollup(metrics)
    embodied_replicates = _embodied_video_replicate_rollup(metrics)
    dandi_neural = _dandi_neural_rollup(metrics)
    dandi_video = _dandi_video_alignment_rollup(metrics)
    dandi_motor = _dandi_motor_response_rollup(metrics)
    dandi_ident = _dandi_identifiability_rollup(metrics)
    dandi_projector = _dandi_projector_semantic_rollup(metrics)
    behavior_alignment = _simzfish_behavior_alignment_rollup(metrics)
    behavior_uncertainty = _simzfish_behavior_uncertainty_rollup(metrics)
    realized = _realized_kinematics_rollup(metrics)
    external_kinematics = _external_kinematics_rollup(metrics)
    high_rate_tail = _high_rate_tail_rollup(metrics)
    high_rate_replay = _high_rate_replay_rollup(metrics)
    return [
        {
            "gate": "G01 source access",
            "criterion": "Primary ZAPBench, simZFish/Z-Robot, DANDI resources are identified and locally cached where needed.",
            "evidence": "source_register.json; z_robot_resource_audit_20260516.md; cache manifests.",
            "result": "pass_with_web_gating",
            "notes": "Science/PMC direct web access can be gated; local PDF and simZFish/DANDI resources are available.",
        },
        {
            "gate": "G02 ZAPBench direct motor labels",
            "criterion": "Use raw motor ephys rather than stimulus-implied labels for calcium-to-action.",
            "evidence": "direct_ephys_labels.npz; direct_ephys_report.md.",
            "result": "pass",
            "notes": "Labels are fictive motor ephys, not free-swimming kinematics.",
        },
        {
            "gate": "G03 held-out decoder metrics",
            "criterion": "Report held-out kick/force/side metrics with caveats.",
            "evidence": "direct_ephys_metrics.json.",
            "result": "pass_with_side_limit",
            "notes": "Kick/force are usable; side-active accuracy is weak.",
        },
        {
            "gate": "G04 50k calcium stability",
            "criterion": "Calcium replay spans >=50k ticks and passes anomaly checks.",
            "evidence": "synthesis_manifest.json calcium_all checks.",
            "result": "pass",
            "notes": "Does not prove biological identity.",
        },
        {
            "gate": "G05 50k video stability",
            "criterion": "Video replay spans >=50k ticks and passes anomaly checks.",
            "evidence": "synthesis_manifest.json video checks.",
            "result": "pass" if not video_nonpass else "warn",
            "notes": (
                f"Sustained-curl gate is {video_curl.get('level')} with value {video_curl.get('value')}; "
                f"non-pass video checks={len(video_nonpass)}."
            ),
        },
        {
            "gate": "G06 exact simZFish port",
            "criterion": "Python/MuJoCo video branch is validated against original simZFish controller outputs.",
            "evidence": "simzfish_controller_regression_20260603/SIMZFISH_CONTROLLER_REGRESSION.md; image_c_retina_regression.csv; robot_c_motor_regression.csv; omr_c_compiled_regression.csv; leaky_integrator_c_compiled_regression.csv.",
            "result": "partial",
            "notes": (
                f"Image.c direct ports pass OFF={metrics.get('simzfish_controller_manifest', {}).get('image_c_off_exact')} "
                f"DSC={metrics.get('simzfish_controller_manifest', {}).get('image_c_dsc_exact')}; current Robot.c target mean corr "
                f"{float(metrics.get('simzfish_controller_manifest', {}).get('robot_target_corr_mean', 0.0)):.4f}; "
                f"compiled OMR.c rows ok={metrics.get('simzfish_controller_manifest', {}).get('omr_c_all_results_ok')}; "
                f"compiled leaky total bouts={metrics.get('simzfish_controller_manifest', {}).get('leaky_c_total_bouts')}; "
                "runtime adapter equivalence and fitted MuJoCo mapping remain missing."
            ),
        },
        {
            "gate": "G07 paired video-calcium-motion chain",
            "criterion": "Natural video frames, whole-brain calcium, ephys/muscle or kinematics are synchronized in one public dataset.",
            "evidence": "No current cache or paper source provides this chain.",
            "result": "missing",
            "notes": "This prevents a strict journal claim of accurate natural-video-to-movement reproduction.",
        },
        {
            "gate": "G08 PAULA biological validation",
            "criterion": "PAULA zebrafish network is fitted/validated against zebrafish circuit activity.",
            "evidence": "Forensics observe PAULA state, but no zebrafish PAULA fitting artifact exists.",
            "result": "missing",
            "notes": "PAULA is a runtime substrate/diagnostic layer for this pipeline.",
        },
        {
            "gate": "G09 simZFish implementation fidelity",
            "criterion": "Current video branch reaches exact or validated equivalence with the public simZFish controller and published behavior tables.",
            "evidence": "simzfish_fidelity_audit_20260603/SIMZFISH_FIDELITY_AUDIT.md; component_fidelity_scorecard.csv.",
            "result": "partial",
            "notes": f"Mean fidelity score {float(metrics.get('simzfish_fidelity_manifest', {}).get('mean_score', 0.0)):.4f}; Image.c direct-port and compiled OMR.c/LeakyIntegrator.c reference tests now pass, while live-adapter equivalence and calibration fitting are still missing.",
        },
        {
            "gate": "G10 published calibration target extraction",
            "criterion": "Published simZFish/Z-Robot data are converted into numeric targets and compared with current 50k runs.",
            "evidence": "simzfish_calibration_targets_20260603/SIMZFISH_CALIBRATION_TARGET_ATLAS.md; current_vs_published_behavior.csv.",
            "result": "pass_for_extraction",
            "notes": f"Extracted {metrics.get('simzfish_calibration_manifest', {}).get('locomotion_targets', 0)} locomotion, {metrics.get('simzfish_calibration_manifest', {}).get('neural_targets', 0)} neural, and {metrics.get('simzfish_calibration_manifest', {}).get('rheotaxis_targets', 0)} rheotaxis targets; fitting remains required.",
        },
        {
            "gate": "G11 deterministic controller regression",
            "criterion": "Original simZFish Image.c/Robot.c equations and compiled OMR.c/LeakyIntegrator.c sources are represented in a regression harness and compared with current adapter outputs.",
            "evidence": "simzfish_controller_regression_20260603/SIMZFISH_CONTROLLER_REGRESSION.md; image_c_retina_winner_regression.png; robot_c_motor_target_comparison.png; omr_c_compiled_reference_comparison.png; leaky_integrator_c_bout_regression.png.",
            "result": "pass_for_harness_warn_for_equivalence",
            "notes": (
                f"Direct Image.c OFF/DSC ports pass; current adapter has L/R retina winner matches "
                f"{metrics.get('simzfish_controller_manifest', {}).get('retina_left_winner_matches')}/"
                f"{metrics.get('simzfish_controller_manifest', {}).get('retina_right_winner_matches')} and Robot.c target RMSE "
                f"{float(metrics.get('simzfish_controller_manifest', {}).get('robot_target_rmse_mean', 0.0)):.4f}; "
                f"compiled OMR side matches {metrics.get('simzfish_controller_manifest', {}).get('omr_current_side_sign_matches')}/"
                f"{metrics.get('simzfish_controller_manifest', {}).get('omr_c_rows')}; "
                f"compiled leaky total bouts {metrics.get('simzfish_controller_manifest', {}).get('leaky_c_total_bouts')}."
            ),
        },
        {
            "gate": "G12 selected-video backend robustness",
            "criterion": "Every cached selected video is decoded through the current backend extractor with no coast-frame force leakage, and event/quality metrics are reported.",
            "evidence": "backend_video_robustness/20260603_all_selected_60s/BACKEND_VIDEO_ROBUSTNESS_AUDIT.md; backend_video_robustness_summary.csv; backend_video_frame_metrics.csv.",
            "result": "pass_for_coast_force_regression_warn_for_video_quality",
            "notes": (
                f"{backend_video['clip_count']} clips, {backend_video['frame_count']} frames, "
                f"max coast-force leak fraction {backend_video['max_coast_force_leak']:.6f}; "
                f"{backend_video['inside_published_band']}/{backend_video['clip_count']} clips inside published p05-p95 event-rate band; "
                f"{backend_video['high_quality_risk']} clips have quality-event fraction >0.50."
            ),
        },
        {
            "gate": "G13 embodied video replicate stability",
            "criterion": "At least two independent selected-video embodied recordings span >=50k ticks and pass gross anomaly gates.",
            "evidence": "embodied_video_replicate_comparison_20260603/EMBODIED_VIDEO_REPLICATE_COMPARISON.md; 20260603_verified_50k; 20260603_black_rockfish_50k.",
            "result": "pass_for_two_replicates",
            "notes": (
                f"{embodied_replicates['tick_pass_count']}/{embodied_replicates['replicate_count']} video replicates pass >=50k ticks; "
                f"non-pass anomaly checks={embodied_replicates['nonpass_anomaly_checks']}; event Hz range "
                f"{embodied_replicates['min_event_hz']:.4f}-{embodied_replicates['max_event_hz']:.4f}; "
                f"quality event range {embodied_replicates['min_quality_events']}-{embodied_replicates['max_quality_events']}."
            ),
        },
        {
            "gate": "G14 DANDI OMR neural target extraction",
            "criterion": "Every cached DANDI 001076 NWB file is parsed into OMR stimulus-response, trial, and accepted-ROI selectivity summaries with nonblank plots.",
            "evidence": "dandi_omr_neural_validation_20260603/DANDI_OMR_NEURAL_VALIDATION.md; manifest.json; stimulus_response_summary.csv; roi_selectivity_summary.csv.",
            "result": "pass_for_class_level_neural_targets",
            "notes": (
                f"{dandi_neural['parsed_files']}/{dandi_neural['requested_files']} NWB files parsed, "
                f"parse errors={dandi_neural['parse_errors']}, trials={dandi_neural['trial_rows']}, "
                f"stimuli={dandi_neural['unique_stimuli']}, accepted ROIs={dandi_neural['total_accepted_rois']}, "
                f"plots={dandi_neural['plot_count']}, top stimulus={dandi_neural['top_stimulus']}."
            ),
        },
        {
            "gate": "G15 DANDI-grounded selected-video neural motif alignment",
            "criterion": "Every cached selected-video backend frame is projected into coarse OMR axes, mapped to a DANDI OMR calcium class, scored, summarized per clip, and visualized.",
            "evidence": "dandi_video_omr_alignment_20260603/DANDI_VIDEO_OMR_ALIGNMENT.md; video_frame_neural_alignment.csv; clip_neural_alignment_summary.csv; manifest.json.",
            "result": "pass_for_audit_warn_for_stimulus_identity",
            "notes": (
                f"{dandi_video['frame_rows']} frames, {dandi_video['clip_count']} clips, "
                f"{dandi_video['stimulus_count']} DANDI classes; mean plausibility "
                f"{dandi_video['mean_dandi_neural_plausibility']:.4f}, quality-weighted "
                f"{dandi_video['mean_quality_weighted_plausibility']:.4f}; "
                f"quality flags {dandi_video['quality_flagged_frames']} ({dandi_video['quality_flagged_fraction']:.3f}); "
                f"top clip {dandi_video['top_clip']} maps mainly to {dandi_video['top_clip_top_stimulus']}."
            ),
        },
        {
            "gate": "G16 DANDI calibrated OMR class motor-response alignment",
            "criterion": "The current simZFish-inspired motor adapter is driven by calibrated projector-like versions of all DANDI OMR classes, with calcium/motor rank agreement reported and visualized.",
            "evidence": "dandi_omr_motor_response_audit_20260603/DANDI_OMR_MOTOR_RESPONSE_AUDIT.md; stimulus_motor_response_summary.csv; stimulus_motor_response_frames.csv; manifest.json.",
            "result": "pass_for_audit_warn_for_alignment",
            "notes": (
                f"{dandi_motor['stimulus_count']} DANDI classes, {dandi_motor['frames']} synthetic frames; "
                f"value corr={dandi_motor['dandi_response_motor_drive_corr']:.4f}, "
                f"rank corr={dandi_motor['dandi_rank_motor_rank_corr']:.4f}; "
                f"top DANDI stimulus {dandi_motor['top_dandi_stimulus']} motor rank {dandi_motor['top_dandi_motor_rank']}; "
                f"top motor stimulus {dandi_motor['top_motor_stimulus']} DANDI rank {dandi_motor['top_motor_dandi_rank']}."
            ),
        },
        {
            "gate": "G17 DANDI stimulus representation identifiability",
            "criterion": "The current selected-video OMR feature representation preserves enough information to distinguish DANDI OMR calcium-response classes.",
            "evidence": "dandi_stimulus_identifiability_audit_20260603/DANDI_STIMULUS_IDENTIFIABILITY_AUDIT.md; coarse_axis_collision_groups.csv; stimulus_feature_model_scores.csv; axis_sign_sensitivity.csv.",
            "result": "warn_current_representation_insufficient",
            "notes": (
                f"{dandi_ident['collision_group_count']} coarse-axis collision groups; worst collision "
                f"`{dandi_ident['max_collision_labels']}` response range {dandi_ident['max_collision_response_range']:.4f}; "
                f"axis-only LOO R2 {dandi_ident['axes_loo_r2']:.4f}; token LOO R2 {dandi_ident['tokens_loo_r2']:.4f}; "
                f"best sign-flip train R2 {dandi_ident['best_sign_axis_train_r2']:.4f}."
            ),
        },
        {
            "gate": "G18 DANDI two-eye/projector semantic bridge validation",
            "criterion": "A richer eye-specific retinal/projector representation improves DANDI alignment and predicts DANDI class-level calcium responses under held-out validation.",
            "evidence": "dandi_projector_semantic_bridge_audit_20260603/DANDI_PROJECTOR_SEMANTIC_BRIDGE_AUDIT.md; semantic_feature_model_scores.csv; semantic_model_permutation_scores.csv; rich_video_clip_alignment_summary.csv.",
            "result": "warn_traceability_improved_prediction_not_validated",
            "notes": (
                f"Best semantic model {dandi_projector['best_order_hypothesis']}/{dandi_projector['best_feature_set']} "
                f"LOO R2 {dandi_projector['best_loo_r2']:.4f} versus coarse {dandi_projector['baseline_coarse_loo_r2']:.4f}; "
                f"permutation p={dandi_projector['best_permutation_p_ge_observed']:.4f}; "
                f"rich mean video plausibility {dandi_projector['mean_rich_quality_weighted_plausibility']:.4f} versus coarse "
                f"{dandi_projector['mean_coarse_quality_weighted_plausibility']:.4f}; plots={dandi_projector['plot_count']}."
            ),
        },
        {
            "gate": "G19 published simZFish/Z-Robot behavior-target alignment",
            "criterion": "Current DANDI synthetic motor outputs, selected-video backend outputs, and >=50k long-run action profiles fall inside published simZFish/Z-Robot behavior envelopes for bout frequency and left/forward/right fractions.",
            "evidence": "simzfish_behavior_target_alignment_audit_20260603/SIMZFISH_BEHAVIOR_TARGET_ALIGNMENT_AUDIT.md; behavior_target_alignment_scores.csv; behavior_nearest_target_scores.csv; published_behavior_condition_profiles.csv.",
            "result": "warn_current_behavior_targets_not_aligned",
            "notes": (
                f"{behavior_alignment['target_condition_count']} published condition profiles; DANDI mean within-target fraction "
                f"{behavior_alignment['mean_dandi_within_target_fraction']:.4f}; selected-video mean "
                f"{behavior_alignment['mean_video_within_target_fraction']:.4f}; long-run mean "
                f"{behavior_alignment['mean_long_run_within_target_fraction']:.4f}; best video "
                f"{behavior_alignment['best_video_label']}->{behavior_alignment['best_video_target']} distance "
                f"{behavior_alignment['best_video_distance']:.4f}; worst DANDI "
                f"{behavior_alignment['worst_dandi_label']}->{behavior_alignment['worst_dandi_expected']} distance "
                f"{behavior_alignment['worst_dandi_distance']:.4f}."
            ),
        },
        {
            "gate": "G20 behavior-target bootstrap robustness",
            "criterion": "Behavior-target alignment conclusions remain stable under temporally autocorrelated block-bootstrap resampling of generated frame/action rows.",
            "evidence": "simzfish_behavior_uncertainty_audit_20260603/SIMZFISH_BEHAVIOR_UNCERTAINTY_AUDIT.md; bootstrap_behavior_alignment_summary.csv; bootstrap_behavior_alignment_samples.csv.",
            "result": "pass_for_uncertainty_quantification_warn_for_alignment",
            "notes": (
                f"{behavior_uncertainty['profile_count']} profiles x {behavior_uncertainty['iterations']} iterations; "
                f"supported={behavior_uncertainty['supported_count']}, suggestive={behavior_uncertainty['suggestive_count']}, "
                f"not_aligned={behavior_uncertainty['not_aligned_count']}; best selected video "
                f"{behavior_uncertainty['best_selected_label']}->{behavior_uncertainty['best_selected_target']} "
                f"distance p50 {behavior_uncertainty['best_selected_distance_p50']:.4f}; best long run "
                f"{behavior_uncertainty['best_long_label']} event Hz p50 {behavior_uncertainty['best_long_event_hz_p50']:.4f}; "
                f"worst DANDI {behavior_uncertainty['worst_dandi_label']} distance p50 "
                f"{behavior_uncertainty['worst_dandi_distance_p50']:.4f}."
            ),
        },
        {
            "gate": "G21 realized MuJoCo kinematics transfer",
            "criterion": "Decoded calcium/video action commands produce measurable tail, muscle, heading, speed, and z-motion responses in recorded MuJoCo state over long runs.",
            "evidence": "realized_kinematics_transfer_audit_20260603/REALIZED_KINEMATICS_TRANSFER_AUDIT.md; realized_kinematics_transfer_summary.csv; command_body_lag_scan.csv; event_triggered_kinematic_responses.csv.",
            "result": "pass_for_physical_transfer_audit_warn_for_biological_validation",
            "notes": (
                f"{realized['run_count']} runs, {realized['lag_scan_rows']} lag-scan rows, "
                f"{realized['event_response_rows']} event-response rows; supported={realized['supported_count']}, "
                f"suggestive={realized['suggestive_count']}, weak={realized['weak_count']}; best tail transfer "
                f"{realized['best_tail_run']} r={realized['best_tail_corr']:.4f} lag={realized['best_tail_lag_s']:.4f}s; "
                f"best muscle transfer {realized['best_muscle_run']} r={realized['best_muscle_corr']:.4f}; "
                f"external measured larval tail kinematics remain missing."
            ),
        },
        {
            "gate": "G22 external larval kinematics validation",
            "criterion": "Current >=50k recordings match external larval zebrafish kinematic anchors for event timing, bout duration, speed, distance, heading, tail amplitude, and tail-beat frequency observability.",
            "evidence": "external_kinematics_validation_audit_20260603/EXTERNAL_KINEMATICS_VALIDATION_AUDIT.md; external_kinematic_run_summary.csv; external_kinematic_target_comparison.csv.",
            "result": "warn_speed_scale_and_tbf_telemetry_not_validated",
            "notes": (
                f"{external_kinematics['comparison_rows']} comparisons: inside={external_kinematics['inside_target_count']}, "
                f"below={external_kinematics['below_target_count']}, above={external_kinematics['above_target_count']}, "
                f"undersampled={external_kinematics['undersampled_count']}; video event Hz mean "
                f"{external_kinematics['video_mean_event_hz']:.4f}, calcium event Hz mean "
                f"{external_kinematics['calcium_mean_event_hz']:.4f}; all mean-speed checks below target; "
                f"max telemetry Nyquist {external_kinematics['max_telemetry_nyquist']:.4f} Hz is insufficient for 20-100 Hz realized TBF validation."
            ),
        },
        {
            "gate": "G23 high-rate tail-beat observability",
            "criterion": "Fresh current-code direct MuJoCo recordings span >=50k ticks, sample at >=190 Hz, pass gross anomaly checks, and resolve realized 20-95 Hz tail-beat spectral content.",
            "evidence": "high_rate_tail_validation_20260603/HIGH_RATE_TAIL_VALIDATION_AUDIT.md; high_rate_run_summary.csv; high_rate_tail_segment_spectra.csv; high_rate_tail_window_spectra.csv; high_rate_anomaly_checks.csv.",
            "result": "pass_for_tbf_observability_warn_for_motion_scale",
            "notes": (
                f"{high_rate_tail['recording_count']} recordings, sample Hz {high_rate_tail['sample_hz_min']:.1f}-"
                f"{high_rate_tail['sample_hz_max']:.1f}, realized median TBF "
                f"{high_rate_tail['tbf_median_min_hz']:.3f}-{high_rate_tail['tbf_median_max_hz']:.3f} Hz, "
                f"anomaly failures={high_rate_tail['failed_anomaly_count']}; external target checks inside="
                f"{high_rate_tail['inside_target_count']}, below={high_rate_tail['below_target_count']}, "
                f"above={high_rate_tail['above_target_count']}. Speed, distance, tail amplitude, and bout timing still need calibration."
            ),
        },
        {
            "gate": "G24 exact replay high-rate observability",
            "criterion": "Exact calcium and backend-video replay branches each span >=50k physics ticks, sample at >=190 Hz, record source/action/neural/muscle/body telemetry, and expose any replay-specific instability.",
            "evidence": "high_rate_replay_validation_20260603/HIGH_RATE_TAIL_VALIDATION_AUDIT.md; high_rate_run_summary.csv; high_rate_scalar_stats.csv; high_rate_matrix_channel_stats.csv; high_rate_bout_events.csv; recordings/*.npz.",
            "result": "pass_for_observability_warn_video_instability_and_motion_scale",
            "notes": (
                f"{high_rate_replay['recording_count']} exact replay recordings, sample Hz "
                f"{high_rate_replay['sample_hz_min']:.1f}-{high_rate_replay['sample_hz_max']:.1f}; "
                f"calcium frames={high_rate_replay['calcium_source_frames']}, event Hz={high_rate_replay['calcium_event_hz']:.3f}, "
                f"TBF={high_rate_replay['calcium_tbf_hz']:.3f}, vertical-instability events="
                f"{high_rate_replay['calcium_vertical_instability_events']}; video frames="
                f"{high_rate_replay['video_source_frames']}, event Hz={high_rate_replay['video_event_hz']:.3f}, "
                f"TBF={high_rate_replay['video_tbf_hz']:.3f}, vertical-instability events="
                f"{high_rate_replay['video_vertical_instability_events']}. External target checks inside="
                f"{high_rate_replay['inside_target_count']}, below={high_rate_replay['below_target_count']}, "
                f"above={high_rate_replay['above_target_count']}."
            ),
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    metrics: dict[str, Any],
    source_path: Path,
    claim_path: Path,
    trace_path: Path,
    gate_path: Path,
) -> None:
    direct = metrics["direct_metrics"]
    runs = {row["name"]: row for row in metrics["forensics"].get("runs", [])}
    calcium = runs.get("calcium_all", {})
    video = runs.get("video_commons_tenggol_underwater", {})
    checks = metrics["synthesis"].get("checks", [])
    video_warns = [c for c in checks if c.get("run") == "video_commons_tenggol_underwater" and c.get("level") != "pass"]
    controller = metrics.get("simzfish_controller_manifest", {})
    backend_video = _backend_video_rollup(metrics)
    embodied_replicates = _embodied_video_replicate_rollup(metrics)
    dandi_neural = _dandi_neural_rollup(metrics)
    dandi_video = _dandi_video_alignment_rollup(metrics)
    dandi_motor = _dandi_motor_response_rollup(metrics)
    dandi_ident = _dandi_identifiability_rollup(metrics)
    dandi_projector = _dandi_projector_semantic_rollup(metrics)
    behavior_alignment = _simzfish_behavior_alignment_rollup(metrics)
    behavior_uncertainty = _simzfish_behavior_uncertainty_rollup(metrics)
    realized = _realized_kinematics_rollup(metrics)
    external_kinematics = _external_kinematics_rollup(metrics)
    high_rate_tail = _high_rate_tail_rollup(metrics)
    high_rate_replay = _high_rate_replay_rollup(metrics)
    lines = [
        "# Two-Paper Zebrafish Pipeline Study",
        "",
        "## Executive Conclusion",
        "",
        "The current zebrafish lab is a paper-grounded, instrumented demonstrator with one comparatively strong ZAPBench calcium/ephys replay branch and one simZFish/Z-Robot-inspired video OMR branch. It is not yet a top-tier-journal-validated zebrafish digital twin. The strongest defensible claim is that both branches feed a shared embodied MuJoCo tail-action interface and have been recorded over >=50k physics ticks with full observability.",
        "",
        "The highest-risk scientific gaps are now explicit: no public paired dataset currently ties natural video frames, whole-brain calcium, ephys/muscle labels, and free-swimming kinematics for the same animal/task, and the current coarse video-to-OMR representation is not identifiable enough to distinguish DANDI stimulus classes. Without those, the video-to-calcium-to-movement chain remains an engineered bridge, not a validated biological reproduction.",
        "",
        "## Evidence Package",
        "",
        f"- Source register: `{source_path.resolve()}`",
        f"- Claim coverage matrix: `{claim_path.resolve()}`",
        f"- Paper-to-code traceability: `{trace_path.resolve()}`",
        f"- Acceptance gates: `{gate_path.resolve()}`",
        f"- Long-run atlas: `{(SYNTHESIS_DIR / 'FULL_ACTIVITY_ATLAS_REPORT.md').resolve()}`",
        f"- Forensic atlas: `{(FORENSICS_DIR / 'FORENSIC_ACTIVITY_STUDY.md').resolve()}`",
        f"- simZFish/Z-Robot fidelity audit: `{(SIMZFISH_FIDELITY_DIR / 'SIMZFISH_FIDELITY_AUDIT.md').resolve()}`",
        f"- simZFish/Z-Robot calibration target atlas: `{(SIMZFISH_CALIBRATION_DIR / 'SIMZFISH_CALIBRATION_TARGET_ATLAS.md').resolve()}`",
        f"- simZFish deterministic controller regression: `{(SIMZFISH_CONTROLLER_DIR / 'SIMZFISH_CONTROLLER_REGRESSION.md').resolve()}`",
        f"- Selected-video backend robustness audit: `{(BACKEND_VIDEO_ROBUSTNESS_DIR / 'BACKEND_VIDEO_ROBUSTNESS_AUDIT.md').resolve()}`",
        f"- Embodied selected-video replicate comparison: `{(EMBODIED_VIDEO_REPLICATE_DIR / 'EMBODIED_VIDEO_REPLICATE_COMPARISON.md').resolve()}`",
        f"- DANDI OMR neural validation atlas: `{(DANDI_NEURAL_VALIDATION_DIR / 'DANDI_OMR_NEURAL_VALIDATION.md').resolve()}`",
        f"- DANDI-grounded selected-video OMR alignment: `{(DANDI_VIDEO_ALIGNMENT_DIR / 'DANDI_VIDEO_OMR_ALIGNMENT.md').resolve()}`",
        f"- DANDI calibrated OMR motor-response audit: `{(DANDI_MOTOR_RESPONSE_DIR / 'DANDI_OMR_MOTOR_RESPONSE_AUDIT.md').resolve()}`",
        f"- DANDI stimulus-identifiability audit: `{(DANDI_IDENTIFIABILITY_DIR / 'DANDI_STIMULUS_IDENTIFIABILITY_AUDIT.md').resolve()}`",
        f"- DANDI two-eye/projector semantic bridge audit: `{(DANDI_PROJECTOR_SEMANTIC_DIR / 'DANDI_PROJECTOR_SEMANTIC_BRIDGE_AUDIT.md').resolve()}`",
        f"- simZFish/Z-Robot behavior target alignment audit: `{(SIMZFISH_BEHAVIOR_ALIGNMENT_DIR / 'SIMZFISH_BEHAVIOR_TARGET_ALIGNMENT_AUDIT.md').resolve()}`",
        f"- simZFish/Z-Robot behavior uncertainty audit: `{(SIMZFISH_BEHAVIOR_UNCERTAINTY_DIR / 'SIMZFISH_BEHAVIOR_UNCERTAINTY_AUDIT.md').resolve()}`",
        f"- Realized MuJoCo kinematics transfer audit: `{(REALIZED_KINEMATICS_DIR / 'REALIZED_KINEMATICS_TRANSFER_AUDIT.md').resolve()}`",
        f"- External larval zebrafish kinematics validation audit: `{(EXTERNAL_KINEMATICS_DIR / 'EXTERNAL_KINEMATICS_VALIDATION_AUDIT.md').resolve()}`",
        f"- High-rate tail/activity validation audit: `{(HIGH_RATE_TAIL_DIR / 'HIGH_RATE_TAIL_VALIDATION_AUDIT.md').resolve()}`",
        f"- Exact replay high-rate activity audit: `{(HIGH_RATE_REPLAY_DIR / 'HIGH_RATE_TAIL_VALIDATION_AUDIT.md').resolve()}`",
        "",
        "## Primary Sources",
        "",
        "- [ZAPBench OpenReview paper](https://openreview.net/forum?id=oCHsDpyawq)",
        "- [ZAPBench project portal](https://zapbench-release.storage.googleapis.com/landing.html)",
        "- [ZAPBench dataset README](https://zapbench-release.storage.googleapis.com/volumes/README.html)",
        "- [ZAPBench code](https://github.com/google-research/zapbench)",
        "- [Google Research ZAPBench article](https://research.google/blog/improving-brain-models-with-zapbench/)",
        "- [Z-Robot / simZFish Science Robotics DOI](https://www.science.org/doi/10.1126/scirobotics.adv4408)",
        "- [simZFish repository](https://ponyo.epfl.ch/proj/zebrafish/simzfish)",
        "- [DANDI 001076](https://dandiarchive.org/dandiset/001076)",
        "",
        "## Pipeline Under Review",
        "",
        "```mermaid",
        "flowchart TD",
        "  Z1[ZAPBench calcium traces + raw tail ephys] --> Z2[calcium/ephys decoder]",
        "  Z2 --> A[ZebrafishActionLatent]",
        "  V1[Natural/sample video] --> V2[OpenCV frame decode + Farneback flow + RANSAC stabilization]",
        "  V2 --> V3[simZFish-style retina and OMR adapter]",
        "  V3 --> A",
        "  A --> S[ACTION_* sensor channels]",
        "  S --> N[PAULA-backed zebrafish nervous wrapper]",
        "  N --> M[MuJoCo tail muscles/body]",
        "  M --> W[REST/WebSocket lab telemetry]",
        "```",
        "",
        "## Quantitative Summary",
        "",
        "| item | value | interpretation |",
        "|---|---:|---|",
        f"| ZAPBench ephys decoder kick accuracy | {direct.get('kick_accuracy', 0):.4f} | usable kick/no-kick signal |",
        f"| ZAPBench ephys decoder kick F1 | {direct.get('direct_extra', {}).get('kick_f1', 0):.4f} | moderate class balance |",
        f"| ZAPBench ephys decoder force R2 | {direct.get('force_r2', 0):.4f} | usable but not high-fidelity force |",
        f"| ZAPBench ephys decoder side-active accuracy | {direct.get('direct_extra', {}).get('side_active_accuracy', 0):.4f} | weak turn-side signal |",
        f"| Calcium replay ticks | {calcium.get('tick_span', 0)} | passes >=50k gate |",
        f"| Calcium replay action samples | {calcium.get('actions', 0)} | ZAPBench replay rows sampled over run |",
        f"| Video replay ticks | {video.get('tick_span', 0)} | passes >=50k tick count |",
        f"| Video backend frames | {video.get('backend_video_frames', 0)} | manually stepped backend decoded frames |",
        f"| Video quality events | {sum(metrics['video_quality_event_counts'].values())} | explains caution around video behavior |",
        f"| simZFish data cache | {metrics['simzfish_data_files']} files / {metrics['simzfish_data_bytes']} bytes | available for next calibration |",
        f"| DANDI 001076 cache manifest | {metrics['dandi_asset_files']} assets / {metrics['dandi_asset_bytes']} bytes | available for OMR calcium validation |",
        f"| simZFish fidelity score | {metrics.get('simzfish_fidelity_manifest', {}).get('mean_score', 0):.4f} | current video branch is motif-level, not exact-controller equivalent |",
        f"| simZFish calibration targets | {metrics.get('simzfish_calibration_manifest', {}).get('locomotion_targets', 0)} locomotion / {metrics.get('simzfish_calibration_manifest', {}).get('neural_targets', 0)} neural / {metrics.get('simzfish_calibration_manifest', {}).get('rheotaxis_targets', 0)} rheotaxis | extracted from public Z-Robot/simZFish data |",
        f"| Image.c direct-port regression | OFF exact {controller.get('image_c_off_exact')} / DSC exact {controller.get('image_c_dsc_exact')} | deterministic retinal equations pass in the harness |",
        f"| Current adapter vs Robot.c targets | mean corr {float(controller.get('robot_target_corr_mean', 0.0)):.4f} / mean RMSE {float(controller.get('robot_target_rmse_mean', 0.0)):.4f} | current MuJoCo tail latent is not Robot.c-equivalent yet |",
        f"| Compiled OMR.c reference | rows {controller.get('omr_c_rows')} / all ok {controller.get('omr_c_all_results_ok')} / current side matches {controller.get('omr_current_side_sign_matches')} | original OMR source now runs, current adapter still mismatches |",
        f"| Compiled LeakyIntegrator.c reference | rows {controller.get('leaky_c_rows')} / total bouts {controller.get('leaky_c_total_bouts')} | original bout-threshold source now runs as a reference surface |",
        f"| Selected-video backend robustness | {backend_video['clip_count']} clips / {backend_video['frame_count']} frames / no coast-force leak {backend_video['max_coast_force_leak']:.6f} | action extractor regression passes across cached video set |",
        f"| Selected-video event-rate coverage | mean {backend_video['mean_event_hz']:.4f} Hz / {backend_video['inside_published_band']}/{backend_video['clip_count']} inside published p05-p95 band | source set is behavior-like but heterogeneous |",
        f"| Selected-video quality burden | mean {backend_video['mean_quality_fraction']:.4f} / max {backend_video['max_quality_fraction']:.4f} quality-event fraction | many public clips need exclusion or down-weighting |",
        f"| Embodied video replicate stability | {embodied_replicates['tick_pass_count']}/{embodied_replicates['replicate_count']} pass >=50k / anomaly non-pass {embodied_replicates['nonpass_anomaly_checks']} | selected-video body stability replicated across two clips |",
        f"| Embodied video replicate quality range | {embodied_replicates['min_quality_events']}-{embodied_replicates['max_quality_events']} quality events | black-rockfish is much cleaner than Tenggol |",
        f"| DANDI OMR calcium files | {dandi_neural['parsed_files']}/{dandi_neural['requested_files']} parsed / {dandi_neural['parse_errors']} errors | public neural OMR resource is structurally usable |",
        f"| DANDI OMR calcium trials | {dandi_neural['trial_rows']} trials / {dandi_neural['unique_stimuli']} stimuli / {dandi_neural['total_accepted_rois']} accepted ROIs | class-level whole-brain calcium target surface extracted |",
        f"| DANDI top calcium stimulus | {dandi_neural['top_stimulus']} / {dandi_neural['top_population_response_mean']:.6g} mean response | strongest aggregate OMR response in the public NWB cache |",
        f"| DANDI validation plots | {dandi_neural['plot_count']} plots / acquisition rate {dandi_neural['rate_min_hz']:.6f}-{dandi_neural['rate_max_hz']:.6f} Hz | visualized neural response, selectivity, centroids, traces, and similarity |",
        f"| DANDI-grounded video alignment | {dandi_video['frame_rows']} frames / {dandi_video['clip_count']} clips / {dandi_video['stimulus_count']} DANDI classes | every selected-video backend frame scored against calcium motif atlas |",
        f"| Video neural plausibility | mean {dandi_video['mean_dandi_neural_plausibility']:.4f} / quality-weighted {dandi_video['mean_quality_weighted_plausibility']:.4f} | selected videos are only weak-to-moderate motif matches after quality weighting |",
        f"| Best DANDI-aligned selected clip | {dandi_video['top_clip']} / {dandi_video['top_clip_quality_weighted_plausibility']:.4f} / {dandi_video['top_clip_top_stimulus']} | black-rockfish remains the best low-artifact candidate, but maps mostly to mixed OMR |",
        f"| DANDI calibrated OMR motor audit | {dandi_motor['stimulus_count']} classes / {dandi_motor['frames']} synthetic frames | current adapter tested against actual DANDI stimulus-class axes |",
        f"| Calcium/motor rank agreement | value corr {dandi_motor['dandi_response_motor_drive_corr']:.4f} / rank corr {dandi_motor['dandi_rank_motor_rank_corr']:.4f} | current adapter is not strongly DANDI neural-rank aligned |",
        f"| Top DANDI vs top motor class | {dandi_motor['top_dandi_stimulus']} motor rank {dandi_motor['top_dandi_motor_rank']} / {dandi_motor['top_motor_stimulus']} DANDI rank {dandi_motor['top_motor_dandi_rank']} | main calibrated-stimulus mismatch |",
        f"| DANDI OMR identifiability | {dandi_ident['stimulus_count']} stimuli / {dandi_ident['axis_group_count']} coarse-axis groups / {dandi_ident['collision_group_count']} collision groups | current selected-video projection loses class semantics |",
        f"| Worst coarse-axis collision | {dandi_ident['max_collision_labels']} / response range {dandi_ident['max_collision_response_range']:.4f} | distinct DANDI calcium classes become indistinguishable to the current axes |",
        f"| Coarse-axis calcium-response prediction | train R2 {dandi_ident['axes_train_r2']:.4f} / LOO R2 {dandi_ident['axes_loo_r2']:.4f} | current axes are not a usable class-level DANDI response model |",
        f"| Label-token calcium-response prediction | train R2 {dandi_ident['tokens_train_r2']:.4f} / LOO R2 {dandi_ident['tokens_loo_r2']:.4f} | DANDI labels contain extra semantics but still need more data/regularization |",
        f"| Sign-convention sensitivity | best axis R2 {dandi_ident['best_sign_axis_train_r2']:.4f} at signs {dandi_ident['best_signs']} | sign flips do not explain the alignment failure |",
        f"| Two-eye/projector semantic bridge | {dandi_projector['stimulus_count']} DANDI stimuli / {dandi_projector['video_frame_count']} selected-video frames / {dandi_projector['plot_count']} plots | paper-derived richer retinal representation audit |",
        f"| Semantic bridge held-out prediction | best LOO R2 {dandi_projector['best_loo_r2']:.4f} vs coarse {dandi_projector['baseline_coarse_loo_r2']:.4f} / permutation p {dandi_projector['best_permutation_p_ge_observed']:.4f} | traceability improves, class-level calcium prediction still not validated |",
        f"| Rich selected-video DANDI plausibility | mean {dandi_projector['mean_rich_quality_weighted_plausibility']:.4f} vs coarse {dandi_projector['mean_coarse_quality_weighted_plausibility']:.4f}; best clip {dandi_projector['top_clip']} {dandi_projector['top_clip_rich_quality_weighted_plausibility']:.4f} | retinal-counter matching improves the diagnostic score but remains uncalibrated |",
        f"| Published behavior target profiles | {behavior_alignment['target_condition_count']} conditions / global bout mean {behavior_alignment['published_global_hz_mean']:.4f} Hz ({behavior_alignment['published_global_hz_p05']:.4f}-{behavior_alignment['published_global_hz_p95']:.4f} p05-p95) | simZFish/Z-Robot locomotion target surface now quantified |",
        f"| Current behavior-target coverage | DANDI {behavior_alignment['mean_dandi_within_target_fraction']:.4f} / selected video {behavior_alignment['mean_video_within_target_fraction']:.4f} / long-run {behavior_alignment['mean_long_run_within_target_fraction']:.4f} mean within-target fraction | current outputs are measurable but not fitted to published envelopes |",
        f"| Best selected-video behavior target | {behavior_alignment['best_video_label']} -> {behavior_alignment['best_video_target']} / event {behavior_alignment['best_video_event_hz']:.4f} Hz / distance {behavior_alignment['best_video_distance']:.4f} / startle {behavior_alignment['best_video_startle_fraction']:.4f} | plausible nearest-condition match, not a validation of arbitrary video identity |",
        f"| Best 50k long-run behavior target | {behavior_alignment['best_long_label']} -> {behavior_alignment['best_long_target']} / event {behavior_alignment['best_long_event_hz']:.4f} Hz / distance {behavior_alignment['best_long_distance']:.4f} | long-run body stability can now be compared to published bout envelopes |",
        f"| Worst mapped DANDI behavior target | {behavior_alignment['worst_dandi_label']} -> {behavior_alignment['worst_dandi_expected']} / distance {behavior_alignment['worst_dandi_distance']:.4f} | current DANDI stimulus-to-motor mapping has a strong direction/fraction mismatch |",
        f"| Behavior-target bootstrap coverage | {behavior_uncertainty['profile_count']} profiles x {behavior_uncertainty['iterations']} iterations = {behavior_uncertainty['sample_count']} resampled profiles | temporal uncertainty is now quantified for selected-video, DANDI synthetic, and 50k action rows |",
        f"| Behavior-target bootstrap classes | supported {behavior_uncertainty['supported_count']} / suggestive {behavior_uncertainty['suggestive_count']} / not aligned {behavior_uncertainty['not_aligned_count']} | no profile passes strict bootstrap support yet |",
        f"| Best selected-video bootstrap result | {behavior_uncertainty['best_selected_label']} -> {behavior_uncertainty['best_selected_target']} / distance p50 {behavior_uncertainty['best_selected_distance_p50']:.4f} [{behavior_uncertainty['best_selected_distance_p025']:.4f}, {behavior_uncertainty['best_selected_distance_p975']:.4f}] / target stability {behavior_uncertainty['best_selected_target_stability']:.4f} | selected-video evidence remains suggestive, not validated |",
        f"| Best 50k long-run bootstrap result | {behavior_uncertainty['best_long_label']} -> {behavior_uncertainty['best_long_target']} / event Hz p50 {behavior_uncertainty['best_long_event_hz_p50']:.4f} [{behavior_uncertainty['best_long_event_hz_p025']:.4f}, {behavior_uncertainty['best_long_event_hz_p975']:.4f}] / distance p50 {behavior_uncertainty['best_long_distance_p50']:.4f} | long-run video event rate is behavior-like but still not a fitted movement model |",
        f"| Worst DANDI bootstrap mismatch | {behavior_uncertainty['worst_dandi_label']} -> {behavior_uncertainty['worst_dandi_target']} / distance p50 {behavior_uncertainty['worst_dandi_distance_p50']:.4f} [{behavior_uncertainty['worst_dandi_distance_p025']:.4f}, {behavior_uncertainty['worst_dandi_distance_p975']:.4f}] | DANDI direction/fraction mismatch is robust to resampling |",
        f"| Realized kinematics transfer coverage | {realized['run_count']} runs / {realized['lag_scan_rows']} lag rows / {realized['event_response_rows']} event-response rows | commands are now tested against actual MuJoCo body state, not only action labels |",
        f"| Realized transfer classes | supported {realized['supported_count']} / suggestive {realized['suggestive_count']} / weak {realized['weak_count']} | video branches express stronger command-to-body transfer than sparse calcium replay |",
        f"| Best force-to-tail transfer | {realized['best_tail_run']} r {realized['best_tail_corr']:.4f} at lag {realized['best_tail_lag_s']:.4f}s | strongest current command-to-realized-tail coupling |",
        f"| Best force-to-muscle transfer | {realized['best_muscle_run']} r {realized['best_muscle_corr']:.4f} at lag {realized['best_muscle_lag_s']:.4f}s | strongest current command-to-realized-muscle coupling |",
        f"| Video vs calcium event-triggered tail response | video mean delta {realized['video_mean_tail_delta']:.6f} / calcium mean delta {realized['calcium_mean_tail_delta']:.6f} | video commands produce larger event-triggered tail response under current replay recordings |",
        f"| Black-rockfish realized body summary | tail r {realized['black_video_tail_corr']:.4f} / muscle r {realized['black_video_muscle_corr']:.4f} / speed {realized['black_video_speed_mean']:.4f} mm/s / z-span p95 {realized['black_video_z_span_p95']:.4f} mm | best current low-artifact video branch has measurable body expression |",
        f"| External kinematics scorecard | inside {external_kinematics['inside_target_count']} / below {external_kinematics['below_target_count']} / above {external_kinematics['above_target_count']} / undersampled {external_kinematics['undersampled_count']} across {external_kinematics['comparison_rows']} comparisons | current motion is partially behavior-like but not biologically calibrated |",
        f"| External kinematics event-rate lane | video mean {external_kinematics['video_mean_event_hz']:.4f} Hz / calcium mean {external_kinematics['calcium_mean_event_hz']:.4f} Hz | selected-video event rates sit in the broad larval bout-rate band; calcium replay is sparse |",
        f"| External kinematics speed lane | black-rockfish video mean {external_kinematics['black_video_speed_mean']:.4f} mm/s, p95 {external_kinematics['black_video_speed_p95']:.4f} mm/s | below ZebraZoom/Dryad larval speed anchors |",
        f"| External kinematics bout-timing lane | black-rockfish video bout p50 {external_kinematics['black_video_bout_ms']:.1f} ms / interbout p50 {external_kinematics['black_video_interbout_ms']:.1f} ms | interbout is plausible, active span is too long versus 100-250 ms target band |",
        f"| External kinematics TBF observability | command TBF inside-target count {external_kinematics['command_tbf_inside_count']} / realized TBF undersampled {external_kinematics['realized_tbf_unmeasurable_count']} / max Nyquist {external_kinematics['max_telemetry_nyquist']:.4f} Hz | command frequencies are plausible but true body tail-beat validation is not measurable from current telemetry |",
        "",
        "## Status By Evidence Lane",
        "",
        "- **ZAPBench literature/data lane:** strong for whole-brain calcium forecasting and fictive motor ephys labels; not a free-swimming movement paper.",
        "- **Z-Robot/simZFish lane:** strong for embodied OMR mechanics and visual-to-bout architecture; deterministic Image.c/Robot.c and compiled OMR.c/LeakyIntegrator.c reference tests now exist, but the current lab adapter is not exact-controller equivalent and still needs runtime fitting/replacement plus calibration.",
        "- **Implementation lane:** common action latent and MuJoCo tail-target path are implemented and observable.",
        "- **Selected-video robustness lane:** the backend video extractor now has an all-cached-clip regression audit; it passes coast-force leakage and reports per-clip event-rate/quality gates.",
        (
            "- **Long-run behavior lane:** calcium branch passes current >=50k checks; "
            + (
                "video branch passes all current gross-anomaly checks for this run and now has a second independent >=50k selected-video replicate."
                if not video_warns
                else "video branch still has non-pass anomaly checks that need tuning."
            )
        ),
        "- **DANDI neural-resource lane:** all cached OMR calcium NWB files now parse into stimulus, trial, and accepted-ROI target surfaces with nonblank visualizations.",
        "- **DANDI/video alignment lane:** selected-video frames now have a motif-level calcium-atlas alignment score; the result exposes weak mixed-stimulus identity rather than strong convergence onto the highest DANDI response class.",
        "- **DANDI calibrated motor lane:** the current simZFish-inspired adapter has now been tested on the DANDI stimulus classes directly; the weak calcium/motor rank agreement is a concrete tuning/replacement target.",
        "- **DANDI stimulus-identifiability lane:** the coarse `forward/side/expansion` projection is now quantified as insufficient for class-level calcium claims, independent of long-run body stability.",
        "- **DANDI two-eye/projector bridge lane:** eye-specific lower-posterior retinal semantics now exist as a reportable representation scaffold; held-out response prediction remains below validation threshold.",
        "- **Published behavior-target lane:** simZFish/Z-Robot bout-frequency and direction envelopes now score DANDI synthetic, selected-video, and >=50k long-run outputs; the score is a warning, not a pass.",
        "- **Behavior uncertainty lane:** block-bootstrap confidence intervals show the behavior-target gap is stable under temporal resampling; no profile reaches strict bootstrap support.",
        "- **Realized-body transfer lane:** decoded commands are now tested against actual MuJoCo tail, muscle, heading, speed, and z-motion telemetry; this passes as a physical-transfer audit but remains only suggestive biologically until measured tail kinematics are added.",
        "- **External kinematics lane:** public larval zebrafish kinematic targets now score current motion directly; event rate, heading, tail amplitude, and command TBF are partly plausible, while speed/scale, active-bout duration, sparse calcium events, and realized TBF telemetry remain warning-level blockers.",
        "- **Scientific validity lane:** current system is not yet a precise biological reproduction because paired natural-video/calcium/kinematics evidence is missing.",
        "",
        "## Video Branch Warning",
        "",
    ]
    if video_warns:
        for warn in video_warns:
            lines.append(
                f"- `{warn.get('check')}` is `{warn.get('level')}` with value `{warn.get('value')}`: {warn.get('detail')}"
            )
    else:
        lines.append("- No non-pass video anomaly checks were present in the synthesis manifest.")
    lines.extend(
        [
            "",
            "The forensics layer also found video-quality events from the selected underwater clip:",
            "",
        ]
    )
    for event, count in sorted(metrics["video_quality_event_counts"].items()):
        lines.append(f"- `{event}`: {count}")
    quality_sorted = sorted(
        backend_video["summary"],
        key=lambda row: _to_float(row.get("quality_event_fraction")),
        reverse=True,
    )
    low_risk = [
        row
        for row in backend_video["summary"]
        if _to_float(row.get("quality_event_fraction")) <= 0.25
        and _to_float(row.get("event_frequency_inside_published_p05_p95")) > 0.5
    ]
    low_risk_text = ", ".join(f"`{row.get('clip')}`" for row in low_risk) if low_risk else "none under current thresholds"
    lines.extend(
        [
            "",
            "## Selected-Video Backend Robustness",
            "",
            "A new offline audit runs every cached selected video through the current backend extractor, preserving the stateful simZFish-style OMR integrator. This isolates source-video and action-decoder robustness from MuJoCo body stability.",
            "",
            f"- Audited clips/frames: `{backend_video['clip_count']}` clips, `{backend_video['frame_count']}` backend-decoded frames.",
            f"- Coast-force leak regression: max coast nonzero-force fraction `{backend_video['max_coast_force_leak']:.6f}`.",
            f"- Event-rate coverage: mean `{backend_video['mean_event_hz']:.4f}` Hz; published simZFish/Z-Robot mean `{backend_video['published_mean_hz']:.4f}` Hz; `{backend_video['inside_published_band']}/{backend_video['clip_count']}` clips inside published p05-p95 band.",
            f"- Quality burden: mean quality-event fraction `{backend_video['mean_quality_fraction']:.4f}`; max `{backend_video['max_quality_fraction']:.4f}`; `{backend_video['high_quality_risk']}` clips exceed quality-event fraction `0.50`.",
            f"- Low-risk clips for the next embodied >=50k runs: {low_risk_text}.",
            "",
            "Highest quality-risk selected clips:",
            "",
            "| clip | quality fraction | event Hz | mean force |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in quality_sorted[:6]:
        lines.append(
            f"| `{row.get('clip')}` | {_to_float(row.get('quality_event_fraction')):.3f} | "
            f"{_to_float(row.get('noncoast_event_frequency_hz')):.3f} | {_to_float(row.get('action_force_mean')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Embodied Selected-Video Replicate Comparison",
            "",
            "The long-run evidence now includes two independent selected-video MuJoCo recordings over >=50k physics ticks: Tenggol and the low-risk black-rockfish stereo-DOV clip selected by the backend robustness audit.",
            "",
            f"- Replicate pass count: `{embodied_replicates['tick_pass_count']}/{embodied_replicates['replicate_count']}` pass >=50k ticks.",
            f"- Gross anomaly result: `{embodied_replicates['nonpass_anomaly_checks']}` non-pass anomaly checks across video replicates.",
            f"- Event-rate range: `{embodied_replicates['min_event_hz']:.4f}`-`{embodied_replicates['max_event_hz']:.4f}` Hz.",
            f"- Quality-event count range: `{embodied_replicates['min_quality_events']}`-`{embodied_replicates['max_quality_events']}`.",
            "",
            "| replicate | tick span | event Hz | quality events | anomaly non-pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in embodied_replicates["records"]:
        lines.append(
            f"| `{row.get('label')}` | {int(_to_float(row.get('tick_span')))} | "
            f"{_to_float(row.get('noncoast_event_frequency_hz')):.4f} | "
            f"{int(_to_float(row.get('quality_event_count')))} | "
            f"{int(_to_float(row.get('nonpass_anomaly_checks')))} |"
        )
    lines.extend(
        [
            "",
            "## DANDI OMR Neural Validation",
            "",
            "The public DANDI 001076 NWB cache has now been parsed into an OMR calcium target atlas. This provides a class-level neural validation surface for the Z-Robot/simZFish visual stimulus classes, separate from the selected-video body recordings.",
            "",
            f"- NWB parse coverage: `{dandi_neural['parsed_files']}/{dandi_neural['requested_files']}` files with `{dandi_neural['parse_errors']}` parse errors.",
            f"- Neural data volume: `{dandi_neural['trial_rows']}` trial rows, `{dandi_neural['unique_stimuli']}` OMR stimulus labels, `{dandi_neural['total_accepted_rois']}` accepted ROIs.",
            f"- Acquisition coverage: `{dandi_neural['rate_min_hz']:.6f}`-`{dandi_neural['rate_max_hz']:.6f}` Hz; accepted ROI range `{dandi_neural['accepted_rois_min']}`-`{dandi_neural['accepted_rois_max']}` per file.",
            f"- Strongest aggregate stimulus: `{dandi_neural['top_stimulus']}` with mean response `{dandi_neural['top_population_response_mean']:.6g}`.",
            f"- Visualization count: `{dandi_neural['plot_count']}` nonblank plot artifacts covering response heatmaps, ranks, stimulus similarity, ROI selectivity, centroid maps, trial-triggered traces, and stimulus embedding.",
            "",
            "| rank | stimulus | primary axis | files | trials | mean response | active ROI fraction |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(dandi_neural["top_rows"], start=1):
        lines.append(
            f"| {rank} | `{row.get('stimulus')}` | `{row.get('primary_axis')}` | "
            f"{int(_to_float(row.get('files')))} | {int(_to_float(row.get('trials')))} | "
            f"{_to_float(row.get('population_response_mean')):.6g} | "
            f"{_to_float(row.get('selective_active_roi_fraction_mean')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: this confirms that public Z-Robot/DANDI resources can supply OMR calcium response motifs for stimulus classes. It still does not provide a frame-exact natural-video, ephys/muscle, and free-swimming kinematics chain for the current MuJoCo body.",
            "",
            "## DANDI-Grounded Selected-Video OMR Alignment",
            "",
            "The selected-video backend frame table has now been projected into coarse DANDI-like OMR axes and scored against the extracted DANDI calcium-response atlas. This is the strongest current bridge between arbitrary selected-video features and public Z-Robot/DANDI neural motifs, but it is still a motif-level audit rather than a trained video-to-calcium model.",
            "",
            f"- Frame coverage: `{dandi_video['frame_rows']}` backend-decoded frames across `{dandi_video['clip_count']}` selected clips.",
            f"- DANDI stimulus coverage: `{dandi_video['stimulus_count']}` OMR calcium classes.",
            f"- Quality burden: `{dandi_video['quality_flagged_frames']}` frames flagged (`{dandi_video['quality_flagged_fraction']:.3f}` of scored frames).",
            f"- Mean neural plausibility: `{dandi_video['mean_dandi_neural_plausibility']:.4f}` raw, `{dandi_video['mean_quality_weighted_plausibility']:.4f}` quality-weighted.",
            f"- Best quality-weighted clip: `{dandi_video['top_clip']}` with score `{dandi_video['top_clip_quality_weighted_plausibility']:.4f}` and top mapped stimulus `{dandi_video['top_clip_top_stimulus']}` / axis `{dandi_video['top_clip_top_axis']}`.",
            "",
            "| rank | clip | top stimulus | top axis | quality-weighted plausibility | quality weight | event Hz |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(dandi_video["rows"][:8], start=1):
        lines.append(
            f"| {rank} | `{row.get('clip')}` | `{row.get('top_stimulus')}` | `{row.get('top_primary_axis')}` | "
            f"{_to_float(row.get('quality_weighted_neural_plausibility')):.4f} | "
            f"{_to_float(row.get('mean_quality_weight')):.4f} | "
            f"{_to_float(row.get('backend_event_frequency_hz')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Alignment interpretation: the best selected videos are the same low-artifact clips already favored by body-stability testing, but their nearest DANDI stimulus assignments are dominated by mixed `backward_forward`/`forward_backward` classes rather than the strongest DANDI aggregate calcium class `converging`. This argues for either calibrated two-eye projector-like stimuli or a stronger video-to-OMR feature transform before claiming neural fidelity.",
            "",
            "## DANDI Calibrated OMR Motor Response",
            "",
            "A stricter audit now bypasses arbitrary underwater video and drives the current simZFish-inspired motor adapter with calibrated, projector-like synthetic versions of the 20 DANDI OMR stimulus classes. This asks whether the current motor path itself ranks OMR classes in a way that resembles the public DANDI calcium target surface.",
            "",
            f"- Stimulus coverage: `{dandi_motor['stimulus_count']}` DANDI classes, `{dandi_motor['frames']}` synthetic frames at `{dandi_motor['sample_hz']:.3f}` Hz.",
            f"- Calcium/motor value correlation: `{dandi_motor['dandi_response_motor_drive_corr']:.4f}`.",
            f"- Calcium/motor rank correlation: `{dandi_motor['dandi_rank_motor_rank_corr']:.4f}`.",
            f"- Top DANDI stimulus `{dandi_motor['top_dandi_stimulus']}` has current motor-drive rank `{dandi_motor['top_dandi_motor_rank']}`.",
            f"- Top current motor stimulus `{dandi_motor['top_motor_stimulus']}` has DANDI rank `{dandi_motor['top_motor_dandi_rank']}`.",
            "",
            "| DANDI rank | stimulus | DANDI response | motor rank | motor drive | event Hz | mean force |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dandi_motor["rows_by_dandi"][:10]:
        lines.append(
            f"| {int(_to_float(row.get('dandi_rank')))} | `{row.get('stimulus')}` | "
            f"{_to_float(row.get('dandi_population_response_mean')):.6g} | "
            f"{int(_to_float(row.get('motor_drive_rank')))} | "
            f"{_to_float(row.get('motor_drive_index')):.6g} | "
            f"{_to_float(row.get('bout_transition_frequency_hz')):.4f} | "
            f"{_to_float(row.get('mean_force')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Motor-response interpretation: this is a concrete current-implementation warning. Stable motion and selected-video plausibility do not yet mean the motor adapter is aligned with DANDI/Z-Robot neural response classes. The adapter needs fitting or replacement against DANDI class-level calcium motifs plus simZFish/ZBot motor targets before a strong neural-fidelity claim.",
            "",
            "## DANDI Stimulus Identifiability",
            "",
            "This audit asks a sharper question: even before fitting the motor adapter, does the current selected-video representation preserve enough visual-stimulus information to identify DANDI OMR calcium classes? The answer is no under the current `forward/side/expansion` projection.",
            "",
            f"- Stimulus compression: `{dandi_ident['stimulus_count']}` DANDI labels collapse to `{dandi_ident['axis_group_count']}` coarse-axis groups.",
            f"- Collision burden: `{dandi_ident['collision_group_count']}` coarse-axis groups contain multiple DANDI labels with different calcium responses.",
            f"- Worst collision: `{dandi_ident['max_collision_labels']}` with DANDI response range `{dandi_ident['max_collision_response_range']:.6g}`.",
            f"- Axis-only prediction: train R2 `{dandi_ident['axes_train_r2']:.4f}`, leave-one-out R2 `{dandi_ident['axes_loo_r2']:.4f}`.",
            f"- Token-label prediction: train R2 `{dandi_ident['tokens_train_r2']:.4f}`, leave-one-out R2 `{dandi_ident['tokens_loo_r2']:.4f}`.",
            f"- Best simple sign-convention axis R2: `{dandi_ident['best_sign_axis_train_r2']:.4f}` with signs `{dandi_ident['best_signs']}`.",
            f"- Visualization count: `{dandi_ident['plot_count']}` nonblank plots covering collisions, model scores, and response scatter.",
            "",
            "| coarse axis | labels | count | response min | response max | response range |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in dandi_ident["collision_rows"][:8]:
        if int(_to_float(row.get("label_count"))) <= 1:
            continue
        lines.append(
            f"| `{row.get('axis_key')}` | `{row.get('labels')}` | {int(_to_float(row.get('label_count')))} | "
            f"{_to_float(row.get('response_min')):.6g} | {_to_float(row.get('response_max')):.6g} | "
            f"{_to_float(row.get('response_range')):.6g} |"
        )
    lines.extend(
        [
            "",
            "| feature set | train R2 | LOO R2 | RMSE | LOO RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in dandi_ident["model_rows"]:
        lines.append(
            f"| `{row.get('feature_set')}` | {_to_float(row.get('train_r2')):.4f} | "
            f"{_to_float(row.get('loo_r2')):.4f} | {_to_float(row.get('rmse')):.4f} | "
            f"{_to_float(row.get('loo_rmse')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Identifiability interpretation: the coarse axes are acceptable as a diagnostic motif projection, but they are not enough for a rigorous DANDI-class calcium bridge. A calibrated two-eye/projector-like representation must encode the stimulus semantics that distinguish labels such as `x_forward`, `forward_x`, and `forward` before video-to-calcium fitting can be judged scientifically.",
            "",
            "## DANDI Two-Eye/Projector Semantic Bridge",
            "",
            "A richer audit now encodes the DANDI OMR labels as eye-specific monocular/binocular/conflict stimuli and lower-posterior retinal direction-channel proxies, then compares selected-video backend retinal counters against this representation. This directly addresses the previous coarse-axis limitation, but it also shows that richer traceability is not the same thing as validated calcium prediction.",
            "",
            f"- Stimulus/video coverage: `{dandi_projector['stimulus_count']}` DANDI stimuli and `{dandi_projector['video_frame_count']}` selected-video backend frames.",
            f"- Best semantic response model: `{dandi_projector['best_order_hypothesis']}` / `{dandi_projector['best_feature_set']}`.",
            f"- Held-out response prediction: LOO R2 `{dandi_projector['best_loo_r2']:.4f}` versus coarse-axis `{dandi_projector['baseline_coarse_loo_r2']:.4f}`.",
            f"- Permutation check: p(LOO R2 >= observed) `{dandi_projector['best_permutation_p_ge_observed']:.4f}`.",
            f"- Collision check: best model collision groups `{dandi_projector['best_collision_group_count']}` versus coarse `{dandi_projector['baseline_collision_group_count']}`; best zero-collision feature set `{dandi_projector['best_zero_collision_feature_set']}` has LOO R2 `{dandi_projector['best_zero_collision_loo_r2']:.4f}`.",
            f"- Selected-video scoring: mean rich quality-weighted plausibility `{dandi_projector['mean_rich_quality_weighted_plausibility']:.4f}` versus coarse `{dandi_projector['mean_coarse_quality_weighted_plausibility']:.4f}`.",
            f"- Top rich-aligned clip: `{dandi_projector['top_clip']}` rich `{dandi_projector['top_clip_rich_quality_weighted_plausibility']:.4f}` versus coarse `{dandi_projector['top_clip_coarse_quality_weighted_plausibility']:.4f}`.",
            "",
            "| rank | clip | top stimulus | semantic class | rich score | coarse score | delta |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(dandi_projector["clip_rows"][:8], start=1):
        lines.append(
            f"| {rank} | `{row.get('clip')}` | `{row.get('top_stimulus')}` | `{row.get('top_semantic_class')}` | "
            f"{_to_float(row.get('quality_weighted_neural_plausibility')):.4f} | "
            f"{_to_float(row.get('coarse_quality_weighted_plausibility')):.4f} | "
            f"{_to_float(row.get('rich_minus_coarse_quality_weighted_plausibility')):.4f} |"
        )
    lines.extend(
        [
            "",
            "| feature set | order | LOO R2 | collisions | permutation/notes |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in dandi_projector["model_rows"][:6]:
        extra = ""
        if row.get("feature_set") == dandi_projector["best_feature_set"] and row.get("order_hypothesis") == dandi_projector["best_order_hypothesis"]:
            extra = f"best; p={dandi_projector['best_permutation_p_ge_observed']:.4f}"
        lines.append(
            f"| `{row.get('feature_set')}` | `{row.get('order_hypothesis')}` | "
            f"{_to_float(row.get('loo_r2')):.4f} | {int(_to_float(row.get('collision_group_count')))} | {extra} |"
        )
    lines.extend(
        [
            "",
            "Semantic-bridge interpretation: this is the right direction for a biologically meaningful visual bridge because it uses eye-specific retinal semantics rather than arbitrary underwater optical-flow axes. It is still not scientifically sufficient for a precise video-to-calcium claim: all held-out response-prediction R2 values remain negative, the best model is not permutation-significant, and exact DANDI projector movies remain unavailable in the public NWB cache.",
            "",
            "## simZFish/Z-Robot Behavior Target Alignment",
            "",
            "A new behavior-target audit now compares current motor outputs against the published simZFish/Z-Robot locomotion envelopes extracted from `Data_Liu_simZFish_2025`. It uses transition-based bout events and left/forward/right bout fractions, because the published targets summarize bout/action events rather than raw frame occupancy.",
            "",
            f"- Published target surface: `{behavior_alignment['target_condition_count']}` condition profiles; global bout-frequency mean `{behavior_alignment['published_global_hz_mean']:.4f}` Hz with p05-p95 `{behavior_alignment['published_global_hz_p05']:.4f}`-`{behavior_alignment['published_global_hz_p95']:.4f}` Hz.",
            f"- DANDI synthetic motor profiles: `{behavior_alignment['dandi_profile_count']}` profiles; mean within-target fraction `{behavior_alignment['mean_dandi_within_target_fraction']:.4f}`.",
            f"- Selected-video backend profiles: `{behavior_alignment['video_profile_count']}` profiles; mean within-target fraction `{behavior_alignment['mean_video_within_target_fraction']:.4f}`.",
            f"- >=50k long-run action profiles: `{behavior_alignment['long_run_profile_count']}` profiles; mean within-target fraction `{behavior_alignment['mean_long_run_within_target_fraction']:.4f}`.",
            f"- Best selected-video nearest target: `{behavior_alignment['best_video_label']}` -> `{behavior_alignment['best_video_target']}` at distance `{behavior_alignment['best_video_distance']:.4f}`.",
            f"- Worst mapped DANDI target: `{behavior_alignment['worst_dandi_label']}` expected `{behavior_alignment['worst_dandi_expected']}` at distance `{behavior_alignment['worst_dandi_distance']:.4f}`.",
            "",
            "| selected video | nearest target | event Hz | L/F/R/startle | distance | inside fraction |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in behavior_alignment["video_rows"][:8]:
        lines.append(
            f"| `{row.get('label')}` | `{row.get('target_condition')}` | {_to_float(row.get('bout_frequency_hz')):.4f} | "
            f"{_to_float(row.get('left_fraction')):.2f}/{_to_float(row.get('forward_fraction')):.2f}/"
            f"{_to_float(row.get('right_fraction')):.2f}/{_to_float(row.get('startle_fraction')):.2f} | "
            f"{_to_float(row.get('alignment_distance')):.4f} | {_to_float(row.get('within_p05_p95_fraction')):.2f} |"
        )
    lines.extend(
        [
            "",
            "| long-run profile | nearest target | event Hz | L/F/R/startle | distance | inside fraction |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in behavior_alignment["long_rows"]:
        lines.append(
            f"| `{row.get('label')}` | `{row.get('target_condition')}` | {_to_float(row.get('bout_frequency_hz')):.4f} | "
            f"{_to_float(row.get('left_fraction')):.2f}/{_to_float(row.get('forward_fraction')):.2f}/"
            f"{_to_float(row.get('right_fraction')):.2f}/{_to_float(row.get('startle_fraction')):.2f} | "
            f"{_to_float(row.get('alignment_distance')):.4f} | {_to_float(row.get('within_p05_p95_fraction')):.2f} |"
        )
    lines.extend(
        [
            "",
            "| worst DANDI mapped stimulus | expected target | event Hz | L/F/R/startle | distance | inside fraction |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in behavior_alignment["dandi_rows"][:8]:
        lines.append(
            f"| `{row.get('label')}` | `{row.get('expected_condition')}` | {_to_float(row.get('bout_frequency_hz')):.4f} | "
            f"{_to_float(row.get('left_fraction')):.2f}/{_to_float(row.get('forward_fraction')):.2f}/"
            f"{_to_float(row.get('right_fraction')):.2f}/{_to_float(row.get('startle_fraction')):.2f} | "
            f"{_to_float(row.get('alignment_distance')):.4f} | {_to_float(row.get('within_p05_p95_fraction')):.2f} |"
        )
    lines.extend(
        [
            "",
            "Behavior-target interpretation: this is the first audit that ties the current live adapter and long-run action logs directly to published simZFish/Z-Robot behavior envelopes. It improves scientific falsifiability but does not make the system validated. Selected videos can land near a published bout-rate envelope while still carrying high startle fractions or arbitrary-video identity problems. DANDI synthetic classes show strong direction/fraction mismatches, so the motor adapter still needs fitting or replacement before movement can be called data-grounded.",
            "",
            "## simZFish/Z-Robot Behavior Uncertainty",
            "",
            "The behavior-target point estimates are now stress-tested with contiguous block bootstrapping. This matters because video/backend frames and calcium/action replay rows are temporally autocorrelated; independent frame resampling would overstate certainty.",
            "",
            f"- Bootstrap coverage: `{behavior_uncertainty['profile_count']}` profiles, `{behavior_uncertainty['iterations']}` iterations per profile, `{behavior_uncertainty['sample_count']}` total resampled profile estimates.",
            f"- Stability classes: supported `{behavior_uncertainty['supported_count']}`, suggestive `{behavior_uncertainty['suggestive_count']}`, not aligned `{behavior_uncertainty['not_aligned_count']}`.",
            f"- Best selected-video bootstrap result: `{behavior_uncertainty['best_selected_label']}` -> `{behavior_uncertainty['best_selected_target']}` distance p50 `{behavior_uncertainty['best_selected_distance_p50']:.4f}` with 95% CI `{behavior_uncertainty['best_selected_distance_p025']:.4f}`-`{behavior_uncertainty['best_selected_distance_p975']:.4f}` and target stability `{behavior_uncertainty['best_selected_target_stability']:.4f}`.",
            f"- Best >=50k long-run bootstrap result: `{behavior_uncertainty['best_long_label']}` -> `{behavior_uncertainty['best_long_target']}` event Hz p50 `{behavior_uncertainty['best_long_event_hz_p50']:.4f}` with 95% CI `{behavior_uncertainty['best_long_event_hz_p025']:.4f}`-`{behavior_uncertainty['best_long_event_hz_p975']:.4f}` and distance p50 `{behavior_uncertainty['best_long_distance_p50']:.4f}`.",
            f"- Worst DANDI bootstrap mismatch: `{behavior_uncertainty['worst_dandi_label']}` -> `{behavior_uncertainty['worst_dandi_target']}` distance p50 `{behavior_uncertainty['worst_dandi_distance_p50']:.4f}` with 95% CI `{behavior_uncertainty['worst_dandi_distance_p025']:.4f}`-`{behavior_uncertainty['worst_dandi_distance_p975']:.4f}`.",
            "",
            "| selected video | target | distance p50 [p025, p975] | inside probability | target stability | class |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in behavior_uncertainty["selected_rows"][:8]:
        lines.append(
            f"| `{row.get('label')}` | `{row.get('target_condition')}` | "
            f"{_to_float(row.get('alignment_distance_p50')):.4f} "
            f"[{_to_float(row.get('alignment_distance_p025')):.4f}, {_to_float(row.get('alignment_distance_p975')):.4f}] | "
            f"{_to_float(row.get('mean_metric_inside_probability')):.4f} | "
            f"{_to_float(row.get('nearest_target_stability_fraction')):.4f} | `{row.get('stability_class')}` |"
        )
    lines.extend(
        [
            "",
            "| long-run profile | target | event Hz p50 [p025, p975] | distance p50 [p025, p975] | inside probability |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in behavior_uncertainty["long_rows"]:
        lines.append(
            f"| `{row.get('label')}` | `{row.get('target_condition')}` | "
            f"{_to_float(row.get('bout_frequency_hz_p50')):.4f} "
            f"[{_to_float(row.get('bout_frequency_hz_p025')):.4f}, {_to_float(row.get('bout_frequency_hz_p975')):.4f}] | "
            f"{_to_float(row.get('alignment_distance_p50')):.4f} "
            f"[{_to_float(row.get('alignment_distance_p025')):.4f}, {_to_float(row.get('alignment_distance_p975')):.4f}] | "
            f"{_to_float(row.get('mean_metric_inside_probability')):.4f} |"
        )
    lines.extend(
        [
            "",
            "| worst DANDI bootstrap mismatch | target | distance p50 [p025, p975] | inside probability | class |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in behavior_uncertainty["dandi_rows"][:8]:
        lines.append(
            f"| `{row.get('label')}` | `{row.get('target_condition')}` | "
            f"{_to_float(row.get('alignment_distance_p50')):.4f} "
            f"[{_to_float(row.get('alignment_distance_p025')):.4f}, {_to_float(row.get('alignment_distance_p975')):.4f}] | "
            f"{_to_float(row.get('mean_metric_inside_probability')):.4f} | `{row.get('stability_class')}` |"
        )
    lines.extend(
        [
            "",
            "Uncertainty interpretation: this strengthens the negative conclusion. The current adapter can produce selected-video outputs that are near a published nearest-condition envelope, but none meet a strict bootstrap-supported criterion. The DANDI synthetic stimulus-to-motor mismatch remains robust under resampling, so behavior fitting remains a required next step.",
            "",
            "## Realized MuJoCo Kinematics Transfer",
            "",
            "The behavior-target and bootstrap audits operate on decoded action rows. This additional audit checks the next physical layer: whether those decoded commands actually appear as realized tail, muscle, heading, speed, and z-motion changes in recorded MuJoCo state over the long recordings.",
            "",
            f"- Coverage: `{realized['run_count']}` long recordings, `{realized['lag_scan_rows']}` command/body lag rows, `{realized['event_response_rows']}` event-triggered response rows.",
            f"- Transfer classes: supported `{realized['supported_count']}`, suggestive `{realized['suggestive_count']}`, weak `{realized['weak_count']}`.",
            f"- Strongest force-to-tail coupling: `{realized['best_tail_run']}` r `{realized['best_tail_corr']:.4f}` at lag `{realized['best_tail_lag_s']:.4f}` s.",
            f"- Strongest force-to-muscle coupling: `{realized['best_muscle_run']}` r `{realized['best_muscle_corr']:.4f}` at lag `{realized['best_muscle_lag_s']:.4f}` s.",
            f"- Strongest side-to-left/right-muscle coupling: `{realized['best_direction_run']}` r `{realized['best_direction_corr']:.4f}`.",
            f"- Mean event-triggered tail response delta: video `{realized['video_mean_tail_delta']:.6f}`, calcium `{realized['calcium_mean_tail_delta']:.6f}`.",
            "",
            "| run | class | events | force-tail r/lag | force-muscle r/lag | side-muscle r | tail event delta | speed mean | z span p95 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in realized["rows"]:
        lines.append(
            f"| `{row.get('run')}` | `{row.get('transfer_class')}` | {int(_to_float(row.get('event_count')))} | "
            f"{_to_float(row.get('tail_yaw_abs_mean_best_corr')):.4f}/{_to_float(row.get('tail_yaw_abs_mean_best_lag_s')):.3f}s | "
            f"{_to_float(row.get('muscle_sum_best_corr')):.4f}/{_to_float(row.get('muscle_sum_best_lag_s')):.3f}s | "
            f"{_to_float(row.get('side_muscle_lr_bias_corr0')):.4f} | "
            f"{_to_float(row.get('tail_yaw_abs_mean_event_delta_mean')):.6f} | "
            f"{_to_float(row.get('speed_xy_mm_s_mean')):.4f} | "
            f"{_to_float(row.get('body_z_span_mm_p95')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Realized-transfer interpretation: this is an important implementation pass because it catches the failure mode where action labels look plausible but the rendered/simulated body does not express them. The two video recordings show supported command-to-body transfer; the two calcium replay recordings are only suggestive because the replay contains far fewer event windows. This is still not biological validation: the next required measurement is realized tail curvature, bout angle, beat frequency, speed, and z-motion against external zebrafish kinematic recordings under matched stimuli.",
            "",
            "## External Larval Zebrafish Kinematics Validation",
            "",
            "The realized-body audit asks whether commands reach the simulated body. This stricter external-kinematics audit asks whether the resulting motion matches measured larval zebrafish scales from ZebraZoom, OMR/prey-capture kinematic studies, and the Dryad larval swimming biomechanics dataset.",
            "",
            f"- Comparisons: `{external_kinematics['comparison_rows']}` target checks across `{external_kinematics['run_count']}` long recordings.",
            f"- Scorecard: inside `{external_kinematics['inside_target_count']}`, below `{external_kinematics['below_target_count']}`, above `{external_kinematics['above_target_count']}`, undersampled `{external_kinematics['undersampled_count']}`.",
            f"- Event-rate result: video mean `{external_kinematics['video_mean_event_hz']:.4f}` Hz is in the broad bout-rate band, while calcium mean `{external_kinematics['calcium_mean_event_hz']:.4f}` Hz is sparse.",
            f"- Speed result: all `{external_kinematics['speed_below_count']}` mean-speed checks and all `{external_kinematics['speed_p95_below_count']}` p95-speed checks are below external larval targets.",
            f"- TBF observability result: command frequencies are in range for `{external_kinematics['command_tbf_inside_count']}` runs, but realized TBF is undersampled in `{external_kinematics['realized_tbf_unmeasurable_count']}` runs; max telemetry Nyquist is `{external_kinematics['max_telemetry_nyquist']:.4f}` Hz.",
            "",
            "| run | event Hz | bout p50 ms | interbout p50 ms | speed mean/p95 mm/s | distance/event mm | heading/event deg | tail p95 deg | command TBF Hz | Nyquist Hz |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in external_kinematics["rows"]:
        lines.append(
            f"| `{row.get('run')}` | {_to_float(row.get('event_frequency_hz')):.4f} | "
            f"{_to_float(row.get('bout_duration_ms_p50')):.1f} | "
            f"{_to_float(row.get('interbout_ms_p50')):.1f} | "
            f"{_to_float(row.get('speed_xy_mm_s_mean')):.3f}/{_to_float(row.get('speed_xy_mm_s_p95')):.3f} | "
            f"{_to_float(row.get('distance_per_event_mm_p50')):.3f} | "
            f"{_to_float(row.get('heading_delta_abs_deg_p50')):.3f} | "
            f"{_to_float(row.get('tail_yaw_abs_max_deg_p95')):.3f} | "
            f"{_to_float(row.get('command_tail_frequency_hz_mean')):.3f} | "
            f"{_to_float(row.get('telemetry_nyquist_hz')):.3f} |"
        )
    lines.extend(
        [
            "",
            "External-kinematics interpretation: this is now the clearest biological-calibration warning in the package. The selected-video branches produce plausible event rates, interbout intervals, heading changes, broad tail-angle envelopes, and command target frequencies. They do not yet produce measured-scale larval swimming speed or distance per bout, their active spans are longer than the 100-250 ms bout-duration target band, and the current telemetry cannot verify 20-100 Hz realized tail-beat execution. A lab presentation should state this as an implementation demonstrator with explicit calibration gaps, not as a biologically faithful motion model.",
            "",
            "## High-Rate Tail And Full Activity Validation",
            "",
            "The external-kinematics audit exposed a telemetry problem: the browser/WebSocket long-run traces were too sparse to resolve larval tail-beat frequencies. The high-rate audit closes that measurement gap for fresh current-code direct MuJoCo recordings by recording every physics tick at 200 Hz.",
            "",
            f"- Coverage: `{high_rate_tail['recording_count']}` fresh current-code recordings, each >=50k ticks; sample rate range `{high_rate_tail['sample_hz_min']:.1f}`-`{high_rate_tail['sample_hz_max']:.1f}` Hz.",
            f"- Tables: `{high_rate_tail['scalar_stats_rows']}` scalar-stat rows, `{high_rate_tail['matrix_channel_stats_rows']}` matrix-channel rows, `{high_rate_tail['bout_events']}` bout rows, `{high_rate_tail['spectral_window_rows']}` spectral-window rows.",
            f"- Visuals: `{high_rate_tail['plot_count']}` plots spanning master timeline, tail spectra, windowed TBF, external target scorecard, bout distributions, phase-space controls, coupling heatmaps, and top channel heatmaps.",
            f"- Realized TBF: baseline `{high_rate_tail['baseline_tbf_hz']:.3f}` Hz and sensorimotor `{high_rate_tail['sensorimotor_tbf_hz']:.3f}` Hz, both inside the external 20-95 Hz larval band.",
            f"- Stability: anomaly failures `{high_rate_tail['failed_anomaly_count']}` across gross checks for >=50k ticks, Nyquist, nonzero bouts, finite TBF, vertical lock, and bounded z-span.",
            f"- Biological target status: inside `{high_rate_tail['inside_target_count']}`, below `{high_rate_tail['below_target_count']}`, above `{high_rate_tail['above_target_count']}` across high-rate external comparisons.",
            "",
            "| run | bouts | event Hz | TBF median Hz | speed mean/p95 mm/s | body z-span p95 mm | anomaly status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in high_rate_tail["rows"]:
        run_name = str(row.get("run", ""))
        failed = [
            a
            for a in high_rate_tail["anomalies"]
            if a.get("run") == run_name and str(a.get("status")) != "pass"
        ]
        lines.append(
            f"| `{run_name}` | {int(_to_float(row.get('bout_count')))} | "
            f"{_to_float(row.get('event_frequency_hz')):.4f} | "
            f"{_to_float(row.get('realized_tail_frequency_tbf_band_hz_median')):.3f} | "
            f"{_to_float(row.get('speed_xy_mm_s_mean')):.3f}/{_to_float(row.get('speed_xy_mm_s_p95')):.3f} | "
            f"{_to_float(row.get('body_z_span_mm_p95')):.3f} | "
            f"{'pass' if not failed else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "High-rate interpretation: this resolves the earlier measurement blocker. The simulator can now be tested at a temporal resolution high enough for larval TBF, and the direct baseline/sensorimotor runs show in-band tail-yaw spectral content without the old vertical-lock/curl failure. The negative result is equally important: the same high-rate runs still have low speed, short distances, small tail amplitudes, and bout/interbout timing that are not calibrated to public larval targets. This is a measurement pass and a calibration warning, not a biological-fidelity pass for the video/calcium branches.",
            "",
            "## Exact Replay High-Rate Calcium/Video Activity",
            "",
            "The exact-replay audit records the two user-facing replay regimes rather than generic direct protocols. Calcium follows the lab's ZAPBench ~1.093 Hz decoded frame schedule and decaying pulse conversion; video follows backend optical-flow/simZFish-style extraction and the rounded 10 Hz video-frame to physics-tick conversion. Every tick stores source frame, source loop, action latent, calcium pulse, video diagnostics, external tail target confidence, muscles, PAULA state, body geometry, MuJoCo qpos/qvel, and derived behavior metrics.",
            "",
            f"- Coverage: `{high_rate_replay['recording_count']}` exact replay recordings, each >=50k ticks; sample rate range `{high_rate_replay['sample_hz_min']:.1f}`-`{high_rate_replay['sample_hz_max']:.1f}` Hz.",
            f"- Tables: `{high_rate_replay['scalar_stats_rows']}` scalar-stat rows, `{high_rate_replay['matrix_channel_stats_rows']}` matrix-channel rows, `{high_rate_replay['bout_events']}` bout rows, `{high_rate_replay['spectral_window_rows']}` spectral-window rows, `{high_rate_replay['window_10s_rows']}` 10s-window rows, `{high_rate_replay['window_30s_rows']}` 30s-window rows.",
            f"- Visuals: `{high_rate_replay['plot_count']}` validated nonblank plots covering source/action timelines, tail spectra, target scorecards, bout distributions, phase-space controls, coupling heatmaps, tail-target heatmaps, muscles, and PAULA channels.",
            f"- Replay source coverage: calcium `{high_rate_replay['calcium_source_frames']}` decoded ZAPBench frames; video `{high_rate_replay['video_source_frames']}` backend frames with source loop max `{high_rate_replay['video_source_loop_index_max']:.0f}`.",
            f"- Stability split: calcium vertical-instability events `{high_rate_replay['calcium_vertical_instability_events']}`; video vertical-instability events `{high_rate_replay['video_vertical_instability_events']}` with body pitch |p95| `{high_rate_replay['video_body_pitch_abs_p95_rad']:.3f}` rad and z-span p95 `{high_rate_replay['video_z_span_p95_mm']:.3f}` mm.",
            f"- Biological target status: inside `{high_rate_replay['inside_target_count']}`, below `{high_rate_replay['below_target_count']}`, above `{high_rate_replay['above_target_count']}` across exact-replay high-rate external comparisons.",
            "",
            "| replay run | source frames | action force/conf mean | bouts | event Hz | TBF median Hz | speed mean/p95 mm/s | z-span p95 mm | vertical-instability events |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in high_rate_replay["rows"]:
        run_name = str(row.get("run", ""))
        vertical = (
            high_rate_replay["calcium_vertical_instability_events"]
            if run_name.startswith("calcium")
            else high_rate_replay["video_vertical_instability_events"]
        )
        lines.append(
            f"| `{run_name}` | {int(_to_float(row.get('source_frames_unique')))} | "
            f"{_to_float(row.get('action_force_mean')):.3f}/{_to_float(row.get('action_confidence_mean')):.3f} | "
            f"{int(_to_float(row.get('bout_count')))} | {_to_float(row.get('event_frequency_hz')):.4f} | "
            f"{_to_float(row.get('realized_tail_frequency_tbf_band_hz_median')):.3f} | "
            f"{_to_float(row.get('speed_xy_mm_s_mean')):.3f}/{_to_float(row.get('speed_xy_mm_s_p95')):.3f} | "
            f"{_to_float(row.get('body_z_span_mm_p95')):.3f} | {vertical} |"
        )
    lines.extend(
        [
            "",
            "Exact-replay interpretation: calcium is the cleaner branch in this 50k run. It expresses ZAPBench/ephys-derived action as short pulses, reaches plausible event frequency and measurable in-band TBF, and does not show vertical-instability events. Video expresses a richer range of force/turn commands and reaches better speed than calcium, but it also generates many high-bend/turn events, 55 vertical-instability events, below-target speed/distance, and too-short interbout intervals. This makes the video branch valuable as an instrumented PoC, but not yet a biologically faithful motion reproduction.",
        ]
    )
    fidelity = metrics.get("simzfish_fidelity_manifest", {})
    fidelity_data = metrics.get("simzfish_fidelity_data", {})
    calibration = metrics.get("simzfish_calibration_manifest", {})
    comparison_rows = metrics.get("simzfish_calibration_comparison", [])
    comparison_lines = []
    for row in comparison_rows:
        comparison_lines.append(
            f"- `{row.get('run')}`: event Hz `{float(row.get('current_noncoast_event_frequency_hz', 0.0)):.4f}`, "
            f"command Hz `{float(row.get('current_noncoast_command_frequency_hz', 0.0)):.4f}`, "
            f"published mean Hz `{float(row.get('published_bout_frequency_mean_hz', 0.0)):.4f}`."
        )
    lines.extend(
        [
            "",
            "## simZFish/Z-Robot Fidelity Check",
            "",
            "The dedicated simZFish audit scores the current Python/MuJoCo video branch at "
            f"`{float(fidelity.get('mean_score', 0.0)):.4f}` mean component fidelity against the public simZFish C/Webots controller "
            "and `Data_Liu_simZFish_2025` resources.  The strongest overlap is the OFF retinal contrast transform. "
            "The deterministic controller-regression harness now shows that direct `Image.c` OFF and DSC equation ports pass, "
            f"while the current MuJoCo adapter has left/right retinal winner matches `{controller.get('retina_left_winner_matches')}`/"
            f"`{controller.get('retina_right_winner_matches')}` on simple synthetic stimuli and direct `Robot.c` target comparison "
            f"mean correlation `{float(controller.get('robot_target_corr_mean', 0.0)):.4f}` with mean RMSE "
            f"`{float(controller.get('robot_target_rmse_mean', 0.0)):.4f}`.  The same harness now compiles the original `OMR.c` "
            f"and `LeakyIntegrator.c`: compiled OMR rows `{controller.get('omr_c_rows')}`, all results ok "
            f"`{controller.get('omr_c_all_results_ok')}`, current-vs-compiled OMR side matches "
            f"`{controller.get('omr_current_side_sign_matches')}/{controller.get('omr_c_rows')}`, and compiled leaky total bouts "
            f"`{controller.get('leaky_c_total_bouts')}`. These results make the boundary sharper: "
            "source-level reference surfaces now exist, but current lab behavior is still not exact simZFish behavior. "
            "The remaining journal-level blockers are runtime use/fitting of those reference trajectories, compiled-C/Webots end-to-end parity, and calibration against the published behavior/neural tables.  "
            f"The public data cache is substantial: {fidelity_data.get('total_files', 0)} files, "
            f"{fidelity_data.get('total_bytes', 0)} bytes, {fidelity_data.get('xlsx_count', 0)} spreadsheets, "
            f"{fidelity_data.get('mat_count', 0)} MAT files, and {fidelity_data.get('mat_nested_numeric_cells', 0)} nested numeric behavior cells.",
            "",
            "## simZFish/Z-Robot Calibration Targets",
            "",
            f"The calibration target atlas extracted `{calibration.get('locomotion_targets', 0)}` locomotion/bout sheets, "
            f"`{calibration.get('neural_targets', 0)}` neural channel summaries, `{calibration.get('rheotaxis_targets', 0)}` rheotaxis/grid traces, "
            f"`{calibration.get('zbot_stats', 0)}` ZBot scalar stats, and `{calibration.get('mat_numeric_cells', 0)}` MAT numeric behavior cells. "
            "This converts the public Z-Robot/simZFish cache into an actual fitting target surface.  Current-vs-published behavior comparison:",
            "",
            *(comparison_lines or ["- No comparison rows were available."]),
            "",
            "## Required Next Experiments Before A Stronger Claim",
            "",
            "1. Replace or fit the live Python video adapter against the compiled `OMR.c`/`LeakyIntegrator.c` trajectories, the DANDI calibrated OMR motor-response target surface, and the published simZFish/Z-Robot behavior-target envelopes.",
            "2. Fit the MuJoCo video branch against `Data_Liu_simZFish_2025` bout-angle, bout-frequency, left/forward/right bout-fraction, OMR, rheotaxis, and ZBot tables.",
            "3. Replace the current coarse selected-video OMR projection with exact recovered DANDI projector movies or a calibrated two-eye/projector generator that preserves eye-specific lower-posterior retinal semantics, then re-run the identifiability, semantic bridge, DANDI-grounded neural motif alignment, and calibrated motor-response audits.",
            "4. Repeat >=50k embodied video runs for low-risk clips identified by both backend robustness and DANDI-alignment audits, and treat high quality-event clips as stress tests rather than validation stimuli.",
            "5. Extend the new 200 Hz high-rate raw segment-angle capture from direct baseline/sensorimotor protocols to the exact browser video/calcium replay branches, then repeat realized TBF and event-scale validation there.",
            "6. Calibrate MuJoCo force, drag, body scale, and action-pulse duration until speed, distance per bout, active-bout duration, and p95 speed match external ZebraZoom/Dryad larval kinematic targets.",
            "7. Add measured larval zebrafish tail curvature, bout angle, beat frequency, swimming speed, and z-motion recordings under matched stimuli, then validate realized MuJoCo kinematics against those traces.",
            "8. Acquire or locate paired natural video, calcium, ephys, and tail/body kinematics before claiming precise video-to-calcium-to-movement reproduction.",
            "",
            "## Bottom Line For A Zebrafish Lab",
            "",
            "This package is suitable as a transparent pre-submission/pre-collaboration audit. It is not yet evidence for a validated digital twin. The honest pitch is: a high-observability MuJoCo zebrafish lab with a ZAPBench-grounded fictive motor replay branch, a simZFish-inspired visual OMR branch, and explicit validation gates for converting it into a biologically rigorous system.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()
    source_register = []
    for source in SOURCES:
        entry = dict(source)
        entry["checked_at"] = "2026-06-03"
        entry["http_probe"] = _fetch_status(source["url"])
        source_register.append(entry)
    source_path = OUT_DIR / "source_register.json"
    source_path.write_text(json.dumps(source_register, indent=2, default=_json_default), encoding="utf-8")

    claim_rows = _claim_rows(metrics)
    trace_rows = _traceability_rows()
    gate_rows = _gate_rows(metrics)
    claim_path = OUT_DIR / "claim_coverage_matrix.csv"
    trace_path = OUT_DIR / "paper_to_code_traceability.csv"
    gate_path = OUT_DIR / "acceptance_gates.csv"
    _write_csv(claim_path, claim_rows)
    _write_csv(trace_path, trace_rows)
    _write_csv(gate_path, gate_rows)

    metrics_path = OUT_DIR / "study_metric_extract.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    report_path = OUT_DIR / "JOURNAL_STYLE_STUDY.md"
    _write_report(
        report_path,
        metrics=metrics,
        source_path=source_path,
        claim_path=claim_path,
        trace_path=trace_path,
        gate_path=gate_path,
    )
    manifest = {
        "out_dir": str(OUT_DIR.resolve()),
        "report": str(report_path.resolve()),
        "source_register": str(source_path.resolve()),
        "claim_coverage_matrix": str(claim_path.resolve()),
        "paper_to_code_traceability": str(trace_path.resolve()),
        "acceptance_gates": str(gate_path.resolve()),
        "study_metric_extract": str(metrics_path.resolve()),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
