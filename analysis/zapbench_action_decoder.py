"""Train a ZAPBench brain-activity-to-tail-action decoder.

This script reads public ZAPBench calcium traces and stimulus covariates from
GCS, creates stimulus-implied tail-action labels from the published covariate
schema, and trains a small ridge decoder from neural trace windows to:

* kick/no kick
* side: none/left/right
* force magnitude in [0, 1]

The labels are not direct tail-electrode labels. ZAPBench's public covariate
matrix encodes the visual stimulus state, while the raw 10-channel stimulus and
ephys file contains the motor/ephys source needed for direct swim-power labels.
The decoder here is therefore a reproducible first bridge from whole-brain
activity to expected fictive motor action under the published stimulus protocol.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_INFERENCE = REPO_ROOT / "active-inference"
if str(ACTIVE_INFERENCE) not in sys.path:
    sys.path.insert(0, str(ACTIVE_INFERENCE))


CONDITION_OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)
CONDITION_NAMES = (
    "gain",
    "dots",
    "flash",
    "taxis",
    "turning",
    "position",
    "open loop",
    "rotation",
    "dark",
)

TRACE_SPEC = {
    "open": True,
    "driver": "zarr3",
    "kvstore": "gs://zapbench-release/volumes/20240930/traces/",
}

STIMULUS_SPEC = {
    "open": True,
    "driver": "zarr",
    "rank": 2,
    "metadata": {"shape": [7879, 26]},
    "kvstore": "gs://zapbench-release/volumes/20240930/stimuli_features/",
}


@dataclass(frozen=True)
class DecoderRun:
    output_dir: Path
    artifact_path: Path
    rows_per_condition: int
    neurons_per_block: int
    neuron_blocks: int
    projection_components: int
    context: int
    ridge_lambda: float
    seed: int
    split: str


def _condition_for_row(row: int) -> int:
    for idx in range(len(CONDITION_OFFSETS) - 1):
        if CONDITION_OFFSETS[idx] <= row < CONDITION_OFFSETS[idx + 1]:
            return idx
    raise ValueError(f"row out of range: {row}")


def _condition_rows(rows_per_condition: int, context: int) -> np.ndarray:
    rows: list[int] = []
    for idx in range(len(CONDITION_NAMES)):
        start = CONDITION_OFFSETS[idx] + max(1, context)
        end = CONDITION_OFFSETS[idx + 1] - 2
        count = min(rows_per_condition, max(1, end - start))
        selected = np.linspace(start, end, count, dtype=np.int32)
        rows.extend(int(v) for v in selected)
    return np.asarray(sorted(set(rows)), dtype=np.int32)


def _train_test_mask(rows: np.ndarray, *, split: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = np.zeros(rows.shape[0], dtype=bool)
    test = np.zeros(rows.shape[0], dtype=bool)
    rng = np.random.default_rng(seed)
    for cond in range(len(CONDITION_NAMES)):
        idx = np.asarray([i for i, row in enumerate(rows) if _condition_for_row(int(row)) == cond], dtype=np.int32)
        if split == "chronological":
            for i in idx:
                row = rows[i]
                start, end = CONDITION_OFFSETS[cond], CONDITION_OFFSETS[cond + 1]
                phase = (int(row) - start) / max(1, end - start)
                if phase < 0.70:
                    train[i] = True
                else:
                    test[i] = True
        elif split == "stratified_random":
            perm = rng.permutation(idx)
            n_train = max(1, int(round(0.70 * len(perm))))
            train[perm[:n_train]] = True
            test[perm[n_train:]] = True
        else:
            raise ValueError(f"unknown split: {split}")
    return train, test


def _side_class(side: np.ndarray, force: np.ndarray) -> np.ndarray:
    cls = np.zeros(side.shape[0], dtype=np.int32)
    active = force >= 0.15
    cls[active & (side < -0.20)] = 1
    cls[active & (side > 0.20)] = 2
    return cls


def _build_action_targets(stimuli: np.ndarray) -> dict[str, np.ndarray]:
    """Build stimulus-implied fictive tail-action labels.

    Dimension references are 0-indexed here. They map to Appendix B.6 of the
    ZAPBench paper, where the dimensions are described using 1-indexed labels.
    """

    s = np.asarray(stimuli, dtype=np.float32)

    gain = s[:, 0]
    dots = s[:, 2]
    flash = s[:, 4]
    taxis_left = s[:, 6]
    taxis_right = s[:, 7]
    turning_velocity = np.abs(s[:, 9])
    turning_sin = s[:, 10]
    turning_cos = s[:, 11]
    position_drive = np.max(np.abs(s[:, 13:17]), axis=1)
    open_loop = s[:, 18]
    rotation = s[:, 19]
    dark = s[:, 21]

    # Lateral action convention: negative -> left, positive -> right.
    # Taxis is modeled as turning toward the bright hemifield; rotation and
    # turning conditions directly specify a lateral stimulus direction.
    side = np.zeros(s.shape[0], dtype=np.float32)
    side += 0.35 * dots
    side += 0.75 * (taxis_right - taxis_left) * 0.5
    side += 0.95 * turning_sin * np.maximum(0.35, turning_velocity)
    side += 0.45 * turning_cos * turning_velocity
    side += 0.85 * rotation
    side = np.clip(side, -1.0, 1.0)

    straight_drive = np.zeros(s.shape[0], dtype=np.float32)
    straight_drive += 0.32 * np.abs(gain)
    straight_drive += 0.60 * (flash > 0).astype(np.float32)
    straight_drive += 0.25 * (flash < 0).astype(np.float32)
    straight_drive += 0.38 * position_drive
    straight_drive += 0.42 * open_loop

    force = np.clip(0.72 * np.abs(side) + straight_drive, 0.0, 1.0)
    force = np.where(dark > 0.5, 0.0, force).astype(np.float32)
    kick = (force >= 0.15).astype(np.float32)
    side_cls = _side_class(side, force)

    return {
        "kick": kick,
        "side": side.astype(np.float32),
        "force": force.astype(np.float32),
        "side_class": side_cls,
    }


def _read_tensorstore(spec: dict[str, Any]) -> Any:
    try:
        import tensorstore as ts
    except ImportError as exc:
        raise SystemExit(
            "tensorstore is required. Run with: "
            "uv run --with tensorstore python -m analysis.zapbench_action_decoder"
        ) from exc
    return ts.open(spec).result()


def _neuron_slices(total_neurons: int, blocks: int, neurons_per_block: int) -> list[slice]:
    max_start = max(0, total_neurons - neurons_per_block)
    starts = np.linspace(0, max_start, blocks, dtype=np.int32)
    return [slice(int(start), int(start) + neurons_per_block) for start in starts]


def _load_selected_traces(trace_ds: Any, blocks: int, neurons_per_block: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    shape = tuple(int(v) for v in trace_ds.shape)
    if len(shape) != 2:
        raise ValueError(f"unexpected trace shape: {shape}")
    slices = _neuron_slices(shape[1], blocks, neurons_per_block)
    arrays: list[np.ndarray] = []
    block_ranges: list[tuple[int, int]] = []
    for sl in slices:
        arr = np.asarray(trace_ds[:, sl].read().result(), dtype=np.float32)
        arrays.append(arr)
        block_ranges.append((int(sl.start), int(sl.stop)))
    traces = np.concatenate(arrays, axis=1)
    return traces, block_ranges


def _window_features(projected: np.ndarray, rows: np.ndarray, context: int) -> np.ndarray:
    features = np.zeros((rows.shape[0], projected.shape[1] * 3), dtype=np.float32)
    for i, row in enumerate(rows):
        start = int(row) - context + 1
        window = projected[start : int(row) + 1]
        features[i] = np.concatenate([window[-1], window.mean(axis=0), window[-1] - window[0]])
    return features


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    gram = xb.T @ xb
    reg = np.eye(gram.shape[0], dtype=np.float32) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(gram + reg, xb.T @ y).astype(np.float32)


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    pred = xb @ weights
    pred[:, 0] = np.clip(pred[:, 0], 0.0, 1.0)
    pred[:, 2] = np.clip(pred[:, 2], 0.0, 1.0)
    return pred


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return 0.0 if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)


def _metrics(
    rows: np.ndarray,
    y: dict[str, np.ndarray],
    pred: np.ndarray,
    *,
    include_per_condition: bool = True,
) -> dict[str, Any]:
    pred_kick = (pred[:, 0] >= 0.5).astype(np.int32)
    true_kick = y["kick"].astype(np.int32)
    pred_side = _side_class(pred[:, 1], pred[:, 2])
    true_side = y["side_class"].astype(np.int32)
    out: dict[str, Any] = {
        "n": int(rows.shape[0]),
        "kick_accuracy": float(np.mean(pred_kick == true_kick)),
        "kick_positive_rate_true": float(np.mean(true_kick)),
        "kick_positive_rate_pred": float(np.mean(pred_kick)),
        "side_accuracy": float(np.mean(pred_side == true_side)),
        "side_non_none_rate_true": float(np.mean(true_side != 0)),
        "side_non_none_rate_pred": float(np.mean(pred_side != 0)),
        "force_mae": float(np.mean(np.abs(pred[:, 2] - y["force"]))),
        "force_rmse": float(math.sqrt(float(np.mean((pred[:, 2] - y["force"]) ** 2)))),
        "force_r2": _r2(y["force"], pred[:, 2]),
        "side_signal_mae": float(np.mean(np.abs(pred[:, 1] - y["side"]))),
    }
    if include_per_condition:
        per_condition: dict[str, Any] = {}
        for cond_id, cond_name in enumerate(CONDITION_NAMES):
            mask = np.asarray([_condition_for_row(int(r)) == cond_id for r in rows], dtype=bool)
            if not np.any(mask):
                continue
            per_condition[cond_name] = _metrics(
                rows[mask],
                {k: v[mask] for k, v in y.items()},
                pred[mask],
                include_per_condition=False,
            )
        out["per_condition"] = per_condition
    return out


def _labels_subset(labels: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in labels.items()}


def _write_report(
    *,
    run: DecoderRun,
    metrics: dict[str, Any],
    block_ranges: list[tuple[int, int]],
    rows: np.ndarray,
    artifact_path: Path,
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# ZAPBench Tail-Action Decoder")
    lines.append("")
    lines.append(f"Generated: 2026-05-13")
    lines.append("")
    lines.append("## Sources Extracted")
    lines.append("")
    lines.append("- ZAPBench landing page: https://zapbench-release.storage.googleapis.com/landing.html")
    lines.append("- Dataset README: https://zapbench-release.storage.googleapis.com/volumes/README.html")
    lines.append("- ICLR/OpenReview paper: https://openreview.net/forum?id=oCHsDpyawq")
    lines.append("- Official code: https://github.com/google-research/zapbench")
    lines.append("- Google Research blog: https://research.google/blog/improving-brain-models-with-zapbench/")
    lines.append("- Public GCS bucket browser: https://storage.googleapis.com/zapbench-release/browse.html#zapbench-release/volumes/")
    lines.append("")
    lines.append("## Data Used")
    lines.append("")
    lines.append("- Calcium traces: `gs://zapbench-release/volumes/20240930/traces/`, zarr3, shape `7879 x 71721`.")
    lines.append("- Stimulus features: `gs://zapbench-release/volumes/20240930/stimuli_features/`, zarr v2, shape `7879 x 26`.")
    lines.append("- Raw stimulus/ephys source exists at `gs://zapbench-release/volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt` and is ~1.73 GB.")
    lines.append("- Selected neuron blocks: " + ", ".join(f"{a}:{b}" for a, b in block_ranges))
    lines.append(f"- Sampled rows: {int(rows.shape[0])} total, {run.rows_per_condition} max per condition.")
    lines.append(f"- Split: `{run.split}`.")
    lines.append("")
    lines.append("## Label Construction")
    lines.append("")
    lines.append("The paper states that the fish was paralyzed for fictive behavior, with left/right tail motor nerve activity recorded by pipettes. The released aligned 26-column covariate matrix encodes the visual stimulus state, not direct tail-power labels. This run therefore trains against stimulus-implied fictive actions:")
    lines.append("")
    lines.append("- `kick`: nonzero expected swim/startle/turn drive.")
    lines.append("- `side`: left/right/none from taxis hemifield contrast, turning direction, dot motion, and rotation direction.")
    lines.append("- `force`: normalized combination of lateral command magnitude plus straight swim/startle drive.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Artifact")
    lines.append("")
    lines.append(f"- Trained decoder artifact: `{artifact_path}`")
    lines.append("- Runtime loader: `active-inference/simulations/zebrafish/action_decoder.py`")
    lines.append("")
    lines.append("## Biological Limit")
    lines.append("")
    lines.append("This is a real ZAPBench calcium-trace decoder, but it is not yet a direct motor-electrode decoder. The next upgrade is to process the raw `stimuli_and_ephys.10chFlt` file into aligned left/right swim-power labels, then retrain the same decoder with direct motor labels instead of stimulus-implied action labels.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_decoder(run: DecoderRun) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.artifact_path.parent.mkdir(parents=True, exist_ok=True)

    trace_ds = _read_tensorstore(TRACE_SPEC)
    stimulus_ds = _read_tensorstore(STIMULUS_SPEC)

    rows = _condition_rows(run.rows_per_condition, run.context)
    stimuli = np.asarray(stimulus_ds[rows, :].read().result(), dtype=np.float32)
    labels = _build_action_targets(stimuli)
    train_mask, test_mask = _train_test_mask(rows, split=run.split, seed=run.seed)

    traces, block_ranges = _load_selected_traces(
        trace_ds, run.neuron_blocks, run.neurons_per_block
    )

    rng = np.random.default_rng(run.seed)
    train_rows = rows[train_mask]
    train_trace_samples = traces[train_rows]
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
    features = _window_features(projected, rows, run.context)
    feature_mean = features[train_mask].mean(axis=0).astype(np.float32)
    feature_std = features[train_mask].std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-4, 1.0, feature_std).astype(np.float32)
    features_z = (features - feature_mean) / feature_std

    y_mat = np.stack([labels["kick"], labels["side"], labels["force"]], axis=1)
    weights = _ridge_fit(features_z[train_mask], y_mat[train_mask], run.ridge_lambda)
    pred_test = _predict(features_z[test_mask], weights)
    metrics = _metrics(rows[test_mask], _labels_subset(labels, test_mask), pred_test)

    pred_train = _predict(features_z[train_mask], weights)
    metrics["train"] = _metrics(rows[train_mask], _labels_subset(labels, train_mask), pred_train)
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
        "source": "ZAPBench public release",
        "trace_spec": TRACE_SPEC["kvstore"],
        "stimulus_spec": STIMULUS_SPEC["kvstore"],
        "condition_offsets": CONDITION_OFFSETS,
        "condition_names": CONDITION_NAMES,
        "selected_neuron_blocks": block_ranges,
        "projection_components": run.projection_components,
        "rows_per_condition": run.rows_per_condition,
        "split": run.split,
        "label_type": "stimulus-implied fictive tail action",
        "biological_limit": "not direct motor-electrode labels; process raw 10ch ephys for that",
    }
    thresholds = {"kick": 0.5, "side_abs": 0.20, "force_none": 0.15}
    np.savez_compressed(
        run.artifact_path,
        projection=projection,
        weights=weights,
        trace_mean=trace_mean,
        trace_std=trace_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        context=np.asarray(run.context, dtype=np.int32),
        rows=rows,
        block_ranges=np.asarray(block_ranges, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        thresholds_json=np.asarray(json.dumps(thresholds, sort_keys=True)),
    )

    # Save a copy of the metrics beside both the report and the model artifact.
    metrics_path = run.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = run.output_dir / "report.md"
    _write_report(
        run=run,
        metrics=metrics,
        block_ranges=block_ranges,
        rows=rows,
        artifact_path=run.artifact_path,
        report_path=report_path,
    )
    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "artifact_path": str(run.artifact_path),
        "selected_neuron_blocks": block_ranges,
    }


def parse_args() -> DecoderRun:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "zapbench_action_decoder",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=ACTIVE_INFERENCE
        / "simulations"
        / "zebrafish"
        / "data"
        / "zapbench_tail_action_decoder.npz",
    )
    parser.add_argument("--rows-per-condition", type=int, default=240)
    parser.add_argument("--neurons-per-block", type=int, default=128)
    parser.add_argument("--neuron-blocks", type=int, default=8)
    parser.add_argument("--projection-components", type=int, default=96)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--split",
        choices=("stratified_random", "chronological"),
        default="stratified_random",
    )
    args = parser.parse_args()
    return DecoderRun(
        output_dir=args.output_dir,
        artifact_path=args.artifact_path,
        rows_per_condition=args.rows_per_condition,
        neurons_per_block=args.neurons_per_block,
        neuron_blocks=args.neuron_blocks,
        projection_components=args.projection_components,
        context=args.context,
        ridge_lambda=args.ridge_lambda,
        seed=args.seed,
        split=args.split,
    )


def main() -> None:
    result = train_decoder(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
