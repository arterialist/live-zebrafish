"""Build compact ZAPBench evidence caches for the lab UI.

The lab server should not pull large remote TensorStore arrays on every
browser request. This script makes two small local caches:

* full 26-column stimulus covariates for every imaging row;
* a sparse trace preview sampled from the same neuron blocks used by the
  direct ephys decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ACTIVE_INFERENCE = REPO_ROOT / "active-inference"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from zapbench_action_decoder import STIMULUS_SPEC, TRACE_SPEC, _read_tensorstore  # noqa: E402


DEFAULT_ARTIFACT = (
    ACTIVE_INFERENCE
    / "simulations"
    / "zebrafish"
    / "data"
    / "zapbench_ephys_tail_action_decoder.npz"
)
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache" / "zapbench"


def _decoder_block_ranges(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return [
            (0, 512),
            (10172, 10684),
            (20345, 20857),
            (30518, 31030),
            (40690, 41202),
            (50863, 51375),
            (61036, 61548),
            (71209, 71721),
        ]
    with np.load(path, allow_pickle=False) as data:
        return [(int(a), int(b)) for a, b in np.asarray(data["block_ranges"], dtype=np.int32)]


def _preview_slices(
    block_ranges: list[tuple[int, int]], neurons_per_block: int
) -> tuple[list[slice], list[int]]:
    slices: list[slice] = []
    neuron_ids: list[int] = []
    for start, stop in block_ranges:
        width = max(1, stop - start)
        count = min(neurons_per_block, width)
        offset = max(0, (width - count) // 2)
        sl = slice(start + offset, start + offset + count)
        slices.append(sl)
        neuron_ids.extend(range(sl.start, sl.stop))
    return slices, neuron_ids


def build_cache(
    *,
    artifact_path: Path,
    cache_dir: Path,
    neurons_per_block: int,
) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    stimulus_ds = _read_tensorstore(STIMULUS_SPEC)
    stimuli = np.asarray(stimulus_ds[:, :].read().result(), dtype=np.float32)
    stimulus_path = cache_dir / "zapbench_stimulus_features.npz"
    np.savez_compressed(
        stimulus_path,
        features=stimuli,
        rows=np.arange(stimuli.shape[0], dtype=np.int32),
        source=np.asarray(str(STIMULUS_SPEC["kvstore"])),
    )

    trace_ds = _read_tensorstore(TRACE_SPEC)
    block_ranges = _decoder_block_ranges(artifact_path)
    slices, neuron_ids = _preview_slices(block_ranges, neurons_per_block)
    traces = np.concatenate(
        [np.asarray(trace_ds[:, sl].read().result(), dtype=np.float32) for sl in slices],
        axis=1,
    )
    trace_path = cache_dir / "zapbench_trace_preview.npz"
    np.savez_compressed(
        trace_path,
        traces=traces,
        rows=np.arange(traces.shape[0], dtype=np.int32),
        neuron_ids=np.asarray(neuron_ids, dtype=np.int32),
        block_ranges=np.asarray(block_ranges, dtype=np.int32),
        source=np.asarray(str(TRACE_SPEC["kvstore"])),
        artifact_path=np.asarray(str(artifact_path)),
    )

    return {
        "stimulus_features": str(stimulus_path),
        "trace_preview": str(trace_path),
        "stimulus_shape": json.dumps(list(stimuli.shape)),
        "trace_shape": json.dumps(list(traces.shape)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--neurons-per-block", type=int, default=12)
    args = parser.parse_args()
    result = build_cache(
        artifact_path=args.artifact_path,
        cache_dir=args.cache_dir,
        neurons_per_block=args.neurons_per_block,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
