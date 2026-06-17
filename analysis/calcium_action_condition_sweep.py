"""Run calcium-action replay across every ZAPBench stimulus condition.

The replay artifact stores decoded calcium-derived action commands.  This
analysis verifies whether those commands produce corresponding body-level
movement in the zebrafish simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from zapbench_calcium_replay import simulate_replay, summarize_replay  # noqa: E402


DEFAULT_CONDITIONS = (
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


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Calcium-To-Action Condition Sweep",
        "",
        "Decoded public ZAPBench calcium traces were replayed into the zebrafish action controller. "
        "The table compares decoded calcium action rates with the movement state measured from the simulation.",
        "",
        "## Replay Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    replay = payload["replay_summary"]
    lines.extend(
        [
            f"| frames | {replay['frames']} |",
            f"| duration s | {replay['duration_s']:.1f} |",
            f"| global decoded kick rate | {replay['kick_rate']:.3f} |",
            f"| global decoded side-active rate | {replay['side_active_rate']:.3f} |",
            f"| mean decoded force | {replay['force']['mean']:.3f} |",
            "",
            "## Condition Simulations",
            "",
            "| condition | frames | ticks | decoded kick | simulated swim | decoded side | simulated side | speed mean mm/s | speed p95 mm/s | turn abs p95 | FE mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, sim in payload["condition_simulations"].items():
        lines.append(
            f"| {condition} | {sim['frames']} | {sim['ticks']} | "
            f"{sim['decoded_kick_rate']:.3f} | {sim['simulated_swim_rate']:.3f} | "
            f"{sim['decoded_side_active_rate']:.3f} | {sim['simulated_side_active_rate']:.3f} | "
            f"{sim['speed_mm_s']['mean']:.3f} | {sim['speed_mm_s']['p95']:.3f} | "
            f"{abs(sim['turn_bias']['p95']):.3f} | {sim['free_energy']['mean']:.5f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-path",
        type=Path,
        default=SCRIPT_DIR / "cache" / "zapbench" / "zapbench_calcium_action_replay.npz",
    )
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    args = parser.parse_args()

    out_dir = args.out_dir or (SCRIPT_DIR / "out" / "calcium_action_conditions" / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    replay_summary = summarize_replay(args.replay_path)
    condition_simulations: dict[str, Any] = {}
    for condition in args.conditions:
        print(json.dumps({"condition": condition, "status": "start"}), flush=True)
        sim = simulate_replay(args.replay_path, seconds=args.seconds, condition=condition, seed=args.seed)
        condition_simulations[condition] = sim
        print(
            json.dumps(
                {
                    "condition": condition,
                    "ticks": sim["ticks"],
                    "decoded_kick_rate": sim["decoded_kick_rate"],
                    "simulated_swim_rate": sim["simulated_swim_rate"],
                    "decoded_side_active_rate": sim["decoded_side_active_rate"],
                    "simulated_side_active_rate": sim["simulated_side_active_rate"],
                    "speed_mean_mm_s": sim["speed_mm_s"]["mean"],
                    "free_energy_mean": sim["free_energy"]["mean"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "replay_path": str(args.replay_path),
        "seconds_per_condition": args.seconds,
        "seed": args.seed,
        "replay_summary": replay_summary,
        "condition_simulations": condition_simulations,
    }
    summary_path = out_dir / "calcium_condition_summary.json"
    report_path = out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary_path": str(summary_path), "report_path": str(report_path)}))


if __name__ == "__main__":
    main()
