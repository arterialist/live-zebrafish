"""Deterministic simZFish controller regression harness.

This script ports the deterministic parts of the public simZFish C/Webots
controller that can be validated locally without Webots:

- ``Image.c`` OFF bipolar cells and direction-selective retinal counters.
- ``Robot.c`` CPG/vSPN tail motor target equations.

It then compares those direct ports with the current Python/MuJoCo adapter.  A
pass here does not mean the whole video branch is exact simZFish; it means the
study now has regression-testable anchors for two important deterministic
controller surfaces and quantitative evidence for the remaining adapter gap.
"""

from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import shutil
import sys
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_INFERENCE = ROOT.parent / "active-inference"
if str(ACTIVE_INFERENCE) not in sys.path:
    sys.path.insert(0, str(ACTIVE_INFERENCE))

from simulations.zebrafish.action_latent import tail_targets_from_scores  # noqa: E402
from simulations.zebrafish.simzfish_omr import SimZFishOMRActionAdapter, SimZFishRetina  # noqa: E402


OUT_DIR = ROOT / "analysis" / "out" / "simzfish_controller_regression_20260603"
SIMZFISH_SRC = ROOT / "analysis" / "cache" / "z_robot" / "simzfish_raw" / "controllers" / "ZebrafishRobot_mini"
C_SHIM_DIR = OUT_DIR / "c_shim"

IMAGE_H = 240
IMAGE_W = 320
GAP = 3
GAP_BI = 2
E_NUMBER = 2.71828


class COmrState(ctypes.Structure):
    _fields_ = [
        ("RPT_left_eye", ctypes.c_float * 9),
        ("LPT_right_eye", ctypes.c_float * 9),
        ("LoB1", ctypes.c_float),
        ("LoB2", ctypes.c_float),
        ("RoB1", ctypes.c_float),
        ("RoB2", ctypes.c_float),
        ("LB1", ctypes.c_float),
        ("LB2", ctypes.c_float),
        ("RB1", ctypes.c_float),
        ("RB2", ctypes.c_float),
        ("LiB", ctypes.c_float),
        ("RiB", ctypes.c_float),
        ("LioB", ctypes.c_float),
        ("RioB", ctypes.c_float),
        ("LiMm", ctypes.c_float),
        ("RiMm", ctypes.c_float),
        ("LMm1", ctypes.c_float),
        ("LMm2", ctypes.c_float),
        ("RMm1", ctypes.c_float),
        ("RMm2", ctypes.c_float),
        ("LoMm", ctypes.c_float),
        ("RoMm", ctypes.c_float),
        ("LS1", ctypes.c_float),
        ("LS2", ctypes.c_float),
        ("RS1", ctypes.c_float),
        ("RS2", ctypes.c_float),
        ("LMLF", ctypes.c_float),
        ("RMLF", ctypes.c_float),
        ("LLHB", ctypes.c_float),
        ("RLHB", ctypes.c_float),
    ]


class CLeakyState(ctypes.Structure):
    _fields_ = [
        ("SS_MLF", ctypes.c_float),
        ("LLHB", ctypes.c_float),
        ("RLHB", ctypes.c_float),
        ("command", ctypes.c_int),
        ("probability_LT", ctypes.c_float),
        ("probability_FB", ctypes.c_float),
        ("probability_RT", ctypes.c_float),
        ("phase", ctypes.c_double),
        ("noise_SS_MLF_deduction_rate", ctypes.c_double),
        ("SS_MLF_countdown", ctypes.c_float),
        ("SS_LVSPNs_countdown", ctypes.c_float),
        ("SS_RVSPNs_countdown", ctypes.c_float),
        ("noise_turning_LSPN", ctypes.c_float),
        ("noise_turning_RSPN", ctypes.c_float),
        ("noise_forward_LSPN", ctypes.c_float),
        ("noise_forward_RSPN", ctypes.c_float),
        ("bout_flag", ctypes.c_int),
        ("swim_current", ctypes.c_float),
        ("left_descending_ampl", ctypes.c_float),
        ("right_descending_ampl", ctypes.c_float),
        ("LVSPNs_current", ctypes.c_float),
        ("RVSPNs_current", ctypes.c_float),
    ]


def _write_c_reference_shim(model_key: str) -> tuple[Path, Path]:
    include_dir = C_SHIM_DIR / "include"
    webots_dir = include_dir / "webots"
    include_dir.mkdir(parents=True, exist_ok=True)
    webots_dir.mkdir(parents=True, exist_ok=True)

    (include_dir / "Utils.h").write_text(
        textwrap.dedent(
            """
            #ifndef SIMZFISH_UTILS_STUB_H
            #define SIMZFISH_UTILS_STUB_H
            #include <stdbool.h>
            #define MAX_LINE_LEN 4096
            #define MAX_PARAM_NAME_LEN 256
            #define MAX_VALUE_LEN 256
            float gaussRand(void);
            double clamp(double value, double min, double max);
            float getMin(float a, float b);
            #endif
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (include_dir / "LeakyIntegrator.h").write_text(
        textwrap.dedent(
            """
            #ifndef SIMZFISH_LEAKY_INTEGRATOR_STUB_H
            #define SIMZFISH_LEAKY_INTEGRATOR_STUB_H
            #include <stdbool.h>
            #endif
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (webots_dir / "robot.h").write_text(
        textwrap.dedent(
            """
            #ifndef WEBOTS_ROBOT_STUB_H
            #define WEBOTS_ROBOT_STUB_H
            #include <stdbool.h>
            double wb_robot_get_time(void);
            #endif
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    combined_c = C_SHIM_DIR / f"simzfish_controller_reference_{model_key}.c"
    lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = C_SHIM_DIR / f"libsimzfish_controller_reference_{model_key}{lib_ext}"
    combined_c.write_text(
        textwrap.dedent(
            f"""
            #include <stdbool.h>
            #include <math.h>
            #include <stdlib.h>
            #include <stdio.h>
            #include <string.h>

            double wb_robot_get_time(void) {{ return 0.0; }}

            #include "{(SIMZFISH_SRC / 'Utils.c').as_posix()}"
            #include "{(SIMZFISH_SRC / 'OMR.c').as_posix()}"
            #include "{(SIMZFISH_SRC / 'LeakyIntegrator.c').as_posix()}"

            typedef struct {{
                float RPT_left_eye[9];
                float LPT_right_eye[9];
                float LoB1, LoB2, RoB1, RoB2, LB1, LB2, RB1, RB2;
                float LiB, RiB, LioB, RioB, LiMm, RiMm, LMm1, LMm2, RMm1, RMm2;
                float LoMm, RoMm, LS1, LS2, RS1, RS2;
                float LMLF, RMLF, LLHB, RLHB;
            }} SimZFishOMRState;

            typedef struct {{
                float SS_MLF;
                float LLHB;
                float RLHB;
                int command;
                float probability_LT;
                float probability_FB;
                float probability_RT;
                double phase;
                double noise_SS_MLF_deduction_rate;
                float SS_MLF_countdown;
                float SS_LVSPNs_countdown;
                float SS_RVSPNs_countdown;
                float noise_turning_LSPN;
                float noise_turning_RSPN;
                float noise_forward_LSPN;
                float noise_forward_RSPN;
                int bout_flag;
                float swim_current;
                float left_descending_ampl;
                float right_descending_ampl;
                float LVSPNs_current;
                float RVSPNs_current;
            }} SimZFishLeakyState;

            void simzfish_seed(unsigned int seed) {{
                srand(seed);
            }}

            int simzfish_omr_step(
                int model_a,
                int bout_flag,
                const float *count_left_eye,
                const float *count_right_eye,
                unsigned int cycles,
                SimZFishOMRState *state
            ) {{
                return OMRs(
                    model_a != 0,
                    bout_flag != 0,
                    count_left_eye,
                    count_right_eye,
                    cycles,
                    state->RPT_left_eye,
                    state->LPT_right_eye,
                    &state->LoB1,
                    &state->LoB2,
                    &state->RoB1,
                    &state->RoB2,
                    &state->LB1,
                    &state->LB2,
                    &state->RB1,
                    &state->RB2,
                    &state->LiB,
                    &state->RiB,
                    &state->LioB,
                    &state->RioB,
                    &state->LiMm,
                    &state->RiMm,
                    &state->LMm1,
                    &state->LMm2,
                    &state->RMm1,
                    &state->RMm2,
                    &state->LoMm,
                    &state->RoMm,
                    &state->LS1,
                    &state->LS2,
                    &state->RS1,
                    &state->RS2,
                    &state->LMLF,
                    &state->RMLF,
                    &state->LLHB,
                    &state->RLHB
                );
            }}

            void simzfish_leaky_reset(SimZFishLeakyState *state) {{
                memset(state, 0, sizeof(*state));
            }}

            void simzfish_leaky_step(
                float bearing,
                float position_x,
                float position_z,
                float accumulate_x,
                float accumulate_z,
                float lmlf,
                float rmlf,
                SimZFishLeakyState *state
            ) {{
                bool bout_flag_bool = state->bout_flag != 0;
                leakyIntegratorModule(
                    false,
                    bearing,
                    position_x,
                    position_z,
                    accumulate_x,
                    accumulate_z,
                    &state->SS_MLF,
                    lmlf,
                    rmlf,
                    &state->LLHB,
                    &state->RLHB,
                    &state->command,
                    &state->probability_LT,
                    &state->probability_FB,
                    &state->probability_RT,
                    &state->phase,
                    &state->noise_SS_MLF_deduction_rate,
                    &state->SS_MLF_countdown,
                    &state->SS_LVSPNs_countdown,
                    &state->SS_RVSPNs_countdown,
                    &state->noise_turning_LSPN,
                    &state->noise_turning_RSPN,
                    &state->noise_forward_LSPN,
                    &state->noise_forward_RSPN,
                    &bout_flag_bool,
                    &state->swim_current,
                    &state->left_descending_ampl,
                    &state->right_descending_ampl,
                    &state->LVSPNs_current,
                    &state->RVSPNs_current
                );
                state->bout_flag = bout_flag_bool ? 1 : 0;
            }}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return combined_c, lib_path


def _compile_c_reference_library(model_key: str) -> Path:
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        raise RuntimeError("No C compiler found for simZFish reference compilation")
    combined_c, lib_path = _write_c_reference_shim(model_key)
    cmd = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c99",
        "-I",
        str(C_SHIM_DIR / "include"),
        str(combined_c),
        "-o",
        str(lib_path),
        "-lm",
    ]
    result = subprocess.run(cmd, cwd=SIMZFISH_SRC, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to compile simZFish C reference shim:\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )
    return lib_path


def _load_c_reference(model_key: str) -> ctypes.CDLL:
    lib_path = _compile_c_reference_library(model_key)
    lib = ctypes.CDLL(str(lib_path))
    lib.simzfish_seed.argtypes = [ctypes.c_uint]
    lib.simzfish_seed.restype = None
    lib.simzfish_omr_step.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint,
        ctypes.POINTER(COmrState),
    ]
    lib.simzfish_omr_step.restype = ctypes.c_int
    lib.simzfish_leaky_reset.argtypes = [ctypes.POINTER(CLeakyState)]
    lib.simzfish_leaky_reset.restype = None
    lib.simzfish_leaky_step.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(CLeakyState),
    ]
    lib.simzfish_leaky_step.restype = None
    return lib


class _Pushd:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.previous: str | None = None

    def __enter__(self) -> None:
        self.previous = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *_exc: object) -> None:
        if self.previous is not None:
            os.chdir(self.previous)


def _sigmoid_c(x: np.ndarray | float, omega: float, bias: float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.power(E_NUMBER, -omega * (np.asarray(x) - bias)))


def _off_bipolar_loop(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    out = np.zeros_like(previous, dtype=np.float64)
    for x in range(1, IMAGE_H - 1):
        for y in range(1, IMAGE_W - 1):
            out[x, y] = 1.0 / (1.0 + E_NUMBER ** (-(float(previous[x, y]) - float(current[x, y]) - 3.0)))
    return out


def _off_bipolar_vectorized(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    out = np.zeros_like(previous, dtype=np.float64)
    out[1 : IMAGE_H - 1, 1 : IMAGE_W - 1] = _sigmoid_c(
        previous[1 : IMAGE_H - 1, 1 : IMAGE_W - 1].astype(float)
        - current[1 : IMAGE_H - 1, 1 : IMAGE_W - 1].astype(float),
        omega=1.0,
        bias=3.0,
    )
    return out


def _dsc_counts_loop(off_prev: np.ndarray, off_curr: np.ndarray) -> dict[str, np.ndarray]:
    left = np.zeros(9, dtype=np.float64)
    right = np.zeros(9, dtype=np.float64)
    for x in range(IMAGE_H // 2 + 25, IMAGE_H - 4, GAP):
        for y in range(3, IMAGE_W // 2, GAP):
            inputs = {
                0: off_curr[x, y] - off_prev[x - GAP_BI, y],
                2: off_curr[x, y] - off_prev[x, y + GAP_BI],
                4: off_curr[x, y] - off_prev[x + GAP_BI, y],
                6: off_curr[x, y] - off_prev[x, y - GAP_BI],
            }
            for z, value in inputs.items():
                left[z] += 1.0 / (1.0 + E_NUMBER ** (-100.0 * (float(value) - 0.3)))
            left[8] += off_curr[x, y]
    for x in range(IMAGE_H // 2 + 25, IMAGE_H - 4, GAP):
        for y in range(IMAGE_W // 2 - 1, IMAGE_W - 4, GAP):
            inputs = {
                0: off_curr[x, y] - off_prev[x - GAP_BI, y],
                2: off_curr[x, y] - off_prev[x, y - GAP_BI],
                4: off_curr[x, y] - off_prev[x + GAP_BI, y],
                6: off_curr[x, y] - off_prev[x, y + GAP_BI],
            }
            for z, value in inputs.items():
                right[z] += 1.0 / (1.0 + E_NUMBER ** (-100.0 * (float(value) - 0.3)))
            right[8] += off_curr[x, y]
    return {"left": left, "right": right}


def _dsc_counts_vectorized(off_prev: np.ndarray, off_curr: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.arange(IMAGE_H // 2 + 25, IMAGE_H - 4, GAP)
    left_cols = np.arange(3, IMAGE_W // 2, GAP)
    right_cols = np.arange(IMAGE_W // 2 - 1, IMAGE_W - 4, GAP)

    def _eye(cols: np.ndarray, *, right_eye: bool) -> np.ndarray:
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        out = np.zeros(9, dtype=np.float64)
        if right_eye:
            inputs = {
                0: off_curr[rr, cc] - off_prev[rr - GAP_BI, cc],
                2: off_curr[rr, cc] - off_prev[rr, cc - GAP_BI],
                4: off_curr[rr, cc] - off_prev[rr + GAP_BI, cc],
                6: off_curr[rr, cc] - off_prev[rr, cc + GAP_BI],
            }
        else:
            inputs = {
                0: off_curr[rr, cc] - off_prev[rr - GAP_BI, cc],
                2: off_curr[rr, cc] - off_prev[rr, cc + GAP_BI],
                4: off_curr[rr, cc] - off_prev[rr + GAP_BI, cc],
                6: off_curr[rr, cc] - off_prev[rr, cc - GAP_BI],
            }
        for z, values in inputs.items():
            out[z] = float(np.sum(_sigmoid_c(values, omega=100.0, bias=0.3)))
        out[8] = float(np.sum(off_curr[rr, cc]))
        return out

    return {"left": _eye(left_cols, right_eye=False), "right": _eye(right_cols, right_eye=True)}


def _synthetic_frames(direction: str, step: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    older = np.full((IMAGE_H, IMAGE_W), 190, dtype=np.uint8)
    previous = older.copy()
    current = older.copy()
    flow = np.zeros((IMAGE_H, IMAGE_W, 2), dtype=np.float32)
    # Draw paired lower-field dark bars so both eye halves receive stimulation.
    centers = [(84, 182), (236, 182)]
    dx, dy = {
        "posterior": (-step, 0),
        "anterior": (step, 0),
        "superior": (0, -step),
        "inferior": (0, step),
    }[direction]
    for cx, cy in centers:
        for frame, offset in [(older, -1), (previous, 0), (current, 1)]:
            x0 = int(cx + offset * dx)
            y0 = int(cy + offset * dy)
            frame[y0 - 10 : y0 + 11, x0 - 12 : x0 + 13] = 30
        current_mask = np.zeros((IMAGE_H, IMAGE_W), dtype=bool)
        x1 = int(cx + dx)
        y1 = int(cy + dy)
        current_mask[y1 - 10 : y1 + 11, x1 - 12 : x1 + 13] = True
        flow[current_mask, 0] = dx
        flow[current_mask, 1] = dy
    return older, previous, current, flow


def _winner_and_margin(counts: np.ndarray) -> tuple[str, float]:
    labels = {0: "superior", 2: "anterior", 4: "inferior", 6: "posterior"}
    ranked = sorted(((float(counts[i]), labels[i]) for i in labels), reverse=True)
    margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else float("nan")
    return ranked[0][1], float(margin)


def _retinal_current_adapter(previous: np.ndarray, current: np.ndarray, flow: np.ndarray) -> dict[str, Any]:
    retina = SimZFishRetina()
    retinal = retina.extract(
        current_gray=current.astype(np.float32) / 255.0,
        previous_gray=previous.astype(np.float32) / 255.0,
        residual_flow=flow.astype(np.float32),
    )
    data = retinal.as_dict()
    left_vec = np.array(
        [data["left_superior"], data["left_anterior"], data["left_inferior"], data["left_posterior"]],
        dtype=float,
    )
    right_vec = np.array(
        [data["right_superior"], data["right_anterior"], data["right_inferior"], data["right_posterior"]],
        dtype=float,
    )
    labels = ["superior", "anterior", "inferior", "posterior"]
    data["left_winner"] = labels[int(np.argmax(left_vec))]
    data["right_winner"] = labels[int(np.argmax(right_vec))]
    return data


def _retina_regression() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction in ["superior", "anterior", "inferior", "posterior"]:
        older, previous, current, flow = _synthetic_frames(direction)
        off_prev_loop = _off_bipolar_loop(older, previous)
        off_curr_loop = _off_bipolar_loop(previous, current)
        off_prev_vec = _off_bipolar_vectorized(older, previous)
        off_curr_vec = _off_bipolar_vectorized(previous, current)
        dsc_loop = _dsc_counts_loop(off_prev_loop, off_curr_loop)
        dsc_vec = _dsc_counts_vectorized(off_prev_vec, off_curr_vec)
        current_adapter = _retinal_current_adapter(previous, current, flow)
        exact_off_error = max(
            float(np.max(np.abs(off_prev_loop - off_prev_vec))),
            float(np.max(np.abs(off_curr_loop - off_curr_vec))),
        )
        exact_dsc_error = max(
            float(np.max(np.abs(dsc_loop["left"] - dsc_vec["left"]))),
            float(np.max(np.abs(dsc_loop["right"] - dsc_vec["right"]))),
        )
        left_winner, left_margin = _winner_and_margin(dsc_loop["left"])
        right_winner, right_margin = _winner_and_margin(dsc_loop["right"])
        rows.append(
            {
                "stimulus_direction": direction,
                "image_c_off_loop_vs_vectorized_max_abs_error": exact_off_error,
                "image_c_dsc_loop_vs_vectorized_max_abs_error": exact_dsc_error,
                "direct_port_left_winner": left_winner,
                "direct_port_right_winner": right_winner,
                "direct_port_left_winner_margin": left_margin,
                "direct_port_right_winner_margin": right_margin,
                "current_adapter_left_winner": current_adapter["left_winner"],
                "current_adapter_right_winner": current_adapter["right_winner"],
                "current_matches_direct_left": current_adapter["left_winner"] == left_winner,
                "current_matches_direct_right": current_adapter["right_winner"] == right_winner,
                "direct_left_counts": dsc_loop["left"].tolist(),
                "direct_right_counts": dsc_loop["right"].tolist(),
                "current_adapter": current_adapter,
            }
        )
    return rows


def _current_action_terminal(direction: str, steps: int = 120) -> dict[str, Any]:
    _older, previous, current, flow = _synthetic_frames(direction)
    adapter = SimZFishOMRActionAdapter()
    latent = None
    diagnostics_out: dict[str, Any] = {}
    motion_energy = float(np.clip(np.mean(np.linalg.norm(flow, axis=2)) / 2.5, 0.0, 1.0))
    for step in range(steps):
        latent, diagnostics_out = adapter.action_from_frame(
            current_gray=current.astype(np.float32) / 255.0,
            previous_gray=previous.astype(np.float32) / 255.0,
            residual_flow=flow.astype(np.float32),
            features={"motion_energy": motion_energy, "startle": 0.0, "asymmetry": 0.0},
            diagnostics={"flow_reliability": 1.0, "residual_flow_coherence": 1.0, "flow_coherence": 1.0},
            file_name=f"synthetic_{direction}",
            frame_index=step,
            video_time_s=step / 40.0,
            sample_hz=40.0,
        )
    if latent is None:
        raise RuntimeError("Current adapter did not produce a latent")
    return {
        "current_kick": float(latent.kick),
        "current_force": float(latent.force),
        "current_side_score": float(latent.side_score),
        "current_confidence": float(latent.confidence),
        "current_bout_type": latent.bout_type,
        "current_tail_frequency_hz": float(latent.tail_frequency_hz),
        "current_left_pt": float(diagnostics_out.get("simzfish_left_pt", 0.0)),
        "current_right_pt": float(diagnostics_out.get("simzfish_right_pt", 0.0)),
        "current_turn_state": float(diagnostics_out.get("simzfish_turn_state", 0.0)),
        "current_bout_state": float(diagnostics_out.get("simzfish_bout_state", 0.0)),
    }


def _sign(value: float, eps: float = 1e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _omr_c_regression(retina_rows: list[dict[str, Any]], steps: int = 1500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_by_direction = {
        str(row["stimulus_direction"]): _current_action_terminal(str(row["stimulus_direction"]))
        for row in retina_rows
    }
    for model_name, model_a in [("A", True), ("C", False)]:
        lib = _load_c_reference(f"omr_model_{model_name.lower()}")
        lib.simzfish_seed(424242 if model_a else 424243)
        with _Pushd(SIMZFISH_SRC):
            for row in retina_rows:
                direction = str(row["stimulus_direction"])
                state = COmrState()
                left_counts = (ctypes.c_float * 9)(*map(float, row["direct_left_counts"]))
                right_counts = (ctypes.c_float * 9)(*map(float, row["direct_right_counts"]))
                result = 0
                for cycle in range(steps):
                    result = int(
                        lib.simzfish_omr_step(
                            1 if model_a else 0,
                            0,
                            left_counts,
                            right_counts,
                            cycle + 1,
                            ctypes.byref(state),
                        )
                    )
                    if result != 0:
                        break
                current = current_by_direction[direction]
                c_side_score = float(state.RLHB - state.LLHB)
                c_forward_drive = float(0.5 * (state.LMLF + state.RMLF))
                rows.append(
                    {
                        "model": model_name,
                        "stimulus_direction": direction,
                        "cycles": steps,
                        "c_result": result,
                        "c_LMLF": float(state.LMLF),
                        "c_RMLF": float(state.RMLF),
                        "c_LLHB": float(state.LLHB),
                        "c_RLHB": float(state.RLHB),
                        "c_forward_drive": c_forward_drive,
                        "c_side_score_RLHB_minus_LLHB": c_side_score,
                        "c_RPT_left_eye": [float(x) for x in state.RPT_left_eye],
                        "c_LPT_right_eye": [float(x) for x in state.LPT_right_eye],
                        "c_pretectum": {
                            "LoB1": float(state.LoB1),
                            "LoB2": float(state.LoB2),
                            "RoB1": float(state.RoB1),
                            "RoB2": float(state.RoB2),
                            "LB1": float(state.LB1),
                            "LB2": float(state.LB2),
                            "RB1": float(state.RB1),
                            "RB2": float(state.RB2),
                            "LiB": float(state.LiB),
                            "RiB": float(state.RiB),
                            "LiMm": float(state.LiMm),
                            "RiMm": float(state.RiMm),
                            "LMm1": float(state.LMm1),
                            "RMm1": float(state.RMm1),
                            "LS1": float(state.LS1),
                            "RS1": float(state.RS1),
                        },
                        **current,
                        "current_side_matches_c": _sign(float(current["current_side_score"])) == _sign(c_side_score),
                        "current_force_minus_c_forward": float(current["current_force"]) - c_forward_drive,
                    }
                )
    return rows


def _leaky_c_regression(steps: int = 2500) -> list[dict[str, Any]]:
    lib = _load_c_reference("leaky_integrator")
    lib.simzfish_seed(98765)
    cases = [
        {"case": "no_drive", "lmlf": 0.0, "rmlf": 0.0, "llhb": 0.0, "rlhb": 0.0},
        {"case": "forward_drive", "lmlf": 0.85, "rmlf": 0.85, "llhb": 0.05, "rlhb": 0.05},
        {"case": "left_hindbrain_drive", "lmlf": 0.55, "rmlf": 0.55, "llhb": 0.85, "rlhb": 0.05},
        {"case": "right_hindbrain_drive", "lmlf": 0.55, "rmlf": 0.55, "llhb": 0.05, "rlhb": 0.85},
    ]
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        state = CLeakyState()
        lib.simzfish_leaky_reset(ctypes.byref(state))
        state.LLHB = float(case["llhb"])
        state.RLHB = float(case["rlhb"])
        previous_bout = 0
        bout_onsets: list[int] = []
        commands: list[int] = []
        duty_samples = 0
        swim_samples: list[float] = []
        ss_peak = 0.0
        for tick in range(steps):
            state.LLHB = float(case["llhb"])
            state.RLHB = float(case["rlhb"])
            lib.simzfish_leaky_step(
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(float(case["lmlf"])),
                ctypes.c_float(float(case["rmlf"])),
                ctypes.byref(state),
            )
            ss_peak = max(ss_peak, float(state.SS_MLF))
            if int(state.bout_flag):
                duty_samples += 1
                swim_samples.append(float(state.swim_current))
            if int(state.bout_flag) and not previous_bout:
                bout_onsets.append(tick)
                commands.append(int(state.command))
            previous_bout = int(state.bout_flag)
        command_counts = {
            "left": sum(1 for command in commands if command == -1),
            "forward": sum(1 for command in commands if command == 0),
            "right": sum(1 for command in commands if command == 1),
        }
        rows.append(
            {
                "case": str(case["case"]),
                "seed_sequence_index": index,
                "steps": steps,
                "lmlf": float(case["lmlf"]),
                "rmlf": float(case["rmlf"]),
                "llhb": float(case["llhb"]),
                "rlhb": float(case["rlhb"]),
                "bout_onsets": bout_onsets,
                "first_bout_tick": bout_onsets[0] if bout_onsets else -1,
                "bout_count": len(bout_onsets),
                "bout_duty_fraction": duty_samples / steps,
                "command_counts": command_counts,
                "mean_swim_current_during_bout": float(np.mean(swim_samples)) if swim_samples else 0.0,
                "max_swim_current": float(np.max(swim_samples)) if swim_samples else 0.0,
                "terminal_SS_MLF": float(state.SS_MLF),
                "peak_SS_MLF": ss_peak,
                "terminal_countdown": float(state.SS_MLF_countdown),
                "terminal_prob_LT": float(state.probability_LT),
                "terminal_prob_FB": float(state.probability_FB),
                "terminal_prob_RT": float(state.probability_RT),
                "terminal_phase": float(state.phase),
            }
        )
    return rows


def _robot_target_positions(
    *,
    phase: float,
    left_ampl: float,
    right_ampl: float,
    lmlf: float,
    rmlf: float,
    lvspn: float,
    rvspn: float,
    bout_flag: bool,
) -> dict[str, Any]:
    seg_ampl = np.array([0.202952981, 0.36415025, 0.481232303, 0.593806825, 0.69596167, 1.0], dtype=float)
    left_cpg = np.zeros(7, dtype=float)
    right_cpg = np.zeros(7, dtype=float)
    left_secondary = np.zeros(7, dtype=float)
    right_secondary = np.zeros(7, dtype=float)
    target = np.zeros(6, dtype=float)
    for i in range(7):
        if bout_flag:
            left_cpg[i] = math.sin(phase + i * (1.2 * math.pi / 6.0)) + 1.01
            right_cpg[i] = math.sin(phase + math.pi + i * (1.2 * math.pi / 6.0)) + 1.01
        else:
            left_cpg[i] = 0.01
            right_cpg[i] = 0.01
        if i == 0 or i == 1:
            left_mod = left_ampl * left_cpg[i] + lmlf + lvspn * 1.2
            right_mod = right_ampl * right_cpg[i] + rmlf + rvspn * 1.2
        else:
            left_mod = left_ampl * left_cpg[i] + lvspn * 1.2
            right_mod = right_ampl * right_cpg[i] + rvspn * 1.2
        left_secondary[i] = 1.0 / (1.0 + E_NUMBER ** (-2.0 * (left_mod - 1.4)))
        right_secondary[i] = 1.0 / (1.0 + E_NUMBER ** (-2.0 * (right_mod - 1.4)))
        if i < 6:
            target[i] = (right_secondary[i] - left_secondary[i]) * 1.5 * seg_ampl[i]
    return {
        "target_position": target,
        "left_cpg": left_cpg,
        "right_cpg": right_cpg,
        "left_secondary_motor_neuron": left_secondary,
        "right_secondary_motor_neuron": right_secondary,
    }


def _tail_angle(target_position: np.ndarray) -> float:
    tp = np.asarray(target_position, dtype=float)
    cumulative = np.cumsum(tp[:6])
    x = 0.126 * float(np.sum(np.cos(cumulative[:5])) + 2.43 * math.cos(cumulative[5]))
    y = 0.126 * float(np.sum(np.sin(cumulative[:5])) + 2.43 * math.sin(cumulative[5]))
    return float(math.degrees(math.atan2(y, x)))


def _resample_current_to_six(values: tuple[float, ...] | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, 6)
    return np.interp(x_new, x_old, arr)


def _robot_regression() -> list[dict[str, Any]]:
    cases = [
        {"name": "forward_bout", "phase": 0.2, "left_ampl": 0.9, "right_ampl": 0.9, "lmlf": 0.5, "rmlf": 0.5, "lvspn": 0.0, "rvspn": 0.0, "side": 0.0},
        {"name": "left_turn", "phase": math.pi, "left_ampl": 0.9, "right_ampl": 0.2, "lmlf": 0.7, "rmlf": 0.1, "lvspn": 0.8, "rvspn": 0.0, "side": -1.0},
        {"name": "right_turn", "phase": 0.0, "left_ampl": 0.2, "right_ampl": 0.9, "lmlf": 0.1, "rmlf": 0.7, "lvspn": 0.0, "rvspn": 0.8, "side": 1.0},
        {"name": "glide", "phase": 0.7, "left_ampl": 0.0, "right_ampl": 0.0, "lmlf": 0.05, "rmlf": 0.05, "lvspn": 0.0, "rvspn": 0.0, "side": 0.0},
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        direct = _robot_target_positions(
            phase=case["phase"],
            left_ampl=case["left_ampl"],
            right_ampl=case["right_ampl"],
            lmlf=case["lmlf"],
            rmlf=case["rmlf"],
            lvspn=case["lvspn"],
            rvspn=case["rvspn"],
            bout_flag=case["name"] != "glide",
        )
        target = np.asarray(direct["target_position"], dtype=float)
        force = float(max(case["left_ampl"], case["right_ampl"], abs(case["lmlf"] - case["rmlf"])))
        current = _resample_current_to_six(
            tail_targets_from_scores(force=force, side_score=case["side"], phase=case["phase"], n_segments=16)
        )
        if np.std(target) > 1e-9 and np.std(current) > 1e-9:
            corr = float(np.corrcoef(target, current)[0, 1])
        else:
            corr = float("nan")
        rmse = float(np.sqrt(np.mean((target - current) ** 2)))
        rows.append(
            {
                "case": case["name"],
                "direct_robot_c_tail_angle_deg": _tail_angle(target),
                "direct_robot_c_target_rms": float(np.sqrt(np.mean(target * target))),
                "current_tail_latent_rms": float(np.sqrt(np.mean(current * current))),
                "current_vs_direct_corr": corr,
                "current_vs_direct_rmse": rmse,
                "direct_target_position": target.tolist(),
                "current_resampled_tail_targets": current.tolist(),
                "direct_left_secondary": direct["left_secondary_motor_neuron"].tolist(),
                "direct_right_secondary": direct["right_secondary_motor_neuron"].tolist(),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def _plot_retina(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["stimulus_direction"] for row in rows]
    left_match = [1.0 if row["current_matches_direct_left"] else 0.0 for row in rows]
    right_match = [1.0 if row["current_matches_direct_right"] else 0.0 for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, left_match, width=0.36, label="left eye")
    ax.bar(x + 0.18, right_match, width=0.36, label="right eye")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("winner match")
    ax.set_title("Current Retinal Adapter Winner vs Direct Image.c DSC Port")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_robot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.8 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    x = np.arange(6)
    for ax, row in zip(axes, rows):
        direct = np.asarray(row["direct_target_position"], dtype=float)
        current = np.asarray(row["current_resampled_tail_targets"], dtype=float)
        ax.plot(x, direct, marker="o", label="direct Robot.c")
        ax.plot(x, current, marker="s", label="current latent resampled")
        ax.set_ylabel("target")
        ax.set_title(f"{row['case']}: corr={row['current_vs_direct_corr']:.3f}, rmse={row['current_vs_direct_rmse']:.3f}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("six simZFish motor coordinates")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_omr_c(rows: list[dict[str, Any]], path: Path) -> None:
    directions = ["superior", "anterior", "inferior", "posterior"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(directions))
    width = 0.18
    for idx, model in enumerate(["A", "C"]):
        model_rows = {row["stimulus_direction"]: row for row in rows if row["model"] == model}
        side = [float(model_rows[d]["c_side_score_RLHB_minus_LLHB"]) for d in directions]
        forward = [float(model_rows[d]["c_forward_drive"]) for d in directions]
        axes[0].bar(x + (idx - 0.5) * width, side, width=width, label=f"C OMR model {model}")
        axes[1].bar(x + (idx - 0.5) * width, forward, width=width, label=f"C OMR model {model}")
    current_rows = {row["stimulus_direction"]: row for row in rows if row["model"] == "A"}
    current_side = [float(current_rows[d]["current_side_score"]) for d in directions]
    current_force = [float(current_rows[d]["current_force"]) for d in directions]
    axes[0].plot(x, current_side, color="black", marker="o", linewidth=1.5, label="current adapter side")
    axes[1].plot(x, current_force, color="black", marker="o", linewidth=1.5, label="current adapter force")
    axes[0].set_ylabel("side score")
    axes[1].set_ylabel("forward / force")
    axes[1].set_xticks(x, directions)
    axes[0].set_title("Compiled OMR.c Reference vs Current Python Adapter")
    for ax in axes:
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_leaky_c(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["case"] for row in rows]
    x = np.arange(len(rows))
    first = [float(row["first_bout_tick"]) if int(row["first_bout_tick"]) >= 0 else np.nan for row in rows]
    duty = [float(row["bout_duty_fraction"]) for row in rows]
    counts = [int(row["bout_count"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, first)
    axes[0].set_ylabel("first bout tick")
    axes[0].set_title("Compiled LeakyIntegrator.c Bout Timing")
    axes[1].bar(x, counts)
    axes[1].set_ylabel("bout count")
    axes[2].bar(x, duty)
    axes[2].set_ylabel("bout duty")
    axes[2].set_xticks(x, labels, rotation=15, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(
    retina_rows: list[dict[str, Any]],
    robot_rows: list[dict[str, Any]],
    omr_rows: list[dict[str, Any]],
    leaky_rows: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> str:
    off_ok = all(float(row["image_c_off_loop_vs_vectorized_max_abs_error"]) < 1e-12 for row in retina_rows)
    dsc_ok = all(float(row["image_c_dsc_loop_vs_vectorized_max_abs_error"]) < 1e-9 for row in retina_rows)
    left_matches = sum(1 for row in retina_rows if row["current_matches_direct_left"])
    right_matches = sum(1 for row in retina_rows if row["current_matches_direct_right"])
    left_margins = [float(row["direct_port_left_winner_margin"]) for row in retina_rows]
    right_margins = [float(row["direct_port_right_winner_margin"]) for row in retina_rows]
    robot_corrs = [float(row["current_vs_direct_corr"]) for row in robot_rows if np.isfinite(float(row["current_vs_direct_corr"]))]
    robot_rmse = [float(row["current_vs_direct_rmse"]) for row in robot_rows]
    omr_success = all(int(row["c_result"]) == 0 for row in omr_rows)
    omr_side_matches = sum(1 for row in omr_rows if row["current_side_matches_c"])
    leaky_bout_cases = sum(1 for row in leaky_rows if int(row["bout_count"]) > 0)
    rows_md = "\n".join(
        f"- `{row['case']}`: corr `{row['current_vs_direct_corr']:.4f}`, rmse `{row['current_vs_direct_rmse']:.4f}`, direct tail angle `{row['direct_robot_c_tail_angle_deg']:.2f}` deg."
        for row in robot_rows
    )
    retina_md = "\n".join(
        f"- `{row['stimulus_direction']}`: direct max labels L/R `{row['direct_port_left_winner']}`/`{row['direct_port_right_winner']}` "
        f"(margins `{row['direct_port_left_winner_margin']:.4g}`/`{row['direct_port_right_winner_margin']:.4g}`), "
        f"current max labels L/R `{row['current_adapter_left_winner']}`/`{row['current_adapter_right_winner']}`."
        for row in retina_rows
    )
    omr_md = "\n".join(
        f"- model `{row['model']}` / `{row['stimulus_direction']}`: C LMLF/RMLF `{row['c_LMLF']:.4f}`/`{row['c_RMLF']:.4f}`, "
        f"C LLHB/RLHB `{row['c_LLHB']:.4f}`/`{row['c_RLHB']:.4f}`, C side `{row['c_side_score_RLHB_minus_LLHB']:.4f}`, "
        f"current side `{row['current_side_score']:.4f}`, current force `{row['current_force']:.4f}`."
        for row in omr_rows
    )
    leaky_md = "\n".join(
        f"- `{row['case']}`: first bout tick `{row['first_bout_tick']}`, bout count `{row['bout_count']}`, duty `{row['bout_duty_fraction']:.4f}`, "
        f"commands `{row['command_counts']}`, mean swim current `{row['mean_swim_current_during_bout']:.4f}`."
        for row in leaky_rows
    )
    return f"""# simZFish Controller Regression Harness

## Working Conclusion

This harness establishes deterministic regression coverage for two public
simZFish controller surfaces and compiled-source coverage for two more:

- `Image.c` OFF bipolar cells and direction-selective retinal counters.
- `Robot.c` CPG/vSPN motor target equations.
- Compiled `OMR.c` model-A/model-C pretectum, MLF, and hindbrain dynamics.
- Compiled `LeakyIntegrator.c` bout-threshold and command dynamics.

The direct Python vectorized ports match literal Python C-like loops for
`Image.c`: OFF exact pass = `{off_ok}`, DSC exact pass = `{dsc_ok}`.  This is a
local equivalence anchor, not yet a compiled C/Webots equivalence test.

The current MuJoCo video adapter remains different from the direct simZFish
controller.  Current retinal max-counter agreement with the direct `Image.c`
DSC port is `{left_matches}/4` for the left eye and `{right_matches}/4` for the
right eye on synthetic motion probes.  These max-counter labels are descriptive,
not a definitive retinal-equivalence score, because simple translating dark-bar
stimuli can produce tied or low-margin direction-selective counters in the
original Barlow-Levick-style inhibition scheme.  Direct-port left/right median
winner margins were `{float(np.median(left_margins)):.4g}` / `{float(np.median(right_margins)):.4g}`.

Current tail latent versus direct `Robot.c` target
correlation mean is `{float(np.mean(robot_corrs)) if robot_corrs else float('nan'):.4f}`;
RMSE mean is `{float(np.mean(robot_rmse)):.4f}`.

Compiled `OMR.c` executed successfully for all rows = `{omr_success}`.  Current
adapter side-sign agreement with compiled `OMR.c` is `{omr_side_matches}/{len(omr_rows)}`
on the synthetic probes.  Compiled `LeakyIntegrator.c` produced at least one
bout in `{leaky_bout_cases}/{len(leaky_rows)}` deterministic drive cases.  This
closes the previous "no OMR/leaky exact surface" gap, but it also shows that
the live Python adapter is still not a behaviorally equivalent replacement for
the original C controller.

## Retinal Winner Comparison

![Retina regression]({artifacts['retina_png']})

{retina_md}

## Tail Motor Target Comparison

![Robot target comparison]({artifacts['robot_png']})

{rows_md}

## Compiled OMR.c Reference

![OMR compiled reference]({artifacts['omr_png']})

{omr_md}

## Compiled LeakyIntegrator.c Reference

![LeakyIntegrator compiled reference]({artifacts['leaky_png']})

{leaky_md}

## Generated Artifacts

- Retinal regression CSV: `{artifacts['retina_csv']}`
- Robot target regression CSV: `{artifacts['robot_csv']}`
- Compiled OMR.c regression CSV: `{artifacts['omr_csv']}`
- Compiled LeakyIntegrator.c regression CSV: `{artifacts['leaky_csv']}`
- Manifest: `{artifacts['manifest']}`

## Remaining Exact-Port Work

This improves the evidence package but does not close the exact simZFish gate.
The next hard steps are a runtime adapter that actually uses or matches the
compiled `OMR.c`/`LeakyIntegrator.c` state trajectory, plus fitted mapping from
the direct six-motor/six-target surface into the current 16-segment MuJoCo
actuator layout.
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    retina_rows = _retina_regression()
    robot_rows = _robot_regression()
    omr_rows = _omr_c_regression(retina_rows)
    leaky_rows = _leaky_c_regression()
    retina_csv = OUT_DIR / "image_c_retina_regression.csv"
    robot_csv = OUT_DIR / "robot_c_motor_regression.csv"
    omr_csv = OUT_DIR / "omr_c_compiled_regression.csv"
    leaky_csv = OUT_DIR / "leaky_integrator_c_compiled_regression.csv"
    retina_png = OUT_DIR / "image_c_retina_winner_regression.png"
    robot_png = OUT_DIR / "robot_c_motor_target_comparison.png"
    omr_png = OUT_DIR / "omr_c_compiled_reference_comparison.png"
    leaky_png = OUT_DIR / "leaky_integrator_c_bout_regression.png"
    report_path = OUT_DIR / "SIMZFISH_CONTROLLER_REGRESSION.md"
    manifest_path = OUT_DIR / "manifest.json"
    _write_csv(retina_csv, retina_rows)
    _write_csv(robot_csv, robot_rows)
    _write_csv(omr_csv, omr_rows)
    _write_csv(leaky_csv, leaky_rows)
    _plot_retina(retina_rows, retina_png)
    _plot_robot(robot_rows, robot_png)
    _plot_omr_c(omr_rows, omr_png)
    _plot_leaky_c(leaky_rows, leaky_png)
    finite_robot_corr = [float(row["current_vs_direct_corr"]) for row in robot_rows if np.isfinite(float(row["current_vs_direct_corr"]))]
    manifest = {
        "out_dir": str(OUT_DIR.resolve()),
        "report": str(report_path.resolve()),
        "retina_csv": str(retina_csv.resolve()),
        "robot_csv": str(robot_csv.resolve()),
        "omr_csv": str(omr_csv.resolve()),
        "leaky_csv": str(leaky_csv.resolve()),
        "retina_png": str(retina_png.resolve()),
        "robot_png": str(robot_png.resolve()),
        "omr_png": str(omr_png.resolve()),
        "leaky_png": str(leaky_png.resolve()),
        "image_c_off_exact": all(float(row["image_c_off_loop_vs_vectorized_max_abs_error"]) < 1e-12 for row in retina_rows),
        "image_c_dsc_exact": all(float(row["image_c_dsc_loop_vs_vectorized_max_abs_error"]) < 1e-9 for row in retina_rows),
        "retina_left_winner_matches": sum(1 for row in retina_rows if row["current_matches_direct_left"]),
        "retina_right_winner_matches": sum(1 for row in retina_rows if row["current_matches_direct_right"]),
        "robot_target_corr_mean": float(np.mean(finite_robot_corr)) if finite_robot_corr else float("nan"),
        "robot_target_rmse_mean": float(np.mean([float(row["current_vs_direct_rmse"]) for row in robot_rows])),
        "omr_c_compiled_available": True,
        "omr_c_rows": len(omr_rows),
        "omr_c_all_results_ok": all(int(row["c_result"]) == 0 for row in omr_rows),
        "omr_current_side_sign_matches": sum(1 for row in omr_rows if row["current_side_matches_c"]),
        "leaky_c_compiled_available": True,
        "leaky_c_rows": len(leaky_rows),
        "leaky_c_bout_cases": sum(1 for row in leaky_rows if int(row["bout_count"]) > 0),
        "leaky_c_total_bouts": sum(int(row["bout_count"]) for row in leaky_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    artifacts = {
        "retina_csv": str(retina_csv.resolve()),
        "robot_csv": str(robot_csv.resolve()),
        "omr_csv": str(omr_csv.resolve()),
        "leaky_csv": str(leaky_csv.resolve()),
        "retina_png": str(retina_png.resolve()),
        "robot_png": str(robot_png.resolve()),
        "omr_png": str(omr_png.resolve()),
        "leaky_png": str(leaky_png.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    report_path.write_text(_report(retina_rows, robot_rows, omr_rows, leaky_rows, artifacts), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
