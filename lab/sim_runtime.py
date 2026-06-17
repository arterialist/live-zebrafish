"""Lab-flavoured simulation runtime.

Owned by a single background thread. Differs from the demo ``SimRuntime``:

* No food, no food commands — the aquatic arena is empty on purpose.
* Play/pause/step transport with a shared ``sim_lock`` so parameter patches
  can be applied safely between ticks.
* Snapshots expose joint angles / velocities, contact forces, muscle
  activations, free-energy samples, and neuromod levels.
* Optional wall-clock pacing (see ``pacing_snapshot`` / ``set_pacing``):
  minimum real ms per physics step and per neural sub-tick (``0`` = no cap).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from loguru import logger

from simulations.zebrafish.body import ZebrafishBody
from simulations.zebrafish.config import (
    ARENA_RADIUS_M,
    BODY_RADIUS_M,
    PHYSICS_TIMESTEP_S,
    TAIL_BEAT_FREQ_MIN_HZ,
    TAIL_BEAT_FREQ_MAX_HZ,
    TWO_PI,
)
from simulations.zebrafish.environment import AquaticArenaEnvironment
from simulations.zebrafish.neuron_mapping import ZebrafishNervousSystem
from simulations.zebrafish.simulation import build_zebrafish_simulation


PatchFn = Callable[[], None]
ZAPBENCH_CALCIUM_FRAME_HZ = 1.093


@dataclass
class TransportState:
    running: bool = True
    tick: int = 0
    step_pending: int = 0


@dataclass
class LatestFrame:
    """Most recent lab snapshot, ready for wire encoding."""

    tick: int = 0
    running: bool = True
    # Body (mm): COM is [cx, cy, cz]; each segment is [x, y, z] (wire v5 triplets).
    com_mm: list[float] = field(default_factory=list)
    segments_mm: list[list[float]] = field(default_factory=list)
    heading_rad: float = 0.0
    pitch_rad: float = 0.0
    tail_angles: list[float] = field(default_factory=list)
    tail_pitch_angles: list[float] = field(default_factory=list)
    # Neurons (paula_id order)
    neuron_s: list[float] = field(default_factory=list)
    neuron_r: list[float] = field(default_factory=list)
    neuron_b: list[float] = field(default_factory=list)
    neuron_tref: list[float] = field(default_factory=list)
    neuron_fired: list[int] = field(default_factory=list)
    # Physics extras (joint order == body.joint_names)
    joint_angles: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    # Touch sensors: (nose, ant, post)
    touch_forces: list[float] = field(default_factory=list)
    # Muscles (order == body.muscle_names)
    muscle_activations: list[float] = field(default_factory=list)
    # Per-neuron M_vector[0] / M_vector[1] (paula_id order, same length as neuron_s)
    neuron_m0: list[float] = field(default_factory=list)
    neuron_m1: list[float] = field(default_factory=list)
    # Neuromodulators
    neuromod: tuple[float, float] = (0.0, 0.0)
    # Free-energy proxy
    free_energy: float = 0.0


class LabSimRuntime:
    """Thread-owned simulation with transport + patch queue."""

    def __init__(
        self,
        *,
        body_settle_steps: int | None = None,
        evol_config: dict[str, Any] | None = None,
        log_free_energy: bool = True,
    ) -> None:
        self.engine, self.loop = build_zebrafish_simulation(
            food_positions=[],
            log_level="WARNING",
            record_neural_states=False,
            suppress_connectome_summary=True,
            max_history=32,
        )
        self.loop.log_free_energy = bool(log_free_energy)
        self.loop.reset()
        logger.info("LabSimRuntime: simulation ready (no food)")

        self._sim_lock = threading.RLock()
        self._patch_queue: queue.SimpleQueue[PatchFn] = queue.SimpleQueue()
        self._transport = TransportState()
        self._running_flag = threading.Event()
        self._running_flag.set()
        self._latest_lock = threading.Lock()
        self._latest: LatestFrame = LatestFrame()
        # Wall-clock pacing (0 = no extra delay — run as fast as the CPU allows).
        self._real_ms_per_physics_step: float = 0.0
        self._real_ms_per_neural_tick: float = 0.0
        self._calcium_replay_path: Path = (
            Path(__file__).resolve().parents[1]
            / "analysis"
            / "cache"
            / "zapbench"
            / "zapbench_calcium_action_replay.npz"
        )
        self._calcium_replay_loaded_path: Path | None = None
        self._calcium_replay_data: dict[str, np.ndarray] | None = None
        self._calcium_replay_metadata: dict[str, Any] = {}
        self._calcium_replay_enabled = False
        self._calcium_replay_loop = True
        self._calcium_replay_condition = "turning"
        self._calcium_replay_gain = 1.0
        self._calcium_replay_start_tick = 0
        self._calcium_replay_last_position = -1
        self._calcium_replay_error = ""
        self._calcium_replay_restore_running = False
        self._manual_video_active = False
        self._video_manual_last_frame_index = -1

        # Cached body / neuron ordering for wire encoding.
        ns0 = self.engine.nervous_system
        if not isinstance(ns0, ZebrafishNervousSystem):
            raise RuntimeError("Lab requires ZebrafishNervousSystem")
        body0 = self.engine.body
        if not isinstance(body0, ZebrafishBody):
            raise RuntimeError("Lab requires ZebrafishBody")
        self.neuron_names: list[str] = ns0.get_neuron_names_paula_order()
        self.joint_names: list[str] = list(body0.joint_names)
        self.muscle_names: list[str] = list(body0.muscle_names)
        self.touch_sensor_names = [
            name for name in body0.get_state().contact_forces if name.startswith("touch_")
        ]
        self.engine.real_ms_per_neural_tick = float(self._real_ms_per_neural_tick)

    # ------------------------------------------------------------------
    # Wall-clock pacing (live lab)
    # ------------------------------------------------------------------

    def pacing_snapshot(self) -> dict[str, float]:
        with self._sim_lock:
            return {
                "real_ms_per_physics_step": float(self._real_ms_per_physics_step),
                "real_ms_per_neural_tick": float(self._real_ms_per_neural_tick),
            }

    def set_pacing(
        self,
        *,
        real_ms_per_physics_step: float | None = None,
        real_ms_per_neural_tick: float | None = None,
    ) -> dict[str, float]:
        with self._sim_lock:
            if real_ms_per_physics_step is not None:
                self._real_ms_per_physics_step = max(0.0, float(real_ms_per_physics_step))
            if real_ms_per_neural_tick is not None:
                self._real_ms_per_neural_tick = max(0.0, float(real_ms_per_neural_tick))
                self.engine.real_ms_per_neural_tick = float(self._real_ms_per_neural_tick)
            return {
                "real_ms_per_physics_step": float(self._real_ms_per_physics_step),
                "real_ms_per_neural_tick": float(self._real_ms_per_neural_tick),
            }

    # ------------------------------------------------------------------
    # ZAPBench calcium replay
    # ------------------------------------------------------------------

    def calcium_replay_snapshot(self) -> dict[str, Any]:
        with self._sim_lock:
            data = self._calcium_replay_data
            condition_names = list(self._calcium_replay_metadata.get("condition_names", []))
            frames = int(data["rows"].shape[0]) if data is not None else 0
            selected = self._calcium_replay_indices_locked()
            env_state: dict[str, Any] = {}
            env = self.engine.environment
            if isinstance(env, AquaticArenaEnvironment):
                env_state = env.calcium_stimulus_state()
            return {
                "available": self._calcium_replay_path.exists(),
                "enabled": bool(self._calcium_replay_enabled),
                "loop": bool(self._calcium_replay_loop),
                "gain": float(self._calcium_replay_gain),
                "condition": self._calcium_replay_condition,
                "path": str(self._calcium_replay_path),
                "source": env_state.get("source", ""),
                "frame_index": int(env_state.get("frame_index", 0)),
                "row": int(env_state.get("row", 0)),
                "calcium_time_s": float(env_state.get("calcium_time_s", 0.0)),
                "kick": float(env_state.get("kick", 0.0)),
                "side": str(env_state.get("side", "none")),
                "side_score": float(env_state.get("side_score", 0.0)),
                "force": float(env_state.get("force", 0.0)),
                "kick_score": float(env_state.get("kick_score", 0.0)),
                "confidence": float(env_state.get("confidence", 0.0)),
                "frames": frames,
                "selected_frames": int(selected.shape[0]) if selected is not None else 0,
                "condition_names": condition_names,
                "error": self._calcium_replay_error,
            }

    def configure_calcium_replay(
        self,
        *,
        enabled: bool | None = None,
        gain: float | None = None,
        condition: str | None = None,
        loop: bool | None = None,
        replay_path: str | Path | None = None,
    ) -> dict[str, Any]:
        with self._sim_lock:
            old_condition = self._calcium_replay_condition
            old_path = self._calcium_replay_path
            if replay_path is not None:
                self._calcium_replay_path = Path(replay_path).expanduser().resolve()
                if self._calcium_replay_path != old_path:
                    self._calcium_replay_data = None
                    self._calcium_replay_metadata = {}
                    self._calcium_replay_loaded_path = None
            if condition is not None:
                self._calcium_replay_condition = str(condition)
            if loop is not None:
                self._calcium_replay_loop = bool(loop)
            if gain is not None:
                self._calcium_replay_gain = float(np.clip(float(gain), 0.0, 3.0))
                env = self.engine.environment
                if isinstance(env, AquaticArenaEnvironment):
                    env.set_calcium_stimulus(gain=self._calcium_replay_gain)
            if enabled is not None:
                next_enabled = bool(enabled)
                if next_enabled and not self._calcium_replay_enabled:
                    self._calcium_replay_restore_running = bool(self._transport.running)
                self._calcium_replay_enabled = next_enabled
                if enabled is False:
                    env = self.engine.environment
                    if isinstance(env, AquaticArenaEnvironment):
                        env.clear_calcium_stimulus()
                    self._transport.running = bool(self._calcium_replay_restore_running)
                    self._calcium_replay_restore_running = False

            if (
                enabled is True
                or old_condition != self._calcium_replay_condition
                or old_path != self._calcium_replay_path
            ):
                self._calcium_replay_start_tick = int(self.engine.tick)
                self._calcium_replay_last_position = -1

            if self._calcium_replay_enabled:
                self._manual_video_active = False
                env = self.engine.environment
                if isinstance(env, AquaticArenaEnvironment):
                    env.clear_video_stimulus()
                self._transport.running = True
                self._load_calcium_replay_locked()
                if isinstance(env, AquaticArenaEnvironment):
                    env.set_calcium_stimulus(
                        enabled=True,
                        gain=self._calcium_replay_gain,
                        source=f"zapbench-calcium:{self._calcium_replay_condition}",
                    )
            return self.calcium_replay_snapshot()

    # ------------------------------------------------------------------
    # Manual video stimulus playback
    # ------------------------------------------------------------------

    def configure_video_stimulus(
        self,
        *,
        enabled: bool | None = None,
        gain: float | None = None,
        file_name: str | None = None,
    ) -> dict[str, Any]:
        with self._sim_lock:
            env = self.engine.environment
            if not isinstance(env, AquaticArenaEnvironment):
                raise RuntimeError("zebrafish environment missing")
            if enabled is True:
                self._calcium_replay_enabled = False
                self._calcium_replay_restore_running = False
                env.clear_calcium_stimulus()
                self._manual_video_active = True
                self._transport.running = False
                self._video_manual_last_frame_index = -1
            elif enabled is False:
                self._manual_video_active = False
            env.set_video_stimulus(
                enabled=enabled,
                gain=gain,
                file_name=file_name,
            )
            return env.video_stimulus_state()

    def clear_video_stimulus(self) -> dict[str, Any]:
        with self._sim_lock:
            self._manual_video_active = False
            self._video_manual_last_frame_index = -1
            env = self.engine.environment
            if not isinstance(env, AquaticArenaEnvironment):
                raise RuntimeError("zebrafish environment missing")
            env.clear_video_stimulus()
            return env.video_stimulus_state()

    def push_video_stimulus_frame_and_step(
        self,
        frame_payload: dict[str, Any],
        *,
        sample_hz: float,
    ) -> dict[str, Any]:
        with self._sim_lock:
            env = self.engine.environment
            if not isinstance(env, AquaticArenaEnvironment):
                raise RuntimeError("zebrafish environment missing")
            if self._calcium_replay_enabled:
                self._manual_video_active = False
                env.clear_video_stimulus()
                return {
                    **env.video_stimulus_state(),
                    "manual_advance_ticks": 0,
                    "manual_sample_hz": float(sample_hz),
                    "ignored": "calcium_replay_active",
                }
            if frame_payload.get("enabled") is True:
                self._manual_video_active = True
                self._transport.running = False
            state = env.push_video_stimulus_frame(frame_payload)
            ticks_to_advance = self._manual_video_ticks_for_frame(
                int(frame_payload.get("frame_index", 0)),
                sample_hz,
            )
            for _ in range(ticks_to_advance):
                self._drain_patches()
                frame = self._build_frame(apply_pacing=False)
                self._transport.tick = frame.tick
                with self._latest_lock:
                    self._latest = frame
            return {
                **state,
                "manual_advance_ticks": int(ticks_to_advance),
                "manual_sample_hz": float(sample_hz),
            }

    def _manual_video_ticks_for_frame(self, frame_index: int, sample_hz: float) -> int:
        hz = max(1.0, float(sample_hz))
        frame_index = max(0, int(frame_index))
        if frame_index <= self._video_manual_last_frame_index:
            self._video_manual_last_frame_index = frame_index - 1
        ticks_per_frame = 1.0 / (hz * PHYSICS_TIMESTEP_S)
        prev_boundary = int(round(frame_index * ticks_per_frame))
        next_boundary = int(round((frame_index + 1) * ticks_per_frame))
        self._video_manual_last_frame_index = frame_index
        return max(1, next_boundary - prev_boundary)

    def clear_calcium_replay(self) -> dict[str, Any]:
        with self._sim_lock:
            self._calcium_replay_enabled = False
            self._calcium_replay_start_tick = int(self.engine.tick)
            self._calcium_replay_last_position = -1
            self._transport.running = bool(self._calcium_replay_restore_running)
            self._calcium_replay_restore_running = False
            env = self.engine.environment
            if isinstance(env, AquaticArenaEnvironment):
                env.clear_calcium_stimulus()
            return self.calcium_replay_snapshot()

    def _load_calcium_replay_locked(self) -> bool:
        if (
            self._calcium_replay_data is not None
            and self._calcium_replay_loaded_path == self._calcium_replay_path
        ):
            return True
        if not self._calcium_replay_path.exists():
            self._calcium_replay_error = f"missing replay artifact: {self._calcium_replay_path}"
            self._calcium_replay_data = None
            self._calcium_replay_metadata = {}
            return False
        try:
            with np.load(self._calcium_replay_path, allow_pickle=False) as npz:
                self._calcium_replay_data = {
                    "rows": np.asarray(npz["rows"], dtype=np.int32),
                    "calcium_time_s": np.asarray(npz["calcium_time_s"], dtype=np.float32),
                    "condition_id": np.asarray(npz["condition_id"], dtype=np.int32),
                    "kick": np.asarray(npz["kick"], dtype=np.float32),
                    "side_score": np.asarray(npz["side_score"], dtype=np.float32),
                    "force": np.asarray(npz["force"], dtype=np.float32),
                    "kick_score": np.asarray(npz["kick_score"], dtype=np.float32),
                    "confidence": np.asarray(npz["confidence"], dtype=np.float32),
                }
                self._calcium_replay_metadata = json.loads(str(npz["metadata_json"]))
            self._calcium_replay_loaded_path = self._calcium_replay_path
            self._calcium_replay_error = ""
            return True
        except Exception as exc:  # noqa: BLE001
            self._calcium_replay_error = str(exc)
            self._calcium_replay_data = None
            self._calcium_replay_metadata = {}
            self._calcium_replay_loaded_path = None
            return False

    def _calcium_replay_indices_locked(self) -> np.ndarray | None:
        data = self._calcium_replay_data
        if data is None:
            return None
        rows = data["rows"]
        if self._calcium_replay_condition == "all":
            return np.arange(rows.shape[0], dtype=np.int32)
        condition_names = list(self._calcium_replay_metadata.get("condition_names", []))
        try:
            cond_id = condition_names.index(self._calcium_replay_condition)
        except ValueError:
            return np.zeros(0, dtype=np.int32)
        return np.flatnonzero(data["condition_id"] == cond_id).astype(np.int32)

    def _advance_calcium_replay(self) -> None:
        if not self._calcium_replay_enabled:
            return
        if not self._load_calcium_replay_locked():
            return
        env = self.engine.environment
        if not isinstance(env, AquaticArenaEnvironment):
            self._calcium_replay_error = "zebrafish environment missing"
            return
        selected = self._calcium_replay_indices_locked()
        data = self._calcium_replay_data
        if data is None or selected is None or selected.shape[0] == 0:
            self._calcium_replay_error = (
                f"no calcium replay frames for condition {self._calcium_replay_condition!r}"
            )
            return

        tick = int(self.engine.tick)
        if tick < self._calcium_replay_start_tick:
            self._calcium_replay_start_tick = tick
            self._calcium_replay_last_position = -1
        elapsed_s = max(0.0, (tick - self._calcium_replay_start_tick) * PHYSICS_TIMESTEP_S)
        elapsed_frames = int(math.floor(elapsed_s * ZAPBENCH_CALCIUM_FRAME_HZ + 1e-9))
        if self._calcium_replay_loop:
            position = int(elapsed_frames % selected.shape[0])
        elif elapsed_frames >= selected.shape[0]:
            self._calcium_replay_enabled = False
            env.set_calcium_stimulus(enabled=False)
            self._transport.running = bool(self._calcium_replay_restore_running)
            self._calcium_replay_restore_running = False
            return
        else:
            position = int(elapsed_frames)
        if position == self._calcium_replay_last_position:
            return
        self._calcium_replay_last_position = position
        row_index = int(selected[position])
        source = f"zapbench-calcium:{self._calcium_replay_condition}"
        env.push_calcium_action_frame(
            {
                "enabled": True,
                "source": source,
                "row": int(data["rows"][row_index]),
                "frame_index": position,
                "calcium_time_s": float(data["calcium_time_s"][row_index]),
                "kick": float(data["kick"][row_index]),
                "side_score": float(data["side_score"][row_index]),
                "force": float(data["force"][row_index]),
                "kick_score": float(data["kick_score"][row_index]),
                "confidence": float(data["confidence"][row_index]),
                "tail_phase": float(
                    TWO_PI
                    * elapsed_s
                    * (
                        TAIL_BEAT_FREQ_MIN_HZ
                        + float(data["force"][row_index])
                        * (TAIL_BEAT_FREQ_MAX_HZ - TAIL_BEAT_FREQ_MIN_HZ)
                    )
                ),
            }
        )

    # ------------------------------------------------------------------
    # Transport

    @property
    def sim_lock(self) -> threading.RLock:
        return self._sim_lock

    def transport_snapshot(self) -> dict[str, Any]:
        with self._sim_lock:
            return {
                "running": bool(
                    self._transport.running
                    or self._manual_video_active
                    or self._calcium_replay_enabled
                ),
                "tick": int(self._transport.tick),
            }

    def play(self) -> dict[str, Any]:
        with self._sim_lock:
            if not self._manual_video_active:
                self._transport.running = True
        return self.transport_snapshot()

    def pause(self) -> dict[str, Any]:
        with self._sim_lock:
            self._transport.running = False
            if self._manual_video_active:
                self._manual_video_active = False
                env = self.engine.environment
                if isinstance(env, AquaticArenaEnvironment):
                    env.clear_video_stimulus()
        return self.transport_snapshot()

    def step_once(self) -> dict[str, Any]:
        """Request exactly one physics+neural step; stays paused afterwards."""
        with self._sim_lock:
            self._transport.step_pending += 1
            self._transport.running = False
        return self.transport_snapshot()

    def stop(self) -> None:
        self._running_flag.clear()

    # ------------------------------------------------------------------
    # Patch queue

    def enqueue_patch(self, fn: PatchFn) -> None:
        self._patch_queue.put(fn)

    def _drain_patches(self) -> None:
        while True:
            try:
                fn = self._patch_queue.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                logger.exception("Live patch failed")

    # ------------------------------------------------------------------
    # Main loop

    def run_loop(self) -> None:
        try:
            while self._running_flag.is_set():
                should_step = False
                with self._sim_lock:
                    if self._transport.step_pending > 0:
                        self._transport.step_pending -= 1
                        should_step = True
                    elif self._transport.running:
                        should_step = True

                if not should_step:
                    time.sleep(1.0 / 120.0)
                    self._refresh_latest_paused()
                    continue

                with self._sim_lock:
                    self._drain_patches()
                    frame = self._build_frame()
                    self._transport.tick = frame.tick
                with self._latest_lock:
                    self._latest = frame
        except Exception:
            logger.exception("Lab simulation thread crashed")
            self._running_flag.clear()

    # ------------------------------------------------------------------
    # Snapshot construction

    def _refresh_latest_paused(self) -> None:
        """Keep broadcasting a fresh snapshot (same tick) while paused."""
        with self._latest_lock:
            if not self._latest.segments_mm:
                return
            self._latest.running = bool(
                self._transport.running
                or self._manual_video_active
                or self._calcium_replay_enabled
            )

    def _build_frame(self, *, apply_pacing: bool = True) -> LatestFrame:
        phy_target_ms = float(self._real_ms_per_physics_step) if apply_pacing else 0.0
        self.engine.real_ms_per_neural_tick = float(self._real_ms_per_neural_tick)
        t_wall0 = time.perf_counter()
        self._advance_calcium_replay()
        step = self.engine.step()
        body = self.engine.body
        ns = self.engine.nervous_system
        assert isinstance(body, ZebrafishBody)
        assert isinstance(ns, ZebrafishNervousSystem)

        bs = step.body_state
        com = bs.position
        com_mm = [
            float(com[0] * 1000.0),
            float(com[1] * 1000.0),
            float(com[2] * 1000.0),
        ]

        segments_mm: list[list[float]] = []
        shape = body.get_body_shape()
        for i in range(shape.shape[0]):
            segments_mm.append(
                [
                    float(shape[i, 0] * 1000.0),
                    float(shape[i, 1] * 1000.0),
                    float(shape[i, 2] * 1000.0),
                ]
            )

        extra = bs.extra or {}
        tail_angles_raw = extra.get("tail_angles")
        tail_pitch_raw = extra.get("tail_pitch_angles")
        tail_angles = (
            [float(x) for x in np.asarray(tail_angles_raw, dtype=float).ravel()]
            if tail_angles_raw is not None
            else []
        )
        tail_pitch_angles = (
            [float(x) for x in np.asarray(tail_pitch_raw, dtype=float).ravel()]
            if tail_pitch_raw is not None
            else []
        )

        # Neurons
        s_list, r_list, b_list, tref_list, f_list = self._neuron_arrays(ns)
        m0_list, m1_list = self._neuron_m01_arrays(ns)

        # Joints
        ja = [float(bs.joint_angles.get(n, 0.0)) for n in self.joint_names]
        jv = [float(bs.joint_velocities.get(n, 0.0)) for n in self.joint_names]

        # Touch (scalar force per sensor)
        tc: list[float] = []
        for sname in self.touch_sensor_names:
            vec = bs.contact_forces.get(sname)
            if vec is None:
                tc.append(0.0)
            else:
                tc.append(float(np.linalg.norm(np.asarray(vec, dtype=float))))

        # Muscles: read post-NMJ clipped ctrl values directly from mjData.
        import mujoco  # local import keeps module-load cost down

        ma: list[float] = []
        mj_model = body.model
        mj_data = body.data
        for mname in self.muscle_names:
            aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, mname)
            if aid < 0:
                ma.append(0.0)
                continue
            force_max = float(mj_model.actuator_forcerange[aid, 1]) or 1.0
            ma.append(float(mj_data.ctrl[aid]) / force_max if force_max else 0.0)

        # Neuromod HUD proxy: M0 = aversive / corrective state, M1 = swim drive.
        m0, m1 = ns.neuromod_levels

        # Free-energy
        fe_val = 0.0
        if self.loop.log_free_energy and self.loop.free_energy_trace.prediction_error:
            fe_val = float(self.loop.free_energy_trace.prediction_error[-1])

        if phy_target_ms > 0.0:
            elapsed_s = time.perf_counter() - t_wall0
            remain_s = phy_target_ms / 1000.0 - elapsed_s
            if remain_s > 0.0:
                time.sleep(remain_s)

        return LatestFrame(
            tick=int(step.tick),
            running=True,
            com_mm=com_mm,
            segments_mm=segments_mm,
            heading_rad=float(extra.get("heading", 0.0)),
            pitch_rad=float(extra.get("pitch_rad", 0.0)),
            tail_angles=tail_angles,
            tail_pitch_angles=tail_pitch_angles,
            neuron_s=s_list,
            neuron_r=r_list,
            neuron_b=b_list,
            neuron_tref=tref_list,
            neuron_fired=f_list,
            joint_angles=ja,
            joint_velocities=jv,
            touch_forces=tc,
            muscle_activations=ma,
            neuron_m0=m0_list,
            neuron_m1=m1_list,
            neuromod=(float(m0), float(m1)),
            free_energy=fe_val,
        )

    @staticmethod
    def _neuron_arrays(
        ns: ZebrafishNervousSystem,
    ) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
        """Parallel per-neuron arrays in paula_id order: (S, r, b, t_ref, fired)."""
        if ns._network is None:  # type: ignore[attr-defined]
            return [], [], [], [], []
        neurons = ns._network.network.neurons  # type: ignore[attr-defined]
        if not neurons:
            return [], [], [], [], []
        ids = sorted(neurons.keys(), key=int)
        n = int(ids[-1]) + 1
        s_out = [0.0] * n
        r_out = [0.0] * n
        b_out = [0.0] * n
        tref_out = [0.0] * n
        f_out = [0] * n
        for i in ids:
            neuron = neurons[i]
            ii = int(i)
            if 0 <= ii < n:
                s_out[ii] = float(neuron.S)
                r_out[ii] = float(neuron.r)
                b_out[ii] = float(neuron.b)
                tref_out[ii] = float(neuron.t_ref)
                f_out[ii] = 1 if float(neuron.O) > 0 else 0
        return s_out, r_out, b_out, tref_out, f_out

    @staticmethod
    def _neuron_m01_arrays(ns: ZebrafishNervousSystem) -> tuple[list[float], list[float]]:
        """Parallel M_vector[0], M_vector[1] in paula_id order (same indexing as S)."""
        if ns._network is None:  # type: ignore[attr-defined]
            return [], []
        neurons = ns._network.network.neurons  # type: ignore[attr-defined]
        if not neurons:
            return [], []
        ids = sorted(neurons.keys(), key=int)
        n = int(ids[-1]) + 1
        m0_out = [0.0] * n
        m1_out = [0.0] * n
        for i in ids:
            neuron = neurons[i]
            ii = int(i)
            if not (0 <= ii < n):
                continue
            mvec = getattr(neuron, "M_vector", None)
            if mvec is None:
                continue
            try:
                arr = np.asarray(mvec, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size >= 1:
                m0_out[ii] = float(arr[0])
            if arr.size >= 2:
                m1_out[ii] = float(arr[1])
        return m0_out, m1_out

    # ------------------------------------------------------------------
    # Broadcast API

    def get_latest(self) -> LatestFrame | None:
        with self._latest_lock:
            if not self._latest.segments_mm:
                return None
            # Return a shallow copy (dataclass replace would be per-field).
            return LatestFrame(
                tick=self._latest.tick,
                running=bool(
                    self._transport.running
                    or self._manual_video_active
                    or self._calcium_replay_enabled
                ),
                com_mm=list(self._latest.com_mm),
                segments_mm=[list(row) for row in self._latest.segments_mm],
                heading_rad=float(self._latest.heading_rad),
                pitch_rad=float(self._latest.pitch_rad),
                tail_angles=list(self._latest.tail_angles),
                tail_pitch_angles=list(self._latest.tail_pitch_angles),
                neuron_s=list(self._latest.neuron_s),
                neuron_r=list(self._latest.neuron_r),
                neuron_b=list(self._latest.neuron_b),
                neuron_tref=list(self._latest.neuron_tref),
                neuron_fired=list(self._latest.neuron_fired),
                joint_angles=list(self._latest.joint_angles),
                joint_velocities=list(self._latest.joint_velocities),
                touch_forces=list(self._latest.touch_forces),
                muscle_activations=list(self._latest.muscle_activations),
                neuron_m0=list(self._latest.neuron_m0),
                neuron_m1=list(self._latest.neuron_m1),
                neuromod=self._latest.neuromod,
                free_energy=self._latest.free_energy,
            )

    def has_frame(self) -> bool:
        with self._latest_lock:
            return bool(self._latest.segments_mm)

    # ------------------------------------------------------------------
    # Statics (for hello frame & REST)

    def plate_radius_mm(self) -> float:
        return float(ARENA_RADIUS_M * 1000.0)

    def zebrafish_radius_mm(self) -> float:
        return float(BODY_RADIUS_M * 1000.0)
