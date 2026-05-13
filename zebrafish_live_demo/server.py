"""WebSocket server for the no-food larval zebrafish live demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from typing import Any

import numpy as np
from loguru import logger
from websockets.asyncio.server import ServerConnection, serve

from simulations.zebrafish.body import ZebrafishBody
from simulations.zebrafish.config import ARENA_RADIUS_M, PHYSICS_TIMESTEP_S
from simulations.zebrafish.environment import AquaticArenaEnvironment
from simulations.zebrafish.neuron_mapping import ZebrafishNervousSystem
from simulations.zebrafish.simulation import build_zebrafish_simulation

PROTOCOL_VERSION = 1
BROADCAST_HZ = 30.0
MAX_CLIENTS = 40


def _stderr_filter(record: Any) -> bool:
    try:
        name = str(record["name"])
    except (KeyError, TypeError):
        return True
    if name.startswith("neuron."):
        try:
            return int(record["level"].no) >= 30
        except (KeyError, TypeError, AttributeError):
            return False
    return True


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
        level=level,
        colorize=True,
        filter=_stderr_filter,
    )


class SimRuntime:
    """Owns the simulation thread and latest no-food snapshot."""

    def __init__(self, *, seed: int | None = 7):
        self.engine, self.loop = build_zebrafish_simulation(
            food_positions=[],
            log_level="WARNING",
            record_neural_states=False,
            suppress_connectome_summary=True,
            max_history=32,
            seed=seed,
        )
        self.loop.reset()
        self._cmd_queue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._latest: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()

    def stop(self) -> None:
        self._running.clear()

    def command_queue(self) -> queue.SimpleQueue[dict[str, Any]]:
        return self._cmd_queue

    def run_loop(self) -> None:
        try:
            while self._running.is_set():
                start = time.perf_counter()
                self._drain_commands()
                snap = self._build_snapshot()
                with self._lock:
                    self._latest = snap
                delay = max(0.0, PHYSICS_TIMESTEP_S - (time.perf_counter() - start))
                if delay > 0.0:
                    time.sleep(delay)
        except Exception:
            logger.exception("Zebrafish simulation thread crashed")
            self._running.clear()
            raise

    def get_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def _drain_commands(self) -> None:
        env = self.engine.environment
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            if cmd.get("type") == "startle" and isinstance(env, AquaticArenaEnvironment):
                env.trigger_startle()
                logger.info("Startle stimulus triggered")

    def _build_snapshot(self) -> dict[str, Any]:
        step = self.engine.step()
        body = self.engine.body
        env = self.engine.environment
        points_m: list[list[float]] = []
        if isinstance(body, ZebrafishBody):
            points_m = body.get_body_shape().round(7).tolist()
        names: list[str] = []
        neural_s: list[float] = []
        neural_f: list[int] = []
        ns = self.engine.nervous_system
        behavior: dict[str, Any] = {}
        if isinstance(ns, ZebrafishNervousSystem):
            names = ns.get_neuron_names_paula_order()
            neural_s, neural_f, _ = ns.get_compact_neural_snapshot()
            behavior = ns.behavior_state
        env_state = env.environment_state() if isinstance(env, AquaticArenaEnvironment) else {}
        return {
            "p": PROTOCOL_VERSION,
            "t": "state",
            "tick": int(step.tick),
            "arena_radius_m": ARENA_RADIUS_M,
            "arena_depth_m": float(env_state.get("arena_depth_m", 0.008)),
            "points_m": points_m,
            "com_m": step.body_state.position.round(7).tolist(),
            "head_m": step.body_state.head_position.round(7).tolist(),
            "heading_rad": float(step.body_state.extra.get("heading", 0.0)),
            "pitch_rad": float(step.body_state.extra.get("pitch_rad", 0.0)),
            "speed_m_s": float(step.body_state.extra.get("speed_m_s", 0.0)),
            "vertical_velocity_m_s": float(step.body_state.extra.get("vertical_velocity_m_s", 0.0)),
            "tail_angles": np.asarray(step.body_state.extra.get("tail_angles", []), dtype=float).round(4).tolist(),
            "tail_pitch_angles": np.asarray(step.body_state.extra.get("tail_pitch_angles", []), dtype=float).round(4).tolist(),
            "behavior": behavior,
            "neural": {
                "names": names,
                "s": [round(float(x), 4) for x in neural_s],
                "fired": neural_f,
            },
        }


CLIENTS: set[ServerConnection] = set()


async def _send_presence() -> None:
    msg = json.dumps({"p": PROTOCOL_VERSION, "t": "presence", "n": len(CLIENTS)})
    stale: list[ServerConnection] = []
    for ws in list(CLIENTS):
        try:
            await ws.send(msg)
        except Exception:
            stale.append(ws)
    for ws in stale:
        CLIENTS.discard(ws)


async def handle_client(ws: ServerConnection, runtime: SimRuntime) -> None:
    if len(CLIENTS) >= MAX_CLIENTS:
        await ws.close(code=1013, reason="too many clients")
        return
    CLIENTS.add(ws)
    await ws.send(
        json.dumps(
            {
                "p": PROTOCOL_VERSION,
                "t": "hello",
                "m": "larval zebrafish live simulation",
            }
        )
    )
    await _send_presence()
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"p": PROTOCOL_VERSION, "t": "error", "m": "invalid JSON"}))
                continue
            mtype = msg.get("t")
            if mtype == "startle":
                runtime.command_queue().put({"type": "startle"})
            elif mtype == "ping":
                await ws.send(json.dumps({"p": PROTOCOL_VERSION, "t": "pong"}))
    finally:
        CLIENTS.discard(ws)
        await _send_presence()


async def broadcast_loop(runtime: SimRuntime) -> None:
    interval = 1.0 / BROADCAST_HZ
    while True:
        await asyncio.sleep(interval)
        snap = runtime.get_snapshot()
        if not snap or not CLIENTS:
            continue
        payload = json.dumps(snap, separators=(",", ":"))
        stale: list[ServerConnection] = []
        for ws in list(CLIENTS):
            try:
                await asyncio.wait_for(ws.send(payload), timeout=0.5)
            except Exception:
                stale.append(ws)
        for ws in stale:
            CLIENTS.discard(ws)


async def async_main(args: argparse.Namespace) -> None:
    runtime = SimRuntime(seed=args.seed)
    _configure_logging(args.log_level)
    sim_thread = threading.Thread(target=runtime.run_loop, name="zebrafish-sim", daemon=True)
    sim_thread.start()
    try:
        async with serve(lambda ws: handle_client(ws, runtime), args.host, args.port):
            logger.info("Zebrafish demo server listening on ws://{}:{}", args.host, args.port)
            await broadcast_loop(runtime)
    finally:
        runtime.stop()
        sim_thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zebrafish live demo WebSocket server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
