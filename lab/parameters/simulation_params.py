"""Zebrafish simulation-level parameter specs."""

from __future__ import annotations

from typing import Any

from simulations.zebrafish import config as zfc
from simulations.zebrafish.body import ZebrafishBody
from simulations.zebrafish.environment import AquaticArenaEnvironment
from simulations.zebrafish.neuron_mapping import ZebrafishNervousSystem

from lab.parameters.registry import Kind, ParameterRegistry, ParameterSpec


def _engine(ctx: Any) -> Any:
    return ctx.runtime.engine


def _ns(ctx: Any) -> ZebrafishNervousSystem:
    ns = _engine(ctx).nervous_system
    if not isinstance(ns, ZebrafishNervousSystem):
        raise RuntimeError("lab requires ZebrafishNervousSystem")
    return ns


def _body(ctx: Any) -> ZebrafishBody:
    body = _engine(ctx).body
    if not isinstance(body, ZebrafishBody):
        raise RuntimeError("lab requires ZebrafishBody")
    return body


def _env(ctx: Any) -> AquaticArenaEnvironment:
    env = _engine(ctx).environment
    if not isinstance(env, AquaticArenaEnvironment):
        raise RuntimeError("lab requires AquaticArenaEnvironment")
    return env


def _coerce(value: Any, kind: Kind) -> Any:
    if kind == "int":
        return int(value)
    if kind == "bool":
        return bool(value)
    if kind == "float":
        return float(value)
    return value


def _engine_attr(
    path: str,
    attr: str,
    label: str,
    *,
    group: str,
    kind: Kind = "float",
    min: float | None = None,
    max: float | None = None,
    step: float | None = None,
    help: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        label=label,
        group=group,
        kind=kind,
        apply="live",
        getter=lambda ctx: getattr(_engine(ctx), attr),
        setter=lambda ctx, value: setattr(_engine(ctx), attr, _coerce(value, kind)),
        min=min,
        max=max,
        step=step,
        help=help,
    )


def _body_attr(
    path: str,
    attr: str,
    label: str,
    *,
    group: str,
    min: float,
    max: float,
    step: float,
    help: str,
) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        label=label,
        group=group,
        kind="float",
        apply="live",
        getter=lambda ctx: float(getattr(_body(ctx), attr)),
        setter=lambda ctx, value: setattr(_body(ctx), attr, float(value)),
        min=min,
        max=max,
        step=step,
        help=help,
    )


def _ns_attr(
    path: str,
    attr: str,
    label: str,
    *,
    group: str,
    min: float,
    max: float,
    step: float,
    help: str,
) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        label=label,
        group=group,
        kind="float",
        apply="live",
        getter=lambda ctx: float(getattr(_ns(ctx), attr)),
        setter=lambda ctx, value: setattr(_ns(ctx), attr, float(value)),
        min=min,
        max=max,
        step=step,
        help=help,
    )


def _ns_bool(
    path: str,
    attr: str,
    label: str,
    *,
    group: str,
    help: str,
) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        label=label,
        group=group,
        kind="bool",
        apply="live",
        getter=lambda ctx: bool(getattr(_ns(ctx), attr)),
        setter=lambda ctx, value: setattr(_ns(ctx), attr, bool(value)),
        help=help,
    )


def register_simulation_specs(registry: ParameterRegistry) -> None:
    specs: list[ParameterSpec] = []

    specs.append(
        _engine_attr(
            "sim.neural_ticks_per_physics",
            "neural_ticks_per_physics_step",
            "Neural ticks / physics tick",
            group="Timing",
            kind="int",
            min=1,
            max=32,
            step=1,
            help="PAULA sub-steps per MuJoCo step.",
        )
    )

    def _get_flow_x(ctx: Any) -> float:
        return float(_env(ctx).environment_state()["water_flow_m_s"][0])

    def _set_flow_x(ctx: Any, value: Any) -> None:
        state = _env(ctx).environment_state()
        _env(ctx).set_water_flow(float(value), float(state["water_flow_m_s"][1]))

    def _get_flow_y(ctx: Any) -> float:
        return float(_env(ctx).environment_state()["water_flow_m_s"][1])

    def _set_flow_y(ctx: Any, value: Any) -> None:
        state = _env(ctx).environment_state()
        _env(ctx).set_water_flow(float(state["water_flow_m_s"][0]), float(value))

    env_specs = [
        ParameterSpec(
            path="sim.environment.water_flow_x_m_s",
            label="Water flow X (m/s)",
            group="Water environment",
            kind="float",
            apply="live",
            getter=_get_flow_x,
            setter=_set_flow_x,
            min=-0.03,
            max=0.03,
            step=0.0005,
            help="Horizontal current along the arena x-axis.",
        ),
        ParameterSpec(
            path="sim.environment.water_flow_y_m_s",
            label="Water flow Y (m/s)",
            group="Water environment",
            kind="float",
            apply="live",
            getter=_get_flow_y,
            setter=_set_flow_y,
            min=-0.03,
            max=0.03,
            step=0.0005,
            help="Horizontal current along the arena y-axis.",
        ),
        ParameterSpec(
            path="sim.environment.temperature_c",
            label="Temperature (C)",
            group="Water environment",
            kind="float",
            apply="live",
            getter=lambda ctx: float(_env(ctx).environment_state()["temperature_c"]),
            setter=lambda ctx, value: _env(ctx).set_temperature(float(value)),
            min=18.0,
            max=34.0,
            step=0.1,
            help="Water temperature used by thermal sensory channels.",
        ),
        ParameterSpec(
            path="sim.environment.light_level",
            label="Light level",
            group="Water environment",
            kind="float",
            apply="live",
            getter=lambda ctx: float(_env(ctx).environment_state()["light_level"]),
            setter=lambda ctx, value: _env(ctx).set_light_level(float(value)),
            min=0.0,
            max=1.0,
            step=0.01,
            help="Scalar illumination level for visual responsiveness.",
        ),
        ParameterSpec(
            path="sim.environment.arena_radius_m",
            label="Arena radius (m)",
            group="Water environment",
            kind="float",
            apply="rebuild",
            getter=lambda _ctx: float(zfc.ARENA_RADIUS_M),
            setter=lambda _ctx, value: setattr(zfc, "ARENA_RADIUS_M", float(value)),
            min=0.01,
            max=0.20,
            step=0.001,
            help="Circular water arena radius; rebuild required.",
        ),
    ]
    specs.extend(env_specs)

    specs.extend(
        [
            ParameterSpec(
                path="sim.neuromod.enable_m0",
                label="Enable M0 (maneuver/stress)",
                group="Neuromodulation",
                kind="bool",
                apply="live",
                getter=lambda ctx: bool(_ns(ctx)._enable_m0),  # noqa: SLF001
                setter=lambda ctx, value: _ns(ctx).set_neuromodulator_enabled(0, bool(value)),
                help=(
                    "Enable the M0 neuromodulator channel used for aversive, "
                    "corrective, and sudden-surprise modulation. Disabling it "
                    "zeros M0 and stops new M0 modulation from sensory inputs."
                ),
            ),
            ParameterSpec(
                path="sim.neuromod.enable_m1",
                label="Enable M1 (swim drive)",
                group="Neuromodulation",
                kind="bool",
                apply="live",
                getter=lambda ctx: bool(_ns(ctx)._enable_m1),  # noqa: SLF001
                setter=lambda ctx, value: _ns(ctx).set_neuromodulator_enabled(1, bool(value)),
                help=(
                    "Enable the M1 neuromodulator channel used for approach "
                    "and swim-drive modulation. Disabling it zeros M1 and "
                    "stops new M1 modulation from sensory inputs."
                ),
            ),
        ]
    )

    for path, attr, label, lo, hi, step, help_text in [
        ("sim.body.linear_drag_per_s", "linear_drag_per_s", "Linear drag", 0.0, 40.0, 0.1, "Horizontal velocity damping."),
        ("sim.body.angular_drag_per_s", "angular_drag_per_s", "Yaw drag", 0.0, 80.0, 0.1, "Yaw-rate damping."),
        ("sim.body.vertical_drag_per_s", "vertical_drag_per_s", "Vertical drag", 0.0, 60.0, 0.1, "Up/down velocity damping."),
        ("sim.body.turn_accel_rad_s2", "turn_accel_rad_s2", "Turn acceleration", 0.0, 160.0, 1.0, "Yaw acceleration gain from left/right tail imbalance."),
        ("sim.body.max_vertical_accel_m_s2", "max_vertical_accel_m_s2", "Vertical acceleration", 0.0, 1.0, 0.01, "Vertical acceleration gain from dorsal/ventral tail imbalance."),
        ("sim.body.pitch_drag_per_s", "pitch_drag_per_s", "Pitch drag", 0.0, 60.0, 0.1, "Pitch-rate damping."),
        ("sim.body.depth_home_gain", "depth_home_gain", "Depth home gain", 0.0, 2.0, 0.01, "Passive correction toward preferred swim depth."),
    ]:
        specs.append(
            _body_attr(
                path,
                attr,
                label,
                group="Hydrodynamics",
                min=lo,
                max=hi,
                step=step,
                help=help_text,
            )
        )

    for path, attr, label, lo, hi, step, help_text in [
        ("sim.nervous.spontaneous_bout_prob", "spontaneous_bout_prob", "Spontaneous bout probability", 0.0, 0.05, 0.0005, "Per-tick probability of initiating a bout during coast."),
        ("sim.nervous.baseline_drive", "baseline_drive", "Baseline swim drive", 0.0, 1.0, 0.01, "Baseline input into the bout generator."),
        ("sim.nervous.visual_approach_gain", "visual_approach_gain", "Visual approach gain", 0.0, 2.0, 0.01, "Target/prey visual drive into approach swimming."),
        ("sim.nervous.optic_flow_drive_gain", "optic_flow_drive_gain", "Optic-flow drive gain", 0.0, 2.0, 0.01, "Optomotor stabilization contribution to swim drive."),
        ("sim.nervous.wall_avoidance_drive_gain", "wall_avoidance_drive_gain", "Wall avoidance drive gain", 0.0, 2.0, 0.01, "Aversive wall/contact contribution to swim drive."),
        ("sim.nervous.visual_turn_gain", "visual_turn_gain", "Visual turn gain", 0.0, 2.0, 0.01, "Left/right visual asymmetry to turn bias."),
        ("sim.nervous.optic_flow_turn_gain", "optic_flow_turn_gain", "Optic-flow turn gain", 0.0, 2.0, 0.01, "Optic-flow asymmetry to turn bias."),
        ("sim.nervous.lateral_line_turn_gain", "lateral_line_turn_gain", "Lateral-line turn gain", 0.0, 2.0, 0.01, "Water-flow asymmetry to turn bias."),
        ("sim.nervous.wall_turn_gain", "wall_turn_gain", "Wall turn gain", 0.0, 2.0, 0.01, "Wall proximity asymmetry to turn bias."),
        ("sim.nervous.paula_activity_gain", "paula_activity_gain", "PAULA activity gain", 0.0, 10.0, 0.05, "Visible PAULA state nudging from decoded behavioral motifs."),
        ("sim.nervous.sensory_input_gain", "sensory_input_gain", "Sensory input gain", 0.0, 5.0, 0.01, "Scale applied before injecting sensory channels into PAULA."),
    ]:
        specs.append(
            _ns_attr(
                path,
                attr,
                label,
                group="Nervous system",
                min=lo,
                max=hi,
                step=step,
                help=help_text,
            )
        )

    specs.extend(
        [
            _ns_bool(
                "sim.nervous.tail_cpg_enabled",
                "tail_cpg_enabled",
                "Tail CPG oscillator",
                group="Nervous system",
                help=(
                    "Enable the explicit spinal tail-wave CPG used to produce "
                    "larval zebrafish bout swimming. When disabled, sensory "
                    "turn and pitch biases can remain, but the rhythmic axial "
                    "wave is suppressed."
                ),
            ),
            ParameterSpec(
                path="sim.muscles.plasticity_disabled",
                label="Disable PAULA synaptic plasticity (eta=0)",
                group="Nervous system",
                kind="bool",
                apply="live",
                getter=lambda ctx: bool(_ns(ctx)._plasticity_disabled),  # noqa: SLF001
                setter=lambda ctx, value: _ns(ctx).set_plasticity_disabled(bool(value)),
                help=(
                    "Globally freeze PAULA eta_post/eta_retro at zero. Turning "
                    "this off restores the zebrafish scaffold's small default "
                    "plasticity rates without editing neuron.py."
                ),
            ),
        ]
    )

    registry.extend(specs)
