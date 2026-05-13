"""Body-aligned 2D layout for the reduced larval zebrafish circuit."""

from __future__ import annotations

from simulations.zebrafish import config as zfc


def _lr_y(name: str, *, scale: float = 0.52) -> float:
    if name.endswith("_L") or "_L_" in name or name.startswith("MOTOR_L"):
        return scale
    if name.endswith("_R") or "_R_" in name or name.startswith("MOTOR_R"):
        return -scale
    return 0.0


def _motor_position(name: str) -> tuple[float, float] | None:
    try:
        seg = int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
    frac = seg / max(1, zfc.N_BODY_SEGMENTS - 1)
    x = 0.42 + 0.55 * frac
    if name.startswith("MOTOR_L"):
        return x, 0.48
    if name.startswith("MOTOR_R"):
        return x, -0.48
    if name.startswith("MOTOR_D"):
        return x, 0.23
    if name.startswith("MOTOR_V"):
        return x, -0.23
    return None


def body_aligned_layout(names: list[str]) -> tuple[list[float], list[float]]:
    """Return parallel ``(ax, ay)`` lists in body-fraction space.

    ``x`` is anterior-to-posterior from nose to tail. ``y`` separates left/right
    and dorsal/ventral channels so the reduced circuit remains legible while
    still matching the larval body plan.
    """
    ax: list[float] = []
    ay: list[float] = []
    for idx, name in enumerate(names):
        motor = _motor_position(name)
        if motor is not None:
            x, y = motor
        elif name.startswith(("RETINA", "OPTIC")):
            x, y = 0.05, _lr_y(name, scale=0.50)
        elif name.startswith("OLFACTORY"):
            x, y = 0.04, _lr_y(name, scale=0.30)
        elif name.startswith("LATERAL_LINE"):
            x, y = 0.18, _lr_y(name, scale=0.58)
        elif name.startswith("WALL"):
            x, y = 0.24, _lr_y(name, scale=0.62)
        elif name.startswith("VESTIBULAR"):
            x, y = 0.18, _lr_y(name, scale=0.26)
            if name.endswith("_UP"):
                y = 0.18
            elif name.endswith("_DOWN"):
                y = -0.18
        elif name.startswith("DEPTH"):
            x = 0.24
            y = 0.18 if name.endswith("SHALLOW") else -0.18 if name.endswith("DEEP") else 0.0
        elif name.startswith("THERMO") or name == "STARTLE":
            x, y = 0.12, 0.0 if name == "STARTLE" else (0.28 if name.endswith("HOT") else -0.28)
        elif name.startswith(("TECTUM", "PRETECTUM", "ARTR")):
            x, y = 0.28, _lr_y(name, scale=0.46)
        elif name.startswith(("HINDBRAIN", "MAUTHNER", "RETICULOSPINAL")):
            x, y = 0.36, _lr_y(name, scale=0.42)
        elif name in {"APPROACH_GATE", "AVOID_GATE", "SWIM_GATE", "BOUT_CLOCK"}:
            x = 0.36
            y = {
                "APPROACH_GATE": 0.18,
                "AVOID_GATE": -0.18,
                "SWIM_GATE": 0.0,
                "BOUT_CLOCK": 0.30,
            }[name]
        elif name in {"DEPTH_HOMEOSTAT", "ASCEND_GATE", "DIVE_GATE"}:
            x = 0.38
            y = {"DEPTH_HOMEOSTAT": 0.0, "ASCEND_GATE": 0.24, "DIVE_GATE": -0.24}[name]
        elif name.startswith("SPINAL_CPG"):
            x, y = 0.50, _lr_y(name, scale=0.34)
        else:
            frac = idx / max(1, len(names) - 1)
            x, y = 0.15 + 0.75 * frac, 0.08 if idx % 2 else -0.08
        ax.append(round(float(x), 4))
        ay.append(round(float(y), 4))
    return ax, ay
