"""Schematic side-view layout for reduced larval zebrafish neuron names.

The reduced PAULA circuit does not currently ship a full 3D cell atlas. Neuron
order is used as an anterior-posterior proxy, while common zebrafish name
suffixes and segment ids separate left/right and dorsal/ventral pools.
"""

from __future__ import annotations


def side_view_layout_normalized(names: list[str]) -> tuple[list[float], list[float]]:
    """Return ``(ax, ay)`` in [0, 1] × roughly [-0.6, 0.6] for each neuron index."""
    n = len(names)
    if n == 0:
        return [], []
    ax: list[float] = []
    ay: list[float] = []
    denom = max(1, n - 1)
    for i, name in enumerate(names):
        ax.append(round(i / denom, 5))
        ay.append(round(_dv_offset(name), 5))
    return ax, ay


def _dv_offset(name: str) -> float:
    if not name:
        return 0.0
    last = name[-1]
    if last == "L":
        return 0.38
    if last == "R":
        return -0.38
    if last == "V":
        return -0.58
    if last == "D":
        return 0.58
    return 0.0
