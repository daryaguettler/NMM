"""triple-decker rectangles, linkage list, and zone/link indexing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_sim.io.schema import N_LINKAGES, N_ZONES

# zone order matches visualization notebook enumerate(ZONES): floor1_front, ...
ZONE_NAMES: tuple[str, ...] = tuple(
    f"floor{f}_{side}" for f in (1, 2, 3) for side in ("front", "back")
)
assert len(ZONE_NAMES) == N_ZONES


def zone_index(name: str) -> int:
    return ZONE_NAMES.index(name)


@dataclass(frozen=True)
class LinkageSpec:
    kind: str
    name: str
    endpoint_a: str
    endpoint_b: str
    zone_a: int  # interior zone index if endpoint is zone
    zone_b: int  # -1 for outdoor
    # segment on the boundary of zone_a when moving a->b (exterior: a is interior)
    # axis: 'x' or 'y' = constant coordinate on interior boundary
    seg_axis: str
    seg_pos: float
    seg_min: float
    seg_max: float
    width_m: float
    facade: str | None  # 'front' | 'back' | None for doorway


def floor_y_world(floor_idx: int) -> float:
    return float(floor_idx * (7.0 + 3.0))  # wide + gap


def build_zone_rects() -> np.ndarray:
    """(6,4) xmin, ymin, xmax, ymax world meters."""
    rects = np.zeros((N_ZONES, 4), dtype=np.float64)
    for f in range(3):
        fy = floor_y_world(f)
        zi_front = f * 2
        zi_back = f * 2 + 1
        rects[zi_front] = (0.0, fy, 12.0, fy + 3.5)
        rects[zi_back] = (0.0, fy + 3.5, 12.0, fy + 7.0)
    return rects


def build_links() -> list[LinkageSpec]:
    links: list[LinkageSpec] = []
    for f in range(3):
        fy = floor_y_world(f)
        zf = f * 2
        zb = f * 2 + 1
        pref = ZONE_NAMES[zf]
        pback = ZONE_NAMES[zb]
        # window front (south y=fy), interior zone zf
        links.append(
            LinkageSpec(
                "window",
                f"win_floor{f+1}_front",
                pref,
                "outdoor",
                zf,
                -1,
                "y",
                fy,
                6.0 - 0.4,
                6.0 + 0.4,
                0.8,
                "front",
            )
        )
        links.append(
            LinkageSpec(
                "window",
                f"win_floor{f+1}_back",
                pback,
                "outdoor",
                zb,
                -1,
                "y",
                fy + 7.0,
                6.0 - 0.4,
                6.0 + 0.4,
                0.8,
                "back",
            )
        )
        # cracks: single effective segment at building center per notebook simplification
        links.append(
            LinkageSpec(
                "crack",
                f"crack_floor{f+1}_front",
                pref,
                "outdoor",
                zf,
                -1,
                "y",
                fy,
                3.0 - 0.1,
                3.0 + 0.1,
                0.2,
                "front",
            )
        )
        links.append(
            LinkageSpec(
                "crack",
                f"crack_floor{f+1}_back",
                pback,
                "outdoor",
                zb,
                -1,
                "y",
                fy + 7.0,
                3.0 - 0.1,
                3.0 + 0.1,
                0.2,
                "back",
            )
        )
        # doorway front->back (+y)
        links.append(
            LinkageSpec(
                "doorway",
                f"door_floor{f+1}",
                pref,
                pback,
                zf,
                zb,
                "y",
                fy + 3.5,
                6.0 - 0.4,
                6.0 + 0.4,
                0.8,
                None,
            )
        )
    assert len(links) == N_LINKAGES
    return links


LINKS: list[LinkageSpec] = build_links()
ZONE_RECTS: np.ndarray = build_zone_rects()

# window link indices in LINKS order: 0,1,5,6,10,11 for floors 1..3 -> pattern 5*f+0, 5*f+1
WINDOW_LINK_INDICES = tuple(5 * f + i for f in range(3) for i in (0, 1))


def passage_base(kind: str) -> float:
    if kind == "window":
        return 1.0
    if kind == "doorway":
        return 1.0
    if kind == "crack":
        return 0.3
    return 0.0


def zone_from_point(pos: np.ndarray) -> np.ndarray:
    """vectorized: (..., 2) -> int32 zone index or -1."""
    x = pos[..., 0]
    y = pos[..., 1]
    r = ZONE_RECTS
    inside = (
        (x[..., None] >= r[None, :, 0])
        & (x[..., None] <= r[None, :, 2])
        & (y[..., None] >= r[None, :, 1])
        & (y[..., None] <= r[None, :, 3])
    )
    # first match
    idx = np.argmax(inside, axis=-1).astype(np.int32)
    none = ~inside.any(axis=-1)
    idx = np.where(none, -1, idx)
    return idx
