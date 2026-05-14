"""canonical triple-decker topology aligned with particle_sim.geometry."""

from __future__ import annotations

from particle_sim.core.geometry import LINKS, ZONE_NAMES
from particle_sim.io.schema import N_LINKAGES, N_ZONES
from surrogate.types import LinkageSpecT, Topology, ZoneSide, ZoneSpec


def _side_for_zone(name: str) -> ZoneSide:
    return "front" if name.endswith("_front") else "back"


def _floor_from_name(name: str) -> int:
    # floor1_front -> 1
    tok = name.split("_", maxsplit=1)[0]
    return int(tok.replace("floor", ""))


def build_default_topology() -> Topology:
    """Zones + 15 linkages in notebook / particle_sim order."""
    zones: list[ZoneSpec] = []
    for name in ZONE_NAMES:
        f = _floor_from_name(name)
        fy = float((f - 1) * (7.0 + 3.0))
        side = _side_for_zone(name)
        if side == "front":
            zones.append(
                ZoneSpec(
                    name=name,
                    floor=f,
                    side=side,
                    x_min=0.0,
                    x_max=12.0,
                    y_min=fy,
                    y_max=fy + 3.5,
                )
            )
        else:
            zones.append(
                ZoneSpec(
                    name=name,
                    floor=f,
                    side=side,
                    x_min=0.0,
                    x_max=12.0,
                    y_min=fy + 3.5,
                    y_max=fy + 7.0,
                )
            )
    assert len(zones) == N_ZONES

    linkages: list[LinkageSpecT] = []
    for lk in LINKS:
        az: float | None = 180.0 if lk.facade == "front" else 0.0 if lk.facade == "back" else None
        seg_cx = 0.5 * (lk.seg_min + lk.seg_max)
        seg_cy = float(lk.seg_pos)
        if lk.seg_axis == "x":
            x, y = seg_cx, seg_cy
        else:
            x, y = seg_cy, seg_cx
        linkages.append(
            LinkageSpecT(
                name=lk.name,
                kind=lk.kind,  # type: ignore[assignment]
                a=lk.endpoint_a,
                b="outdoor" if lk.zone_b < 0 else lk.endpoint_b,
                x=float(x),
                y=float(y),
                width=float(lk.width_m),
                facade_azimuth=az,
            )
        )
    assert len(linkages) == N_LINKAGES
    return Topology(name="triple_decker_v0", zones=zones, linkages=linkages)


def linkage_kinds(linkages: list[LinkageSpecT]) -> tuple[str, ...]:
    return tuple(lk.kind for lk in linkages)


def outdoor_wind_azimuth_deg(link: LinkageSpecT) -> float:
    if link.facade_azimuth is None:
        return 180.0
    return float(link.facade_azimuth)


def normalize_axis_deg(a: float) -> float:
    x = float(a) % 360.0
    if x < 0:
        x += 360.0
    return x


def met_wind_to_blowing_from(met_deg: float) -> float:
    """Convert meteorological wind dir to blowing-from degrees."""
    return normalize_axis_deg(met_deg + 180.0)
