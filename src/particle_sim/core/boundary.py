"""segment/box exit for boundary resolve."""

from __future__ import annotations

import numpy as np


def segment_last_inside_param(
    p0: np.ndarray,
    p1: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
    iterations: int = 22,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary search last interior parameter t in [0,1] along p0->p1."""
    lo = np.zeros(p0.shape[:-1], dtype=p0.dtype)
    hi = np.ones(p0.shape[:-1], dtype=p0.dtype)
    d = p1 - p0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        pm = p0 + mid[..., None] * d
        ins = (
            (pm[..., 0] >= xmin)
            & (pm[..., 0] <= xmax)
            & (pm[..., 1] >= ymin)
            & (pm[..., 1] <= ymax)
        )
        lo = np.where(ins, mid, lo)
        hi = np.where(ins, hi, mid)
    t_exit = lo
    hit = p0 + t_exit[..., None] * d
    return t_exit, hit


def classify_exit_edge(
    hit: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """0=left x=xmin, 1=right x=xmax, 2=bottom y=ymin, 3=top y=ymax (-1 if unknown)."""
    x, y = hit[..., 0], hit[..., 1]
    left = np.abs(x - xmin) < eps
    right = np.abs(x - xmax) < eps
    bottom = np.abs(y - ymin) < eps
    top = np.abs(y - ymax) < eps
    edge = np.full(hit.shape[:-1], -1, dtype=np.int32)
    edge = np.where(left, 0, edge)
    edge = np.where(right, 1, edge)
    edge = np.where(bottom, 2, edge)
    edge = np.where(top, 3, edge)
    return edge


def hit_on_segment(
    hit: np.ndarray,
    edge: np.ndarray,
    axis: str,
    pos: float,
    seg_lo: float,
    seg_hi: float,
    edge_need: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """True if hit is on axis=pos line, segment span, and edge code matches."""
    if axis == "y":
        along = hit[..., 0]
        on_axis = np.abs(hit[..., 1] - pos) < eps
    else:
        along = hit[..., 1]
        on_axis = np.abs(hit[..., 0] - pos) < eps
    in_span = (along >= seg_lo - eps) & (along <= seg_hi + eps)
    return (edge == edge_need) & on_axis & in_span
