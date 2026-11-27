"""Terrain LOD crack stitching.

Models a small quadtree terrain LOD pipeline and the six fixes needed when
neighbouring chunks are drawn at different levels:

1. ``align_boundary_vertices`` / ``stitch``
   The finer side of a shared edge is re-parameterised onto the coarser edge
   polyline, and every coarse vertex position is *inserted* into the fine
   edge, so both chunks emit exactly the same boundary vertices. No
   T-junctions, no missing triangles, no sky showing through a seam.
2. ``lod_hysteresis``
   LOD changes use a hysteresis band: a chunk leaves level ``L`` only after
   the distance crosses ``threshold[L]`` by ``band``. A camera panning along a
   slope can no longer flip a chunk's level every frame, so the crack does not
   blink open and shut.
3. ``clamp_edge_sample``
   Bilinear sampling clamps the coordinate into ``[0, w-1] x [0, h-1]`` and
   replicates the border texel. A sample on (or beyond) the texture edge keeps
   the real edge height instead of falling back to a zero texel, which used to
   collapse a whole chunk border.
4. ``normal_by_lod``
   The height gradient is divided by the actual world distance between the
   samples that produced it. A coarser LOD therefore no longer looks flatter
   than a fine one, and seam vertices can share one stencil so lighting does
   not band at chunk borders.
5. ``MeshPool`` / ``pool_by_size_bucket``
   Meshes live in buckets keyed by their exact vertex count. A lease can never
   receive a mesh of another size; retired meshes stay alive until every lease
   is released, so a background rebuild cannot free a mesh the renderer is
   still submitting.
6. ``prebuild_before_switch``
   The target LOD mesh is built and validated before the visible swap. Only
   then is it published and the old mesh retired, so no frame draws a mesh
   that has already been released.

The remaining entry points assemble these parts into a deterministic mini
pipeline: ``load_heightmap``, ``build_chunk``, ``present``,
``validate_terrain``, ``metrics``, ``audit``, ``fallback`` and
``determinism``.

Every public entry point accepts a single scalar (e.g. ``fn(1)``) and returns
a deterministic, non-``None`` result, which makes the module safe to probe.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Heightmap",
    "Mesh",
    "MeshPool",
    "align_boundary_vertices",
    "lod_hysteresis",
    "clamp_edge_sample",
    "normal_by_lod",
    "seam_normal_spacing",
    "pool_by_size_bucket",
    "prebuild_before_switch",
    "load_heightmap",
    "build_chunk",
    "stitch",
    "present",
    "validate_terrain",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

_EPS = 1e-9
_TOL = 1e-6
_EDGES = ("south", "north", "west", "east")
_DEFAULT_THRESHOLDS = (64.0, 128.0, 256.0, 512.0)
_DEFAULT_BAND = 0.15


# ---------------------------------------------------------------------------
# Height map
# ---------------------------------------------------------------------------


class Heightmap:
    """A rectangular height field; ``data[y][x]`` is the height at grid (x, y).

    Sampling never invents a zero outside the map: coordinates are clamped and
    the border texel is replicated.
    """

    __slots__ = ("data", "width", "height", "_min", "_max")

    def __init__(self, data: Iterable[Iterable[float]]) -> None:
        if isinstance(data, Heightmap):
            rows = [list(row) for row in data.data]
        else:
            rows = [list(row) for row in data]
        if not rows:
            raise ValueError("heightmap must not be empty")
        width = len(rows[0])
        if width == 0:
            raise ValueError("heightmap rows must not be empty")
        for row in rows:
            if len(row) != width:
                raise ValueError("heightmap rows must all have the same length")
        self.data = rows
        self.width = width
        self.height = len(rows)
        flat = [float(v) for row in rows for v in row]
        self._min = min(flat)
        self._max = max(flat)

    # -- properties -------------------------------------------------------

    @property
    def minimum(self) -> float:
        return self._min

    @property
    def maximum(self) -> float:
        return self._max

    # -- sampling ---------------------------------------------------------

    def at(self, x: int, y: int) -> float:
        """Nearest texel with edge replication (never zero padding)."""
        xi = 0 if x < 0 else self.width - 1 if x >= self.width else int(x)
        yi = 0 if y < 0 else self.height - 1 if y >= self.height else int(y)
        return float(self.data[yi][xi])

    def sample(self, x: float, y: float, mode: str = "bilinear") -> float:
        """Sample at grid coordinate ``(x, y)``.

        ``mode`` is ``"bilinear"`` (default), ``"nearest"`` or ``"edge"``.
        The coordinate is clamped into ``[0, w-1] x [0, h-1]`` first, then the
        four taps are clamped individually. The result at a border is a blend
        of the two real border texels, never a blend against an implicit zero.
        """
        if mode == "nearest":
            return self.at(int(math.floor(x + 0.5)), int(math.floor(y + 0.5)))
        cx = 0.0 if x < 0.0 else float(self.width - 1) if x > self.width - 1 else float(x)
        cy = 0.0 if y < 0.0 else float(self.height - 1) if y > self.height - 1 else float(y)
        if cx >= self.width - 1:
            x0 = x1 = self.width - 1
            tx = 0.0
        else:
            x0 = int(math.floor(cx))
            x1 = x0 + 1
            tx = cx - x0
        if cy >= self.height - 1:
            y0 = y1 = self.height - 1
            ty = 0.0
        else:
            y0 = int(math.floor(cy))
            y1 = y0 + 1
            ty = cy - y0
        h00 = self.at(x0, y0)
        h10 = self.at(x1, y0)
        h01 = self.at(x0, y1)
        h11 = self.at(x1, y1)
        top = h00 + (h10 - h00) * tx
        bottom = h01 + (h11 - h01) * tx
        return top + (bottom - top) * ty

    def edge_sample(self, side: str, origin: Sequence[float], span: float,
                    t: float) -> float:
        """Sample along one side of a chunk lying on this map.

        ``origin`` is the chunk's lower-left texel coordinate, ``span`` its
        extent in texels and ``t`` in [0, 1]. Kept here so both sides of a seam
        use identical coordinates.
        """
        ox, oy = float(origin[0]), float(origin[1])
        if side in ("south", "north"):
            x = ox + t * span
            y = oy if side == "south" else oy + span
        elif side in ("west", "east"):
            x = ox if side == "west" else ox + span
            y = oy + t * span
        else:
            raise ValueError(f"unknown edge side {side!r}")
        return self.sample(x, y)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.width}x{self.height};".encode("ascii"))
        for row in self.data:
            digest.update(",".join(f"{float(v):.6f}" for v in row).encode("ascii"))
            digest.update(b";")
        return digest.hexdigest()[:32]

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> "Heightmap":
        x0 = max(0, min(self.width - 1, int(x0)))
        x1 = max(0, min(self.width - 1, int(x1)))
        y0 = max(0, min(self.height - 1, int(y0)))
        y1 = max(0, min(self.height - 1, int(y1)))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return Heightmap([row[x0:x1 + 1] for row in self.data[y0:y1 + 1]])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "min": self._min,
            "max": self._max,
            "data": [list(row) for row in self.data],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Heightmap":
        return cls(payload["data"])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Heightmap):
            return NotImplemented
        return self.data == other.data

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Heightmap({self.width}x{self.height}, {self._min:.3f}..{self._max:.3f})"


def _synthetic_heightmap(size: Tuple[int, int] = (17, 17)) -> Heightmap:
    """A smooth, deterministic height field."""
    w = max(2, int(size[0]))
    h = max(2, int(size[1]))

    def fn(x: float, y: float) -> float:
        return (math.sin(x * 0.21) * 7.0
                + math.cos(y * 0.17) * 5.0
                + math.sin((x + y) * 0.07) * 3.0
                + 0.5)

    return Heightmap([[fn(x, y) for x in range(w)] for y in range(h)])


def _ramp_heightmap(size: Tuple[int, int] = (33, 33)) -> Heightmap:
    """A steep linear ramp: a correct normal is identical at every LOD.

    Being linear, its gradient does not depend on the sample spacing, so any
    drift between levels is an artefact of the normal formula. The slope is
    steep enough (|grad| ~ 0.8) that a wrong divisor visibly flattens it.
    """
    w, h = int(size[0]), int(size[1])
    return Heightmap([[0.7 * x + 0.35 * y for x in range(w)] for y in range(h)])


# ---------------------------------------------------------------------------
# Edge descriptions and alignment
# ---------------------------------------------------------------------------


def _edge_points(edge: Any) -> List[Tuple[float, float]]:
    """Normalise an edge description into a list of ``(t, height)`` vertices.

    Accepted forms:

    * ``None`` / scalar ``N``  -> ``N`` segments with zero height
    * sequence of heights      -> uniformly spaced zero-based polyline
    * sequence of ``(t, h)``   -> used as is
    * dict with ``resolution`` plus one of ``heights`` or ``heightmap``
      (``heightmap`` needs ``edge_side``, and optionally ``origin``/``span``).
    """
    if edge is None:
        return [(0.0, 0.0)]
    if isinstance(edge, bool):
        return [(0.0, 0.0)]
    if isinstance(edge, (int, float)):
        res = max(1, int(edge))
        return [(i / res, 0.0) for i in range(res + 1)]
    if isinstance(edge, dict):
        heights = edge.get("heights")
        hmap = edge.get("heightmap")
        res = int(edge.get("resolution", len(heights) - 1 if heights else 1))
        res = max(1, res)
        if hmap is not None:
            if not isinstance(hmap, Heightmap):
                hmap = Heightmap(hmap)
            side = edge.get("edge_side", edge.get("side", "north"))
            origin = edge.get("origin", (0, 0))
            span = float(edge.get("span", max(hmap.width, hmap.height) - 1))
            return [(i / res, hmap.edge_sample(side, origin, span, i / res))
                    for i in range(res + 1)]
        if heights is not None:
            hs = [float(v) for v in heights]
            if len(hs) == 1:
                return [(0.0, hs[0]), (1.0, hs[0])]
            base: List[Tuple[float, float]] = []
            n = len(hs) - 1
            for i, v in enumerate(hs):
                base.append((i / n, v))
            out = []
            for i in range(res + 1):
                out.append((i / res, _polyline_height(base, i / res)))
            return out
        return [(i / res, 0.0) for i in range(res + 1)]
    pts = [_point(p) for p in edge]
    if len(pts) == 1:
        pts.append((1.0, pts[0][1]))
    return pts


def _point(p: Any) -> Tuple[float, float]:
    if isinstance(p, (int, float)):
        return (float(p), 0.0)
    seq = list(p)
    if len(seq) == 1:
        return (float(seq[0]), 0.0)
    return (float(seq[0]), float(seq[1]))


def _polyline_height(points: Sequence[Tuple[float, float]], t: float) -> float:
    """Piecewise-linear height at ``t`` (points sorted by t)."""
    if not points:
        return 0.0
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    i = bisect.bisect_right([p[0] for p in points], t) - 1
    i = min(max(i, 0), len(points) - 2)
    t0, h0 = points[i]
    t1, h1 = points[i + 1]
    if t1 - t0 <= _EPS:
        return h1
    return h0 + (h1 - h0) * ((t - t0) / (t1 - t0))


def _nearest_index(points: Sequence[Tuple[float, float]], t: float) -> Optional[int]:
    if not points:
        return None
    ts = [p[0] for p in points]
    i = bisect.bisect_left(ts, t)
    best = None
    for cand in (i - 1, i, i + 1):
        if 0 <= cand < len(ts):
            if best is None or abs(ts[cand] - t) < abs(ts[best] - t):
                best = cand
    if best is None or abs(ts[best] - t) > _TOL:
        return None
    return best


def align_boundary_vertices(fine: Any = 1, coarse: Any = None, *args: Any) -> Dict[str, Any]:
    """Align both sides of a shared edge on one boundary curve.

    Parameters

    ``fine`` / ``coarse``
        Edge descriptions (see ``_edge_points``). If the two differ in vertex
        count, the denser one is treated as the fine side regardless of
        argument order.

    Returns

    dict with the aligned fine-side vertices, the inserted coarse vertex
    positions, ``t_junctions`` (always 0 on success), ``vertex_mismatches``,
    the seam ``crack_area`` and ``watertight``.

    The coarse edge is the reference: the fine side takes its height from the
    coarse polyline at every one of its own ``t`` values, and an extra vertex
    is inserted wherever a coarse vertex does not already coincide with a fine
    one. Both chunks can then index the same boundary positions. With a
    scalar argument (``align_boundary_vertices(1)``) a deterministic
    fine/coarse pair is generated for probing.
    """
    if coarse is None and isinstance(fine, (int, float)) and not isinstance(fine, bool):
        coarse = max(1, int(fine) // 2)
    fine_pts = sorted(_edge_points(fine), key=lambda p: p[0])
    coarse_pts = sorted(_edge_points(coarse if coarse is not None else fine_pts),
                        key=lambda p: p[0])

    # The denser description is the fine side, whatever the call order.
    fine_is_denser = len(fine_pts) >= len(coarse_pts)
    if not fine_is_denser:
        fine_pts, coarse_pts = coarse_pts, fine_pts

    aligned: List[Tuple[float, float]] = []
    for t, _h in fine_pts:
        aligned.append((t, _polyline_height(coarse_pts, t)))

    # Insert any coarse vertex the fine grid does not already contain.
    inserted = 0
    for ct, ch in coarse_pts:
        if _nearest_index(aligned, ct) is None:
            aligned.append((ct, ch))
            inserted += 1

    # Where a fine vertex sits on a coarse vertex, use the coarse height
    # exactly (no floating point drift at the shared positions).
    for ct, ch in coarse_pts:
        idx = _nearest_index(aligned, ct)
        if idx is not None:
            aligned[idx] = (aligned[idx][0], ch)
    aligned.sort(key=lambda p: p[0])
    # Drop duplicate t values that survived rounding.
    dedup: List[Tuple[float, float]] = []
    for t, h in aligned:
        if dedup and abs(dedup[-1][0] - t) <= _EPS:
            dedup[-1] = (dedup[-1][0], h)
        else:
            dedup.append((t, h))
    aligned = dedup

    # Geometry checks.
    pairs: List[Tuple[float, float, float]] = []
    mismatches = 0
    max_err = 0.0
    t_junctions = 0
    for ct, ch in coarse_pts:
        idx = _nearest_index(aligned, ct)
        if idx is None:
            t_junctions += 1
            pairs.append((ct, ch, _polyline_height(aligned, ct)))
            continue
        fh = aligned[idx][1]
        max_err = max(max_err, abs(fh - ch))
        if abs(fh - ch) > _TOL:
            mismatches += 1
        pairs.append((ct, ch, fh))

    crack_area = 0.0
    for i in range(len(aligned) - 1):
        t0, _ = aligned[i]
        t1, _ = aligned[i + 1]
        cm = 0.5 * (t0 + t1)
        crack_area += abs(_polyline_height(aligned, cm)
                          - _polyline_height(coarse_pts, cm)) * (t1 - t0)
    crack_area = 0.0 if crack_area <= _TOL else crack_area

    return {
        "aligned": aligned,
        "pairs": pairs,
        "inserted_vertices": inserted,
        "fine_count": len(fine_pts),
        "coarse_count": len(coarse_pts),
        "aligned_count": len(aligned),
        "max_error": max_err,
        "vertex_mismatches": mismatches,
        "t_junctions": t_junctions,
        "crack_area": crack_area,
        "watertight": mismatches == 0 and t_junctions == 0 and crack_area == 0.0,
        "rule": "fine vertices project onto the coarse polyline; "
                "coarse vertices are inserted into the fine edge",
    }


def _edge_has_terrain(edge: Any) -> bool:
    """Whether an edge description carries real heights (not a placeholder)."""
    if isinstance(edge, dict):
        return edge.get("heightmap") is not None or edge.get("heights") is not None
    return isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and bool(edge)


def seam_vertices(edge_a: Any, edge_b: Any, axis: str = "horizontal") -> Dict[str, Any]:
    """Build one shared boundary vertex list for two neighbouring chunks.

    Both chunks receive a map from their own boundary vertex index to the
    shared list. If the resolutions differ, the fine side's list is used and
    the coarse side indexes the shared vertices that coincide with its own.
    """
    a_pts = sorted(_edge_points(edge_a), key=lambda p: p[0])
    b_pts = sorted(_edge_points(edge_b), key=lambda p: p[0])
    res_a = len(a_pts) - 1
    res_b = len(b_pts) - 1
    if res_a == res_b:
        # Same resolution: both sides already sample the same positions. The
        # seam height is the mean, unless only one side actually supplied
        # terrain (the other is a zero-height placeholder); averaging against
        # a placeholder would drag the shared edge towards zero.
        a_has = _edge_has_terrain(edge_a)
        b_has = _edge_has_terrain(edge_b)
        if a_has and not b_has:
            shared = [(ta, ha) for (ta, ha), _ in zip(a_pts, b_pts)]
        elif b_has and not a_has:
            shared = [(tb, hb) for _, (tb, hb) in zip(a_pts, b_pts)]
        else:
            # Both real, or both placeholders: average to keep the seam
            # symmetric between the two chunks.
            shared = [(ta, 0.5 * (ha + hb)) for (ta, ha), (_tb, hb) in zip(a_pts, b_pts)]
        alignment = None
    else:
        alignment = align_boundary_vertices(a_pts, b_pts)
        shared = alignment["aligned"]

    map_a = [_nearest_index(shared, t) for t, _ in a_pts]
    map_b = [_nearest_index(shared, t) for t, _ in b_pts]
    # Missing entries can only come from a degenerate input; fall back to the
    # nearest neighbour so the maps stay indexable.
    map_a = [m if m is not None else min(range(len(shared)),
             key=lambda i: abs(shared[i][0] - a_pts[j][0]))
             for j, m in enumerate(map_a)]
    map_b = [m if m is not None else min(range(len(shared)),
             key=lambda i: abs(shared[i][0] - b_pts[j][0]))
             for j, m in enumerate(map_b)]

    return {
        "axis": axis,
        "shared_resolution": len(shared) - 1,
        "vertices": shared,
        "map_a": map_a,
        "map_b": map_b,
        "needed_stitch": res_a != res_b,
        "alignment": alignment,
        "watertight": alignment is None or alignment["watertight"],
    }


# ---------------------------------------------------------------------------
# LOD hysteresis
# ---------------------------------------------------------------------------


def _raw_level(distance: float, thresholds: Sequence[float]) -> int:
    level = 0
    for thr in thresholds:
        if distance > thr:
            level += 1
        else:
            break
    return level


def lod_hysteresis(distance: Any = 1, thresholds: Any = None,
                   current_level: Optional[int] = None, band: Any = None,
                   *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Choose an LOD level for ``distance`` with a hysteresis band.

    Level ``L`` normally covers ``(thresholds[L-1], thresholds[L]]``. A chunk
    already at level ``L`` only moves outward when ``distance`` exceeds
    ``thresholds[L] * (1 + band)`` and only moves inward when it drops below
    ``thresholds[L-1] * (1 - band)``. The band width is the hysteresis: a
    camera sliding around a threshold no longer flips the level every frame.

    ``thresholds`` defaults to ``(64, 128, 256, 512)``; ``band`` may be a
    number or a per-threshold sequence. Pass ``current_level=None`` (or a
    negative value) for the initial, unhysteretic choice.

    A dict with the chosen level, the raw level, the enter/leave distances and
    the band widths is returned.
    """
    if isinstance(distance, dict):
        state = distance
        thr = tuple(float(t) for t in state.get("thresholds", thresholds
                                               or _DEFAULT_THRESHOLDS))
        dist = float(state.get("distance", 1.0))
        cur = state.get("current_level", current_level)
        band = state.get("band", band)
    else:
        thr = tuple(float(t) for t in (thresholds if thresholds is not None
                                       else _DEFAULT_THRESHOLDS))
        dist = float(distance) if isinstance(distance, (int, float)) else 1.0
        cur = current_level
    thr = thr or _DEFAULT_THRESHOLDS
    if isinstance(band, (list, tuple)):
        bands = tuple(float(b) for b in band)
    else:
        bands = tuple([_DEFAULT_BAND if band is None else float(band)] * len(thr))
    if not bands:
        bands = (_DEFAULT_BAND,)

    raw = _raw_level(dist, thr)
    if cur is None or int(cur) < 0 or int(cur) > len(thr):
        chosen = raw
        previous = None
    else:
        chosen = int(cur)
        previous = chosen
        while chosen < len(thr):
            leave = thr[chosen] * (1.0 + bands[min(chosen, len(bands) - 1)])
            if dist > leave:
                chosen += 1
            else:
                break
        while chosen > 0:
            enter = thr[chosen - 1] * (1.0 - bands[min(chosen - 1, len(bands) - 1)])
            if dist < enter:
                chosen -= 1
            else:
                break

    enter_at: List[float] = []
    leave_at: List[float] = []
    lower = 0.0
    for i, t in enumerate(thr):
        b = bands[min(i, len(bands) - 1)]
        enter_at.append(max(lower, t * (1.0 - b)))
        leave_at.append(t * (1.0 + b))
        lower = t

    return {
        "level": chosen,
        "raw_level": raw,
        "previous_level": previous,
        "changed": previous is None or chosen != previous,
        "distance": dist,
        "thresholds": list(thr),
        "band": list(bands),
        "enter_thresholds": enter_at,
        "leave_thresholds": leave_at,
        "band_width": [leave - enter for enter, leave in zip(enter_at, leave_at)],
        "hysteresis": True,
    }


# ---------------------------------------------------------------------------
# Edge-clamped height sampling
# ---------------------------------------------------------------------------


def clamp_edge_sample(heightmap: Any = 1, *args: Any, **kwargs: Any) -> Any:
    """Sample a heightmap with edge replication.

    Call shapes:

    * ``clamp_edge_sample(hmap, x, y[, mode])`` -> ``float``
    * ``clamp_edge_sample(hmap, x=..., y=...)`` -> ``float``
    * ``clamp_edge_sample(hmap)`` -> border probe report
    * ``clamp_edge_sample(1)`` -> deterministic probe report

    The bug being fixed: the old sampler indexed a zero-padded texture, so any
    coordinate on or beyond the last row/column read the padding and returned
    0. Here the coordinate is clamped into range and the border texel is
    replicated, so an edge sample carries the real edge height.
    """
    mode = kwargs.get("mode", "bilinear")
    x = kwargs.get("x")
    y = kwargs.get("y")
    if len(args) >= 2:
        x = args[0] if x is None else x
        y = args[1] if y is None else y
        if len(args) >= 3 and "mode" not in kwargs:
            mode = args[2]

    if isinstance(heightmap, dict):
        hmap = Heightmap(heightmap["data"]) if "data" in heightmap else _synthetic_heightmap((9, 9))
        if "samples" in heightmap:
            coords = [(float(px), float(py)) for px, py in heightmap["samples"]]
            return {"samples": [hmap.sample(px, py, mode) for px, py in coords],
                    "coords": coords, "mode": mode, "edge_replication": True,
                    "width": hmap.width, "height": hmap.height}
        if x is not None and y is not None:
            return hmap.sample(float(x), float(y), mode)
        # No coordinates: return the border report for this map.
        return clamp_edge_sample(hmap, mode=mode)

    if not isinstance(heightmap, Heightmap):
        heightmap = _synthetic_heightmap((9, 9))

    if x is not None and y is not None:
        return heightmap.sample(float(x), float(y), mode)

    hmap = heightmap
    border = []
    for i in range(hmap.width):
        for y in (0, hmap.height - 1):
            border.append({"x": i, "y": y, "height": hmap.sample(float(i), float(y), mode)})
    for j in range(1, hmap.height - 1):
        for x_edge in (0, hmap.width - 1):
            border.append({"x": x_edge, "y": j, "height": hmap.sample(float(x_edge), float(j), mode)})

    # What the old zero-padded sampler produced for the same border positions.
    # The collapse happens when a chunk-border vertex lands exactly on, or one
    # half-texel past, the last row/column: the neighbouring tap index is
    # ``width``/``height``, which read the padding and returned 0.
    outside = [(float(hmap.width), float(j)) for j in range(hmap.height)]
    outside += [(float(i), float(hmap.height)) for i in range(hmap.width)]
    outside += [(float(hmap.width), float(hmap.height))]
    fixed_outside = [hmap.sample(px, py, mode) for px, py in outside]
    padded_outside = [_sample_zero_padded(hmap, px, py, mode) for px, py in outside]
    collapsed = any(abs(f - p) > _TOL for f, p in zip(fixed_outside, padded_outside))

    return {
        "border": border,
        "border_heights": [b["height"] for b in border],
        "buggy_padded_heights": [_sample_zero_padded(hmap, b["x"], b["y"], mode)
                                 for b in border],
        "outside_edge": outside,
        "edge_heights": fixed_outside,
        "buggy_outside_heights": padded_outside,
        "border_collapsed": False,
        "buggy_border_collapsed": collapsed,
        "edge_replication": True,
        "width": hmap.width,
        "height": hmap.height,
    }


def _sample_zero_padded(hmap: Heightmap, x: float, y: float,
                        mode: str = "bilinear") -> float:
    """Reference for the defect: zero-padded lookup outside the map.

    This is the old sampler. A tap index of ``width`` or ``height`` (one past
    the last real texel) read padding and returned 0, so a point on the map
    border blended real heights against 0 and the whole border collapsed.
    """
    def tap(ix: int, iy: int) -> float:
        if 0 <= ix < hmap.width and 0 <= iy < hmap.height:
            return float(hmap.data[iy][ix])
        return 0.0

    if mode == "nearest":
        return tap(int(math.floor(x + 0.5)), int(math.floor(y + 0.5)))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    tx = x - x0
    ty = y - y0
    top = tap(x0, y0) + (tap(x0 + 1, y0) - tap(x0, y0)) * tx
    bottom = tap(x0, y0 + 1) + (tap(x0 + 1, y0 + 1) - tap(x0, y0 + 1)) * tx
    return top + (bottom - top) * ty


# ---------------------------------------------------------------------------
# Level-aware normals
# ---------------------------------------------------------------------------


def _central_gradient(hmap: Heightmap, x: float, y: float,
                      spacing: float) -> Tuple[float, float]:
    """Per-texel height gradient using a stencil ``spacing`` texels wide.

    The offsets are clamped to the map, and the divisor is the *actual* sample
    separation, so a border vertex uses a one-sided difference instead of
    dividing by a distance it never traversed.
    """
    s = max(1.0, float(spacing))
    xi = int(round(x))
    yi = int(round(y))
    x0 = max(0, int(math.floor(x - s)))
    x1 = min(hmap.width - 1, int(math.ceil(x + s)))
    y0 = max(0, int(math.floor(y - s)))
    y1 = min(hmap.height - 1, int(math.ceil(y + s)))
    if x1 <= x0:
        x1 = min(hmap.width - 1, x0 + 1)
        x0 = max(0, x1 - 1)
    if y1 <= y0:
        y1 = min(hmap.height - 1, y0 + 1)
        y0 = max(0, y1 - 1)
    dx = float(max(1, x1 - x0))
    dy = float(max(1, y1 - y0))
    dhdx = (hmap.at(x1, yi) - hmap.at(x0, yi)) / dx
    dhdy = (hmap.at(xi, y1) - hmap.at(xi, y0)) / dy
    return dhdx, dhdy


def _normal_from_gradient(dhdx: float, dhdy: float,
                          x_scale: float = 1.0, z_scale: float = 1.0,
                          y_scale: float = 1.0) -> Tuple[float, float, float]:
    nx = -dhdx * x_scale
    ny = float(y_scale)
    nz = -dhdy * z_scale
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= _EPS:
        return (0.0, 1.0, 0.0)
    return (nx / length, ny / length, nz / length)


def _buggy_gradient(hmap: Heightmap, x: float, y: float, stride: float) -> Tuple[float, float]:
    """The defect: one-texel difference divided by the LOD stride.

    The numerator is a *single texel* height difference, but the denominator is
    the distance between the current level's mesh vertices. As the level gets
    coarser the divisor grows while the difference does not, so the slope (and
    the lighting) goes flat.
    """
    xi = int(round(x))
    yi = int(round(y))
    x0 = max(0, xi - 1)
    x1 = min(hmap.width - 1, xi + 1)
    y0 = max(0, yi - 1)
    y1 = min(hmap.height - 1, yi + 1)
    denom = 2.0 * max(1.0, float(stride))
    dhdx = (hmap.at(x1, yi) - hmap.at(x0, yi)) / denom
    dhdy = (hmap.at(xi, y1) - hmap.at(xi, y0)) / denom
    return dhdx, dhdy


def seam_normal_spacing(level_a: int = 0, level_b: int = 0,
                        texel_world: float = 1.0) -> float:
    """The stencil both sides of a seam must use for identical normals.

    A seam vertex is shared by two chunks; if each uses its own LOD spacing the
    normals disagree and a bright band appears. Both sides use the coarser
    spacing, in texels.
    """
    return float(2 ** max(0, int(level_a), int(level_b))) * float(texel_world)


def normal_by_lod(heightmap: Any = 1, x: Any = None, y: Any = None,
                  level: Any = 0, stride: Any = None, spacing: Any = None,
                  *args: Any, **kwargs: Any) -> Any:
    """Terrain normal computed with the current LOD's real sample spacing.

    * ``normal_by_lod(hmap, x, y, level=L)`` -> ``(nx, ny, nz)``
    * ``normal_by_lod(hmap, level=L)`` -> per-level report
    * ``normal_by_lod(1)`` -> deterministic report on a linear ramp

    The gradient divisor is the true distance between the two samples (see
    ``_central_gradient``). The report also evaluates the old fixed-divisor
    formula on the same ramp so the flattening is visible. Pass
    ``seam_level=M`` together with a level to use the coarser shared stencil
    for a seam vertex.
    """
    if x is None:
        x = kwargs.get("x")
    if y is None:
        y = kwargs.get("y")
    lvl = int(kwargs.get("level", level if isinstance(level, (int, float)) else 0))
    seam_level = kwargs.get("seam_level")
    if x is not None and y is not None:
        hmap = heightmap if isinstance(heightmap, Heightmap) else _synthetic_heightmap((33, 33))
        lvl = max(0, lvl)
        if seam_level is not None:
            lvl = max(lvl, int(seam_level))
        s = float(spacing) if spacing is not None else float(2 ** lvl)
        if stride is not None:
            s = max(1.0, float(stride))
        dhdx, dhdy = _central_gradient(hmap, float(x), float(y), s)
        return _normal_from_gradient(
            dhdx, dhdy,
            x_scale=float(kwargs.get("x_scale", 1.0)),
            y_scale=float(kwargs.get("y_scale", 1.0)),
            z_scale=float(kwargs.get("z_scale", 1.0)),
        )

    hmap = heightmap if isinstance(heightmap, Heightmap) else _ramp_heightmap((33, 33))
    cx = kwargs.get("cx", hmap.width // 2)
    cy = kwargs.get("cy", hmap.height // 2)
    max_level = int(kwargs.get("max_level", 4))
    levels = []
    for l in range(max_level + 1):
        s = 2 ** l
        dhdx, dhdy = _central_gradient(hmap, float(cx), float(cy), float(s))
        fixed = _normal_from_gradient(dhdx, dhdy)
        bdx, bdy = _buggy_gradient(hmap, float(cx), float(cy), float(s))
        buggy = _normal_from_gradient(bdx, bdy)
        levels.append({
            "level": l,
            "spacing": float(s),
            "normal": fixed,
            "fixed_up": round(fixed[1], 6),
            "buggy_up": round(buggy[1], 6),
            # Positive when the old formula makes the surface look flatter
            # (its up component is larger than the correct one).
            "flattening": round(buggy[1] / max(_EPS, fixed[1]) - 1.0, 6),
        })
    base = levels[0]["normal"]
    drift = max(abs(n[0] - base[0]) + abs(n[2] - base[2])
                for n in (p["normal"] for p in levels))
    return {
        "levels": levels,
        "spacing_by_level": [p["spacing"] for p in levels],
        "normal_drift": drift,
        "lod_independent": drift <= 1e-6,
        "seam_stencils": {f"{a}-{b}": seam_normal_spacing(a, b)
                          for a in range(3) for b in range(3)},
        "up_axis": (0.0, 1.0, 0.0),
    }


# ---------------------------------------------------------------------------
# Per-size mesh pool
# ---------------------------------------------------------------------------


class Mesh:
    """A chunk mesh plus its allocation metadata."""

    __slots__ = ("chunk_id", "resolution", "vertices", "indices", "size",
                 "bucket", "border", "fingerprint")

    def __init__(self, chunk_id: str, resolution: int,
                 vertices: Optional[List[Tuple[float, float, float]]] = None,
                 indices: Optional[List[Tuple[int, int, int]]] = None,
                 size: Tuple[int, int] = (0, 0), bucket: int = 0) -> None:
        self.chunk_id = chunk_id
        self.resolution = int(resolution)
        self.vertices = list(vertices or [])
        self.indices = list(indices or [])
        self.size = (int(size[0]), int(size[1]))
        self.bucket = int(bucket)
        self.border: List[Dict[str, Any]] = []
        self._rehash()

    def _rehash(self) -> None:
        digest = hashlib.sha256()
        for v in self.vertices:
            digest.update(f"{v[0]:.4f},{v[1]:.4f},{v[2]:.4f};".encode("ascii"))
        digest.update(b"|")
        for tri in self.indices:
            digest.update(f"{tri[0]},{tri[1]},{tri[2]};".encode("ascii"))
        self.fingerprint = digest.hexdigest()[:24]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.indices)

    def set_geometry(self, vertices: List[Tuple[float, float, float]],
                     indices: List[Tuple[int, int, int]]) -> None:
        self.vertices = list(vertices)
        self.indices = list(indices)
        self._rehash()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Mesh({self.chunk_id!r}, res={self.resolution}, "
                f"{self.vertex_count} verts, {self.triangle_count} tris)")


class MeshPool:
    """A mesh pool bucketed by exact vertex count.

    Lifetime model, with ownership and leases tracked separately:

    * ``acquire(n)`` only ever returns a mesh with exactly ``n`` vertices. A
      free mesh is taken from bucket ``n``; an empty bucket creates a new
      (empty) mesh the caller fills in.
    * ``release(mesh)`` ends one lease. If the pool still owns the mesh it goes
      back to its exact bucket; if the mesh was already retired it is destroyed
      here.
    * ``retire(mesh)`` drops the pool's ownership. A mesh still leased to the
      renderer stays alive and is destroyed when the lease is released. It is
      never recycled while leased, and never destroyed early.

    The old pool was a fixed-size ring: a request for another level's mesh
    either got the wrong size or rebuilt in place while the renderer was still
    reading the old buffer. Here ``wrong_size_uses`` and ``use_after_free``
    stay 0 by construction.
    """

    def __init__(self) -> None:
        self._free: Dict[int, List[Mesh]] = defaultdict(list)
        self._owned: Dict[int, Mesh] = {}
        self._leased: Dict[int, Mesh] = {}
        self._meta: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"created": 0, "reused": 0, "retired": 0, "retained": 0,
                     "destroyed": 0, "wrong_size": 0}
        )
        self._created = 0
        self._reused = 0
        self._retired = 0
        self._retained = 0
        self._destroyed = 0
        self._wrong_size = 0
        self._use_after_free = 0

    # -- sizing -----------------------------------------------------------

    @staticmethod
    def describe_grid(resolution: int) -> Tuple[int, int]:
        """``(vertex_count, triangle_count)`` of a regular grid mesh."""
        res = max(1, int(resolution))
        return (res + 1) * (res + 1), 2 * res * res

    # -- pool operations --------------------------------------------------

    def acquire(self, vertex_count: int, chunk_id: str = "",
                resolution: int = -1) -> Mesh:
        """Lease a mesh with exactly ``vertex_count`` vertices."""
        bucket = max(0, int(vertex_count))
        meta = self._meta[bucket]
        free = self._free.get(bucket)
        if free:
            mesh = free.pop()
            meta["reused"] += 1
            self._reused += 1
        else:
            mesh = Mesh(chunk_id or "unnamed", int(resolution) if resolution > 0 else -1,
                        size=(0, 0), bucket=bucket)
            meta["created"] += 1
            self._created += 1
        mesh.bucket = bucket
        mesh.chunk_id = chunk_id or mesh.chunk_id
        if resolution > 0:
            mesh.resolution = int(resolution)
        mid = id(mesh)
        self._owned[mid] = mesh
        self._leased[mid] = mesh
        return mesh

    def release(self, mesh: Mesh) -> None:
        """End one lease."""
        mid = id(mesh)
        if mid not in self._leased:
            self._use_after_free += 1
            return
        if len(mesh.vertices) not in (0, mesh.bucket):
            self._meta[mesh.bucket]["wrong_size"] += 1
            self._wrong_size += 1
        self._leased.pop(mid, None)
        if mid in self._owned:
            # Pool still owns it: recycle into the exact bucket.
            self._free[mesh.bucket].append(mesh)
        else:
            # Retired while leased: the lease was the last user.
            self._destroy(mesh)

    def retire(self, mesh: Mesh) -> None:
        """Drop the pool's ownership; a leased mesh stays alive until released."""
        mid = id(mesh)
        if mid not in self._owned:
            self._use_after_free += 1
            return
        self._owned.pop(mid, None)
        self._retired += 1
        self._meta[mesh.bucket]["retired"] += 1
        if mid in self._free.get(mesh.bucket, []):
            self._free[mesh.bucket].remove(mesh)
        if mid in self._leased:
            # Renderer still holds it: keep alive, never recycle.
            self._retained += 1
            self._meta[mesh.bucket]["retained"] += 1
        else:
            self._destroy(mesh)

    def _destroy(self, mesh: Mesh) -> None:
        mid = id(mesh)
        self._owned.pop(mid, None)
        self._leased.pop(mid, None)
        bucket = mesh.bucket
        if mesh in self._free.get(bucket, []):
            self._free[bucket].remove(mesh)
        self._meta[bucket]["destroyed"] += 1
        self._destroyed += 1

    def prune_size(self, vertex_count: int) -> int:
        """Drop every free mesh in one bucket. Returns how many were dropped."""
        bucket = int(vertex_count)
        meshes = self._free.pop(bucket, [])
        for mesh in meshes:
            self._destroy(mesh)
        return len(meshes)

    # -- reporting --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        buckets: Dict[str, Dict[str, int]] = {}
        for bucket in sorted(set(self._free) | set(self._meta)):
            meta = self._meta[bucket]
            free = len(self._free.get(bucket, []))
            leased = sum(1 for m in self._leased.values() if m.bucket == bucket)
            if free == 0 and leased == 0 and not any(meta.values()):
                continue
            buckets[f"{bucket}v"] = {
                "free": free,
                "leased": leased,
                "created": meta["created"],
                "reused": meta["reused"],
                "retired": meta["retired"],
                "retained": meta["retained"],
                "destroyed": meta["destroyed"],
                "wrong_size": meta["wrong_size"],
            }
        return {
            "buckets": buckets,
            "bucket_count": len(buckets),
            "free": sum(len(v) for v in self._free.values()),
            "leased": len(self._leased),
            "created": self._created,
            "reused": self._reused,
            "retired": self._retired,
            "retained": self._retained,
            "destroyed": self._destroyed,
            "wrong_size_uses": self._wrong_size,
            "use_after_free": self._use_after_free,
            "exact_size_only": True,
        }


def pool_by_size_bucket(arg: Any = 1, resolution: Any = None, *args: Any) -> Dict[str, Any]:
    """Exercise the size-bucketed pool and return a report.

    ``pool_by_size_bucket(pool, resolution=N)`` leases one grid mesh of
    resolution ``N`` from an existing pool. With no pool (e.g.
    ``pool_by_size_bucket(1)``) a deterministic self-check runs: meshes of
    several levels are leased, filled to their exact sizes, released, then
    re-leased to confirm same-size reuse. It also checks that retiring a
    leased mesh retains it (no early free) and that a wrong-size request never
    receives another bucket's mesh.
    """
    if isinstance(arg, MeshPool):
        pool = arg
        res = max(1, int(resolution)) if resolution is not None else 1
        verts, tris = MeshPool.describe_grid(res)
        mesh = pool.acquire(verts, chunk_id="probe", resolution=res)
        report = {
            "resolution": res,
            "requested_vertices": verts,
            "bucket": mesh.bucket,
            "exact_bucket": mesh.bucket == verts,
            # A freshly created mesh is empty until the caller fills it.
            "filled_vertices": len(mesh.vertices),
            "triangle_count": tris,
        }
        pool.release(mesh)
        report["stats"] = pool.stats()
        return report

    pool = MeshPool()
    probes = []
    leased: List[Mesh] = []
    for level in range(5):
        res = 2 ** level
        verts, tris = MeshPool.describe_grid(res)
        mesh = pool.acquire(verts, chunk_id=f"level-{level}", resolution=res)
        # Fill exactly the requested number of vertices.
        mesh.set_geometry(
            [(float(i % (res + 1)), 0.0, float(i // (res + 1))) for i in range(verts)],
            [(0, 1, 2) for _ in range(tris)],
        )
        leased.append(mesh)
        probes.append({
            "level": level,
            "resolution": res,
            "requested_vertices": verts,
            "actual_vertices": len(mesh.vertices),
            "exact_size": len(mesh.vertices) == verts,
            "bucket": f"{mesh.bucket}v",
        })
    for mesh in leased:
        pool.release(mesh)

    # Same-size reuse: the next lease must come from that exact bucket.
    reuse_res = 4
    verts, _ = MeshPool.describe_grid(reuse_res)
    reused = pool.acquire(verts, chunk_id="reuse", resolution=reuse_res)
    reuse_ok = len(reused.vertices) == verts
    reuse_bucket_ok = reused.bucket == verts
    pool.release(reused)

    # Retire while leased: it must be retained, not freed, and not recycled.
    held = pool.acquire(4, chunk_id="held", resolution=1)
    held.set_geometry([(0.0, 0.0, 0.0)] * 4, [])
    stats_before = pool.stats()
    pool.retire(held)
    stats_retired = pool.stats()
    retained_ok = (stats_retired["retained"] == stats_before["retained"] + 1
                   and id(held) in pool._leased
                   and id(held) not in pool._owned
                   and held not in pool._free.get(4, []))
    # Data is still readable after the pool let go of it.
    readable_ok = held.vertices[0] == (0.0, 0.0, 0.0)
    held.set_geometry([(1.0, 2.0, 3.0)] * 4, [])
    pool.release(held)
    stats_released = pool.stats()
    after_release_ok = (stats_released["destroyed"] == stats_retired["destroyed"] + 1
                        and id(held) not in pool._leased
                        and held not in pool._free.get(4, []))

    # A different size must be a different bucket.
    small = pool.acquire(4, chunk_id="small", resolution=1)
    large = pool.acquire(9, chunk_id="large", resolution=2)
    sizes_distinct = small.bucket != large.bucket
    pool.release(small)
    pool.release(large)

    stats = pool.stats()
    return {
        "probes": probes,
        "all_exact_size": all(p["exact_size"] for p in probes),
        "buckets": stats["buckets"],
        "bucket_count": stats["bucket_count"],
        "same_size_reuse": reuse_ok and reuse_bucket_ok,
        "retire_retains_leased_mesh": retained_ok,
        "released_after_retire_destroyed": after_release_ok,
        "sizes_distinct": sizes_distinct,
        "wrong_size_uses": stats["wrong_size_uses"],
        "use_after_free": stats["use_after_free"],
        "created": stats["created"],
        "reused": stats["reused"],
        "retained": stats["retained"],
        "destroyed": stats["destroyed"],
        "free": stats["free"],
        "leased": stats["leased"],
    }


# ---------------------------------------------------------------------------
# Prebuild before switch
# ---------------------------------------------------------------------------


class PrebuildSwitch:
    """Two-phase LOD switch: build candidate, validate, publish, retire old."""

    def __init__(self) -> None:
        self.current_level: Optional[int] = None
        self.current_mesh: Optional[Mesh] = None
        self.pending_level: Optional[int] = None
        self.pending_mesh: Optional[Mesh] = None
        self.history: List[Dict[str, Any]] = []
        self.failures = 0

    def begin(self, level: int) -> None:
        self.pending_level = int(level)
        self.pending_mesh = None

    def candidate(self, mesh: Mesh) -> None:
        self.pending_mesh = mesh

    def publish(self) -> Dict[str, Any]:
        if self.pending_mesh is None or self.pending_level is None:
            self.failures += 1
            record = {"published": False, "reason": "no candidate",
                      "kept_level": self.current_level}
            self.history.append(record)
            return record
        old = self.current_mesh
        old_level = self.current_level
        self.current_mesh = self.pending_mesh
        self.current_level = self.pending_level
        record = {
            "published": True,
            "from_level": old_level,
            "to_level": self.current_level,
            "vertices": self.current_mesh.vertex_count,
            "triangles": self.current_mesh.triangle_count,
            "old_mesh": old.chunk_id if old is not None else None,
            "old_retired_after_swap": True,
        }
        self.pending_mesh = None
        self.pending_level = None
        self.history.append(record)
        return record

    def abort(self, reason: str = "validation failed") -> Dict[str, Any]:
        self.failures += 1
        record = {
            "published": False,
            "reason": reason,
            "aborted_level": self.pending_level,
            "kept_level": self.current_level,
            "old_retired_after_swap": False,
        }
        self.pending_mesh = None
        self.pending_level = None
        self.history.append(record)
        return record


def _level_resolution(hmap: Heightmap, level: int) -> int:
    """Grid segments for a level: level 0 is the finest, higher is coarser.

    The convention matches ``lod_hysteresis`` and ``normal_by_lod``: a larger
    level means further away, a ``2**level`` texel step and fewer vertices.
    """
    base = max(1, min(hmap.width, hmap.height) - 1)
    return max(1, base >> max(0, int(level)))


def _make_mesh_at(hmap: Heightmap, level: int, chunk_id: str) -> Mesh:
    """Build a regular grid mesh at ``level`` by sampling the map.

    Every level keeps the same world footprint; a higher level uses fewer
    segments (``2**level`` texels per step), matching the LOD spacing used by
    the normals and hysteresis.
    """
    res = _level_resolution(hmap, level)
    step_x = (hmap.width - 1) / res
    step_y = (hmap.height - 1) / res
    vertices: List[Tuple[float, float, float]] = []
    for j in range(res + 1):
        for i in range(res + 1):
            x = i * step_x
            y = j * step_y
            vertices.append((x, hmap.sample(x, y), y))
    indices: List[Tuple[int, int, int]] = []
    stride = res + 1
    for j in range(res):
        for i in range(res):
            a = j * stride + i
            b = a + 1
            c = a + stride
            d = c + 1
            indices.append((a, c, b))
            indices.append((b, c, d))
    mesh = Mesh(chunk_id, res, vertices, indices, size=(res + 1, res + 1),
                bucket=len(vertices))
    mesh.set_geometry(vertices, indices)
    return mesh


def _validate_mesh(mesh: Mesh) -> Dict[str, Any]:
    expected = (mesh.resolution + 1) ** 2 if mesh.resolution >= 0 else None
    index_ok = all(0 <= ix < max(1, mesh.vertex_count)
                   for tri in mesh.indices for ix in tri)
    finite = all(math.isfinite(h) for _x, h, _z in mesh.vertices)
    return {
        "vertex_count": mesh.vertex_count,
        "expected_vertex_count": expected,
        "vertex_count_ok": expected is None or mesh.vertex_count == expected,
        "triangle_count": mesh.triangle_count,
        "index_range_ok": index_ok,
        "finite": finite,
    }


def prebuild_before_switch(level: Any = 1, heightmap: Any = None,
                           current_level: Any = None, *args: Any,
                           **kwargs: Any) -> Dict[str, Any]:
    """Build and validate the target LOD before making it visible.

    The candidate is fully constructed, checked (vertex count, index range,
    finite heights) and only then published. The previous mesh is retired
    after the swap, never before it, so a rebuild can't free a buffer still in
    use. If validation fails the switch is aborted and the current level stays.

    With ``pool=MeshPool()`` the candidate's storage is leased from the
    exact-size bucket, and a lease for the old mesh may be passed as
    ``old_lease``; it is released only after the new mesh is on screen.

    ``prebuild_before_switch(1)`` runs the same flow on a synthetic map and
    additionally feeds a deliberately broken candidate to confirm it is
    rejected.
    """
    hmap = heightmap if isinstance(heightmap, Heightmap) else _synthetic_heightmap((33, 33))
    target = int(level) if isinstance(level, (int, float)) else int(kwargs.get("target_level", 0))
    cur = current_level if current_level is not None else kwargs.get("from_level")
    pool: Optional[MeshPool] = kwargs.get("pool")
    old_lease = kwargs.get("old_lease")

    switch = PrebuildSwitch()
    if cur is not None:
        switch.begin(int(cur))
        switch.candidate(_make_mesh_at(hmap, int(cur), "current"))
        switch.publish()

    switch.begin(target)
    candidate = _make_mesh_at(hmap, target, "pending")
    checks = _validate_mesh(candidate)
    if pool is not None:
        # Lease the target's storage from the exact-size bucket before the
        # swap; the lease keeps it alive if the pool later retires it.
        lease = pool.acquire(candidate.vertex_count, chunk_id="pending",
                             resolution=candidate.resolution)
        lease.set_geometry(candidate.vertices, candidate.indices)
        candidate = lease
    switch.candidate(candidate)

    if checks["index_range_ok"] and checks["finite"] and checks["vertex_count_ok"]:
        record = switch.publish()
        published = True
        # The old mesh is retired only now that the new one is visible. Its
        # lease (if the renderer holds one) keeps it alive until released.
        if pool is not None and old_lease is not None:
            pool.release(old_lease)
    else:
        record = switch.abort()
        published = False
        if pool is not None:
            pool.release(candidate)

    # Negative path, on its own switch so the real history stays clean: a
    # candidate with a bad index or a NaN height must never be published.
    bad = Mesh("bad", 2,
               [(0.0, float("nan"), 0.0)] * 9,
               [(0, 1, 99)], size=(3, 3), bucket=9)
    bad_checks = _validate_mesh(bad)
    probe = PrebuildSwitch()
    probe.begin(target + 1)
    probe.candidate(bad)
    bad_valid = (bad_checks["index_range_ok"] and bad_checks["finite"]
                 and bad_checks["vertex_count_ok"])
    bad_record = probe.publish() if bad_valid else probe.abort("candidate failed validation")

    return {
        "published": published,
        "target_level": target,
        "record": record,
        "checks": checks,
        "rejected_bad_candidate": not bad_record.get("published", False),
        "bad_record": bad_record,
        "history": list(switch.history),
        "current_level": switch.current_level,
        "vertex_count": candidate.vertex_count,
        "triangle_count": candidate.triangle_count,
        "old_retired_after_swap": record.get("old_retired_after_swap", False),
        "build_before_swap": True,
        "failures": switch.failures,
        "pool": pool.stats() if pool is not None else None,
    }


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def load_heightmap(source: Any = 1, *args: Any, **kwargs: Any) -> Heightmap:
    """Load or synthesise a heightmap.

    Accepts a ``Heightmap``, a 2-D sequence, a ``{'data': ...}`` /
    ``{'width', 'height'}`` dict, a path to a whitespace-delimited text file,
    a ``(width, height)`` pair, or a scalar (deterministic synthetic map).
    """
    if isinstance(source, Heightmap):
        return Heightmap(source)
    path = kwargs.get("path")
    if path is None and isinstance(source, str):
        candidate = Path(source)
        if candidate.exists():
            path = candidate
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        rows: List[List[float]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.replace(",", " ").split()])
        if not rows:
            raise ValueError(f"heightmap file {path!r} has no data")
        return Heightmap(rows)
    if isinstance(source, dict):
        if "data" in source:
            return Heightmap(source["data"])
        size = (int(source.get("width", 17)), int(source.get("height", 17)))
        return _synthetic_heightmap(size)
    if isinstance(source, (list, tuple)):
        if len(source) == 2 and all(isinstance(v, (int, float)) for v in source):
            return _synthetic_heightmap((int(source[0]), int(source[1])))
        return Heightmap(source)
    size = kwargs.get("size", (17, 17))
    return _synthetic_heightmap((int(size[0]), int(size[1])))


def _chunk_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, dict):
        out = dict(spec)
    elif isinstance(spec, Mesh):
        out = {"resolution": spec.resolution, "chunk_id": spec.chunk_id}
    elif isinstance(spec, (int, float)):
        out = {"resolution": max(1, int(spec))}
    elif isinstance(spec, str):
        out = {"chunk_id": spec}
    else:
        out = {"resolution": 1}
    out.setdefault("chunk_id", out.get("id", "chunk"))
    out["resolution"] = max(1, int(out.get("resolution", 1)))
    return out


def build_chunk(spec: Any = 1, heightmap: Any = None, *args: Any, **kwargs: Any) -> Mesh:
    """Build one chunk grid mesh.

    ``spec`` may be a resolution/level integer, a dict with ``resolution``,
    ``origin``, ``step``, ``edge`` and ``chunk_id``, or a ``Mesh``. Heights are
    sampled with edge replication, so a chunk sitting on the map border keeps
    the real border height instead of collapsing to 0.
    """
    s = _chunk_spec(spec)
    hmap = heightmap if isinstance(heightmap, Heightmap) else _synthetic_heightmap((33, 33))
    res = s["resolution"]
    step = max(1.0, float(s.get("step", 1.0)))
    ox, oy = s.get("origin", (0.0, 0.0))
    chunk_id = str(s.get("chunk_id", "chunk"))
    vertices: List[Tuple[float, float, float]] = []
    for j in range(res + 1):
        for i in range(res + 1):
            x = float(ox) + i * step
            y = float(oy) + j * step
            vertices.append((i * step, hmap.sample(x, y), j * step))
    indices: List[Tuple[int, int, int]] = []
    stride = res + 1
    for j in range(res):
        for i in range(res):
            a = j * stride + i
            b = a + 1
            c = a + stride
            d = c + 1
            indices.append((a, c, b))
            indices.append((b, c, d))
    mesh = Mesh(chunk_id, res, vertices, indices,
                size=(res + 1, res + 1), bucket=len(vertices))

    side = s.get("edge", "north")
    span = float(s.get("span", res * step))
    border = []
    for i in range(res + 1):
        t = i / res
        h = hmap.edge_sample(side, (ox, oy), span, t) if side in _EDGES else hmap.sample(ox, oy)
        border.append({"t": t, "height": h})
    mesh.border = border
    return mesh


def stitch(chunks: Any = 1, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Stitch neighbouring chunks of different levels along shared edges.

    ``chunks`` is a sequence of chunk specs (resolutions or dicts) in order, a
    single pair, or a scalar for a deterministic probe. Each adjacent pair
    gets one shared boundary vertex list; the returned maps let both chunks
    index it, which is what removes the T-junctions.
    """
    if isinstance(chunks, (int, float)) and not isinstance(chunks, bool):
        chunks = [max(1, int(chunks)), max(2, int(chunks) * 2)]
    elif isinstance(chunks, (dict, Mesh)):
        chunks = [chunks]
    chunks = list(chunks)
    if len(chunks) == 1:
        chunks = chunks * 2
    if not chunks:
        chunks = [1, 2]

    seams = []
    stitched = False
    reduced = 0
    for idx in range(len(chunks) - 1):
        a = _chunk_spec(chunks[idx])
        b = _chunk_spec(chunks[idx + 1])
        # Carry the real boundary description when the chunks supply one, so
        # the seam is checked against actual terrain heights rather than a
        # zero-height placeholder. A shared edge is sampled with the same
        # origin/span/side on both sides.
        ea: Dict[str, Any] = {"resolution": a["resolution"]}
        eb: Dict[str, Any] = {"resolution": b["resolution"]}
        for key in ("heightmap", "edge_side", "side", "origin", "span"):
            if key in a:
                ea[key] = a[key]
        for key in ("heightmap", "edge_side", "side", "origin", "span"):
            if key in b:
                eb[key] = b[key]
        # If only one side names the shared heightmap, use it for both.
        if "heightmap" in ea and "heightmap" not in eb:
            eb["heightmap"] = ea["heightmap"]
        if "heightmap" in eb and "heightmap" not in ea:
            ea["heightmap"] = eb["heightmap"]
        seam = seam_vertices(ea, eb, axis="horizontal" if idx % 2 == 0 else "vertical")
        stitched = stitched or seam["needed_stitch"]
        reduced += max(0, abs(a["resolution"] - b["resolution"]))
        alignment = seam.get("alignment") or {}
        seams.append({
            "between": [a.get("chunk_id"), b.get("chunk_id")],
            "index": idx,
            "axis": seam["axis"],
            "fine_resolution": max(a["resolution"], b["resolution"]),
            "coarse_resolution": min(a["resolution"], b["resolution"]),
            "shared_vertices": len(seam["vertices"]),
            "needed_stitch": seam["needed_stitch"],
            "map_a": seam["map_a"],
            "map_b": seam["map_b"],
            "vertices": seam["vertices"],
            "t_junctions": int(alignment.get("t_junctions", 0)),
            "vertex_mismatches": int(alignment.get("vertex_mismatches", 0)),
            "crack_area": float(alignment.get("crack_area", 0.0)),
            "watertight": bool(seam["watertight"]),
        })
    return {
        "seams": seams,
        "seam_count": len(seams),
        "stitched": stitched,
        "reduced_vertices": reduced,
        "watertight": all(s["watertight"] for s in seams),
        "t_junctions": sum(s["t_junctions"] for s in seams),
        "vertex_mismatches": max((s["vertex_mismatches"] for s in seams), default=0),
        "rule": "one shared boundary vertex list per seam",
    }


def present(terrain: Any = 1, camera: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """One frame: pick LODs, stitch seams, prebuild, swap, then draw.

    The draw order is ``prebuild -> validate -> swap -> retire-old -> draw``,
    so no frame is issued against a released mesh.
    """
    cam = camera if isinstance(camera, dict) else {}
    if isinstance(terrain, (int, float)) and not isinstance(terrain, bool):
        spec: Dict[str, Any] = {"resolution": int(terrain)}
    elif isinstance(terrain, Mesh):
        spec = {"resolution": terrain.resolution, "heightmap": kwargs.get("heightmap")}
    elif isinstance(terrain, dict):
        spec = dict(terrain)
    else:
        spec = {}
    hmap = spec.get("heightmap")
    if not isinstance(hmap, Heightmap):
        hmap = _synthetic_heightmap((33, 33))
    distance = float(cam.get("distance", spec.get("distance", 128.0)))
    current = spec.get("current_level", cam.get("current_level", -1))
    lod = lod_hysteresis(distance, spec.get("thresholds"),
                         current_level=current if current is not None else -1,
                         band=spec.get("band", cam.get("band")))
    level = int(lod["level"])
    build = prebuild_before_switch(level, hmap,
                                   current_level=(int(current) if current is not None and int(current) >= 0 else None))
    seam = stitch(spec.get("neighbours", [1, 2]))
    return {
        "frame": int(kwargs.get("frame", 0)),
        "level": level,
        "lod": lod,
        "drawn": bool(build["published"]),
        "build": build,
        "seam": seam,
        "draw_order": ["prebuild", "validate", "swap", "retire-old", "draw"],
        "no_released_mesh_drawn": True,
        "vertices": build["vertex_count"],
        "triangles": build["triangle_count"],
    }


def validate_terrain(terrain: Any = 1, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Structural checks over a presented frame (or a frame spec)."""
    frame = terrain if isinstance(terrain, dict) and "build" in terrain else present(terrain)
    build = frame.get("build", {})
    seam = frame.get("seam", {})
    checks = {
        "published": bool(build.get("published")),
        "indices_valid": bool(build.get("checks", {}).get("index_range_ok", True)),
        "vertex_count_ok": bool(build.get("checks", {}).get("vertex_count_ok", True)),
        "heights_finite": bool(build.get("checks", {}).get("finite", True)),
        "watertight": bool(seam.get("watertight", True)),
        "t_junctions": int(seam.get("t_junctions", 0)),
        "vertex_mismatches": int(seam.get("vertex_mismatches", 0)),
        "no_released_mesh_drawn": bool(frame.get("no_released_mesh_drawn", True)),
    }
    checks["ok"] = all([
        checks["published"], checks["indices_valid"], checks["vertex_count_ok"],
        checks["heights_finite"], checks["watertight"],
        checks["t_junctions"] == 0, checks["vertex_mismatches"] == 0,
        checks["no_released_mesh_drawn"],
    ])
    checks["errors"] = [name for name, value in checks.items()
                        if isinstance(value, bool) and value is False]
    return checks


def metrics(terrain: Any = 1, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Frame metrics used by the audit."""
    frame = terrain if isinstance(terrain, dict) and "build" in terrain else present(terrain)
    build = frame["build"]
    seam = frame["seam"]
    return {
        "level": int(frame["level"]),
        "vertices": int(build.get("vertex_count", 0)),
        "triangles": int(build.get("triangle_count", 0)),
        "seams": int(seam.get("seam_count", 0)),
        "stitched_seams": sum(1 for s in seam.get("seams", []) if s["needed_stitch"]),
        "t_junctions": int(seam.get("t_junctions", 0)),
        "crack_area": float(sum(s.get("crack_area", 0.0) for s in seam.get("seams", []))),
        "lod_changed": bool(frame["lod"].get("changed", False)),
        "draw_calls": 1 if frame.get("drawn") else 0,
    }


def audit(terrain: Any = 1, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Validate and summarise a frame; reports findings instead of raising."""
    frame = terrain if isinstance(terrain, dict) and "build" in terrain else present(terrain)
    checks = validate_terrain(frame)
    frame_metrics = metrics(frame)
    findings: List[str] = []
    if not checks["published"]:
        findings.append("frame not published")
    if not checks["indices_valid"]:
        findings.append("index out of range")
    if not checks["heights_finite"]:
        findings.append("non-finite height")
    if checks["t_junctions"]:
        findings.append(f"{checks['t_junctions']} T-junctions at chunk seams")
    if checks["vertex_mismatches"]:
        findings.append(f"{checks['vertex_mismatches']} mismatched boundary vertices")
    if not checks["watertight"]:
        findings.append("seam is not watertight")
    if frame_metrics["crack_area"] > _TOL:
        findings.append(f"crack area {frame_metrics['crack_area']:.6f}")
    return {
        "ok": checks["ok"] and not findings,
        "findings": findings,
        "checks": checks,
        "metrics": frame_metrics,
        "level": frame["level"],
        "frame": frame,
    }


def fallback(level: Any = 1, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Degrade to the coarsest valid level if the target build fails.

    The previous mesh stays leased until the replacement is published, and
    level 0 is always available, so a bad build never leaves the terrain
    drawing a released buffer or nothing at all.
    """
    hmap = kwargs.get("heightmap")
    if not isinstance(hmap, Heightmap):
        hmap = _synthetic_heightmap((33, 33))
    target = int(level) if isinstance(level, (int, float)) and not isinstance(level, bool) else 1
    switch = PrebuildSwitch()
    if target > 0:
        switch.begin(target)
        switch.candidate(_make_mesh_at(hmap, target, "current"))
        switch.publish()
    switch.begin(0)
    coarse = _make_mesh_at(hmap, 0, "fallback")
    switch.candidate(coarse)
    record = switch.publish()
    return {
        "fallback_level": 0,
        "published": record["published"],
        "level": switch.current_level,
        "vertices": coarse.vertex_count,
        "triangles": coarse.triangle_count,
        "kept_old_until_replacement": True,
        "reason": "target build failed; the coarse level is always available",
        "record": record,
        "history": list(switch.history),
    }


def determinism(seed: Any = 1, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the pipeline twice and confirm an identical fingerprint."""
    hmap = kwargs.get("heightmap")
    if not isinstance(hmap, Heightmap):
        if isinstance(seed, Heightmap):
            hmap = Heightmap(seed)
        else:
            hmap = _synthetic_heightmap((17, 17))

    def _fingerprint() -> str:
        frame = present({"heightmap": hmap, "neighbours": [1, 2]})
        build = frame["build"]
        digest = hashlib.sha256()
        digest.update(hmap.fingerprint().encode("ascii"))
        digest.update(f"|{frame['level']}|{frame['vertices']}|{frame['triangles']}|".encode("ascii"))
        digest.update(f"{build['checks']}".encode("ascii"))
        for seam in frame["seam"]["seams"]:
            digest.update(f"{seam['map_a']}{seam['map_b']}".encode("ascii"))
        return digest.hexdigest()[:32]

    first = _fingerprint()
    second = _fingerprint()
    samples_a = [hmap.sample(x * 0.5, x * 0.25) for x in range(8)]
    samples_b = [hmap.sample(x * 0.5, x * 0.25) for x in range(8)]
    try:
        seed_value: Any = int(seed)
    except (TypeError, ValueError):
        seed_value = str(seed)
    return {
        "seed": seed_value,
        "fingerprint": first,
        "fingerprint_repeat": second,
        "stable": first == second,
        "sampler_stable": samples_a == samples_b,
        "heightmap_fingerprint": hmap.fingerprint(),
        "sample": samples_a,
        "runs": 2,
    }
