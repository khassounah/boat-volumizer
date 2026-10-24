#!/usr/bin/env python3
"""
step_envelope_volume.py - envelope ("outside") volume of a STEP model by slicing.

Per slice at height z:
  1. Section the model with the plane z = const and discretize the curves.
  2. Dilate the curves by a small tolerance (eps) so near-touching pieces join
     and open surface curves become thin bands.
  3. Fill every closed outer loop (all holes are discarded -> outermost line only).
  4. Close openings: for wall-like pieces (thin walls, open surfaces), fill the
     pockets of the piece's convex hull, i.e. bridge each gap with a straight chord.
     Optionally only pockets whose mouth (opening width) <= --max-gap. An opening
     closed in one slice stays closed in the neighbouring slices it overlaps, so the
     stubby top of a thick rim is still treated as part of the wall.
  5. Erode by eps to undo step 2 and take the area A(z).

Volume = integral of A(z) dz, using Gauss-Legendre quadrature on intervals split
at geometric breakpoints (vertex heights and edge z-extrema), so kinks in A(z)
(e.g. where a sheer line starts, horizontal faces) fall on interval boundaries
and no slice is ever taken exactly on a horizontal face.

Lengths: the OCCT STEP reader converts to millimetres, so volumes are mm^3.
"""
import argparse
import math
import sys

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_TangentialDeflection
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Trsf, gp_Ax1
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

# TopoDS down-casts are TopoDS.Edge_s in OCP <= 7.x and TopoDS.Edge in OCP >= 8
_as_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
_as_vertex = getattr(TopoDS, "Vertex_s", None) or TopoDS.Vertex


# ---------------------------------------------------------------- geometry IO

def load_step(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        sys.exit(f"error: cannot read STEP file '{path}'")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        sys.exit("error: STEP file contains no transferable geometry")
    return shape


def orient_to_z(shape, axis):
    """Rotate the model so the requested slicing axis becomes +Z."""
    if axis == "z":
        return shape
    t = gp_Trsf()
    o = gp_Pnt(0, 0, 0)
    if axis == "x":    # rotate about Y by -90 deg: z' = x
        t.SetRotation(gp_Ax1(o, gp_Dir(0, 1, 0)), -math.pi / 2)
    else:              # rotate about X by +90 deg: z' = y
        t.SetRotation(gp_Ax1(o, gp_Dir(1, 0, 0)), math.pi / 2)
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def bounding_box(shape):
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    lo, hi = box.CornerMin(), box.CornerMax()
    return lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()


def iter_edges(shape):
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        yield _as_edge(exp.Current())
        exp.Next()


def edge_points(edge, lin_defl, ang_defl=0.1):
    """Discretize an edge into an (N, 3) array of points."""
    try:
        curve = BRepAdaptor_Curve(edge)
        d = GCPnts_TangentialDeflection(curve, ang_defl, lin_defl)
        n = d.NbPoints()
    except Exception:
        return None
    if n < 2:
        return None
    pts = np.empty((n, 3))
    for i in range(n):
        p = d.Value(i + 1)
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


def material_volume(shape):
    """Exact volume of any solids in the model (None if it has no solids)."""
    total, found = 0.0, False
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(exp.Current(), props)
        total += abs(props.Mass())
        found = True
        exp.Next()
    return total if found else None


# ---------------------------------------------------------------- slicing

def section_lines(shape, z, lin_defl):
    """2D polylines of the cross-section of `shape` at height z."""
    sec = BRepAlgoAPI_Section(shape, gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), False)
    sec.Build()
    if not sec.IsDone():
        return None
    lines = []
    for e in iter_edges(sec.Shape()):
        pts = edge_points(e, lin_defl)
        if pts is not None:
            xy = pts[:, :2]
            if np.ptp(xy, axis=0).max() > 0:
                lines.append(LineString(xy))
    return lines


def polygons(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]


def fill_holes(geom):
    return unary_union([Polygon(p.exterior) for p in polygons(geom)])


def analyze_slice(lines, eps, wall_ratio, max_gap, close=True):
    """
    Split one slice into filled outer loops ("pieces") and candidate openings
    ("pockets" = convex-hull pockets of each piece, i.e. regions a straight chord
    across a gap would enclose). Each pocket carries a seed flag: True when its
    piece looks like a wall (thin relative to its size).
    All geometry is in eps-dilated space; finalize_region() erodes it back.
    """
    if not lines:
        return [], []
    band = MultiLineString(lines).buffer(eps, quad_segs=4)
    pieces = polygons(fill_holes(band))
    pockets = []
    if close:
        for p in pieces:
            hull = p.convex_hull
            t_eq = 2.0 * p.area / p.length                 # equivalent wall thickness
            seed = t_eq <= wall_ratio * math.sqrt(hull.area)
            rim = hull.exterior.buffer(0.5 * eps)
            for k in polygons(hull.difference(p)):
                if k.area < 4 * eps * eps:
                    continue
                if not math.isinf(max_gap) and k.exterior.intersection(rim).length > max_gap + 2 * eps:
                    continue
                pockets.append([k, seed])
    return pieces, pockets


def propagate_openings(slices, overlap=0.5, max_iter=6):
    """
    Openings persist between neighbouring slices: a pocket is filled if it is
    seeded, or if at least `overlap` of its area lies inside a filled pocket of
    the slice directly below or above. Fixes the top of thick rims, where the
    last stub of wall is too stubby to look like a wall on its own.
    """
    def filled_union(pockets):
        return unary_union([k for k, f in pockets if f])

    for _ in range(max_iter):
        changed = False
        for order in (range(1, len(slices)), range(len(slices) - 2, -1, -1)):
            for i in order:
                j = i - 1 if order.step == 1 else i + 1
                ref = filled_union(slices[j]["pockets"])
                if ref.is_empty:
                    continue
                for pk in slices[i]["pockets"]:
                    if not pk[1] and pk[0].intersection(ref).area >= overlap * pk[0].area:
                        pk[1] = True
                        changed = True
        if not changed:
            break


def finalize_region(pieces, pockets, eps):
    if not pieces:
        return Polygon()
    region = fill_holes(unary_union(pieces + [k for k, f in pockets if f]))
    return region.buffer(-eps, quad_segs=4)


# ---------------------------------------------------------------- integration

def breakpoints(shape, zmin, zmax, lin_defl):
    """Heights where A(z) may have kinks: vertices and edge z-extrema."""
    zs = [zmin, zmax]
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        zs.append(BRep_Tool.Pnt_s(_as_vertex(exp.Current())).Z())
        exp.Next()
    for e in iter_edges(shape):
        pts = edge_points(e, lin_defl)
        if pts is not None:
            zs += [pts[:, 2].min(), pts[:, 2].max()]
    return np.clip(np.array(zs), zmin, zmax)


def build_intervals(bps, zmin, zmax, min_intervals, max_intervals):
    span = zmax - zmin
    min_dz = span / max_intervals
    pts = [zmin]
    for z in np.sort(bps):
        if z - pts[-1] >= min_dz:
            pts.append(z)
    if zmax - pts[-1] < min_dz:
        pts[-1] = zmax
    else:
        pts.append(zmax)
    max_dz = span / min_intervals
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, math.ceil((b - a) / max_dz - 1e-9))
        edges = np.linspace(a, b, n + 1)
        out += list(zip(edges[:-1], edges[1:]))
    return out


# ---------------------------------------------------------------- main

def run(args):
    shape = orient_to_z(load_step(args.step), args.axis)
    xmin, ymin, zmin, xmax, ymax, zmax = bounding_box(shape)
    diag = math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))
    if diag <= 0:
        sys.exit("error: empty geometry")

    lin_defl = args.deflection if args.deflection else 2e-5 * diag
    eps = args.gap_tol if args.gap_tol else 1e-4 * diag
    max_gap = args.max_gap if args.max_gap is not None else math.inf

    intervals = build_intervals(breakpoints(shape, zmin, zmax, lin_defl),
                                zmin, zmax, args.min_intervals, args.max_intervals)
    nodes, weights = np.polynomial.legendre.leggauss(args.gauss)
    n_slices = len(intervals) * args.gauss
    print(f"bbox {xmax - xmin:.3f} x {ymax - ymin:.3f} x {zmax - zmin:.3f} mm "
          f"(slicing along {args.axis}); {n_slices} slices", file=sys.stderr)

    slices, failed = [], 0
    for a, b in intervals:
        half = 0.5 * (b - a)
        for x, w in zip(nodes, weights):
            z = half * x + 0.5 * (a + b)
            lines = section_lines(shape, z, lin_defl)
            if lines is None:
                failed += 1
                lines = []
            pieces, pockets = analyze_slice(lines, eps, args.wall_ratio, max_gap, not args.no_close)
            slices.append(dict(z=z, w=half * w, lines=lines, pieces=pieces, pockets=pockets))
            if not args.quiet and len(slices) % max(1, n_slices // 10) == 0:
                print(f"  {100 * len(slices) // n_slices:3d}%", file=sys.stderr)

    propagate_openings(slices)
    volume, samples = 0.0, []
    for s in slices:
        region = finalize_region(s["pieces"], s["pockets"], eps)
        volume += s["w"] * region.area
        samples.append((s["z"], region.area, s["lines"], region))

    mat = material_volume(shape)
    print(f"envelope volume : {volume:,.3f} mm^3  ({volume * 1e-6:,.6f} L)")
    if mat is not None:
        print(f"material volume : {mat:,.3f} mm^3  ({mat * 1e-6:,.6f} L)  [exact, solids only]")
    if failed:
        print(f"warning: {failed} section(s) failed and counted as empty", file=sys.stderr)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("z_mm,area_mm2\n")
            for z, area, _, _ in samples:
                f.write(f"{z:.6f},{area:.6f}\n")
    if args.plot:
        plot_slices(samples, args.plot, args.plot_count)
    return volume


def plot_slices(samples, path, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.unique(np.linspace(0, len(samples) - 1, count).round().astype(int))
    cols = min(4, len(idx))
    rows = math.ceil(len(idx) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, i in zip(axes.flat, idx):
        z, area, lines, region = samples[i]
        for p in polygons(region):
            ax.fill(*p.exterior.xy, color="#4c8bf5", alpha=0.35, lw=0)
        for ln in lines:
            ax.plot(*ln.xy, color="k", lw=0.8)
        ax.set_title(f"z={z:.2f}  A={area:,.1f}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_axis_on()
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"slice plot written to {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Envelope volume of a STEP model by slicing "
                                             "(outermost loop per slice, openings closed by chords).")
    ap.add_argument("step", help="input .step/.stp file")
    ap.add_argument("--axis", choices="xyz", default="z", help="slicing axis (default z)")
    ap.add_argument("--max-gap", type=float, default=None,
                    help="only close openings up to this width in mm (default: close all)")
    ap.add_argument("--wall-ratio", type=float, default=0.15,
                    help="pieces with thickness/size below this are treated as walls and "
                         "have their openings closed (default 0.15)")
    ap.add_argument("--no-close", action="store_true",
                    help="only fill closed outer loops, never bridge openings")
    ap.add_argument("--gap-tol", type=float, default=None,
                    help="tolerance for joining near-touching curves, mm (default 1e-4 x diagonal)")
    ap.add_argument("--deflection", type=float, default=None,
                    help="curve discretization tolerance, mm (default 2e-5 x diagonal)")
    ap.add_argument("--min-intervals", type=int, default=30,
                    help="minimum number of integration intervals (default 30)")
    ap.add_argument("--max-intervals", type=int, default=200,
                    help="cap on breakpoint-driven intervals (default 200)")
    ap.add_argument("--gauss", type=int, default=3, help="Gauss points per interval (default 3)")
    ap.add_argument("--csv", help="write the A(z) profile to this CSV")
    ap.add_argument("--plot", help="write a PNG grid of sample slices")
    ap.add_argument("--plot-count", type=int, default=8, help="slices in the plot (default 8)")
    ap.add_argument("-q", "--quiet", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
