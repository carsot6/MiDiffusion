#!/usr/bin/env python3
# ruff: noqa: E501
"""
Read-only validator for GPC -> MiDiffusion `boxes.npz` files.

Checks a GPC-generated boxes.npz (or a directory of them) against the on-disk
contract that the MiDiffusion / ThreedFront cached loader actually expects, and
diffs value ranges against a known-good reference boxes.npz.

Nothing is mutated. The script only reads, validates, and prints a report.

Contract is derived from (not guessed):
  - ThreedFront/threed_front/datasets/threed_front.py :: _parse_room_params
      reads exactly: class_labels, translations, sizes, angles, room_layout,
      and the fpbpn key `floor_plan_boundary_points_normals` (NOT "fpbpn").
  - ThreedFront/scripts/preprocess_data.py
      translations = centroid in METERS (centered on scene centroid),
      sizes        = HALF-extents in METERS,
      angles       = radians, shape (N, 1).
  - midiffusion/datasets/threed_front_encoding.py + threed_front_encoding_base.py
      Scale_CosinAngle normalizes translations/sizes to [-1, 1] AT LOAD TIME using
      the GLOBAL bounds in dataset_stats.txt, and converts angles -> (cos, sin).
      => on disk, translations/sizes must be raw meters, NOT per-room [0, 1].
  - preprocess_floorplan.py : room_layout is uint8 {0, 255}; fpbpn normals are
      unit-length and point INWARD toward the room center.

Usage:
    python scripts/validate_gpc_boxes.py PATH [--reference REF.npz] [--room-side R]
      PATH         a single .npz file or a directory containing *.npz
      --reference  reference boxes.npz to diff ranges against
                   (default: ThreedFront bedroom_TEST001/boxes.npz if found)
      --room-side  expected global room_side in meters (half-canvas). If given,
                   translations/fpbpn extents are checked against [-R, +R].
      --stats      optional dataset_stats.txt to validate class_labels width.
"""
import argparse
import os
import sys
import glob

import numpy as np

try:
    from shapely.geometry import Point, Polygon
    _HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _HAS_SHAPELY = False


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------
PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_SYM = {PASS: "\033[32mPASS\033[0m", FAIL: "\033[31mFAIL\033[0m",
        WARN: "\033[33mWARN\033[0m", INFO: "\033[36mINFO\033[0m"}

REQUIRED_KEYS = ["class_labels", "translations", "sizes", "angles",
                 "room_layout", "floor_plan_boundary_points_normals"]
FPBPN_KEY = "floor_plan_boundary_points_normals"


class Report:
    def __init__(self, name):
        self.name = name
        self.rows = []  # (status, check, detail)

    def add(self, status, check, detail=""):
        self.rows.append((status, check, detail))

    @property
    def failed(self):
        return any(s == FAIL for s, _, _ in self.rows)

    @property
    def n(self):
        c = {PASS: 0, FAIL: 0, WARN: 0, INFO: 0}
        for s, _, _ in self.rows:
            c[s] += 1
        return c

    def print(self, color=True):
        print(f"\n=== {self.name} ===")
        for s, check, detail in self.rows:
            sym = _SYM[s] if color else s
            line = f"  [{sym}] {check}"
            if detail:
                line += f"  ->  {detail}"
            print(line)
        c = self.n
        print(f"  ---- {c[PASS]} pass / {c[FAIL]} fail / {c[WARN]} warn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rng(a):
    a = np.asarray(a, dtype=np.float64)
    return float(a.min()), float(a.max())


def _looks_unit_box(a, lo=-1e-6, hi=1.0 + 1e-6):
    """True if every value sits within [0, 1] (signature of per-room normalization)."""
    a = np.asarray(a, dtype=np.float64)
    return bool(a.min() >= lo and a.max() <= hi)


def _point_in_poly(poly_xy, pts_xy):
    """Vectorized ray-casting fallback if shapely is unavailable.
    poly_xy: (M, 2) ordered polygon. pts_xy: (K, 2). returns bool (K,)."""
    poly_xy = np.asarray(poly_xy, dtype=np.float64)
    pts = np.asarray(pts_xy, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    j = len(poly_xy) - 1
    for i in range(len(poly_xy)):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        inside ^= cond
        j = i
    return inside


def _poly_area(poly_xy):
    p = np.asarray(poly_xy, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _degenerate(poly_xy):
    """True if the boundary polygon has ~no area (placeholder/collinear points),
    in which case inside/outside tests are meaningless."""
    return _poly_area(poly_xy) < 1e-6


def _contains(poly_xy, pts_xy):
    if _HAS_SHAPELY:
        poly = Polygon(poly_xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return np.array([poly.contains(Point(p)) for p in pts_xy])
    return _point_in_poly(poly_xy, pts_xy)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_keys(rep, d):
    keys = set(d.keys())
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        rep.add(FAIL, "required keys present", f"missing: {missing}")
    else:
        rep.add(PASS, "required keys present", f"{len(REQUIRED_KEYS)} keys")
    if "fpbpn" in keys and FPBPN_KEY not in keys:
        rep.add(FAIL, "fpbpn key name",
                f"found short key 'fpbpn'; loader only reads '{FPBPN_KEY}'")
    extra = keys - set(REQUIRED_KEYS) - {"fpbpn"}
    if extra:
        rep.add(INFO, "extra keys (ignored by training loader)", f"{sorted(extra)}")


def check_class_labels(rep, d, n_classes_stats=None):
    if "class_labels" not in d:
        return
    cl = np.asarray(d["class_labels"])
    if cl.ndim != 2:
        rep.add(FAIL, "class_labels shape", f"expected (N, C), got {cl.shape}")
        return
    N, C = cl.shape
    rep.add(INFO, "class_labels shape", f"(N={N}, C={C})")
    if cl.dtype != np.float32:
        rep.add(WARN, "class_labels dtype", f"{cl.dtype} (reference is float32)")
    sums = cl.sum(axis=1)
    if np.allclose(sums, 1.0, atol=1e-4):
        rep.add(PASS, "class_labels one-hot (rows sum to 1)")
    else:
        rep.add(FAIL, "class_labels one-hot",
                f"row sums range {_rng(sums)} (expected 1.0)")
    if n_classes_stats is not None:
        if C == n_classes_stats:
            rep.add(PASS, "class_labels width == dataset_stats", f"C={C}")
        else:
            rep.add(FAIL, "class_labels width == dataset_stats",
                    f"npz C={C} vs stats n_classes={n_classes_stats}")
    # last two columns (start/end) must be zero for real objects (+2 scheme)
    if C >= 2:
        tail = cl[:, -2:]
        if np.any(tail > 0):
            rep.add(WARN, "start/end columns empty",
                    "last 2 cols nonzero (ok only if you don't use the +2 scheme)")
        else:
            rep.add(PASS, "start/end columns empty for real objects")


def check_translations(rep, d):
    if "translations" not in d:
        return
    t = np.asarray(d["translations"])
    if t.ndim != 2 or t.shape[1] != 3:
        rep.add(FAIL, "translations shape", f"expected (N, 3), got {t.shape}")
        return
    if t.dtype != np.float32:
        rep.add(WARN, "translations dtype", f"{t.dtype} (reference is float32)")
    rep.add(INFO, "translations range (m)", f"{_rng(t)}")
    # THE bug: per-room normalization to [0, 1] instead of centered meters.
    if _looks_unit_box(t):
        rep.add(FAIL, "translations are raw meters (centered)",
                "all values in [0,1] -> looks per-room normalized. Loader's "
                "Scale uses GLOBAL bounds; store centered meters instead.")
    else:
        rep.add(PASS, "translations are raw meters (not [0,1])")
    # centered-on-centroid sanity (mean should be near 0 relative to extent)
    span = max(t.max() - t.min(), 1e-6)
    if np.abs(t[:, [0, 2]].mean()) > 0.5 * span:
        rep.add(WARN, "translations centered on room centroid",
                f"xz mean {t[:, [0, 2]].mean():.3f} far from 0")


def check_sizes(rep, d, ref=None):
    if "sizes" not in d:
        return
    s = np.asarray(d["sizes"])
    if s.ndim != 2 or s.shape[1] != 3:
        rep.add(FAIL, "sizes shape", f"expected (N, 3), got {s.shape}")
        return
    if s.dtype != np.float32:
        rep.add(WARN, "sizes dtype", f"{s.dtype} (reference is float32)")
    rep.add(INFO, "sizes range (half-extents, m)", f"{_rng(s)}")
    if (s < 0).any():
        rep.add(FAIL, "sizes non-negative", f"min {s.min()}")
    else:
        rep.add(PASS, "sizes non-negative")
    if _looks_unit_box(s) and s.max() <= 1.0 + 1e-6:
        rep.add(WARN, "sizes are raw meters (half-extents)",
                "all values in [0,1]; check this isn't per-room normalization "
                "(reference half-extents go up to ~3m).")


def check_angles(rep, d):
    if "angles" not in d:
        return
    a = np.asarray(d["angles"])
    if a.ndim != 2:
        rep.add(FAIL, "angles shape", f"expected (N, 1), got {a.shape}")
        return
    N, W = a.shape
    if W == 1:
        rep.add(PASS, "angles shape (N, 1) radians on disk")
        lo, hi = _rng(a)
        if lo < -2 * np.pi - 1e-3 or hi > 2 * np.pi + 1e-3:
            rep.add(WARN, "angles in radian range", f"range {(lo, hi)}")
        else:
            rep.add(PASS, "angles within radian range", f"{(round(lo,3), round(hi,3))}")
    elif W == 2:
        rep.add(FAIL, "angles shape (N, 1) radians on disk",
                "got (N, 2). cos/sin is the LOAD-TIME form; store radians on disk.")
    else:
        rep.add(FAIL, "angles shape", f"got (N, {W})")


def check_room_layout(rep, d, expect_hw=None):
    if "room_layout" not in d:
        return
    r = np.asarray(d["room_layout"])
    rep.add(INFO, "room_layout shape/dtype", f"{r.shape} {r.dtype}")
    if r.dtype != np.uint8:
        rep.add(WARN, "room_layout dtype uint8", f"got {r.dtype}")
    uniq = np.unique(r)
    if set(uniq.tolist()).issubset({0, 255}):
        rep.add(PASS, "room_layout strictly {0,255}")
    else:
        rep.add(FAIL, "room_layout strictly {0,255}",
                f"found {uniq[:8].tolist()}... (anti-aliasing? must be binary)")
    # loader does room_layout[:, :, 0] -> needs a channel dim
    if r.ndim == 3 and r.shape[2] == 1:
        rep.add(PASS, "room_layout has channel dim (H,W,1)")
        hw = r.shape[:2]
    elif r.ndim == 2:
        rep.add(WARN, "room_layout has channel dim (H,W,1)",
                "got (H,W); loader indexes [:,:,0] and will error without channel")
        hw = r.shape
    else:
        rep.add(FAIL, "room_layout has channel dim (H,W,1)", f"got {r.shape}")
        hw = r.shape[:2]
    if expect_hw is not None and tuple(hw) != tuple(expect_hw):
        rep.add(WARN, "room_layout size == config",
                f"{hw} vs expected {tuple(expect_hw)} (resized at load -> greys)")


def check_fpbpn(rep, d, room_side=None):
    if FPBPN_KEY not in d:
        return
    f = np.asarray(d[FPBPN_KEY], dtype=np.float64)
    rep.add(INFO, "fpbpn shape", f"{f.shape}")
    if f.ndim != 2 or f.shape[1] != 4:
        rep.add(FAIL, "fpbpn shape (K, 4)", f"got {f.shape}")
        return
    if f.shape[0] != 256:
        rep.add(WARN, "fpbpn has 256 points", f"got {f.shape[0]}")
    pts, nrm = f[:, :2], f[:, 2:]
    # unit-length normals
    nl = np.linalg.norm(nrm, axis=1)
    if np.allclose(nl, 1.0, atol=1e-2):
        rep.add(PASS, "fpbpn normals unit-length")
    else:
        rep.add(FAIL, "fpbpn normals unit-length", f"|n| range {_rng(nl)}")
    # inward test: step along +normal must land inside the boundary polygon
    if _degenerate(pts):
        rep.add(WARN, "fpbpn normals point INWARD",
                "boundary polygon is degenerate (area~0); cannot test "
                "(placeholder fpbpn?)")
    else:
        span = np.linalg.norm(pts.max(0) - pts.min(0))
        eps = max(span * 0.01, 1e-3)
        inside_plus = _contains(pts, pts + eps * nrm)
        inside_minus = _contains(pts, pts - eps * nrm)
        fp = inside_plus.mean()
        if fp >= 0.95 and inside_minus.mean() <= 0.05:
            rep.add(PASS, "fpbpn normals point INWARD", f"{fp*100:.0f}% +step inside")
        elif inside_minus.mean() >= 0.95:
            rep.add(FAIL, "fpbpn normals point INWARD",
                    "normals point OUTWARD (-step lands inside)")
        else:
            rep.add(WARN, "fpbpn normals point INWARD",
                    f"ambiguous (+inside {fp*100:.0f}%, -inside {inside_minus.mean()*100:.0f}%)")
    rep.add(INFO, "fpbpn xz range (m)", f"{_rng(pts)}")
    if room_side is not None:
        m = float(np.abs(pts).max())
        if m <= room_side * (1 + 1e-3):
            rep.add(PASS, "fpbpn within [-room_side, room_side]", f"max|xz|={m:.3f} <= {room_side}")
        else:
            rep.add(FAIL, "fpbpn within [-room_side, room_side]",
                    f"max|xz|={m:.3f} > room_side={room_side} (will be clipped)")


def check_cross_consistency(rep, d):
    """translations xz must fall inside the fpbpn boundary polygon."""
    if "translations" not in d or FPBPN_KEY not in d:
        return
    t = np.asarray(d["translations"], dtype=np.float64)
    f = np.asarray(d[FPBPN_KEY], dtype=np.float64)
    if t.shape[1] != 3 or f.shape[1] != 4:
        return
    poly = f[:, :2]
    txz = t[:, [0, 2]]
    if _degenerate(poly):
        rep.add(WARN, "object centers inside floor boundary",
                "boundary polygon degenerate (area~0); cannot test")
        return
    inside = _contains(poly, txz)
    frac = inside.mean()
    if frac >= 0.99:
        rep.add(PASS, "object centers inside floor boundary", f"{frac*100:.0f}%")
    elif frac >= 0.8:
        rep.add(WARN, "object centers inside floor boundary",
                f"{frac*100:.0f}% (some out-of-bounds; check axis/orientation)")
    else:
        rep.add(FAIL, "object centers inside floor boundary",
                f"only {frac*100:.0f}% inside -> translations/fpbpn scale or axis mismatch")


def diff_against_reference(rep, d, ref):
    if ref is None:
        return
    for key, label in [("translations", "transl"), ("sizes", "sizes")]:
        if key in d and key in ref:
            rep.add(INFO, f"range diff {label} (npz vs ref)",
                    f"{tuple(round(x,2) for x in _rng(d[key]))} vs "
                    f"{tuple(round(x,2) for x in _rng(ref[key]))}")
    if "class_labels" in d and "class_labels" in ref:
        rep.add(INFO, "class width (npz vs ref)",
                f"{d['class_labels'].shape[1]} vs {ref['class_labels'].shape[1]}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def validate_one(path, ref, room_side, expect_hw, n_classes_stats):
    rep = Report(os.path.relpath(path))
    try:
        d = dict(np.load(path, allow_pickle=True))
    except Exception as e:
        rep.add(FAIL, "loadable", str(e))
        return rep
    check_keys(rep, d)
    check_class_labels(rep, d, n_classes_stats)
    check_translations(rep, d)
    check_sizes(rep, d, ref)
    check_angles(rep, d)
    check_room_layout(rep, d, expect_hw)
    check_fpbpn(rep, d, room_side)
    check_cross_consistency(rep, d)
    diff_against_reference(rep, d, ref)
    return rep


def default_reference():
    here = os.path.dirname(os.path.realpath(__file__))
    cand = os.path.join(
        here, "..", "..", "ThreedFront", "output", "3d_front_processed",
        "bedroom", "bedroom_TEST001", "boxes.npz")
    return cand if os.path.isfile(cand) else None


def load_n_classes(stats_path):
    if not stats_path or not os.path.isfile(stats_path):
        return None
    import json
    with open(stats_path) as f:
        return len(json.load(f)["class_labels"])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="boxes.npz file or directory of *.npz")
    ap.add_argument("--reference", default=None, help="reference boxes.npz")
    ap.add_argument("--room-side", type=float, default=None,
                    help="expected global room_side (m); checks coord extent")
    ap.add_argument("--stats", default=None, help="dataset_stats.txt to check class width")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args(argv)

    ref_path = args.reference or default_reference()
    ref = None
    if ref_path and os.path.isfile(ref_path):
        ref = dict(np.load(ref_path, allow_pickle=True))
        print(f"Reference: {ref_path}")
    else:
        print("Reference: (none)")
    if not _HAS_SHAPELY:
        print("NOTE: shapely not found; using numpy ray-casting fallback.")

    n_classes_stats = load_n_classes(args.stats)
    expect_hw = None  # could be wired from a config later

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "**", "*.npz"), recursive=True))
        files = [f for f in files if ref_path is None or os.path.realpath(f) != os.path.realpath(ref_path)]
    else:
        files = [args.path]
    if not files:
        print("No .npz files found.")
        return 2

    any_fail = False
    agg = {PASS: 0, FAIL: 0, WARN: 0}
    for fp in files:
        rep = validate_one(fp, ref, args.room_side, expect_hw, n_classes_stats)
        rep.print(color=not args.no_color)
        any_fail |= rep.failed
        c = rep.n
        for k in agg:
            agg[k] += c[k]

    print(f"\n==== TOTAL over {len(files)} file(s): "
          f"{agg[PASS]} pass / {agg[FAIL]} fail / {agg[WARN]} warn ====")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
