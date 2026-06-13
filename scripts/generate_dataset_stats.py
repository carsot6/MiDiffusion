#!/usr/bin/env python3
# ruff: noqa: E501
"""
Generate a combined-model `dataset_stats.txt` for GPC -> MiDiffusion training.

The cached loader (`CachedThreedFront._parse_train_stats`,
ThreedFront/threed_front/datasets/threed_front.py:407-466) requires this file in
the dataset directory and reads, at minimum:

    bounds_translations  [min_xyz, max_xyz]   GLOBAL over train(+val), meters (centered)
    bounds_sizes         [min_xyz, max_xyz]   GLOBAL, meters (HALF-extents)
    bounds_angles        [min, max]           radians
    bounds_fpbpn         [min4, max4]         optional (xy bounds + [-1,1] normals)
    class_labels         object_types + ["start","end"]   (the +2 scheme)
    object_types         ordered furniture vocabulary
    class_frequencies    {name: count/total}
    class_order          {name: rank}
    count_furniture      {name: count}

Semantics mirror ThreedFront's own stats computation
(threed_front.py:_compute_bounds / count_furniture / class_frequencies /
class_order) so a from-scratch GPC model trains the same way the reference does.

This script is the single source of truth for the class-column ORDER. Martin's
one-hot `class_labels` columns MUST be built in the same `object_types` order
this script prints (column i  <->  object_types[i]).

Layout expected (same as the loader):
    <data_dir>/<tag>/boxes.npz       where tag.split("_")[1] == scene_id

Usage:
    python scripts/generate_dataset_stats.py <data_dir> \
        --frozen-vocab config/gpc_categories.json \
        --splits gpc_splits.csv \
        [--scheme 2] [--out <data_dir>/dataset_stats.txt] [--dry-run]
"""
import argparse
import json
import os
import sys
import glob
from collections import Counter, OrderedDict

import numpy as np

FPBPN_KEY = "floor_plan_boundary_points_normals"


# ---------------------------------------------------------------------------
def load_categories(path):
    """Return an ordered list of furniture category names (no start/end).
    Accepts either a JSON list, or a {name: description} dict."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        names = list(obj.keys())
    elif isinstance(obj, list):
        names = list(obj)
    else:
        raise ValueError(f"Unsupported categories JSON type: {type(obj)}")
    # drop any accidental sentinel/none entries
    names = [n for n in names if n not in ("start", "end", "__none__", "")]
    # deterministic order (matches reference which sorts object_types)
    return sorted(names)


def load_frozen_vocab(path):
    """Return the ordered category list VERBATIM (no sorting) from a frozen vocab
    file (config/gpc_categories.json). Order is authoritative: it defines the
    one-hot column indices and must never be re-sorted.

    Accepts either {"object_types": [...]} or a plain JSON list of names."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "object_types" in obj:
        names = list(obj["object_types"])
    elif isinstance(obj, list):
        names = list(obj)
    else:
        raise ValueError(
            "frozen vocab must be a list or have an 'object_types' key")
    bad = [n for n in names if n in ("start", "end", "__none__", "")]
    if bad:
        raise ValueError(f"frozen vocab must not contain sentinels/none: {bad}")
    if len(set(names)) != len(names):
        raise ValueError("frozen vocab contains duplicate categories")
    return names


def load_splits(path):
    """Parse a `scene_id,split` CSV -> {scene_id: split}."""
    import csv
    mapping = {}
    with open(path) as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            mapping[row[0].strip()] = row[1].strip()
    return mapping


def scene_id_from_tag(tag):
    parts = tag.split("_")
    return parts[1] if len(parts) > 1 else tag


def find_rooms(data_dir):
    rooms = []
    for tag in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, tag, "boxes.npz")
        if os.path.isfile(p):
            rooms.append((tag, p))
    # also allow flat *.npz files directly under data_dir
    if not rooms:
        for p in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            rooms.append((os.path.splitext(os.path.basename(p))[0], p))
    return rooms


# ---------------------------------------------------------------------------
def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="directory of <tag>/boxes.npz")
    ap.add_argument("--frozen-vocab", default=None,
                    help="config/gpc_categories.json: ordered category list used "
                         "VERBATIM as object_types (recommended; prevents column "
                         "drift). Takes precedence over --categories.")
    ap.add_argument("--categories", default=None,
                    help="target_categories_v2.json (dict or list); names are "
                         "SORTED to define column order. Use only without "
                         "--frozen-vocab.")
    ap.add_argument("--splits", default=None,
                    help="scene_id,split CSV. Bounds use train+val only. "
                         "If omitted, all rooms are used (with a warning).")
    ap.add_argument("--bounds-splits", default="train,val",
                    help="splits used to compute bounds (default: train,val)")
    ap.add_argument("--scheme", type=int, choices=[1, 2], default=2,
                    help="extra label columns: 2 = +[start,end] (reference), "
                         "1 = +[end] only (default: 2)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <data_dir>/dataset_stats.txt)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print but do not write the file")
    ap.add_argument("--force", action="store_true",
                    help="write even if some rooms fail width validation")
    args = ap.parse_args(argv)

    if args.frozen_vocab:
        object_types = load_frozen_vocab(args.frozen_vocab)
        print(f"Vocabulary: frozen ({args.frozen_vocab}), order preserved verbatim")
    elif args.categories:
        object_types = load_categories(args.categories)
        print(f"Vocabulary: {args.categories} (sorted to define column order)")
    else:
        ap.error("provide --frozen-vocab (recommended) or --categories")
    n_obj = len(object_types)
    extra = ["start", "end"] if args.scheme == 2 else ["end"]
    class_labels = object_types + extra
    n_classes = len(class_labels)
    expected_C = n_obj + args.scheme
    name_of = {i: object_types[i] for i in range(n_obj)}

    print(f"Categories: {n_obj} furniture types "
          f"(+{args.scheme} -> n_classes={n_classes}, expected one-hot width={expected_C})")

    splits = load_splits(args.splits) if args.splits else None
    if splits is None:
        print("WARNING: no --splits given; using ALL rooms for bounds.")
    bounds_splits = set(s.strip() for s in args.bounds_splits.split(","))

    rooms = find_rooms(args.data_dir)
    if not rooms:
        print(f"No boxes.npz found under {args.data_dir}")
        return 2
    print(f"Found {len(rooms)} room(s) under {args.data_dir}")

    tr_all, sz_all, ang_all, fpxy_all = [], [], [], []
    counts = Counter()
    n_bound_rooms = 0
    width_errors = []

    for tag, path in rooms:
        sid = scene_id_from_tag(tag)
        split = splits.get(sid) if splits else "train"
        if split is None:
            continue  # scene not listed in splits CSV -> skip entirely
        d = np.load(path, allow_pickle=True)
        cl = np.asarray(d["class_labels"])
        if cl.ndim != 2 or cl.shape[1] != expected_C:
            width_errors.append((tag, None if cl.ndim != 2 else cl.shape[1]))
            continue

        use_for_bounds = split in bounds_splits
        if use_for_bounds:
            n_bound_rooms += 1
            tr_all.append(np.asarray(d["translations"], np.float64))
            sz_all.append(np.asarray(d["sizes"], np.float64))
            ang_all.append(np.asarray(d["angles"], np.float64).reshape(-1))
            if FPBPN_KEY in d:
                fpxy_all.append(np.asarray(d[FPBPN_KEY], np.float64)[:, :2])
            # class counts only from bounds splits (train/val), like reference
            idx = cl[:, :n_obj].argmax(axis=1)
            real = cl[:, :n_obj].sum(axis=1) > 0  # ignore empty/start/end rows
            for i in idx[real]:
                counts[name_of[int(i)]] += 1

    if width_errors:
        print(f"\nWIDTH MISMATCH in {len(width_errors)} room(s): one-hot width != {expected_C}")
        for tag, c in width_errors[:10]:
            print(f"  {tag}: got C={c} (expected {expected_C} = {n_obj} categories + {args.scheme})")
        print("  -> Martin's one-hot must be (N, n_categories + start/end). "
              "Fix the npz builder or pass --scheme/--categories to match.")
        if not args.force:
            print("Aborting (use --force to write stats from the valid rooms only).")
            return 1

    if n_bound_rooms == 0:
        print("No rooms in bounds splits; cannot compute bounds.")
        return 1

    tr = np.vstack(tr_all)
    sz = np.vstack(sz_all)
    ang = np.concatenate(ang_all)
    tr_min, tr_max = tr.min(0), tr.max(0)
    sz_min, sz_max = sz.min(0), sz.max(0)

    # count_furniture sorted desc by count; class_order rank; class_frequencies
    count_furniture = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
    # ensure every category appears even if count 0 (stable for n_classes math)
    for name in object_types:
        count_furniture.setdefault(name, 0)
    total = sum(count_furniture.values()) or 1
    class_order = OrderedDict(
        (name, rank) for rank, name in enumerate(count_furniture.keys()))
    class_frequencies = OrderedDict(
        (name, count_furniture[name] / total) for name in count_furniture)

    stats = {
        "bounds_translations": tr_min.tolist() + tr_max.tolist(),
        "bounds_sizes": sz_min.tolist() + sz_max.tolist(),
        "bounds_angles": [float(ang.min()), float(ang.max())],
        "class_labels": class_labels,
        "object_types": object_types,
        "class_frequencies": class_frequencies,
        "class_order": class_order,
        "count_furniture": count_furniture,
    }
    if fpxy_all:
        fp = np.vstack(fpxy_all)
        fp_min = np.round(fp.min(0), 5).tolist()
        fp_max = np.round(fp.max(0), 5).tolist()
        stats["bounds_fpbpn"] = fp_min + [-1.0, -1.0] + fp_max + [1.0, 1.0]

    # ---- report ----
    print(f"\nBounds computed from {n_bound_rooms} room(s) "
          f"({sum(count_furniture.values())} objects):")
    print(f"  bounds_translations: {[round(x,3) for x in stats['bounds_translations']]}")
    print(f"  bounds_sizes:        {[round(x,3) for x in stats['bounds_sizes']]}")
    print(f"  bounds_angles:       {[round(x,3) for x in stats['bounds_angles']]}")
    if "bounds_fpbpn" in stats:
        print(f"  bounds_fpbpn:        {stats['bounds_fpbpn']}")
    top = list(count_furniture.items())[:8]
    print(f"  top categories:      {top}")
    zero = [n for n, c in count_furniture.items() if c == 0]
    if zero:
        print(f"  WARNING: {len(zero)} categories have 0 occurrences in bounds splits: {zero[:12]}")

    if _looks_normalized(tr) or _looks_normalized(sz):
        print("\n  !! translations/sizes look normalized to [0,1]. dataset_stats expects "
              "RAW METERS. Run scripts/validate_gpc_boxes.py first.")

    out = args.out or os.path.join(args.data_dir, "dataset_stats.txt")
    if args.dry_run:
        print(f"\n[dry-run] would write {out}")
    else:
        with open(out, "w") as f:
            json.dump(stats, f)
        print(f"\nWrote {out}")
    print(f"\nColumn order (use this for one-hot columns): "
          f"{n_obj} categories then {extra}")
    return 0


def _looks_normalized(a):
    a = np.asarray(a, np.float64)
    return bool(a.min() >= -1e-6 and a.max() <= 1.0 + 1e-6)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
