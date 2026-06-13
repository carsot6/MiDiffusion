#!/usr/bin/env python3
# ruff: noqa: E501
"""
Build a train/val/test splits CSV for a GPC -> MiDiffusion dataset, and report
the values you need to fill into config/gpc_mixed.yaml.

The cached loader matches each room subdir to the splits file via
`tag.split("_")[1]` (ThreedFront/threed_front/datasets/threed_front.py:268-269),
so subdirs must be named `<prefix>_<scene_id>` and the CSV rows are
`<scene_id>,<split>`.

What it does:
  - scans <data_dir>/<tag>/boxes.npz
  - assigns each scene to train/val/test by ratio (seeded, deterministic)
  - optionally drops rooms with too few / too many objects (min/max boxes)
  - writes `<scene_id>,<split>` CSV
  - prints the max objects-per-room  -> the minimum `sample_num_points`

Usage:
    python scripts/make_gpc_splits.py <data_dir> --out gpc_threed_front_splits.csv \
        [--ratios 0.8,0.1,0.1] [--seed 0] [--min-boxes 3] [--max-boxes 0]

By convention the CSV belongs in ../ThreedFront/dataset_files/ (PATH_TO_DATASET_FILES),
NOT inside <data_dir> (the loader would choke on a stray file there).
"""
import argparse
import os
import sys
import csv

import numpy as np


def scene_id_from_tag(tag):
    parts = tag.split("_")
    if len(parts) < 2:
        return None
    return parts[1]


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="directory of <tag>/boxes.npz")
    ap.add_argument("--out", required=True, help="output splits CSV path")
    ap.add_argument("--ratios", default="0.8,0.1,0.1",
                    help="train,val,test fractions (default 0.8,0.1,0.1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-boxes", type=int, default=3,
                    help="drop rooms with fewer objects (default 3; 0 = no min)")
    ap.add_argument("--max-boxes", type=int, default=0,
                    help="drop rooms with more objects (0 = no cap)")
    args = ap.parse_args(argv)

    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        print("--ratios must be 3 fractions summing to 1.0")
        return 2
    if os.path.realpath(os.path.dirname(os.path.abspath(args.out))) == \
            os.path.realpath(os.path.abspath(args.data_dir)):
        print("WARNING: writing the CSV inside the dataset dir will break the loader "
              "(tag.split('_')[1]). Put it in ../ThreedFront/dataset_files/ instead.")

    rooms = []  # (scene_id, n_boxes)
    dropped_min = dropped_max = dup = bad = 0
    seen = set()
    for tag in sorted(os.listdir(args.data_dir)):
        p = os.path.join(args.data_dir, tag, "boxes.npz")
        if not os.path.isfile(p):
            continue
        sid = scene_id_from_tag(tag)
        if sid is None:
            bad += 1
            print(f"  skip '{tag}': no '_' in name -> loader would crash on it")
            continue
        if sid in seen:
            dup += 1
            print(f"  WARNING: duplicate scene_id '{sid}' (tag '{tag}')")
            continue
        seen.add(sid)
        try:
            n = int(np.load(p, allow_pickle=True)["class_labels"].shape[0])
        except Exception as e:
            bad += 1
            print(f"  skip '{tag}': unreadable ({e})")
            continue
        if args.min_boxes and n < args.min_boxes:
            dropped_min += 1
            continue
        if args.max_boxes and n > args.max_boxes:
            dropped_max += 1
            continue
        rooms.append((sid, n))

    if not rooms:
        print(f"No usable rooms found under {args.data_dir}")
        return 1

    # deterministic shuffle + split
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rooms))
    n = len(rooms)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    split_of = {}
    for rank, idx in enumerate(order):
        sid = rooms[idx][0]
        if rank < n_train:
            split_of[sid] = "train"
        elif rank < n_train + n_val:
            split_of[sid] = "val"
        else:
            split_of[sid] = "test"

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        for sid, _ in rooms:
            w.writerow([sid, split_of[sid]])

    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_of.values():
        counts[s] += 1
    max_boxes = max(nb for _, nb in rooms)
    box_counts = np.array([nb for _, nb in rooms])

    print(f"\nWrote {args.out}")
    print(f"  rooms kept:   {n}   (dropped: <min {dropped_min}, >max {dropped_max}, "
          f"bad {bad}, dup {dup})")
    print(f"  splits:       train {counts['train']} / val {counts['val']} / test {counts['test']}")
    print(f"  objects/room: min {box_counts.min()}, median {int(np.median(box_counts))}, "
          f"mean {box_counts.mean():.1f}, max {max_boxes}")
    print(f"\n  >> set network.sample_num_points >= {max_boxes} in config/gpc_mixed.yaml")
    print(f"     (it is used as max_length; training asserts it covers the largest room)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
