#!/usr/bin/env python3
import argparse
import json
import os
import pickle

import hashlib

import numpy as np
import trimesh


def get_color_for_label(label):
    h = hashlib.md5(label.encode('utf-8')).digest()
    r = int(h[0]) % 180 + 50
    g = int(h[1]) % 180 + 50
    b = int(h[2]) % 180 + 50
    return [r, g, b, 255]


def _rotation_y(theta):
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _translation(t):
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def _export_one(room, layout, object_types, out_glb, out_labels):
    cls = np.asarray(layout["class_labels"], dtype=np.float32)
    tr = np.asarray(layout["translations"], dtype=np.float32)
    sz = np.asarray(layout["sizes"], dtype=np.float32)
    ang = np.asarray(layout["angles"], dtype=np.float32).reshape(-1)

    scene = trimesh.Scene()
    labels = []

    # 1. Export Floor plan
    if room is not None and hasattr(room, "floor_plan"):
        vertices, faces = room.floor_plan
        vertices = vertices - room.floor_plan_centroid
        floor_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # default floor color (RGBA: 180, 180, 180, 255)
        floor_mesh.visual.face_colors = np.tile([180, 180, 180, 255], (faces.shape[0], 1)).astype(np.uint8)
        scene.add_geometry(floor_mesh, node_name="floor_plan", geom_name="floor_plan")

    # 2. Export Furniture boxes
    for i in range(cls.shape[0]):
        class_idx = int(cls[i].argmax())
        if class_idx < 0 or class_idx >= len(object_types):
            continue

        label = object_types[class_idx]
        if label in ("start", "end"):
            continue

        # Model outputs can be noisy in smoke runs; use absolute half-extents so
        # boxes remain visible in GLB viewers even if a dimension flips negative.
        extents = np.maximum(np.abs(sz[i]) * 2.0, 1e-4).astype(np.float64)
        mesh = trimesh.creation.box(extents=extents)
        # Color code boxes deterministically based on label
        mesh.visual.face_colors = np.tile(get_color_for_label(label), (mesh.faces.shape[0], 1)).astype(np.uint8)

        name = f"{label}_{i:03d}"
        mesh.metadata = {
            "label": label,
            "index": i,
            "translation": tr[i].tolist(),
            "half_extent": sz[i].tolist(),
            "angle_radians": float(ang[i]),
        }

        transform = _translation(tr[i]) @ _rotation_y(ang[i])
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform)

        labels.append(
            {
                "name": name,
                "label": label,
                "index": i,
                "translation": tr[i].tolist(),
                "half_extent": sz[i].tolist(),
                "angle_radians": float(ang[i]),
            }
        )

    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    scene.export(out_glb)

    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump({"objects": labels}, f, indent=2)

    return len(labels)


def main():
    parser = argparse.ArgumentParser(description="Export MiDiffusion results.pkl to box-only GLB scenes")
    parser.add_argument("result_file", help="Path to results.pkl")
    parser.add_argument("--output-dir", required=True, help="Directory to write scene_XXX.glb files")
    parser.add_argument("--limit", type=int, default=None, help="Export only first N scenes")
    args = parser.parse_args()

    with open(args.result_file, "rb") as f:
        results = pickle.load(f)

    object_types = list(results.test_dataset.object_types)
    total = len(results)
    n_export = total if args.limit is None else min(total, args.limit)

    summary = []
    for i in range(n_export):
        scene_idx, layout = results[i]
        room = results.test_dataset[scene_idx] if hasattr(results, "test_dataset") else None
        out_glb = os.path.join(args.output_dir, f"scene_{i:03d}.glb")
        out_labels = os.path.join(args.output_dir, f"scene_{i:03d}.labels.json")
        n_boxes = _export_one(room, layout, object_types, out_glb, out_labels)
        summary.append({"scene_index": i, "boxes_exported": n_boxes, "glb": out_glb})

    print(json.dumps({"exported_scenes": n_export, "total_scenes": total, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
