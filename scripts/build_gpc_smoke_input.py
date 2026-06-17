#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def _room_json(room_id, rng):
    width_mm = rng.randint(3600, 5600)
    depth_mm = rng.randint(3200, 5200)

    children = [
        {"className": "GPC-Wall", "id": "w1", "height": 2400},
        {
            "className": "GPC-Room",
            "id": room_id,
            "roomName": "Living room",
            "wallIds": ["w1"],
            "_profile": {
                "contour": [
                    {"sp": {"x": 0, "y": 0}, "ep": {"x": width_mm, "y": 0}},
                    {"sp": {"x": width_mm, "y": 0}, "ep": {"x": width_mm, "y": depth_mm}},
                    {"sp": {"x": width_mm, "y": depth_mm}, "ep": {"x": 0, "y": depth_mm}},
                    {"sp": {"x": 0, "y": depth_mm}, "ep": {"x": 0, "y": 0}},
                ]
            },
        },
    ]

    products = ["P1", "P2", "P3"]
    for i, product_no in enumerate(products, start=1):
        children.append(
            {
                "className": "GPC-Furniture",
                "id": f"f{i}",
                "userData": {"productNo": product_no},
                "position": {
                    "x": rng.randint(600, width_mm - 600),
                    "y": 0,
                    "z": rng.randint(600, depth_mm - 600),
                },
                "rotation": {"y": rng.choice([-0.9, -0.4, 0.0, 0.35, 0.8])},
                "size": {
                    "x": rng.randint(650, 1200),
                    "y": rng.randint(600, 1000),
                    "z": rng.randint(500, 900),
                },
                "modelOriginType": "BottomCenter",
                "modelUrl": f"model{i}.glb",
            }
        )

    return {"rootNodes": [{"children": children}]}


def main():
    parser = argparse.ArgumentParser(description="Build tiny synthetic GPC input for smoke tests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.output_dir)
    in_dir = out / "input_json"
    in_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for i in range(args.count):
        room_id = f"smokeRoom{i:03d}"
        payload = _room_json(room_id, rng)
        (in_dir / f"room_{i:03d}.json").write_text(json.dumps(payload), encoding="utf-8")

    (out / "item_map.json").write_text(
        json.dumps({"P1": "pt_chair", "P2": "pt_table", "P3": "pt_stool"}),
        encoding="utf-8",
    )
    (out / "pt_map.csv").write_text(
        "productType,category\n"
        "pt_chair,dining_chair\n"
        "pt_table,dining_table\n"
        "pt_stool,stool\n"
        "ignored,__none__\n",
        encoding="utf-8",
    )

    print(str(out))
    print(f"json_count={args.count}")


if __name__ == "__main__":
    main()
