#!/usr/bin/env python3
"""Export MiDiffusion results.pkl to a browser-friendly scene manifest JSON."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_manifest_lib import build_manifest, load_results_from_path


def main():
    parser = argparse.ArgumentParser(description="Export viewer manifest from MiDiffusion results.pkl")
    parser.add_argument("result_file", help="Path to results.pkl")
    parser.add_argument("--output-file", required=True, help="Path to output scene manifest JSON")
    parser.add_argument(
        "--asset-map",
        default=None,
        help=(
            "Optional JSON mapping labels/names to asset URLs. "
            "Supports {by_label:{...},by_name:{...},default_asset_url:'...'}"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Export first N scenes")
    parser.add_argument("--wall-height", type=float, default=2.6, help="Wall height in meters")
    parser.add_argument("--wall-thickness", type=float, default=0.08, help="Wall thickness in meters")
    args = parser.parse_args()

    results = load_results_from_path(args.result_file)
    asset_map_obj = None
    if args.asset_map:
        asset_map_obj = json.loads(Path(args.asset_map).read_text(encoding="utf-8"))

    payload = build_manifest(
        results=results,
        asset_map_obj=asset_map_obj,
        limit=args.limit,
        wall_height=args.wall_height,
        wall_thickness=args.wall_thickness,
    )
    payload["source_file"] = str(Path(args.result_file).resolve())

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out.resolve()),
                "scenes": len(payload["scenes"]),
                "objects": int(sum(len(s["objects"]) for s in payload["scenes"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
