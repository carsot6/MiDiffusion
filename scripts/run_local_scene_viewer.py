#!/usr/bin/env python3
"""Run a local web app for MiDiffusion scene visualization."""

import argparse
import json
import sys
import urllib.parse
import urllib.request
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, Response

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_manifest_lib import build_manifest, load_results_from_bytes
from viewer_manifest_lib import build_manifest_from_gpc_json_payload


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "viewer_web"


def _fetch_sugarcube_model_url(product_no):
    """Resolve product_no to GLB url via internal Sugarcube API."""
    base_url = os.environ.get(
        "SUGARCUBE_FURNITURE_API_URL",
        "https://api.cte.home-design.ikea.com/de/en/ideas/sugarcube/api/furniture/",
    )
    query = {
        "local_item_number": str(product_no),
        "geometry_type": "3d,2d_wall,wall_placeable",
        "combinables_default_products_noinline": "true",
    }
    url = base_url + "?" + urllib.parse.urlencode(query)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MiDiffusionLocalViewer/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if isinstance(payload, list):
        results = payload
    elif isinstance(payload, dict):
        results = payload.get("results", [])
    else:
        results = []
    if not results:
        return None
    first = results[0]
    model_url = first.get("modelUrl") if isinstance(first, dict) else None
    if isinstance(model_url, str) and model_url.strip():
        return model_url.strip()
    return None


def _parse_float(value, default_value):
    if value is None or value == "":
        return default_value
    return float(value)


def _parse_int(value, default_value=None):
    if value is None or value == "":
        return default_value
    return int(value)


def create_app(max_upload_mb):
    app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = int(max_upload_mb * 1024 * 1024)

    @app.route("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/api/parse_results", methods=["POST"])
    def parse_results():
        if "results_file" not in request.files:
            return jsonify({"error": "results_file is required"}), 400

        upload = request.files["results_file"]
        file_name = upload.filename or "results.pkl"
        if not file_name.lower().endswith(".pkl"):
            return jsonify({"error": "results_file must be a .pkl file"}), 400

        try:
            raw = upload.read()
            asset_map_json = request.form.get("asset_map_json", "").strip()
            asset_map_obj = json.loads(asset_map_json) if asset_map_json else None

            limit = _parse_int(request.form.get("limit", ""), None)
            wall_height = _parse_float(request.form.get("wall_height", ""), 2.6)
            wall_thickness = _parse_float(request.form.get("wall_thickness", ""), 0.08)

            results = load_results_from_bytes(raw)
            manifest = build_manifest(
                results=results,
                asset_map_obj=asset_map_obj,
                limit=limit,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
            )
            manifest["source_file"] = file_name
            return jsonify(manifest)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Invalid asset_map_json: {exc}"}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to parse results.pkl: {exc}"}), 500

    @app.route("/api/parse_gpc_json", methods=["POST"])
    def parse_gpc_json():
        if "gpc_file" not in request.files:
            return jsonify({"error": "gpc_file is required"}), 400

        upload = request.files["gpc_file"]
        file_name = upload.filename or "room.json"
        if not file_name.lower().endswith(".json"):
            return jsonify({"error": "gpc_file must be a .json file"}), 400

        try:
            raw = upload.read()
            payload = json.loads(raw.decode("utf-8"))
            asset_map_json = request.form.get("asset_map_json", "").strip()
            asset_map_obj = json.loads(asset_map_json) if asset_map_json else None

            wall_height = _parse_float(request.form.get("wall_height", ""), 2.6)
            wall_thickness = _parse_float(request.form.get("wall_thickness", ""), 0.12)

            manifest = build_manifest_from_gpc_json_payload(
                payload=payload,
                asset_map_obj=asset_map_obj,
                product_asset_resolver=_fetch_sugarcube_model_url,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
            )
            manifest["source_file"] = file_name
            return jsonify(manifest)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Invalid JSON: {exc}"}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to parse GPC JSON: {exc}"}), 500

    @app.route("/api/proxy_asset")
    def proxy_asset():
        # Flask's request.args treats + as space (form-urlencoded behavior)
        # Use request.query_string to get raw query and parse manually preserving +
        query_string = request.query_string.decode("utf-8")
        # Parse manually: find url= parameter
        raw_url = ""
        for part in query_string.split("&"):
            if part.startswith("url="):
                raw_url = urllib.parse.unquote(part[4:])  # Remove "url=" and unquote
                break
        
        if not raw_url:
            return jsonify({"error": "url query parameter is required"}), 400

        print(f"[PROXY] Received URL: {raw_url}")

        try:
            parsed = urllib.parse.urlparse(raw_url)
        except Exception:
            return jsonify({"error": "invalid url"}), 400

        if parsed.scheme not in ("http", "https"):
            return jsonify({"error": "only http/https urls are supported"}), 400

        # Reconstruct URL keeping + as-is in path (IKEA needs literal +)
        safe_path = urllib.parse.quote(parsed.path, safe="/:@!$&'()*+,;=+")
        safe_query = urllib.parse.quote(parsed.query, safe="/:@!$&'()*+,;=+") if parsed.query else ""
        safe_url = urllib.parse.urlunparse((
            parsed.scheme, parsed.netloc, safe_path, parsed.params, safe_query, parsed.fragment
        ))
        
        print(f"[PROXY] Reconstructed URL: {safe_url}")

        try:
            req = urllib.request.Request(
                safe_url,
                headers={
                    "User-Agent": "MiDiffusionLocalViewer/1.0",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as upstream:
                body = upstream.read()
                content_type = upstream.headers.get("Content-Type", "application/octet-stream")

            print(f"[PROXY] Success: {len(body)} bytes, type: {content_type}")

            return Response(
                body,
                status=200,
                content_type=content_type,
                headers={"Cache-Control": "public, max-age=3600"},
            )
        except Exception as exc:
            print(f"[PROXY] Error: {exc}")
            return jsonify({"error": f"asset fetch failed: {exc}"}), 502

    @app.route("/<path:path>")
    def static_proxy(path):
        return send_from_directory(WEB_DIR, path)

    return app


def main():
    parser = argparse.ArgumentParser(description="Run local scene viewer web app")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--max-upload-mb", type=int, default=200, help="Max uploaded pkl size")
    args = parser.parse_args()

    if not WEB_DIR.exists():
        raise FileNotFoundError(f"Missing web assets directory: {WEB_DIR}")

    app = create_app(args.max_upload_mb)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
