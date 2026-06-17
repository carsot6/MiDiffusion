"""Utilities to convert MiDiffusion results.pkl into viewer scene manifests."""

import json
import io
import pickle

import numpy as np


class ThreedFrontResultsShim:
    """Placeholder class for robust unpickling without evaluation imports."""

    def __init__(self, *args, **kwargs):
        pass


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "threed_front.evaluation" and name == "ThreedFrontResults":
            return ThreedFrontResultsShim
        return super().find_class(module, name)


def load_results_from_path(path):
    with open(path, "rb") as f:
        return CompatUnpickler(f).load()


def load_results_from_bytes(raw_bytes):
    return CompatUnpickler(io.BytesIO(raw_bytes)).load()


def load_asset_map(asset_map_obj):
    if asset_map_obj is None:
        return {"by_label": {}, "by_name": {}, "by_product_no": {}, "default": None}

    if isinstance(asset_map_obj, str):
        raw = json.loads(asset_map_obj)
    elif isinstance(asset_map_obj, dict):
        raw = asset_map_obj
    else:
        raise ValueError("asset map must be dict or JSON string")

    by_label = raw.get("by_label", raw)
    by_name = raw.get("by_name", {})
    by_product_no = raw.get("by_product_no", {})
    default_asset = raw.get("default_asset_url", None)

    def _coerce_map(mapping):
        out = {}
        if not isinstance(mapping, dict):
            return out
        for key, value in mapping.items():
            if isinstance(value, str):
                out[key] = {"asset_url": value}
            elif isinstance(value, dict):
                if "asset_url" not in value:
                    continue
                item = {"asset_url": str(value["asset_url"])}
                if "scale" in value:
                    item["scale"] = [float(x) for x in value["scale"]]
                if "offset" in value:
                    item["offset"] = [float(x) for x in value["offset"]]
                out[key] = item
        return out

    default = None
    if isinstance(default_asset, str) and default_asset:
        default = {"asset_url": default_asset}

    return {
        "by_label": _coerce_map(by_label),
        "by_name": _coerce_map(by_name),
        "by_product_no": _coerce_map(by_product_no),
        "default": default,
    }


def _asset_for(asset_map, name, label, product_no=None):
    if product_no is not None:
        key = str(product_no)
        if key in asset_map["by_product_no"]:
            return dict(asset_map["by_product_no"][key])
    if name in asset_map["by_name"]:
        return dict(asset_map["by_name"][name])
    if label in asset_map["by_label"]:
        return dict(asset_map["by_label"][label])
    if asset_map["default"] is not None:
        return dict(asset_map["default"])
    return None


def _extract_boundary_loop(faces):
    """Return one boundary loop from triangulated floor mesh as vertex indices."""
    edge_count = {}
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    if not boundary_edges:
        return []

    adjacency = {}
    for u, v in boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    start = min(adjacency.keys())
    loop = [start]
    prev = None
    cur = start

    for _ in range(len(boundary_edges) + 3):
        neighbors = adjacency.get(cur, [])
        if not neighbors:
            break

        if prev is None:
            nxt = neighbors[0]
        elif len(neighbors) == 1:
            nxt = neighbors[0]
        else:
            nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]

        if nxt == start:
            break

        loop.append(nxt)
        prev, cur = cur, nxt

    return loop


def _coerce_results_view(results):
    """Normalize multiple ThreedFrontResults shapes into one iterable view."""
    if hasattr(results, "_scene_indices") and hasattr(results, "_predicted_layouts"):
        pairs = list(zip(results._scene_indices, results._predicted_layouts))
        dataset = getattr(results, "_test_dataset", None)
        return pairs, dataset

    if hasattr(results, "test_dataset") and hasattr(results, "__getitem__"):
        pairs = [results[i] for i in range(len(results))]
        dataset = getattr(results, "test_dataset", None)
        return pairs, dataset

    raise RuntimeError("Unsupported results object shape")


def _build_scene(
    scene_order,
    scene_idx,
    room,
    layout,
    object_types,
    asset_map,
    wall_height,
    wall_thickness,
):
    cls = np.asarray(layout["class_labels"], dtype=np.float32)
    tr = np.asarray(layout["translations"], dtype=np.float32)
    sz = np.asarray(layout["sizes"], dtype=np.float32)
    ang = np.asarray(layout["angles"], dtype=np.float32).reshape(-1)

    room_obj = {
        "wall_height": float(wall_height),
        "wall_thickness": float(wall_thickness),
    }
    if room is not None:
        room_id = getattr(room, "scene_id", None)
        if room_id is not None:
            room_obj["room_id"] = str(room_id)

        if hasattr(room, "floor_plan"):
            floor_vertices, floor_faces = room.floor_plan
            floor_vertices = np.asarray(floor_vertices, dtype=np.float64)
            floor_faces = np.asarray(floor_faces, dtype=np.int32)

            room_obj["floor_vertices"] = floor_vertices.tolist()
            room_obj["floor_faces"] = floor_faces.tolist()

            loop = _extract_boundary_loop(floor_faces)
            if loop:
                boundary = [[float(floor_vertices[i, 0]), float(floor_vertices[i, 2])] for i in loop]
                room_obj["boundary_xz"] = boundary

    objects = []
    for i in range(cls.shape[0]):
        class_idx = int(cls[i].argmax())
        if class_idx < 0 or class_idx >= len(object_types):
            continue

        label = str(object_types[class_idx])
        if label in ("start", "end"):
            continue

        name = f"{label}_{i:03d}"
        obj = {
            "name": name,
            "label": label,
            "class_index": class_idx,
            "source_index": int(i),
            "translation": [float(x) for x in tr[i]],
            "rotation_y": float(ang[i]),
            "half_extent": [float(abs(x)) for x in sz[i]],
        }

        asset_info = _asset_for(asset_map, name=name, label=label)
        if asset_info is not None:
            obj.update(asset_info)

        objects.append(obj)

    return {
        "scene_order": int(scene_order),
        "scene_index": int(scene_idx),
        "room": room_obj,
        "objects": objects,
    }


def build_manifest(results, asset_map_obj=None, limit=None, wall_height=2.6, wall_thickness=0.08):
    scene_pairs, dataset = _coerce_results_view(results)
    object_types = list(getattr(dataset, "object_types", [])) if dataset is not None else []
    asset_map = load_asset_map(asset_map_obj)

    n = len(scene_pairs) if limit is None else min(len(scene_pairs), int(limit))
    scenes = []
    for out_idx in range(n):
        scene_idx, layout = scene_pairs[out_idx]

        room = None
        if dataset is not None:
            try:
                room = dataset[scene_idx]
            except Exception:
                room = None

        scene = _build_scene(
            scene_order=out_idx,
            scene_idx=scene_idx,
            room=room,
            layout=layout,
            object_types=object_types,
            asset_map=asset_map,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
        )
        scenes.append(scene)

    return {
        "version": "1.0",
        "format": "midiffusion_viewer_scene_manifest",
        "object_types": object_types,
        "scenes": scenes,
    }


def _dedupe_boundary(points, eps=1e-6):
    out = []
    for p in points:
        x, z = float(p[0]), float(p[1])
        if not out:
            out.append([x, z])
            continue
        if abs(out[-1][0] - x) <= eps and abs(out[-1][1] - z) <= eps:
            continue
        out.append([x, z])

    if len(out) >= 2:
        if abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
            out.pop()
    return out


def _triangulate_fan(vertices):
    faces = []
    if len(vertices) < 3:
        return faces
    for i in range(1, len(vertices) - 1):
        faces.append([0, i, i + 1])
    return faces


def build_manifest_from_gpc_json_payload(
    payload,
    asset_map_obj=None,
    product_asset_resolver=None,
    wall_height=2.6,
    wall_thickness=0.12,
):
    """Build a one-scene viewer manifest from a raw GPC room JSON payload."""
    root_nodes = payload.get("rootNodes", [])
    if not isinstance(root_nodes, list) or len(root_nodes) == 0:
        raise ValueError("JSON does not contain rootNodes")

    children = []
    for root in root_nodes:
        if isinstance(root, dict):
            children.extend(root.get("children", []))

    if not children:
        raise ValueError("GPC JSON has no root children")

    room_node = next((c for c in children if c.get("className") == "GPC-Room"), None)
    if room_node is None:
        raise ValueError("GPC JSON missing GPC-Room node")

    contour = room_node.get("_profile", {}).get("contour", [])
    raw_points = []
    for seg in contour:
        sp = seg.get("sp", {})
        raw_points.append([float(sp.get("x", 0.0)), float(sp.get("y", 0.0))])
    if contour:
        ep = contour[-1].get("ep", {})
        raw_points.append([float(ep.get("x", 0.0)), float(ep.get("y", 0.0))])

    boundary_mm = _dedupe_boundary(raw_points)
    if len(boundary_mm) < 3:
        raise ValueError("GPC room contour has fewer than 3 unique points")

    xs = [p[0] for p in boundary_mm]
    zs = [p[1] for p in boundary_mm]
    cx_mm = 0.5 * (min(xs) + max(xs))
    cz_mm = 0.5 * (min(zs) + max(zs))

    boundary_xz = [[(x - cx_mm) / 1000.0, (z - cz_mm) / 1000.0] for x, z in boundary_mm]

    floor_vertices = [[p[0], 0.0, p[1]] for p in boundary_xz]
    floor_faces = _triangulate_fan(floor_vertices)

    wall_nodes = {str(c.get("id")): c for c in children if c.get("className") == "GPC-Wall"}
    wall_ids = room_node.get("wallIds", [])
    wall_heights = []
    for wid in wall_ids:
        node = wall_nodes.get(str(wid))
        if not node:
            continue
        h = node.get("height")
        try:
            wall_heights.append(float(h) / 1000.0)
        except (TypeError, ValueError):
            pass

    final_wall_height = float(wall_height)
    if wall_heights:
        final_wall_height = float(max(wall_heights))

    asset_map = load_asset_map(asset_map_obj)

    furniture_nodes = [c for c in children if c.get("className") == "GPC-Furniture"]
    objects = []
    for i, furn in enumerate(furniture_nodes):
        position = furn.get("position", {})
        size = furn.get("size", {})
        rotation = furn.get("rotation", {})
        user_data = furn.get("userData", {}) or {}

        sx = float(size.get("x", 0.0)) / 1000.0
        sy = float(size.get("y", 0.0)) / 1000.0
        sz = float(size.get("z", 0.0)) / 1000.0

        tx = (float(position.get("x", 0.0)) - cx_mm) / 1000.0
        ty = float(position.get("y", 0.0)) / 1000.0 + sy * 0.5
        tz = (float(position.get("z", 0.0)) - cz_mm) / 1000.0

        product_no = user_data.get("productNo")
        label = (
            user_data.get("productType")
            or user_data.get("category")
            or (str(product_no) if product_no is not None else None)
            or "furniture"
        )

        name = str(furn.get("id") or f"furniture_{i:03d}")
        obj = {
            "name": name,
            "label": str(label),
            "source_index": int(i),
            "translation": [tx, ty, tz],
            "rotation_y": float(rotation.get("y", 0.0)),
            "half_extent": [max(0.01, sx * 0.5), max(0.01, sy * 0.5), max(0.01, sz * 0.5)],
        }

        if product_no is not None:
            obj["product_no"] = str(product_no)

        model_url = furn.get("modelUrl")
        if isinstance(model_url, str) and model_url.strip():
            obj["asset_url"] = model_url.strip()

        if product_asset_resolver is not None and product_no is not None:
            has_http_asset = isinstance(obj.get("asset_url"), str) and obj["asset_url"].startswith(("http://", "https://"))
            if not has_http_asset:
                resolved_url = product_asset_resolver(str(product_no))
                if isinstance(resolved_url, str) and resolved_url.strip():
                    obj["asset_url"] = resolved_url.strip()

        mapped = _asset_for(asset_map, name=name, label=str(label), product_no=product_no)
        if mapped is not None:
            obj.update(mapped)

        objects.append(obj)

    scene = {
        "scene_order": 0,
        "scene_index": 0,
        "room": {
            "room_id": str(room_node.get("id", "gpc_room")),
            "wall_height": float(final_wall_height),
            "wall_thickness": float(wall_thickness),
            "boundary_xz": boundary_xz,
            "floor_vertices": floor_vertices,
            "floor_faces": floor_faces,
        },
        "objects": objects,
    }

    return {
        "version": "1.0",
        "format": "midiffusion_viewer_scene_manifest",
        "object_types": [],
        "scenes": [scene],
    }
