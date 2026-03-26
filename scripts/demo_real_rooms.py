#!/usr/bin/env python3
"""
Generate 3D room visualizations from 3D-FRONT test data.

This script loads actual room layouts from the 3D-FRONT dataset, finds rooms
that are in the test split (not seen during training), loads real furniture 
meshes from 3D-FUTURE, and exports interactive 3D GLB files and PNG previews.

Usage:
    python scripts/demo_real_rooms.py --room_type living --test_only --num_rooms 30
    python scripts/demo_real_rooms.py --room_type bedroom --test_only --num_rooms 15
    python scripts/demo_real_rooms.py --room_type living --non_rectangular --test_only
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import trimesh
except ImportError:
    print("trimesh is required: pip install trimesh")
    sys.exit(1)

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, "midiffusion", "datasets", "data")
THREED_FRONT_DIR = os.path.join(DATA_DIR, "3D-FRONT")
THREED_FUTURE_DIR = os.path.join(DATA_DIR, "3D-FUTURE-model")
OUTPUT_DIR = os.path.join(PROJ_DIR, "output", "test_rooms")

# Color palette for furniture categories
CATEGORY_COLORS = {
    "sofa": [70, 130, 180, 255],
    "table": [34, 139, 34, 255],
    "chair": [139, 90, 43, 255],
    "cabinet": [205, 133, 63, 255],
    "shelf": [160, 82, 45, 255],
    "desk": [107, 142, 35, 255],
    "bed": [100, 149, 237, 255],
    "lamp": [255, 215, 0, 255],
    "tv": [105, 105, 105, 255],
    "wardrobe": [139, 69, 19, 255],
    "nightstand": [188, 143, 143, 255],
    "dresser": [210, 180, 140, 255],
    "default": [150, 150, 150, 255],
}


def get_category_color(category):
    """Get color for a furniture category."""
    cat = (category or '').lower()
    for key, color in CATEGORY_COLORS.items():
        if key in cat:
            return color
    return CATEGORY_COLORS["default"]


def load_model_info():
    """Load 3D-FUTURE model info."""
    info_path = os.path.join(THREED_FUTURE_DIR, "model_info.json")
    if not os.path.exists(info_path):
        print(f"Warning: model_info.json not found at {info_path}")
        return {}
    with open(info_path) as f:
        models = json.load(f)
    return {m['model_id']: m for m in models}


def load_test_scene_ids(room_type="living"):
    """Load test scene IDs from ATISS splits."""
    if room_type == "living":
        splits_file = os.path.join(PROJ_DIR, "config", "splits", 
                                    "livingroom_threed_front_splits.csv")
    elif room_type == "bedroom":
        splits_file = os.path.join(PROJ_DIR, "config", "splits",
                                    "bedroom_threed_front_splits.csv")
    else:
        raise ValueError(f"Unknown room type: {room_type}")
    
    if not os.path.exists(splits_file):
        print(f"Splits file not found: {splits_file}")
        print("Download from: https://github.com/nv-tlabs/ATISS/tree/master/config")
        return set()
    
    test_ids = set()
    with open(splits_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1].strip() == 'test':
                test_ids.add(parts[0].strip())
    
    print(f"Loaded {len(test_ids)} test scene IDs for {room_type}")
    return test_ids


def quat_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def load_furniture_mesh(jid):
    """Load a furniture mesh from 3D-FUTURE."""
    obj_path = os.path.join(THREED_FUTURE_DIR, jid, "raw_model.obj")
    if os.path.exists(obj_path):
        try:
            mesh = trimesh.load(obj_path, force='mesh')
            return mesh
        except Exception:
            pass
    return None


def create_procedural_mesh(category, size):
    """Create a simple box mesh as fallback when actual mesh is unavailable."""
    w, d, h = [max(float(s), 0.1) for s in size]
    return trimesh.creation.box(extents=[w, h, d])


def find_rooms_in_dataset(room_type="living", test_ids=None, 
                           non_rectangular=False, max_rooms=30):
    """
    Find rooms in 3D-FRONT dataset matching criteria.
    
    Args:
        room_type: "living" or "bedroom"
        test_ids: set of test scene IDs (from ATISS splits)
        non_rectangular: if True, prefer non-rectangular (L-shaped) rooms
        max_rooms: maximum number of rooms to return
    
    Returns:
        List of dicts with room info
    """
    model_info = load_model_info()
    
    # Room type matching patterns
    if room_type == "living":
        room_patterns = ["livingroom", "livingdiningroom", "living room", 
                         "living dining room"]
    elif room_type == "bedroom":
        room_patterns = ["bedroom", "masterbedroom", "secondbedroom",
                         "master bedroom", "second bedroom"]
    else:
        room_patterns = [room_type.lower()]
    
    rooms = []
    house_files = [f for f in os.listdir(THREED_FRONT_DIR) if f.endswith('.json')]
    
    for hf in sorted(house_files):
        house_path = os.path.join(THREED_FRONT_DIR, hf)
        try:
            with open(house_path) as f:
                house = json.load(f)
        except Exception:
            continue
        
        furniture_map = {}
        for fu in house.get('furniture', []):
            if fu.get('valid', False):
                furniture_map[fu['uid']] = fu
        
        mesh_map = {}
        for m in house.get('mesh', []):
            mesh_map[m['uid']] = m
        
        for room in house.get('scene', {}).get('room', []):
            scene_id = room.get('instanceid', '')
            room_type_str = (room.get('type', '') or '').lower().replace(' ', '')
            
            # Check if room type matches
            matches_type = any(p.replace(' ', '') in room_type_str 
                              for p in room_patterns)
            if not matches_type:
                continue
            
            # Check if in test set
            if test_ids and scene_id not in test_ids:
                continue
            
            # Count furniture and find floor
            children = room.get('children', [])
            furniture_items = []
            floor_vertices = []
            
            for child in children:
                ref = child.get('ref', '')
                if ref in furniture_map:
                    fu = furniture_map[ref]
                    jid = fu.get('jid', '')
                    cat = (model_info.get(jid, {}).get('category', '') or '')
                    # Skip ceiling items
                    if 'ceiling' in cat.lower():
                        continue
                    furniture_items.append({
                        'jid': jid,
                        'category': cat,
                        'pos': child.get('pos', [0, 0, 0]),
                        'rot': child.get('rot', [0, 0, 0, 1]),
                        'scale': child.get('scale', [1, 1, 1]),
                        'bbox': fu.get('bbox', {}),
                    })
                elif ref in mesh_map:
                    mesh_type = (mesh_map[ref].get('type', '') or '').lower()
                    if mesh_type == 'floor':
                        xyz = np.array(mesh_map[ref].get('xyz', [])).reshape(-1, 3)
                        if len(xyz) > 0:
                            floor_vertices.append(xyz)
            
            if len(furniture_items) < 3:
                continue
            
            # Check floor shape for non-rectangular filter
            is_non_rect = False
            if floor_vertices:
                all_floor = np.vstack(floor_vertices)
                x_coords = all_floor[:, 0]
                z_coords = all_floor[:, 2]
                
                # Compute convex hull area vs bounding box area
                x_range = x_coords.max() - x_coords.min()
                z_range = z_coords.max() - z_coords.min()
                bbox_area = x_range * z_range
                
                if bbox_area > 0:
                    try:
                        from scipy.spatial import ConvexHull
                        pts_2d = np.column_stack([x_coords, z_coords])
                        hull = ConvexHull(pts_2d)
                        hull_area = hull.volume  # 2D: volume = area
                        ratio = hull_area / bbox_area
                        is_non_rect = ratio < 0.85
                    except Exception:
                        pass
            
            if non_rectangular and not is_non_rect:
                continue
            
            rooms.append({
                'scene_id': scene_id,
                'house_file': hf,
                'num_furniture': len(furniture_items),
                'is_non_rectangular': is_non_rect,
                'furniture': furniture_items,
                'floor_vertices': floor_vertices,
            })
    
    # Sort by number of furniture (more interesting rooms first)
    rooms.sort(key=lambda r: r['num_furniture'], reverse=True)
    return rooms[:max_rooms]


def render_room_scene(room_info, model_info, output_dir):
    """
    Render a room to GLB and PNG.
    
    Args:
        room_info: dict with room data
        model_info: dict mapping model_id to model metadata
        output_dir: output directory
    
    Returns:
        True if successful
    """
    scene_id = room_info['scene_id']
    out_glb = os.path.join(output_dir, f"{scene_id}.glb")
    out_png = os.path.join(output_dir, f"{scene_id}.png")
    
    if os.path.exists(out_png):
        print(f"  {scene_id} - already exists, skipping")
        return True
    
    furniture = room_info['furniture']
    floor_verts = room_info.get('floor_vertices', [])
    
    # Determine floor bounds
    if floor_verts:
        all_floor = np.vstack(floor_verts)
        x_min, x_max = all_floor[:, 0].min() - 0.5, all_floor[:, 0].max() + 0.5
        z_min, z_max = all_floor[:, 2].min() - 0.5, all_floor[:, 2].max() + 0.5
    else:
        x_min, x_max, z_min, z_max = -3, 3, -3, 3
    
    # Build 3D scene
    scene = trimesh.Scene()
    
    # Floor
    floor_mesh = trimesh.Trimesh(
        vertices=np.array([
            [x_min, 0, z_min], [x_max, 0, z_min],
            [x_max, 0, z_max], [x_min, 0, z_max]
        ]),
        faces=np.array([[0, 2, 1], [0, 3, 2]])
    )
    floor_mesh.visual.face_colors = [240, 230, 220, 255]
    scene.add_geometry(floor_mesh, node_name='floor')
    
    actual_meshes = 0
    procedural_meshes = 0
    
    for i, fu in enumerate(furniture):
        if 'ceiling' in (fu.get('category', '') or '').lower():
            continue
        
        mesh = load_furniture_mesh(fu['jid'])
        if mesh is not None:
            mesh.apply_scale(fu['scale'])
            actual_meshes += 1
        else:
            # Fallback to procedural box
            bbox = fu.get('bbox', {})
            if isinstance(bbox, dict) and bbox:
                size = [
                    bbox.get('xLen', 1) * fu['scale'][0],
                    bbox.get('yLen', 1) * fu['scale'][1],
                    bbox.get('zLen', 1) * fu['scale'][2]
                ]
            else:
                size = [s * 0.8 for s in fu['scale']]
            mesh = create_procedural_mesh(fu.get('category', ''), size)
            procedural_meshes += 1
        
        # Apply rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_rotation_matrix(fu['rot'])
        mesh.apply_transform(T)
        
        # Apply translation
        mesh.apply_translation(fu['pos'])
        
        # Apply color
        color = get_category_color(fu.get('category', ''))
        mesh.visual.face_colors = color
        
        scene.add_geometry(mesh, node_name=f'furniture_{i}')
    
    # Export GLB
    scene.export(out_glb)
    
    # Create PNG preview
    fig = plt.figure(figsize=(10, 4))
    
    # Top-down view
    ax1 = fig.add_subplot(121)
    ax1.fill([x_min, x_max, x_max, x_min], 
             [z_min, z_min, z_max, z_max], 
             color='lightgray', alpha=0.5)
    ax1.plot([x_min, x_max, x_max, x_min, x_min],
             [z_min, z_min, z_max, z_max, z_min], 'k-', lw=2)
    
    for fu in furniture:
        if 'ceiling' in (fu.get('category', '') or '').lower():
            continue
        color = np.array(get_category_color(fu.get('category', ''))[:3]) / 255
        ax1.plot(fu['pos'][0], fu['pos'][2], 'o', markersize=8, color=color)
    
    ax1.set_aspect('equal')
    ax1.set_title(scene_id, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    
    # 3D perspective view
    ax2 = fig.add_subplot(122, projection='3d')
    for fu in furniture:
        if 'ceiling' in (fu.get('category', '') or '').lower():
            continue
        color = np.array(get_category_color(fu.get('category', ''))[:3]) / 255
        ax2.scatter([fu['pos'][0]], [fu['pos'][1]], [fu['pos'][2]], 
                   s=60, c=[color])
    ax2.view_init(elev=25, azim=-60)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"  {scene_id} - OK ({actual_meshes} mesh, {procedural_meshes} procedural)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D room visualizations from 3D-FRONT test data"
    )
    parser.add_argument("--room_type", default="living", 
                       choices=["living", "bedroom"],
                       help="Type of room to generate")
    parser.add_argument("--test_only", action="store_true",
                       help="Only use rooms from the test split")
    parser.add_argument("--non_rectangular", action="store_true",
                       help="Prefer non-rectangular (L-shaped) rooms")
    parser.add_argument("--num_rooms", type=int, default=30,
                       help="Maximum number of rooms to generate")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: output/test_rooms)")
    parser.add_argument("--rooms_json", default=None,
                       help="Path to pre-selected rooms JSON file")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    model_info = load_model_info()
    
    if args.rooms_json:
        # Load pre-selected rooms
        with open(args.rooms_json) as f:
            rooms = json.load(f)
        print(f"Loaded {len(rooms)} rooms from {args.rooms_json}")
        
        # Need to reload full room data from house files
        for room in rooms:
            hf = room['house_file']
            sid = room['scene_id']
            house_path = os.path.join(THREED_FRONT_DIR, hf)
            
            try:
                with open(house_path) as f:
                    house = json.load(f)
            except Exception as e:
                print(f"  {sid} - error loading house: {e}")
                continue
            
            fm = {fu['uid']: fu for fu in house.get('furniture', []) 
                  if fu.get('valid')}
            mm = {m['uid']: m for m in house.get('mesh', [])}
            
            target_room = None
            for r in house.get('scene', {}).get('room', []):
                if r.get('instanceid') == sid:
                    target_room = r
                    break
            
            if not target_room:
                print(f"  {sid} - room not found in house")
                continue
            
            furniture = []
            floor_vertices = []
            
            for child in target_room.get('children', []):
                ref = child.get('ref', '')
                if ref in fm:
                    fu = fm[ref]
                    cat = (model_info.get(fu['jid'], {}).get('category', '') or '')
                    furniture.append({
                        'jid': fu['jid'],
                        'category': cat,
                        'pos': child.get('pos', [0, 0, 0]),
                        'rot': child.get('rot', [0, 0, 0, 1]),
                        'scale': child.get('scale', [1, 1, 1]),
                        'bbox': fu.get('bbox', {}),
                    })
                elif ref in mm:
                    mesh_type = (mm[ref].get('type', '') or '').lower()
                    if mesh_type == 'floor':
                        xyz = np.array(mm[ref].get('xyz', [])).reshape(-1, 3)
                        if len(xyz) > 0:
                            floor_vertices.append(xyz)
            
            room['furniture'] = furniture
            room['floor_vertices'] = floor_vertices
    else:
        # Find rooms in dataset
        test_ids = None
        if args.test_only:
            test_ids = load_test_scene_ids(args.room_type)
            if not test_ids:
                print("No test IDs found. Run without --test_only or download splits.")
                return
        
        print(f"Scanning 3D-FRONT for {args.room_type} rooms...")
        rooms = find_rooms_in_dataset(
            room_type=args.room_type,
            test_ids=test_ids,
            non_rectangular=args.non_rectangular,
            max_rooms=args.num_rooms
        )
    
    if not rooms:
        print("No matching rooms found.")
        return
    
    print(f"\nFound {len(rooms)} rooms. Generating 3D visualizations...")
    print(f"Output: {output_dir}\n")
    
    success = 0
    for idx, room in enumerate(rooms):
        print(f"  [{idx+1}/{len(rooms)}]", end=" ")
        try:
            if render_room_scene(room, model_info, output_dir):
                success += 1
        except Exception as e:
            print(f"  {room.get('scene_id', 'unknown')} - error: {e}")
    
    print(f"\nDone! Generated {success}/{len(rooms)} rooms to {output_dir}")


if __name__ == "__main__":
    main()
