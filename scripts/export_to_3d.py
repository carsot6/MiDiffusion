#!/usr/bin/env python
"""
Export MiDiffusion generated layouts to 3D formats (GLTF/GLB) for Blender rendering.

This script generates layouts and exports them as 3D files that can be imported
into Blender for photorealistic rendering.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import yaml
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trimesh
from trimesh.transformations import rotation_matrix

from midiffusion.networks.feature_extractors import get_feature_extractor
from midiffusion.networks.diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed


LIVINGROOM_CLASSES = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa",
    "chinese_chair", "coffee_table", "console_table", "corner_side_table",
    "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp",
    "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet",
    "start", "end"
]

# Furniture colors (R, G, B, A) in 0-255
FURNITURE_COLORS = {
    "sofa": [70, 130, 180, 255],       # Steel blue
    "chair": [139, 90, 43, 255],       # Saddle brown
    "table": [160, 82, 45, 255],       # Sienna
    "cabinet": [205, 133, 63, 255],    # Peru
    "shelf": [210, 180, 140, 255],     # Tan
    "desk": [139, 69, 19, 255],        # Saddle brown
    "lamp": [255, 215, 0, 255],        # Gold
    "default": [150, 150, 150, 255],   # Gray
}

FURNITURE_CATEGORY_MAP = {
    "armchair": "chair", "bookshelf": "shelf", "cabinet": "cabinet",
    "ceiling_lamp": "lamp", "chaise_longue_sofa": "sofa", "chinese_chair": "chair",
    "coffee_table": "table", "console_table": "table", "corner_side_table": "table",
    "desk": "desk", "dining_chair": "chair", "dining_table": "table",
    "l_shaped_sofa": "sofa", "lazy_sofa": "sofa", "lounge_chair": "chair",
    "loveseat_sofa": "sofa", "multi_seat_sofa": "sofa", "pendant_lamp": "lamp",
    "round_end_table": "table", "shelf": "shelf", "stool": "chair",
    "tv_stand": "cabinet", "wardrobe": "cabinet", "wine_cabinet": "cabinet",
}

DEFAULT_STATS = {
    "bounds_translations": [-4.0, -4.0, 0.0, 4.0, 4.0, 3.0],
    "bounds_sizes": [0.1, 0.1, 0.1, 3.0, 3.0, 3.0],
    "class_labels": LIVINGROOM_CLASSES,
}


def create_synthetic_floor_plan(shape="rectangular"):
    H, W = 64, 64
    mask = np.zeros((H, W), dtype=np.float32)
    
    if shape == "rectangular":
        margin = 8
        mask[margin:H-margin, margin:W-margin] = 1.0
    elif shape == "l_shaped":
        margin = 8
        mask[margin:H-margin, margin:W//2+10] = 1.0
        mask[margin:H//2+10, margin:W-margin] = 1.0
    elif shape == "irregular":
        margin = 10
        mask[margin:H-margin, margin:W-margin] = 1.0
        mask[margin:margin+15, W-margin-15:W-margin] = 0.0
    
    return mask


def mask_to_boundary_points(mask, num_points=256):
    H, W = mask.shape
    eroded = ndimage.binary_erosion(mask > 0.5)
    boundary = (mask > 0.5) & ~eroded
    boundary_coords = np.argwhere(boundary)
    
    if len(boundary_coords) == 0:
        boundary_coords = np.array([[8, 8], [8, W-8], [H-8, W-8], [H-8, 8]])
    
    if len(boundary_coords) > num_points:
        indices = np.linspace(0, len(boundary_coords)-1, num_points, dtype=int)
        boundary_coords = boundary_coords[indices]
    elif len(boundary_coords) < num_points:
        repeats = num_points // len(boundary_coords) + 1
        boundary_coords = np.tile(boundary_coords, (repeats, 1))[:num_points]
    
    x = (boundary_coords[:, 1] / W) * 2 - 1
    y = (boundary_coords[:, 0] / H) * 2 - 1
    
    nx = np.zeros(num_points)
    ny = np.zeros(num_points)
    for i in range(num_points):
        prev_idx = (i - 1) % num_points
        next_idx = (i + 1) % num_points
        tx = x[next_idx] - x[prev_idx]
        ty = y[next_idx] - y[prev_idx]
        length = np.sqrt(tx**2 + ty**2) + 1e-8
        nx[i] = -ty / length
        ny[i] = tx / length
    
    return np.stack([x, y, nx, ny], axis=1).astype(np.float32)


def load_model(config_path, weight_path, device):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    stats_dir = os.path.dirname(weight_path)
    stats_path = os.path.join(stats_dir, "dataset_stats.txt")
    if not os.path.exists(stats_path):
        with open(stats_path, 'w') as f:
            json.dump(DEFAULT_STATS, f)
    
    config["data"]["dataset_directory"] = stats_dir
    config["data"]["train_stats"] = "dataset_stats.txt"
    
    n_object_types = len(LIVINGROOM_CLASSES) - 2
    feature_extractor = get_feature_extractor(**config["feature_extractor"])
    
    network = DiffusionSceneLayout_Mixed(
        n_object_types, feature_extractor, config["network"], stats_path
    )
    
    state_dict = torch.load(weight_path, map_location=device)
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()
    
    return network, config


def generate_scene(network, floor_mask, device):
    batch_size = 1
    boundary_points = mask_to_boundary_points(floor_mask, num_points=256)
    room_feature = torch.from_numpy(boundary_points).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        layout = network.generate_layout(
            room_feature=room_feature,
            batch_size=batch_size,
            clip_denoised=True,
            device=device,
        )
    return layout


def create_furniture_mesh(furniture_type, size, color=None):
    """Create detailed procedural furniture mesh."""
    w, d, h = float(size[0]), float(size[1]), float(size[2])
    w, d, h = max(w, 0.2), max(d, 0.2), max(h, 0.2)
    
    category = FURNITURE_CATEGORY_MAP.get(furniture_type, "default")
    if color is None:
        color = FURNITURE_COLORS.get(category, FURNITURE_COLORS["default"])
    
    meshes = []
    
    if category == "sofa":
        # Base
        base = trimesh.creation.box(extents=[w, d, h * 0.4])
        base.apply_translation([0, 0, h * 0.2])
        meshes.append(base)
        
        # Back
        back = trimesh.creation.box(extents=[w, d * 0.15, h * 0.55])
        back.apply_translation([0, -d * 0.425, h * 0.525])
        meshes.append(back)
        
        # Arms
        arm1 = trimesh.creation.box(extents=[w * 0.08, d * 0.85, h * 0.35])
        arm1.apply_translation([-w * 0.46, 0, h * 0.375])
        meshes.append(arm1)
        
        arm2 = trimesh.creation.box(extents=[w * 0.08, d * 0.85, h * 0.35])
        arm2.apply_translation([w * 0.46, 0, h * 0.375])
        meshes.append(arm2)
        
        # Cushions
        for i in range(min(3, max(1, int(w / 0.6)))):
            cushion_w = w / min(3, max(1, int(w / 0.6))) * 0.9
            cushion = trimesh.creation.box(extents=[cushion_w, d * 0.7, h * 0.15])
            offset_x = (i - (min(3, max(1, int(w / 0.6))) - 1) / 2) * (w / min(3, max(1, int(w / 0.6))))
            cushion.apply_translation([offset_x, d * 0.05, h * 0.475])
            meshes.append(cushion)
        
    elif category == "chair":
        # Seat
        seat = trimesh.creation.box(extents=[w, d, h * 0.08])
        seat.apply_translation([0, 0, h * 0.46])
        meshes.append(seat)
        
        # Back
        back = trimesh.creation.box(extents=[w * 0.9, d * 0.08, h * 0.45])
        back.apply_translation([0, -d * 0.46, h * 0.725])
        meshes.append(back)
        
        # Legs
        leg_h = h * 0.42
        leg_r = min(w, d) * 0.06
        for dx, dy in [(-0.4, -0.4), (-0.4, 0.4), (0.4, -0.4), (0.4, 0.4)]:
            leg = trimesh.creation.cylinder(radius=leg_r, height=leg_h)
            leg.apply_translation([dx * w, dy * d, leg_h / 2])
            meshes.append(leg)
        
    elif category == "table":
        # Top
        top = trimesh.creation.box(extents=[w, d, h * 0.06])
        top.apply_translation([0, 0, h * 0.97])
        meshes.append(top)
        
        # Legs
        leg_h = h * 0.94
        leg_r = min(w, d) * 0.04
        for dx, dy in [(-0.42, -0.42), (-0.42, 0.42), (0.42, -0.42), (0.42, 0.42)]:
            leg = trimesh.creation.cylinder(radius=leg_r, height=leg_h)
            leg.apply_translation([dx * w, dy * d, leg_h / 2])
            meshes.append(leg)
        
    elif category == "cabinet":
        # Main body
        body = trimesh.creation.box(extents=[w, d, h])
        body.apply_translation([0, 0, h / 2])
        meshes.append(body)
        
        # Door line (visual detail)
        line = trimesh.creation.box(extents=[w * 0.02, d * 0.02, h * 0.9])
        line.apply_translation([0, d * 0.51, h * 0.5])
        meshes.append(line)
        
    elif category == "desk":
        # Top
        top = trimesh.creation.box(extents=[w, d, h * 0.04])
        top.apply_translation([0, 0, h * 0.98])
        meshes.append(top)
        
        # Side panels
        panel1 = trimesh.creation.box(extents=[w * 0.04, d * 0.85, h * 0.96])
        panel1.apply_translation([-w * 0.48, 0, h * 0.48])
        meshes.append(panel1)
        
        panel2 = trimesh.creation.box(extents=[w * 0.04, d * 0.85, h * 0.96])
        panel2.apply_translation([w * 0.48, 0, h * 0.48])
        meshes.append(panel2)
        
        # Back panel
        back = trimesh.creation.box(extents=[w * 0.92, d * 0.02, h * 0.3])
        back.apply_translation([0, -d * 0.42, h * 0.15])
        meshes.append(back)
        
    elif category == "lamp":
        # Base
        base = trimesh.creation.cylinder(radius=w * 0.25, height=h * 0.05)
        base.apply_translation([0, 0, h * 0.025])
        meshes.append(base)
        
        # Pole
        pole = trimesh.creation.cylinder(radius=w * 0.04, height=h * 0.7)
        pole.apply_translation([0, 0, h * 0.4])
        meshes.append(pole)
        
        # Shade
        shade = trimesh.creation.cone(radius=w * 0.35, height=h * 0.25)
        shade.apply_translation([0, 0, h * 0.875])
        meshes.append(shade)
        
    else:
        # Default box
        box = trimesh.creation.box(extents=[w, d, h])
        box.apply_translation([0, 0, h / 2])
        meshes.append(box)
    
    # Combine and color
    mesh = trimesh.util.concatenate(meshes)
    mesh.visual.face_colors = color
    
    return mesh


def create_floor_mesh(floor_mask, room_scale=6.0):
    """Create floor mesh from mask."""
    H, W = floor_mask.shape
    ys, xs = np.where(floor_mask > 0.5)
    
    if len(xs) == 0:
        return None
    
    x_min = (xs.min() / W - 0.5) * room_scale
    x_max = (xs.max() / W - 0.5) * room_scale
    y_min = (ys.min() / H - 0.5) * room_scale
    y_max = (ys.max() / H - 0.5) * room_scale
    
    vertices = np.array([
        [x_min, y_min, 0], [x_max, y_min, 0],
        [x_max, y_max, 0], [x_min, y_max, 0]
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    
    floor = trimesh.Trimesh(vertices=vertices, faces=faces)
    floor.visual.face_colors = [240, 230, 220, 255]  # Light wood color
    
    return floor


def create_walls(floor_mask, room_scale=6.0, wall_height=2.8):
    """Create wall meshes."""
    H, W = floor_mask.shape
    ys, xs = np.where(floor_mask > 0.5)
    
    if len(xs) == 0:
        return []
    
    x_min = (xs.min() / W - 0.5) * room_scale
    x_max = (xs.max() / W - 0.5) * room_scale
    y_min = (ys.min() / H - 0.5) * room_scale
    y_max = (ys.max() / H - 0.5) * room_scale
    
    walls = []
    wall_color = [250, 250, 245, 255]  # Off-white
    
    # Back wall
    back = trimesh.creation.box(extents=[x_max - x_min, 0.1, wall_height])
    back.apply_translation([(x_max + x_min) / 2, y_min - 0.05, wall_height / 2])
    back.visual.face_colors = wall_color
    walls.append(back)
    
    # Left wall
    left = trimesh.creation.box(extents=[0.1, y_max - y_min, wall_height])
    left.apply_translation([x_min - 0.05, (y_max + y_min) / 2, wall_height / 2])
    left.visual.face_colors = wall_color
    walls.append(left)
    
    return walls


def export_scene_to_3d(layout, floor_mask, output_path, room_scale=6.0):
    """Export generated layout to 3D file."""
    scene = trimesh.Scene()
    
    # Add floor
    floor = create_floor_mesh(floor_mask, room_scale)
    if floor:
        scene.add_geometry(floor, node_name='floor')
    
    # Add walls
    walls = create_walls(floor_mask, room_scale)
    for i, wall in enumerate(walls):
        scene.add_geometry(wall, node_name=f'wall_{i}')
    
    # Parse layout
    translations = layout['translations'][0].cpu().numpy()
    sizes = layout['sizes'][0].cpu().numpy()
    angles = layout['angles'][0].cpu().numpy()
    class_probs = layout['class_labels'][0].cpu().numpy()
    
    furniture_list = []
    
    for i in range(translations.shape[0]):
        class_idx = np.argmax(class_probs[i])
        if class_idx >= len(LIVINGROOM_CLASSES) - 2:
            continue
        
        furniture_type = LIVINGROOM_CLASSES[class_idx]
        
        # Convert to world coordinates
        x = translations[i, 0] * room_scale / 2
        y = translations[i, 1] * room_scale / 2
        z = 0
        
        # Scale sizes
        w = np.abs(sizes[i, 0]) * 1.5
        d = np.abs(sizes[i, 1]) * 1.5
        h = np.abs(sizes[i, 2]) * 1.5
        
        w = np.clip(w, 0.3, 2.5)
        d = np.clip(d, 0.3, 2.5)
        h = np.clip(h, 0.3, 2.5)
        
        angle = np.arctan2(angles[i, 1], angles[i, 0])
        
        # Create mesh
        mesh = create_furniture_mesh(furniture_type, [w, d, h])
        
        # Apply rotation
        rot_matrix = rotation_matrix(angle, [0, 0, 1])
        mesh.apply_transform(rot_matrix)
        
        # Apply translation
        mesh.apply_translation([x, y, z])
        
        scene.add_geometry(mesh, node_name=f'{furniture_type}_{i}')
        
        furniture_list.append({
            'type': furniture_type,
            'position': [float(x), float(y), float(z)],
            'size': [float(w), float(d), float(h)],
            'rotation': float(angle)
        })
    
    # Export scene
    scene.export(output_path)
    
    # Also save metadata
    meta_path = output_path.replace('.glb', '_metadata.json').replace('.gltf', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'furniture': furniture_list,
            'room_scale': room_scale
        }, f, indent=2)
    
    return furniture_list


def main():
    parser = argparse.ArgumentParser(description="Export MiDiffusion layouts to 3D")
    parser.add_argument("--model_dir", 
                        default="pretrained/midiffusion/MiDiffusion_model_weights/floor_conditioned/livingroom")
    parser.add_argument("--output_dir", default="output/3d_exports")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--format", choices=['glb', 'gltf', 'obj'], default='glb')
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    config_path = os.path.join(args.model_dir, "config.yaml")
    weight_path = os.path.join(args.model_dir, "model.pt")
    
    print("Loading MiDiffusion model...")
    network, config = load_model(config_path, weight_path, device)
    print("Model loaded!")
    
    floor_shapes = ["rectangular", "l_shaped", "irregular"]
    
    for i in range(args.num_examples):
        print(f"\n--- Example {i+1}/{args.num_examples} ---")
        
        shape = floor_shapes[i % len(floor_shapes)]
        floor_mask = create_synthetic_floor_plan(shape=shape)
        print(f"Floor plan: {shape}")
        
        print("Generating layout...")
        layouts = generate_scene(network, floor_mask, device)
        layout = layouts[0]
        
        output_path = os.path.join(args.output_dir, f"scene_{i+1}_{shape}.{args.format}")
        
        print("Exporting to 3D...")
        furniture_list = export_scene_to_3d(layout, floor_mask, output_path)
        
        print(f"Exported {len(furniture_list)} furniture items:")
        for f in furniture_list:
            print(f"  - {f['type']}")
        print(f"Saved to: {output_path}")
    
    print(f"\n✅ Done! 3D files saved to: {args.output_dir}")
    print("\nTo render in Blender:")
    print("  1. Open Blender")
    print("  2. File > Import > glTF 2.0 (.glb/.gltf)")
    print("  3. Select the exported file")
    print("  4. Add lighting and materials as desired")
    print("  5. Render (F12)")


if __name__ == "__main__":
    main()
