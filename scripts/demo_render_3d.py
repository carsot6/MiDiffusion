#!/usr/bin/env python
"""
Demo script to generate and render living room layouts using MiDiffusion.

This script generates furniture layouts and renders them with actual 3D furniture
meshes using trimesh for visualization.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import yaml
from PIL import Image
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import trimesh
    from trimesh.transformations import rotation_matrix
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not installed. Install with: pip install trimesh")

try:
    import pyrender
    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False
    print("Warning: pyrender not installed. Install with: pip install pyrender")

from scipy import ndimage

from midiffusion.networks.feature_extractors import get_feature_extractor
from midiffusion.networks.diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed


# Living room class labels (from 3D-FRONT)
LIVINGROOM_CLASSES = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa",
    "chinese_chair", "coffee_table", "console_table", "corner_side_table",
    "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp",
    "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet",
    "start", "end"
]

# Map furniture classes to simplified categories for mesh selection
FURNITURE_CATEGORY_MAP = {
    "armchair": "chair",
    "bookshelf": "shelf",
    "cabinet": "cabinet",
    "ceiling_lamp": "lamp",
    "chaise_longue_sofa": "sofa",
    "chinese_chair": "chair",
    "coffee_table": "table",
    "console_table": "table",
    "corner_side_table": "table",
    "desk": "desk",
    "dining_chair": "chair",
    "dining_table": "table",
    "l_shaped_sofa": "sofa",
    "lazy_sofa": "sofa",
    "lounge_chair": "chair",
    "loveseat_sofa": "sofa",
    "multi_seat_sofa": "sofa",
    "pendant_lamp": "lamp",
    "round_end_table": "table",
    "shelf": "shelf",
    "stool": "chair",
    "tv_stand": "cabinet",
    "wardrobe": "cabinet",
    "wine_cabinet": "cabinet",
}

# Approximate stats for descaling
DEFAULT_STATS = {
    "bounds_translations": [-4.0, -4.0, 0.0, 4.0, 4.0, 3.0],
    "bounds_sizes": [0.1, 0.1, 0.1, 3.0, 3.0, 3.0],
    "bounds_angles": [-1.0, 1.0],
    "class_labels": LIVINGROOM_CLASSES,
}


def create_synthetic_floor_plan(shape="rectangular"):
    """Create a synthetic floor plan mask."""
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
    """Convert floor plan mask to boundary points with normals."""
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
    """Load the pretrained MiDiffusion model."""
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
        n_object_types,
        feature_extractor,
        config["network"],
        stats_path
    )
    
    print(f"Loading weights from {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()
    
    return network, config


def generate_scene(network, floor_mask, device):
    """Generate a scene layout given a floor plan."""
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


def create_procedural_furniture_mesh(furniture_type, size):
    """
    Create a simple procedural 3D mesh for a furniture type.
    Returns a trimesh object.
    """
    w, d, h = size  # width, depth, height
    
    category = FURNITURE_CATEGORY_MAP.get(furniture_type, "box")
    
    if category == "sofa":
        # Create a sofa-like shape: base + back
        base = trimesh.creation.box(extents=[w, d, h * 0.4])
        base.apply_translation([0, 0, h * 0.2])
        
        back = trimesh.creation.box(extents=[w, d * 0.2, h * 0.6])
        back.apply_translation([0, -d * 0.4, h * 0.5])
        
        # Arms
        arm1 = trimesh.creation.box(extents=[w * 0.1, d * 0.8, h * 0.5])
        arm1.apply_translation([-w * 0.45, 0, h * 0.3])
        
        arm2 = trimesh.creation.box(extents=[w * 0.1, d * 0.8, h * 0.5])
        arm2.apply_translation([w * 0.45, 0, h * 0.3])
        
        mesh = trimesh.util.concatenate([base, back, arm1, arm2])
        
    elif category == "chair":
        # Chair: seat + back + legs
        seat = trimesh.creation.box(extents=[w, d, h * 0.1])
        seat.apply_translation([0, 0, h * 0.45])
        
        back = trimesh.creation.box(extents=[w, d * 0.1, h * 0.5])
        back.apply_translation([0, -d * 0.45, h * 0.75])
        
        # Legs
        leg_h = h * 0.4
        leg_r = min(w, d) * 0.05
        legs = []
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            leg = trimesh.creation.cylinder(radius=leg_r, height=leg_h)
            leg.apply_translation([dx * w * 0.4, dy * d * 0.4, leg_h/2])
            legs.append(leg)
        
        mesh = trimesh.util.concatenate([seat, back] + legs)
        
    elif category == "table":
        # Table: top + legs
        top = trimesh.creation.box(extents=[w, d, h * 0.1])
        top.apply_translation([0, 0, h * 0.95])
        
        leg_h = h * 0.9
        leg_r = min(w, d) * 0.05
        legs = []
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            leg = trimesh.creation.cylinder(radius=leg_r, height=leg_h)
            leg.apply_translation([dx * w * 0.4, dy * d * 0.4, leg_h/2])
            legs.append(leg)
        
        mesh = trimesh.util.concatenate([top] + legs)
        
    elif category == "cabinet" or category == "shelf":
        # Cabinet: box with slight inset
        outer = trimesh.creation.box(extents=[w, d, h])
        outer.apply_translation([0, 0, h/2])
        mesh = outer
        
    elif category == "desk":
        # Desk: top + side panels
        top = trimesh.creation.box(extents=[w, d, h * 0.05])
        top.apply_translation([0, 0, h * 0.95])
        
        panel1 = trimesh.creation.box(extents=[w * 0.05, d * 0.8, h * 0.9])
        panel1.apply_translation([-w * 0.45, 0, h * 0.45])
        
        panel2 = trimesh.creation.box(extents=[w * 0.05, d * 0.8, h * 0.9])
        panel2.apply_translation([w * 0.45, 0, h * 0.45])
        
        mesh = trimesh.util.concatenate([top, panel1, panel2])
        
    elif category == "lamp":
        # Lamp: pole + shade
        pole = trimesh.creation.cylinder(radius=w * 0.1, height=h * 0.8)
        pole.apply_translation([0, 0, h * 0.4])
        
        shade = trimesh.creation.cone(radius=w * 0.4, height=h * 0.3)
        shade.apply_translation([0, 0, h * 0.85])
        
        mesh = trimesh.util.concatenate([pole, shade])
        
    else:
        # Default: simple box
        mesh = trimesh.creation.box(extents=[w, d, h])
        mesh.apply_translation([0, 0, h/2])
    
    return mesh


def create_floor_mesh(floor_mask, scale=6.0):
    """Create a floor plane mesh from the floor mask."""
    H, W = floor_mask.shape
    
    # Find floor boundaries
    ys, xs = np.where(floor_mask > 0.5)
    if len(xs) == 0:
        return None
    
    x_min, x_max = xs.min() / W * scale - scale/2, xs.max() / W * scale - scale/2
    y_min, y_max = ys.min() / H * scale - scale/2, ys.max() / H * scale - scale/2
    
    # Create floor plane
    vertices = np.array([
        [x_min, y_min, 0],
        [x_max, y_min, 0],
        [x_max, y_max, 0],
        [x_min, y_max, 0]
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    
    floor = trimesh.Trimesh(vertices=vertices, faces=faces)
    floor.visual.face_colors = [200, 200, 200, 255]  # Light gray floor
    
    return floor


def render_scene_trimesh(meshes, floor_mesh, output_path, camera_angle='perspective'):
    """
    Render a scene using trimesh's built-in viewer or export to image.
    """
    # Combine all meshes into a scene
    scene = trimesh.Scene()
    
    # Add floor
    if floor_mesh is not None:
        scene.add_geometry(floor_mesh, node_name='floor')
    
    # Add furniture
    for i, mesh in enumerate(meshes):
        scene.add_geometry(mesh, node_name=f'furniture_{i}')
    
    # Try to render using pyrender if available
    if HAS_PYRENDER:
        try:
            # Convert to pyrender scene
            pr_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
            
            # Add meshes
            for name, geom in scene.geometry.items():
                mesh = pyrender.Mesh.from_trimesh(geom)
                pr_scene.add(mesh)
            
            # Add lights
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            pr_scene.add(light, pose=np.eye(4))
            
            # Add camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            
            if camera_angle == 'top':
                camera_pose = np.array([
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 8],
                    [0, 0, 0, 1]
                ])
            else:  # perspective
                camera_pose = np.array([
                    [0.7, 0, 0.7, 5],
                    [0, 1, 0, 2],
                    [-0.7, 0, 0.7, 5],
                    [0, 0, 0, 1]
                ])
            
            pr_scene.add(camera, pose=camera_pose)
            
            # Render
            renderer = pyrender.OffscreenRenderer(800, 600)
            color, depth = renderer.render(pr_scene)
            renderer.delete()
            
            # Save image
            Image.fromarray(color).save(output_path)
            return True
            
        except Exception as e:
            print(f"Pyrender failed: {e}, falling back to trimesh export")
    
    # Fallback: export scene and use trimesh's scene.save
    try:
        # Export as PNG using trimesh's built-in (requires pyglet)
        png_data = scene.save_image(resolution=[800, 600], visible=False)
        with open(output_path, 'wb') as f:
            f.write(png_data)
        return True
    except Exception as e:
        print(f"Trimesh rendering failed: {e}")
        return False


def visualize_with_3d_furniture(floor_mask, layout, class_labels, output_path, example_num):
    """
    Create visualization with procedural 3D furniture.
    """
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_labels)))
    
    fig = plt.figure(figsize=(20, 8))
    
    # Parse layout
    translations = layout['translations'][0].cpu().numpy()
    sizes = layout['sizes'][0].cpu().numpy()
    angles = layout['angles'][0].cpu().numpy()
    class_probs = layout['class_labels'][0].cpu().numpy()
    
    num_objects = translations.shape[0]
    H, W = floor_mask.shape
    
    # Scale factors (normalized coords to world coords)
    world_scale = 6.0  # 6 meters room
    
    # Convert to world coordinates
    trans_world = np.zeros((num_objects, 3))
    trans_world[:, 0] = translations[:, 0] * world_scale / 2  # x
    trans_world[:, 1] = translations[:, 1] * world_scale / 2  # y
    trans_world[:, 2] = 0  # z at ground
    
    # Scale sizes
    sizes_world = np.abs(sizes) * 1.5  # scale factor for visibility
    sizes_world = np.clip(sizes_world, 0.2, 2.0)  # min/max size
    
    # ----- Panel 1: Input Floor Plan -----
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(floor_mask, cmap='gray_r', origin='lower')
    ax1.set_title(f'Input: Floor Plan (Example {example_num})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # ----- Panel 2: Top-Down Layout with Furniture Icons -----
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(floor_mask, cmap='gray_r', alpha=0.3, origin='lower')
    
    legend_handles = []
    legend_labels = []
    furniture_info = []
    
    for i in range(num_objects):
        class_idx = np.argmax(class_probs[i])
        if class_idx >= len(class_labels) - 2:
            continue
        
        furniture_type = class_labels[class_idx]
        
        # Image coordinates
        x_img = (translations[i, 0] + 1) / 2 * W
        y_img = (translations[i, 1] + 1) / 2 * H
        w_img = sizes_world[i, 0] / world_scale * W
        h_img = sizes_world[i, 1] / world_scale * H
        
        angle_rad = np.arctan2(angles[i, 1], angles[i, 0])
        
        color = colors[class_idx % len(colors)]
        
        # Draw furniture representation
        rect = FancyBboxPatch(
            (x_img - w_img/2, y_img - h_img/2), w_img, h_img,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='black', alpha=0.8, linewidth=1.5
        )
        t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(
            x_img, y_img, np.degrees(angle_rad)) + ax2.transData
        rect.set_transform(t)
        ax2.add_patch(rect)
        
        # Add furniture label
        ax2.annotate(furniture_type[:6], (x_img, y_img), fontsize=6, 
                    ha='center', va='center', color='white', fontweight='bold')
        
        if furniture_type not in legend_labels:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8))
            legend_labels.append(furniture_type)
        
        furniture_info.append({
            'type': furniture_type,
            'pos': trans_world[i],
            'size': sizes_world[i],
            'angle': angle_rad,
            'color': color
        })
    
    ax2.set_xlim(0, W)
    ax2.set_ylim(0, H)
    ax2.set_title('Output: Top-Down Layout', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # ----- Panel 3: 3D Rendered View -----
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Render 3D furniture
    for info in furniture_info:
        x, y, z = info['pos']
        w, d, h = info['size']
        angle = info['angle']
        color = info['color'][:3]
        
        # Create simple 3D box representation
        category = FURNITURE_CATEGORY_MAP.get(info['type'], 'box')
        
        # Vertices of a unit box
        if category == 'sofa':
            # Sofa: lower and wider
            h = h * 0.5
        elif category == 'table':
            # Table: thinner top
            pass
        elif category == 'chair':
            # Chair: taller back
            pass
        
        # Simple 3D box
        xs = np.array([0, w, w, 0, 0, w, w, 0]) - w/2
        ys = np.array([0, 0, d, d, 0, 0, d, d]) - d/2
        zs = np.array([0, 0, 0, 0, h, h, h, h])
        
        # Apply rotation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xs_rot = xs * cos_a - ys * sin_a
        ys_rot = xs * sin_a + ys * cos_a
        
        # Translate
        xs_rot += x
        ys_rot += y
        
        # Draw box faces
        verts = [
            [list(zip(xs_rot[:4], ys_rot[:4], zs[:4]))],  # bottom
            [list(zip(xs_rot[4:], ys_rot[4:], zs[4:]))],  # top
        ]
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Bottom face
        ax3.add_collection3d(Poly3DCollection(
            [list(zip(xs_rot[:4], ys_rot[:4], zs[:4]))],
            facecolors=[color], alpha=0.7, edgecolors='black', linewidths=0.5))
        
        # Top face
        ax3.add_collection3d(Poly3DCollection(
            [list(zip(xs_rot[4:], ys_rot[4:], zs[4:]))],
            facecolors=[color], alpha=0.7, edgecolors='black', linewidths=0.5))
        
        # Side faces
        for j in range(4):
            k = (j + 1) % 4
            face = [
                (xs_rot[j], ys_rot[j], zs[j]),
                (xs_rot[k], ys_rot[k], zs[k]),
                (xs_rot[k+4], ys_rot[k+4], zs[k+4]),
                (xs_rot[j+4], ys_rot[j+4], zs[j+4])
            ]
            ax3.add_collection3d(Poly3DCollection(
                [face], facecolors=[color], alpha=0.7, edgecolors='black', linewidths=0.5))
    
    # Draw floor
    floor_x = np.array([-3, 3, 3, -3])
    floor_y = np.array([-3, -3, 3, 3])
    floor_z = np.zeros(4)
    ax3.add_collection3d(Poly3DCollection(
        [list(zip(floor_x, floor_y, floor_z))],
        facecolors=['lightgray'], alpha=0.5, edgecolors='gray'))
    
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.set_zlim(0, 3)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('Output: 3D View', fontsize=14, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    
    # Legend
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='center right',
                  bbox_to_anchor=(1.02, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved visualization to: {output_path}")
    return furniture_info


def main():
    parser = argparse.ArgumentParser(description="Demo: Generate and render living room layouts")
    parser.add_argument("--model_dir", 
                        default="pretrained/midiffusion/MiDiffusion_model_weights/floor_conditioned/livingroom",
                        help="Directory containing model.pt and config.yaml")
    parser.add_argument("--output_dir", default="output/demo_results_3d",
                        help="Output directory for visualizations")
    parser.add_argument("--num_examples", type=int, default=2,
                        help="Number of examples to generate")
    parser.add_argument("--device", default="cpu",
                        help="Device to use (cpu or cuda)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    config_path = os.path.join(args.model_dir, "config.yaml")
    weight_path = os.path.join(args.model_dir, "model.pt")
    
    if not os.path.exists(weight_path):
        print(f"Error: Model weights not found at {weight_path}")
        return
    
    print("Loading MiDiffusion model...")
    try:
        network, config = load_model(config_path, weight_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    floor_shapes = ["rectangular", "l_shaped", "irregular"]
    
    for i in range(args.num_examples):
        print(f"\n{'='*50}")
        print(f"Example {i+1}/{args.num_examples}")
        print(f"{'='*50}")
        
        shape = floor_shapes[i % len(floor_shapes)]
        floor_mask = create_synthetic_floor_plan(shape=shape)
        print(f"  Floor plan: {shape}")
        
        print("  Generating layout...")
        layouts = generate_scene(network, floor_mask, device)
        layout = layouts[0]
        
        num_objects = layout['translations'].shape[1]
        print(f"  Generated {num_objects} furniture objects")
        
        # Print furniture details
        class_probs = layout['class_labels'][0].cpu().numpy()
        for j in range(num_objects):
            class_idx = np.argmax(class_probs[j])
            if class_idx < len(LIVINGROOM_CLASSES) - 2:
                print(f"    - {LIVINGROOM_CLASSES[class_idx]}")
        
        output_path = os.path.join(args.output_dir, f"example_{i+1}_{shape}.png")
        visualize_with_3d_furniture(floor_mask, layout, LIVINGROOM_CLASSES, output_path, i+1)
    
    print(f"\n✅ Done! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
