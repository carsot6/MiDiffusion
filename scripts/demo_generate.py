#!/usr/bin/env python
"""
Demo script to generate and visualize living room layouts using MiDiffusion.

This script demonstrates the model with synthetic floor plans when the full
preprocessed dataset is not available.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Approximate bounds from typical living room scenes
DEFAULT_STATS = {
    "bounds_translations": [-4.0, -4.0, 0.0, 4.0, 4.0, 3.0],
    "bounds_sizes": [0.1, 0.1, 0.1, 3.0, 3.0, 3.0],
    "bounds_angles": [-1.0, 1.0],  # cos/sin
    "class_labels": LIVINGROOM_CLASSES,
    "object_types": LIVINGROOM_CLASSES[:-2],  # exclude start/end
    "class_frequencies": {c: 0.05 for c in LIVINGROOM_CLASSES[:-2]},
    "class_order": list(range(len(LIVINGROOM_CLASSES))),
    "count_furniture": {str(i): 100 for i in range(1, 15)}
}


def create_synthetic_floor_plan(shape="rectangular", size=(6, 8)):
    """Create a synthetic floor plan mask."""
    H, W = 64, 64  # Standard size from config
    mask = np.zeros((H, W), dtype=np.float32)
    
    if shape == "rectangular":
        # Simple rectangular room
        margin = 8
        mask[margin:H-margin, margin:W-margin] = 1.0
    elif shape == "l_shaped":
        # L-shaped room
        margin = 8
        mask[margin:H-margin, margin:W//2+10] = 1.0
        mask[margin:H//2+10, margin:W-margin] = 1.0
    elif shape == "irregular":
        # More organic shape
        margin = 10
        mask[margin:H-margin, margin:W-margin] = 1.0
        # Cut out a corner
        mask[margin:margin+15, W-margin-15:W-margin] = 0.0
    
    return mask


def mask_to_boundary_points(mask, num_points=256):
    """
    Convert a floor plan mask to boundary points with normals.
    
    The PointNet_Simple feature extractor expects floor plan boundary points
    in shape [num_points, 4] where 4 = (x, y, nx, ny).
    
    Args:
        mask: 2D numpy array (H, W) with 1s inside the room
        num_points: Number of boundary points to sample
        
    Returns:
        boundary_points: [num_points, 4] array with (x, y, nx, ny)
    """
    from scipy import ndimage
    
    H, W = mask.shape
    
    # Find boundary by edge detection
    # Erode the mask and subtract to get boundary
    eroded = ndimage.binary_erosion(mask > 0.5)
    boundary = (mask > 0.5) & ~eroded
    
    # Get boundary pixel coordinates
    boundary_coords = np.argwhere(boundary)  # (N, 2) in (row, col) = (y, x)
    
    if len(boundary_coords) == 0:
        # Fallback: create rectangular boundary
        boundary_coords = np.array([
            [8, 8], [8, W-8], [H-8, W-8], [H-8, 8]
        ])
    
    # Sample points if we have more than needed
    if len(boundary_coords) > num_points:
        indices = np.linspace(0, len(boundary_coords)-1, num_points, dtype=int)
        boundary_coords = boundary_coords[indices]
    elif len(boundary_coords) < num_points:
        # Repeat points to reach num_points
        repeats = num_points // len(boundary_coords) + 1
        boundary_coords = np.tile(boundary_coords, (repeats, 1))[:num_points]
    
    # Convert to (x, y) and normalize to [-1, 1]
    x = (boundary_coords[:, 1] / W) * 2 - 1  # col -> x
    y = (boundary_coords[:, 0] / H) * 2 - 1  # row -> y
    
    # Compute normals by finite differences along boundary
    # Simple approach: normal perpendicular to boundary direction
    nx = np.zeros(num_points)
    ny = np.zeros(num_points)
    
    for i in range(num_points):
        # Get neighboring points
        prev_idx = (i - 1) % num_points
        next_idx = (i + 1) % num_points
        
        # Tangent direction
        tx = x[next_idx] - x[prev_idx]
        ty = y[next_idx] - y[prev_idx]
        
        # Normal is perpendicular to tangent (rotated 90 degrees)
        length = np.sqrt(tx**2 + ty**2) + 1e-8
        nx[i] = -ty / length
        ny[i] = tx / length
    
    # Stack into [num_points, 4]
    boundary_points = np.stack([x, y, nx, ny], axis=1).astype(np.float32)
    
    return boundary_points


def load_model(config_path, weight_path, device):
    """Load the pretrained MiDiffusion model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create temporary stats file
    stats_dir = os.path.dirname(weight_path)
    stats_path = os.path.join(stats_dir, "dataset_stats.txt")
    if not os.path.exists(stats_path):
        with open(stats_path, 'w') as f:
            json.dump(DEFAULT_STATS, f)
    
    # Update config paths
    config["data"]["dataset_directory"] = stats_dir
    config["data"]["train_stats"] = "dataset_stats.txt"
    
    n_object_types = len(LIVINGROOM_CLASSES) - 2  # exclude start/end
    
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


def generate_scene(network, floor_mask, device, num_steps=1000):
    """Generate a scene layout given a floor plan."""
    batch_size = 1
    
    # Convert floor mask to boundary points for PointNet_Simple
    # The feature extractor expects [batch_size, num_points, 4] where 4 = (x, y, nx, ny)
    boundary_points = mask_to_boundary_points(floor_mask, num_points=256)
    room_feature = torch.from_numpy(boundary_points).unsqueeze(0).float().to(device)
    
    # The model's generate_layout will pass room_feature through its feature_extractor internally
    # So we pass the raw boundary points, not extracted features
    
    # Generate layout
    with torch.no_grad():
        layout = network.generate_layout(
            room_feature=room_feature,
            batch_size=batch_size,
            clip_denoised=True,
            device=device,
        )
    
    return layout


def descale_layout(layout, bounds_trans, bounds_sizes, bounds_angles):
    """Convert normalized layout back to real coordinates."""
    # layout shape: (B, N, D) where D = trans(3) + size(3) + angle(2) + class(C)
    translations = layout[..., :3]
    sizes = layout[..., 3:6]
    angles = layout[..., 6:8]  # cos, sin
    class_probs = layout[..., 8:]
    
    # Descale translations
    trans_min = np.array(bounds_trans[:3])
    trans_max = np.array(bounds_trans[3:])
    translations = translations * (trans_max - trans_min) + trans_min
    
    # Descale sizes
    size_min = np.array(bounds_sizes[:3])
    size_max = np.array(bounds_sizes[3:])
    sizes = sizes * (size_max - size_min) + size_min
    
    return translations, sizes, angles, class_probs


def visualize_results(floor_mask, layout, class_labels, output_path, example_num):
    """Create visualization of floor plan input and generated furniture layout.
    
    Args:
        floor_mask: 2D numpy array (H, W) with floor plan
        layout: dict with keys 'translations', 'sizes', 'angles', 'class_labels'
                each value is tensor of shape [1, N, ?]
        class_labels: list of class names
        output_path: path to save the visualization
        example_num: example number for title
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color map for furniture types
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_labels)))
    
    # --- Panel 1: Input Floor Plan ---
    ax1 = axes[0]
    ax1.imshow(floor_mask, cmap='gray_r', origin='lower')
    ax1.set_title(f'Input: Floor Plan (Example {example_num})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    
    # --- Panel 2: Generated Layout (Top-Down) ---
    ax2 = axes[1]
    ax2.imshow(floor_mask, cmap='gray_r', alpha=0.3, origin='lower')
    
    # Parse layout from dictionary format
    # layout is a dict with tensors of shape [1, N, ?]
    translations = layout['translations'][0].cpu().numpy()  # [N, 3]
    sizes = layout['sizes'][0].cpu().numpy()  # [N, 3]
    angles = layout['angles'][0].cpu().numpy()  # [N, 2]
    class_probs = layout['class_labels'][0].cpu().numpy()  # [N, num_classes]
    
    num_objects = translations.shape[0]
    print(f"  Generated {num_objects} furniture objects")
    
    H, W = floor_mask.shape
    
    # Scale translations from normalized [-1,1] to image space
    # The model outputs in normalized coordinates
    translations_scaled = np.zeros_like(translations)
    translations_scaled[:, 0] = (translations[:, 0] + 1) / 2 * W  # x
    translations_scaled[:, 1] = (translations[:, 1] + 1) / 2 * H  # y
    
    # Scale sizes (normalized to [0,1] range typically)
    sizes_scaled = np.zeros_like(sizes)
    sizes_scaled[:, 0] = np.abs(sizes[:, 0]) * W * 0.15  # width
    sizes_scaled[:, 1] = np.abs(sizes[:, 1]) * H * 0.15  # height
    sizes_scaled[:, 2] = np.abs(sizes[:, 2]) * 2.0  # height (for 3D)
    
    legend_handles = []
    legend_labels = []
    
    for i in range(num_objects):
        class_idx = np.argmax(class_probs[i])
        if class_idx >= len(class_labels) - 2:  # skip start/end
            continue
            
        x, y = translations_scaled[i, 0], translations_scaled[i, 1]
        w, h = max(sizes_scaled[i, 0], 3), max(sizes_scaled[i, 1], 3)  # minimum size
        
        # Get rotation angle from cos/sin
        angle = np.arctan2(angles[i, 1], angles[i, 0]) * 180 / np.pi
        
        color = colors[class_idx % len(colors)]
        
        # Draw rotated rectangle
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='black', alpha=0.7, linewidth=1
        )
        
        # Apply rotation
        t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(x, y, angle) + ax2.transData
        rect.set_transform(t)
        ax2.add_patch(rect)
        
        # Add to legend (only once per class)
        class_name = class_labels[class_idx]
        if class_name not in legend_labels:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7))
            legend_labels.append(class_name)
    
    ax2.set_xlim(0, W)
    ax2.set_ylim(0, H)
    ax2.set_title(f'Output: Generated Layout (Top-Down)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')
    
    # --- Panel 3: 3D Perspective View ---
    ax3 = axes[2]
    ax3.set_title(f'Output: 3D Perspective View', fontsize=14, fontweight='bold')
    
    # Create simple 3D-like view
    for i in range(num_objects):
        class_idx = np.argmax(class_probs[i])
        if class_idx >= len(class_labels) - 2:
            continue
            
        x, y = translations_scaled[i, 0], translations_scaled[i, 1]
        w, h = max(sizes_scaled[i, 0], 3), max(sizes_scaled[i, 1], 3)
        z = max(sizes_scaled[i, 2], 0.3)
        
        color = colors[class_idx % len(colors)]
        
        # Simple isometric projection
        iso_x = x - y * 0.3
        iso_y = y * 0.5 + z * 10
        
        rect = FancyBboxPatch(
            (iso_x - w/2, iso_y), w, z * 10,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='black', alpha=0.7, linewidth=1
        )
        ax3.add_patch(rect)
    
    # Draw floor outline
    floor_y, floor_x = np.where(floor_mask > 0.5)
    if len(floor_x) > 0:
        ax3.fill(
            [floor_x.min() - floor_y.min()*0.3, floor_x.max() - floor_y.min()*0.3,
             floor_x.max() - floor_y.max()*0.3, floor_x.min() - floor_y.max()*0.3],
            [floor_y.min()*0.5, floor_y.min()*0.5, floor_y.max()*0.5, floor_y.max()*0.5],
            alpha=0.2, color='gray'
        )
    
    ax3.set_xlim(-20, W+20)
    ax3.set_ylim(-10, H+40)
    ax3.axis('off')
    
    # Add legend
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='center right', 
                   bbox_to_anchor=(1.15, 0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo: Generate living room layouts")
    parser.add_argument("--model_dir", 
                        default="pretrained/midiffusion/MiDiffusion_model_weights/floor_conditioned/livingroom",
                        help="Directory containing model.pt and config.yaml")
    parser.add_argument("--output_dir", default="output/demo_results",
                        help="Output directory for visualizations")
    parser.add_argument("--num_examples", type=int, default=2,
                        help="Number of examples to generate")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of diffusion steps (fewer = faster but lower quality)")
    parser.add_argument("--device", default="cpu",
                        help="Device to use (cpu or cuda)")
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    config_path = os.path.join(args.model_dir, "config.yaml")
    weight_path = os.path.join(args.model_dir, "model.pt")
    
    if not os.path.exists(weight_path):
        print(f"Error: Model weights not found at {weight_path}")
        print("Please download from: https://drive.google.com/drive/folders/14N87Ap90KNaDlRv5u6UeCV1h_MT9QqaN")
        return
    
    # Load model
    print("Loading MiDiffusion model...")
    try:
        network, config = load_model(config_path, weight_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nFalling back to visualization-only demo...")
        network = None
    
    # Generate examples with different floor plans
    floor_shapes = ["rectangular", "l_shaped", "irregular"]
    
    for i in range(args.num_examples):
        print(f"\n--- Example {i+1}/{args.num_examples} ---")
        
        # Create floor plan
        shape = floor_shapes[i % len(floor_shapes)]
        floor_mask = create_synthetic_floor_plan(shape=shape)
        print(f"Created {shape} floor plan")
        
        if network is not None:
            # Generate layout using model
            print(f"Generating layout with {args.num_steps} diffusion steps...")
            layouts = generate_scene(network, floor_mask, device, args.num_steps)
            layout = layouts[0]  # Get first layout from list
        else:
            # Create dummy layout for visualization demo (dict format)
            print("Creating demo layout (model not loaded)...")
            n_objects = np.random.randint(5, 12)
            n_classes = len(LIVINGROOM_CLASSES)
            
            translations = torch.zeros(1, n_objects, 3)
            sizes = torch.zeros(1, n_objects, 3)
            angles = torch.zeros(1, n_objects, 2)
            class_labels_tensor = torch.zeros(1, n_objects, n_classes)
            
            # Random positions within floor
            floor_y, floor_x = np.where(floor_mask > 0.5)
            for j in range(n_objects):
                idx = np.random.randint(len(floor_x))
                translations[0, j, 0] = (floor_x[idx] / 64.0) * 2 - 1  # x in [-1, 1]
                translations[0, j, 1] = (floor_y[idx] / 64.0) * 2 - 1  # y in [-1, 1]
                translations[0, j, 2] = 0.0  # z
                sizes[0, j] = torch.rand(3) * 0.3 + 0.1  # size
                angle = np.random.rand() * 2 * np.pi
                angles[0, j, 0] = np.cos(angle)  # cos
                angles[0, j, 1] = np.sin(angle)  # sin
                class_idx = np.random.randint(n_classes - 2)
                class_labels_tensor[0, j, class_idx] = 1.0
            
            layout = {
                'translations': translations,
                'sizes': sizes,
                'angles': angles,
                'class_labels': class_labels_tensor
            }
        
        # Visualize
        output_path = os.path.join(args.output_dir, f"example_{i+1}_{shape}.png")
        visualize_results(floor_mask, layout, LIVINGROOM_CLASSES, output_path, i+1)
    
    print(f"\n✅ Done! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
