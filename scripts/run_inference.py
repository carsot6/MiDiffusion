#!/usr/bin/env python
"""
MiDiffusion Inference Script — Self-contained scene layout generation.

Generates 3D indoor scene layouts using pretrained MiDiffusion models WITHOUT
requiring the full ThreedFront preprocessing pipeline. Works with:
  - Synthetic floor plans (rectangular, L-shaped, irregular)
  - Locally downloaded pretrained weights from Google Drive

Usage:
    python scripts/run_inference.py --room_type livingroom --num_scenes 5
    python scripts/run_inference.py --room_type bedroom --device mps
    python scripts/run_inference.py --room_type livingroom --floor_shape l_shaped

Requirements:
    - Pretrained weights in pretrained_local/MiDiffusion_model_weights/floor_conditioned/<room_type>/
    - Python packages: torch, numpy, scipy, matplotlib, yaml
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import yaml
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from scipy import ndimage

# Add project root to path
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)

from midiffusion.networks.feature_extractors import get_feature_extractor
from midiffusion.networks.diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed

# ============================================================================
# Room type configurations
# ============================================================================

LIVINGROOM_LABELS = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa",
    "chinese_chair", "coffee_table", "console_table", "corner_side_table",
    "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp",
    "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet",
    "start", "end"
]

BEDROOM_LABELS = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair",
    "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair",
    "dressing_table", "kids_bed", "nightstand", "pendant_lamp", "shelf",
    "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe",
    "start", "end"
]

DININGROOM_LABELS = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa",
    "chinese_chair", "coffee_table", "console_table", "corner_side_table",
    "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp",
    "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet",
    "start", "end"
]

ROOM_CONFIGS = {
    "livingroom": {
        "labels": LIVINGROOM_LABELS,
        "room_side": 6.1,
        "bounds_translations": [-5.0, -5.0, 0.0, 5.0, 5.0, 4.0],
        "bounds_sizes": [0.05, 0.05, 0.05, 4.5, 4.0, 3.5],
        "bounds_angles": [-3.14159, 3.14159],
    },
    "bedroom": {
        "labels": BEDROOM_LABELS,
        "room_side": 3.1,
        "bounds_translations": [-3.5, -3.5, 0.0, 3.5, 3.5, 3.0],
        "bounds_sizes": [0.05, 0.05, 0.05, 3.0, 3.0, 2.5],
        "bounds_angles": [-3.14159, 3.14159],
    },
    "diningroom": {
        "labels": DININGROOM_LABELS,
        "room_side": 6.1,
        "bounds_translations": [-5.0, -5.0, 0.0, 5.0, 5.0, 4.0],
        "bounds_sizes": [0.05, 0.05, 0.05, 4.5, 4.0, 3.5],
        "bounds_angles": [-3.14159, 3.14159],
    },
}

# Color map for furniture categories (for visualization)
CATEGORY_COLORS = {
    "armchair": "#4682B4",
    "bookshelf": "#8B4513",
    "cabinet": "#CD853F",
    "ceiling_lamp": "#FFD700",
    "chair": "#A0522D",
    "chaise_longue_sofa": "#4169E1",
    "chinese_chair": "#8B6914",
    "coffee_table": "#228B22",
    "console_table": "#2E8B57",
    "corner_side_table": "#3CB371",
    "desk": "#DEB887",
    "dining_chair": "#A0522D",
    "dining_table": "#006400",
    "double_bed": "#6495ED",
    "dressing_chair": "#D2691E",
    "dressing_table": "#BC8F8F",
    "kids_bed": "#87CEEB",
    "l_shaped_sofa": "#1E90FF",
    "lazy_sofa": "#4169E1",
    "lounge_chair": "#8B4513",
    "loveseat_sofa": "#5B9BD5",
    "multi_seat_sofa": "#4682B4",
    "nightstand": "#8B7355",
    "pendant_lamp": "#FFC125",
    "round_end_table": "#2E8B57",
    "shelf": "#A0522D",
    "single_bed": "#87CEFA",
    "stool": "#D2691E",
    "table": "#3CB371",
    "tv_stand": "#696969",
    "wardrobe": "#8B6914",
    "wine_cabinet": "#722F37",
}


# ============================================================================
# Floor plan generation
# ============================================================================

def create_floor_plan(shape="rectangular", size=(64, 64), room_dims=None):
    """Create a synthetic floor plan mask.
    
    Args:
        shape: 'rectangular', 'l_shaped', 'irregular', 'large_rectangular'
        size: (H, W) mask dimensions
        room_dims: optional (width_m, height_m) room dimensions in meters
    
    Returns:
        mask: binary numpy array of shape (H, W)
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.float32)
    
    if shape == "rectangular":
        margin = 8
        mask[margin:H-margin, margin:W-margin] = 1.0
    elif shape == "large_rectangular":
        margin = 4
        mask[margin:H-margin, margin:W-margin] = 1.0
    elif shape == "l_shaped":
        margin = 8
        mask[margin:H-margin, margin:W//2+10] = 1.0
        mask[margin:H//2+10, margin:W-margin] = 1.0
    elif shape == "irregular":
        margin = 10
        mask[margin:H-margin, margin:W-margin] = 1.0
        # Cut out a corner
        mask[margin:margin+15, W-margin-15:W-margin] = 0.0
    elif shape == "t_shaped":
        margin = 8
        mask[margin:H-margin, W//4:3*W//4] = 1.0  # vertical bar
        mask[margin:H//3, margin:W-margin] = 1.0    # top horizontal bar
    else:
        raise ValueError(f"Unknown floor plan shape: {shape}")
    
    return mask


def mask_to_boundary_points(mask, num_points=256, room_side=6.1):
    """Convert floor plan mask to boundary points with normals for PointNet.
    
    The PointNet_Simple feature extractor expects [num_points, 4] where 4 = (x, y, nx, ny).
    
    IMPORTANT: During training, boundary points are SCALED to [-1, 1] by 
    Scale_CosinAngle using bounds_fpbpn. So we output points already in [-1, 1].
    
    Args:
        mask: 2D binary array (H, W)
        num_points: number of boundary points to sample
        room_side: room side length in meters (6.1 for living/dining, 3.1 for bedroom)
    
    Returns:
        boundary_points: [num_points, 4] array with (x, y, nx, ny) in [-1, 1]
    """
    H, W = mask.shape
    
    # Find boundary pixels
    eroded = ndimage.binary_erosion(mask > 0.5)
    boundary = (mask > 0.5) & ~eroded
    boundary_coords = np.argwhere(boundary)  # (N, 2) in (row, col)
    
    if len(boundary_coords) == 0:
        # Fallback
        boundary_coords = np.array([[H//4, W//4], [H//4, 3*W//4], 
                                     [3*H//4, 3*W//4], [3*H//4, W//4]])
    
    # Order boundary points by angle from centroid
    center = boundary_coords.mean(axis=0)
    angles = np.arctan2(boundary_coords[:, 0] - center[0], 
                        boundary_coords[:, 1] - center[1])
    order = np.argsort(angles)
    boundary_coords = boundary_coords[order]
    
    # Resample to desired number of points
    if len(boundary_coords) > num_points:
        indices = np.linspace(0, len(boundary_coords)-1, num_points, dtype=int)
        boundary_coords = boundary_coords[indices]
    elif len(boundary_coords) < num_points:
        repeats = num_points // len(boundary_coords) + 1
        boundary_coords = np.tile(boundary_coords, (repeats, 1))[:num_points]
    
    # Convert pixel coords directly to [-1, 1] (normalized, as expected by model)
    # During training, raw coords in meters are scaled to [-1, 1] via bounds_fpbpn
    # We generate boundary points directly in [-1, 1] to match training input
    x = (boundary_coords[:, 1] / W) * 2 - 1   # col -> x in [-1, 1]
    y = (boundary_coords[:, 0] / H) * 2 - 1   # row -> y in [-1, 1]
    
    # Compute outward normals (already in [-1, 1] naturally)
    nx = np.zeros(num_points)
    ny = np.zeros(num_points)
    for i in range(num_points):
        prev_idx = (i - 1) % num_points
        next_idx = (i + 1) % num_points
        tx = x[next_idx] - x[prev_idx]
        ty = y[next_idx] - y[prev_idx]
        length = np.sqrt(tx**2 + ty**2) + 1e-8
        nx[i] = -ty / length  # perpendicular
        ny[i] = tx / length
    
    return np.stack([x, y, nx, ny], axis=1).astype(np.float32)


# ============================================================================
# Dataset stats generation
# ============================================================================

def create_dataset_stats(room_type, output_path):
    """Create a dataset_stats.txt file for the given room type.
    
    This file is required by the DiffusionSceneLayout model to initialize
    properly. It contains bounding box statistics from the training data.
    """
    cfg = ROOM_CONFIGS[room_type]
    labels = cfg["labels"]
    object_types = labels[:-2]  # exclude start/end
    
    stats = {
        "bounds_translations": cfg["bounds_translations"],
        "bounds_sizes": cfg["bounds_sizes"],
        "bounds_angles": cfg["bounds_angles"],
        "class_labels": labels,
        "object_types": object_types,
        "class_frequencies": {c: round(1.0/len(object_types), 4) 
                              for c in object_types},
        "class_order": list(range(len(labels))),
        "count_furniture": {str(i): 100 for i in range(1, 25)},
        "bounds_fpbpn": [-10.0, -10.0, -1.0, -1.0, 10.0, 10.0, 1.0, 1.0],
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


# ============================================================================
# Model loading
# ============================================================================

def load_model(room_type, model_dir, device):
    """Load pretrained MiDiffusion model.
    
    Args:
        room_type: 'livingroom', 'bedroom', or 'diningroom'
        model_dir: path to directory containing model.pt and config.yaml
        device: torch device
    
    Returns:
        network: loaded model
        config: model configuration dict
    """
    config_path = os.path.join(model_dir, "config.yaml")
    weight_path = os.path.join(model_dir, "model.pt")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Model weights not found at {weight_path}\n"
            f"Download from: https://drive.google.com/drive/folders/14N87Ap90KNaDlRv5u6UeCV1h_MT9QqaN"
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset_stats.txt in model directory if missing
    stats_path = os.path.join(model_dir, "dataset_stats.txt")
    if not os.path.exists(stats_path):
        print(f"Creating dataset_stats.txt for {room_type}...")
        create_dataset_stats(room_type, stats_path)
    
    cfg = ROOM_CONFIGS[room_type]
    labels = cfg["labels"]
    n_object_types = len(labels) - 2  # exclude start/end
    
    # Build feature extractor
    feature_extractor = get_feature_extractor(**config["feature_extractor"])
    
    # Build network
    network = DiffusionSceneLayout_Mixed(
        n_object_types,
        feature_extractor,
        config["network"],
        stats_path
    )
    
    print(f"Loading weights from {weight_path}...")
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()
    
    print(f"Model loaded: {n_object_types} object types, "
          f"max {config['network']['sample_num_points']} objects/scene")
    
    return network, config


# ============================================================================
# Layout generation
# ============================================================================

@torch.no_grad()
def generate_layouts(network, floor_plans, room_type, device, batch_size=4):
    """Generate scene layouts from floor plan boundary points.
    
    Args:
        network: loaded MiDiffusion model
        floor_plans: list of (mask, boundary_points) tuples
        room_type: room type string
        device: torch device
        batch_size: batch size for inference
    
    Returns:
        layouts: list of dicts with keys 'class_labels', 'translations', 'sizes', 'angles'
    """
    cfg = ROOM_CONFIGS[room_type]
    labels = cfg["labels"]
    
    layouts = []
    for i in range(0, len(floor_plans), batch_size):
        batch = floor_plans[i:i+batch_size]
        B = len(batch)
        
        # Stack boundary points into batch
        room_feature = torch.from_numpy(
            np.stack([bp for _, bp in batch], axis=0)
        ).float().to(device)
        
        # Generate layout via reverse diffusion
        t0 = time.time()
        bbox_params_list = network.generate_layout(
            room_feature=room_feature,
            batch_size=B,
            clip_denoised=True,
            device=device,
        )
        dt = time.time() - t0
        print(f"  Batch {i//batch_size + 1}: generated {B} scenes in {dt:.1f}s "
              f"({dt/B:.1f}s/scene)")
        
        # Post-process: descale from [-1, 1] back to original coordinates
        for bbox_dict in bbox_params_list:
            layout = descale_layout(bbox_dict, cfg)
            layouts.append(layout)
    
    return layouts


def descale_layout(bbox_dict, room_cfg):
    """Descale model output from [-1, 1] back to original metric space.
    
    Args:
        bbox_dict: dict with tensors 'translations', 'sizes', 'angles', 'class_labels'
                   each of shape [1, N_i, ?]
        room_cfg: room configuration with bounds
    
    Returns:
        layout: dict with descaled numpy arrays
    """
    trans = bbox_dict['translations'][0].cpu().numpy()  # [N, 3]
    sizes = bbox_dict['sizes'][0].cpu().numpy()          # [N, 3]
    angles_cs = bbox_dict['angles'][0].cpu().numpy()     # [N, 2] (cos, sin)
    class_labels = bbox_dict['class_labels'][0].cpu().numpy()  # [N, C]
    
    bounds_t = room_cfg['bounds_translations']
    bounds_s = room_cfg['bounds_sizes']
    bounds_a = room_cfg['bounds_angles']
    
    # Descale: v_real = (v_scaled + 1) / 2 * (max - min) + min
    trans_min = np.array(bounds_t[:3])
    trans_max = np.array(bounds_t[3:])
    trans = (trans + 1) / 2 * (trans_max - trans_min) + trans_min
    
    size_min = np.array(bounds_s[:3])
    size_max = np.array(bounds_s[3:])
    sizes = (sizes + 1) / 2 * (size_max - size_min) + size_min
    sizes = np.abs(sizes)  # sizes must be positive
    
    # Convert cos/sin back to angle in radians
    angles = np.arctan2(angles_cs[:, 1], angles_cs[:, 0])
    
    # Get class indices
    class_indices = np.argmax(class_labels, axis=-1)
    
    return {
        'translations': trans,
        'sizes': sizes,
        'angles': angles,
        'angles_cos_sin': angles_cs,
        'class_labels': class_labels,
        'class_indices': class_indices,
        'num_objects': trans.shape[0],
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_layout(floor_mask, layout, room_type, output_path, scene_idx=0):
    """Create detailed top-down visualization of a generated scene layout.
    
    Args:
        floor_mask: 2D binary array (H, W)
        layout: dict from descale_layout
        room_type: room type string
        output_path: path to save visualization
        scene_idx: scene index for title
    """
    cfg = ROOM_CONFIGS[room_type]
    labels = cfg["labels"]
    room_side = cfg["room_side"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    trans = layout['translations']
    sizes = layout['sizes']
    angles = layout['angles']
    class_indices = layout['class_indices']
    n_objects = layout['num_objects']
    
    # --- Panel 1: Floor plan with bounding boxes in metric space ---
    ax = axes[0]
    
    # Draw floor plan in metric space
    H, W = floor_mask.shape
    extent = [-room_side, room_side, -room_side, room_side]
    ax.imshow(floor_mask, cmap='Greys', alpha=0.15, origin='lower', extent=extent)
    
    # Draw boundary
    eroded = ndimage.binary_erosion(floor_mask > 0.5)
    boundary = (floor_mask > 0.5) & ~eroded
    by, bx = np.where(boundary)
    bx_m = (bx / W - 0.5) * 2 * room_side
    by_m = (by / H - 0.5) * 2 * room_side
    ax.scatter(bx_m, by_m, s=0.5, c='gray', alpha=0.3)
    
    legend_handles = {}
    
    for i in range(n_objects):
        cls_idx = class_indices[i]
        if cls_idx >= len(labels) - 2:  # skip start/end/empty
            continue
        
        cls_name = labels[cls_idx]
        x, y, z = trans[i]
        sx, sy, sz = sizes[i]
        angle_rad = angles[i]
        
        # Get color
        color = CATEGORY_COLORS.get(cls_name, '#808080')
        
        # Draw rotated bounding box
        angle_deg = np.degrees(angle_rad)
        rect = patches.FancyBboxPatch(
            (-sx/2, -sy/2), sx, sy,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor='black', alpha=0.7, linewidth=0.8
        )
        t = (plt.matplotlib.transforms.Affine2D()
             .rotate_deg(angle_deg)
             .translate(x, y) + ax.transData)
        rect.set_transform(t)
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y, cls_name[:3], fontsize=5, ha='center', va='center',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.8))
        
        if cls_name not in legend_handles:
            legend_handles[cls_name] = patches.Patch(facecolor=color, label=cls_name)
    
    ax.set_xlim(-room_side, room_side)
    ax.set_ylim(-room_side, room_side)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'{room_type.title()} Scene {scene_idx+1} — Top-Down View\n'
                 f'{n_objects} objects generated', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # --- Panel 2: Object summary ---
    ax2 = axes[1]
    ax2.axis('off')
    
    # Count objects by category
    category_counts = {}
    for i in range(n_objects):
        cls_idx = class_indices[i]
        if cls_idx >= len(labels) - 2:
            continue
        cls_name = labels[cls_idx]
        category_counts[cls_name] = category_counts.get(cls_name, 0) + 1
    
    # Sort by count
    sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
    
    # Draw summary table
    y_pos = 0.95
    ax2.text(0.05, y_pos, f"Generated Layout Summary", fontsize=14, fontweight='bold',
             transform=ax2.transAxes, verticalalignment='top')
    y_pos -= 0.06
    ax2.text(0.05, y_pos, f"Room type: {room_type}", fontsize=11,
             transform=ax2.transAxes, verticalalignment='top')
    y_pos -= 0.04
    ax2.text(0.05, y_pos, f"Total objects: {sum(category_counts.values())}", fontsize=11,
             transform=ax2.transAxes, verticalalignment='top')
    y_pos -= 0.06
    
    ax2.text(0.05, y_pos, "Category", fontsize=10, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.55, y_pos, "Count", fontsize=10, fontweight='bold',
             transform=ax2.transAxes)
    y_pos -= 0.02
    ax2.plot([0.05, 0.75], [y_pos, y_pos], transform=ax2.transAxes, 
             color='gray', linewidth=0.5)
    y_pos -= 0.03
    
    for cat_name, count in sorted_cats:
        color = CATEGORY_COLORS.get(cat_name, '#808080')
        ax2.add_patch(patches.FancyBboxPatch(
            (0.05, y_pos - 0.008), 0.03, 0.02,
            transform=ax2.transAxes, boxstyle="round,pad=0.002",
            facecolor=color, edgecolor='black', linewidth=0.5, clip_on=False
        ))
        ax2.text(0.1, y_pos, cat_name, fontsize=9, transform=ax2.transAxes,
                 verticalalignment='top')
        ax2.text(0.55, y_pos, str(count), fontsize=9, transform=ax2.transAxes,
                 verticalalignment='top')
        y_pos -= 0.035
    
    # Add generation details
    y_pos -= 0.04
    ax2.text(0.05, y_pos, "Object Details:", fontsize=10, fontweight='bold',
             transform=ax2.transAxes)
    y_pos -= 0.04
    
    for i in range(min(n_objects, 15)):  # Show first 15 objects
        cls_idx = class_indices[i]
        if cls_idx >= len(labels) - 2:
            continue
        cls_name = labels[cls_idx]
        x, y, z = trans[i]
        sx, sy, sz = sizes[i]
        ax2.text(0.05, y_pos, 
                 f"  {cls_name}: pos=({x:.1f},{y:.1f},{z:.1f}) "
                 f"size=({sx:.1f},{sy:.1f},{sz:.1f})",
                 fontsize=7, transform=ax2.transAxes, verticalalignment='top',
                 fontfamily='monospace')
        y_pos -= 0.025
    
    if n_objects > 15:
        ax2.text(0.05, y_pos, f"  ... and {n_objects - 15} more objects",
                 fontsize=7, transform=ax2.transAxes, verticalalignment='top',
                 fontfamily='monospace', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def save_layout_json(layout, room_type, output_path, scene_idx=0):
    """Save generated layout as a JSON file for downstream processing."""
    cfg = ROOM_CONFIGS[room_type]
    labels = cfg["labels"]
    
    objects = []
    for i in range(layout['num_objects']):
        cls_idx = layout['class_indices'][i]
        if cls_idx >= len(labels) - 2:
            continue
        objects.append({
            "category": labels[cls_idx],
            "class_index": int(cls_idx),
            "position": layout['translations'][i].tolist(),
            "size": layout['sizes'][i].tolist(),
            "angle_rad": float(layout['angles'][i]),
            "angle_cos_sin": layout['angles_cos_sin'][i].tolist(),
        })
    
    result = {
        "room_type": room_type,
        "scene_index": scene_idx,
        "num_objects": len(objects),
        "objects": objects,
        "bounds_translations": cfg['bounds_translations'],
        "bounds_sizes": cfg['bounds_sizes'],
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MiDiffusion Inference — Generate 3D room layouts"
    )
    parser.add_argument(
        "--room_type", default="livingroom",
        choices=["livingroom", "bedroom", "diningroom"],
        help="Room type to generate"
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Path to model directory (default: pretrained_local/.../<room_type>)"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: output/inference_results/<room_type>)"
    )
    parser.add_argument(
        "--num_scenes", type=int, default=5,
        help="Number of scenes to generate"
    )
    parser.add_argument(
        "--floor_shape", default="mixed",
        choices=["rectangular", "l_shaped", "irregular", "t_shaped",
                 "large_rectangular", "mixed"],
        help="Floor plan shape to use"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to run on (auto detects MPS on Apple Silicon)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--save_json", action="store_true",
        help="Also save layouts as JSON files"
    )
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Setup paths
    if args.model_dir is None:
        args.model_dir = os.path.join(
            PROJ_DIR, "pretrained_local", "MiDiffusion_model_weights",
            "floor_conditioned", args.room_type
        )
    
    if args.output_dir is None:
        args.output_dir = os.path.join(
            PROJ_DIR, "output", "inference_results", args.room_type
        )
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\n{'='*60}")
    print(f"MiDiffusion Inference — {args.room_type.title()}")
    print(f"{'='*60}")
    network, config = load_model(args.room_type, args.model_dir, device)
    
    # Create floor plans
    room_side = ROOM_CONFIGS[args.room_type]["room_side"]
    
    if args.floor_shape == "mixed":
        shapes = ["rectangular", "l_shaped", "irregular", "large_rectangular", "t_shaped"]
    else:
        shapes = [args.floor_shape]
    
    floor_plans = []
    for i in range(args.num_scenes):
        shape = shapes[i % len(shapes)]
        mask = create_floor_plan(shape)
        boundary_points = mask_to_boundary_points(mask, num_points=256, room_side=room_side)
        floor_plans.append((mask, boundary_points))
        print(f"  Floor plan {i+1}: {shape}")
    
    # Generate layouts
    print(f"\nGenerating {args.num_scenes} scene layouts...")
    t0 = time.time()
    layouts = generate_layouts(network, floor_plans, args.room_type, device, 
                               batch_size=args.batch_size)
    total_time = time.time() - t0
    print(f"\nTotal generation time: {total_time:.1f}s "
          f"({total_time/args.num_scenes:.1f}s/scene)")
    
    # Visualize and save
    print(f"\nSaving results to: {args.output_dir}")
    for i, (layout, (mask, _)) in enumerate(zip(layouts, floor_plans)):
        shape = shapes[i % len(shapes)]
        
        # Save visualization
        vis_path = os.path.join(args.output_dir, 
                                f"scene_{i+1:03d}_{shape}.png")
        visualize_layout(mask, layout, args.room_type, vis_path, scene_idx=i)
        
        # Save JSON
        if args.save_json:
            json_path = os.path.join(args.output_dir, 
                                     f"scene_{i+1:03d}_{shape}.json")
            save_layout_json(layout, args.room_type, json_path, scene_idx=i)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Room type:   {args.room_type}")
    print(f"Scenes:      {args.num_scenes}")
    print(f"Device:      {device}")
    print(f"Time:        {total_time:.1f}s total, {total_time/args.num_scenes:.1f}s/scene")
    
    total_objects = sum(l['num_objects'] for l in layouts)
    print(f"Objects:     {total_objects} total, "
          f"{total_objects/args.num_scenes:.1f} avg/scene")
    
    # Category distribution across all scenes
    all_cats = {}
    labels = ROOM_CONFIGS[args.room_type]["labels"]
    for layout in layouts:
        for idx in layout['class_indices']:
            if idx < len(labels) - 2:
                name = labels[idx]
                all_cats[name] = all_cats.get(name, 0) + 1
    
    print(f"\nCategory distribution:")
    for name, count in sorted(all_cats.items(), key=lambda x: -x[1]):
        print(f"  {name:30s} {count:3d}")
    
    print(f"\nOutput: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
