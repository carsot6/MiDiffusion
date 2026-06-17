#!/usr/bin/env python3
import argparse
import json

import numpy as np
import torch

from utils import load_config, update_data_file_paths
from midiffusion.datasets.threed_front_encoding import get_encoded_dataset
from midiffusion.networks import build_network, validate_on_batch


def _to_device(sample, device):
    for key, value in sample.items():
        if not isinstance(value, list):
            sample[key] = value.to(device)
    return sample


def main():
    parser = argparse.ArgumentParser(description="Run one explicit forward pass for smoke verification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    if args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    config = load_config(args.config)
    dataset = get_encoded_dataset(
        update_data_file_paths(config["data"]),
        path_to_bounds=None,
        augmentations=None,
        split=config["training"].get("splits", ["train"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=(config["network"]["room_mask_condition"] and config["feature_extractor"]["name"] == "resnet18"),
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty for forward pass")

    idx = max(0, min(args.index, len(dataset) - 1))
    batch = dataset.collate_fn([dataset[idx]])
    batch = _to_device(batch, device)

    network, _, _ = build_network(
        dataset.n_object_types,
        config,
        weight_file=args.weights,
        device=device,
    )
    network.eval()

    loss = validate_on_batch(network, batch)
    print(json.dumps({
        "device": str(device),
        "dataset_len": len(dataset),
        "sample_index": idx,
        "forward_loss": float(loss),
    }, indent=2))


if __name__ == "__main__":
    main()
