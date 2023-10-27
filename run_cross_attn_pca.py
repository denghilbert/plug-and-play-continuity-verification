import argparse, os
from tqdm import trange
import torch
from einops import rearrange
from omegaconf import OmegaConf
from run_features_extraction import load_model_from_config
from torch import einsum
from pnp_utils import visualize_and_save_features_pca
import numpy as np
from tqdm import tqdm
import json

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_experiments_cross_attn_maps(unet_model, feature_map_paths, block, t):
    cross_attn_maps = []
    block_idx = int(block.split('_')[-1])
    for i, feature_map_path in enumerate(feature_map_paths):
        cross_attn_q = torch.load(os.path.join(feature_map_path, f"{block}_cross_attn_q_time_{t}.pt"))
        cross_attn_k = torch.load(os.path.join(feature_map_path, f"{block}_cross_attn_k_time_{t}.pt"))
        if "output_block" in block:
          scale = unet_model.output_blocks[block_idx][1].transformer_blocks[0].attn1.scale
        else:
          scale = unet_model.input_blocks[block_idx][1].transformer_blocks[0].attn1.scale
        sim = einsum('b i d, b j d -> b i j', cross_attn_q, cross_attn_k) * scale
        cross_attn_map = sim.softmax(dim=-1)
        cross_attn_map = rearrange(cross_attn_map, 'h n m -> n (h m)')
        cross_attn_maps.append(cross_attn_map)

    return cross_attn_maps


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--block",
        type=str,
        nargs="?",
        default="output_block_4",
        help="the name of the visualized feature block"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default = '',
        nargs="?",
        help="the name of the experiment to visualize cross-attention for"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/pnp/setup.yaml")
    exp_path_root = setup_config.config.exp_path_root
    experiment = opt.experiment

    with open(os.path.join(exp_path_root, experiment, "args.json"), "r") as f:
        args = json.load(f)
        ddim_steps = args["save_feature_timesteps"][-1]

    print(f"visualizing features PCA experiments: block - {opt.block}; experiment - {experiment}")

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    total_steps = sampler.ddim_timesteps.shape[0]
    iterator = tqdm(time_range, desc="visualizing cross-attention maps", total=total_steps)
    unet_model = model.model.diffusion_model

    transform_feature_maps_paths = fit_feature_maps_paths = [os.path.join(exp_path_root, experiment, "feature_maps")]

    pca_folder_path = os.path.join(exp_path_root, "PCA_cross_attention_vis", experiment)
    block_cross_attn_map_pca_path = os.path.join(pca_folder_path, f"{opt.block}_cross_attn_map")

    os.makedirs(pca_folder_path, exist_ok=True)
    os.makedirs(block_cross_attn_map_pca_path, exist_ok=True)

    for t in iterator:
        fit_cross_attn_maps = load_experiments_cross_attn_maps(unet_model, fit_feature_maps_paths, opt.block, t)  # T X (H T)
        transform_cross_attn_maps = load_experiments_cross_attn_maps(unet_model, transform_feature_maps_paths, opt.block, t)  # T X (H T)
        visualize_and_save_features_pca(
            torch.cat(fit_cross_attn_maps, dim=0),
            torch.cat(transform_cross_attn_maps, dim=0),
            [experiment],
            t,
            block_cross_attn_map_pca_path
        )


if __name__ == "__main__":
    main()
