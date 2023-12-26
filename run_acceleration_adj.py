import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange, tqdm
from einops import rearrange
#from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
from pnp_utils import check_safety

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from run_features_extraction import load_model_from_config

import random
def seed_everything(seed=42):
    # Seed Python's random module
    random.seed(seed)

    # Seed NumPy's random number generator
    np.random.seed(seed)

    # Seed PyTorch's random number generator
    torch.manual_seed(seed)

    # If CUDA is available, seed its random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.

        # Additional configurations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = 'configs/pnp/pnp-real.yaml')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--model_config', type=str, default='configs/stable-diffusion/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument("--check-safety", action='store_true')
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="name of this experiment",
        default="default_test"
    )
    parser.add_argument(
        "--exp_path_root",
        type=str,
        default="default_test"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="name of this experiment",
        default="default_test"
    )
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--use_3layer_replace", action='store_true')
    parser.add_argument("--replace_with_timestep1", action='store_true')
    parser.add_argument("--duplicate_cross_attn_cond", action='store_true')
    parser.add_argument("--classifier_free_scale", nargs="+", type=int, default=[5, 150])
    parser.add_argument("--condition_timestep", nargs="+", type=int, default=[-1, 49])
    opt = parser.parse_args()
    exp_config = OmegaConf.load(opt.config)
    if opt.experiment_name != "default_test":
        exp_config.source_experiment_name = opt.experiment_name
        exp_config.prompts = [opt.prompt]

    exp_path_root_config = OmegaConf.load("./configs/pnp/setup.yaml")
    if opt.exp_path_root != 'default_test':
        exp_path_root = opt.exp_path_root
    else:
        exp_path_root = exp_path_root_config.config.exp_path_root

    # read seed from args.json of source experiment
    with open(os.path.join(exp_path_root, exp_config.source_experiment_name, "args.json"), "r") as f:
        args = json.load(f)
        #seed = args["seed"]
        seed = opt.seed
        source_prompt = args["prompt"]
    negative_prompt = source_prompt if exp_config.negative_prompt is None else exp_config.negative_prompt

    seed_everything(seed)
    #seed = torch.initial_seed()
    possible_ddim_steps = args["save_feature_timesteps"]
    assert exp_config.num_ddim_sampling_steps in possible_ddim_steps or exp_config.num_ddim_sampling_steps is None, f"possible sampling steps for this experiment are: {possible_ddim_steps}; for {exp_config.num_ddim_sampling_steps} steps, run 'run_features_extraction.py' with save_feature_timesteps = {exp_config.num_ddim_sampling_steps}"
    ddim_steps = exp_config.num_ddim_sampling_steps if exp_config.num_ddim_sampling_steps is not None else possible_ddim_steps[-1]

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)


    translation_folders = [p.replace(' ', '_') for p in exp_config.prompts]
    uncond_list = []
    timesteps_list = []
    for i in range(opt.classifier_free_scale[0], opt.classifier_free_scale[1], 5):
        if i != 10:
            uncond_list.append(i / 10)
    for i in range(opt.condition_timestep[0], opt.condition_timestep[1], 10):
        timesteps_list.append(i)
    for uncond_scale in uncond_list:
        exp_config.scale = uncond_scale
        for threshold in timesteps_list:
            qk_threshold = threshold
            outpaths = [os.path.join(f"{exp_path_root}/{exp_config.source_experiment_name}/translations_{translation_folder}_seed_{opt.seed}", f"{exp_config.scale}_{translation_folder}") for translation_folder in translation_folders]
            step_qk = 20 * (qk_threshold + 1)
            out_label = f"{(exp_config.prompts[0].replace(' ', '_'))}_uncond_scale{uncond_scale}_timestep{qk_threshold+1}"
            print(out_label)

            predicted_samples_paths = [os.path.join(outpath, f"{out_label}") for outpath in outpaths]
            for i in range(len(outpaths)):
                os.makedirs(outpaths[i], exist_ok=True)
                os.makedirs(predicted_samples_paths[i], exist_ok=True)
                # save args in experiment dir
                with open(os.path.join(outpaths[i], "args.json"), "w") as f:
                    json.dump(OmegaConf.to_container(exp_config), f)

            def save_sampled_img(x, i, save_paths):
                for im in range(x.shape[0]):
                    x_samples_ddim = model.decode_first_stage(x[im].unsqueeze(0))
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_sample = x_image_torch[0]

                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(save_paths[im], f"{i}.png"))

            def ddim_sampler_callback(pred_x0, xt, i):
                save_sampled_img(pred_x0, i, predicted_samples_paths)

            def load_target_features(qk_threshold):
                self_attn_output_block_indices = [4,5,6,7,8,9,10,11]
                cross_attn_input_block_indices = [1,2,4,5,7,8] # 3,6,9 is independent downsampling layers, 1/2, 4/5, 7/8 are paired
                cross_attn_output_block_indices = [3,4,5,6,7,8,9,10,11] # 3/4/5, 6/7/8, 9/10/11 are paired, upsampling layers are inserted into 2,5,8
                out_layers_output_block_indices = [3, 6, 9]
                #out_layers_output_block_indices = [4]
                output_block_self_attn_map_injection_thresholds = [-1] * len(self_attn_output_block_indices)
                #output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(self_attn_output_block_indices)
                cross_attn_input_injection_thresholds = [-1] * len(cross_attn_input_block_indices)
                cross_attn_output_injection_thresholds = [-1] * len(cross_attn_output_block_indices)
                feature_injection_thresholds = [qk_threshold] * len(out_layers_output_block_indices)
                #feature_injection_thresholds = [exp_config.feature_injection_threshold] * len(out_layers_output_block_indices)
                target_features = []

                source_experiment_out_layers_path = os.path.join(exp_path_root, exp_config.source_experiment_name, "feature_maps")
                source_experiment_qkv_path = os.path.join(exp_path_root, exp_config.source_experiment_name, "feature_maps")

                time_range = np.flip(sampler.ddim_timesteps)
                total_steps = sampler.ddim_timesteps.shape[0]

                iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)

                for i, t in enumerate(iterator):
                    current_features = {}


                    for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                        if i <= int(output_block_self_attn_map_injection_threshold):
                            # qk from self-attn
                            output_q = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                            output_k = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                            current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                            current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k


                    for (output_block_idx, cross_attn_map_injection_threshold) in zip(cross_attn_output_block_indices,  cross_attn_output_injection_thresholds):
                        if i <= int(cross_attn_map_injection_threshold):
                            # qkv from cross-attn
                            output_k = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_cross_attn_k_time_{t}.pt"))
                            output_v = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_cross_attn_v_time_{t}.pt"))
                            #output_q = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_cross_attn_q_time_{t}.pt"))
                            current_features[f'output_block_{output_block_idx}_cross_attn_k'] = output_k
                            current_features[f'output_block_{output_block_idx}_cross_attn_v'] = output_v
                            # we don't want to use q since it is noise and vk represent cond
                            #current_features[f'output_block_{output_block_idx}_cross_attn_q'] = output_q


                    for (input_block_idx, cross_attn_map_injection_threshold) in zip(cross_attn_input_block_indices,  cross_attn_input_injection_thresholds):
                        if i <= int(cross_attn_map_injection_threshold):
                            input_k = torch.load(os.path.join(source_experiment_qkv_path, f"input_block_{input_block_idx}_cross_attn_k_time_{t}.pt"))
                            input_v = torch.load(os.path.join(source_experiment_qkv_path, f"input_block_{input_block_idx}_cross_attn_v_time_{t}.pt"))
                            #input_q = torch.load(os.path.join(source_experiment_qkv_path, f"input_block_{input_block_idx}_cross_attn_q_time_{t}.pt"))
                            current_features[f'input_block_{input_block_idx}_cross_attn_k'] = input_k
                            current_features[f'input_block_{input_block_idx}_cross_attn_v'] = input_v
                            # we don't want to use q since it is noise and vk represent cond
                            #current_features[f'output_block_{output_block_idx}_cross_attn_q'] = output_q


                    for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
                        if i <= int(feature_injection_threshold):
                            output = torch.load(os.path.join(source_experiment_out_layers_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
                            current_features[f'output_block_{output_block_idx}_out_layers'] = output
                            #output = torch.load(os.path.join(source_experiment_out_layers_path, f"output_block_{output_block_idx}_in_layers_features_time_{t}.pt"))
                            #current_features[f'output_block_{output_block_idx}_in_layers'] = output

                    target_features.append(current_features)

                return target_features

            batch_size = len(exp_config.prompts)
            prompts = exp_config.prompts
            assert prompts is not None

            start_code_path = f"{exp_path_root}/{exp_config.source_experiment_name}/z_enc.pt"
            start_code = torch.load(start_code_path).cuda() if os.path.exists(start_code_path) else None
            if start_code is not None:
                start_code = start_code.repeat(batch_size, 1, 1, 1)

            precision_scope = autocast if opt.precision=="autocast" else nullcontext
            injected_features = load_target_features(qk_threshold)
            unconditional_prompt = ""
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        uc = None
                        nc = None
                        if exp_config.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [unconditional_prompt])
                            nc = model.get_learned_conditioning(batch_size * [negative_prompt])
                        if not isinstance(prompts, list):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        injected_features_3layer = []

                        # load previous x_inter to accelerate
                        timestep_start = 0
                        x_inter = None
                        for i in range(qk_threshold, 0, -1):
                            path_x_inter = f"{exp_path_root}/{exp_config.source_experiment_name}/translations_{translation_folders[0]}_seed_{opt.seed}/{exp_config.prompts[0].replace(' ', '_')}_uncond_scale{uncond_scale}_{i}.pt"
                            if os.path.exists(path_x_inter):
                                x_inter = torch.load(path_x_inter)
                                timestep_start = i
                                break
                        samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         negative_conditioning=nc,
                                                         batch_size=len(prompts),
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=exp_config.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         img_callback=ddim_sampler_callback,
                                                         injected_features=injected_features,
                                                         timestep_cond=qk_threshold,
                                                         timestep_start=timestep_start,
                                                         x_inter=x_inter,
                                                         injected_features_3layer=injected_features_3layer,
                                                         negative_prompt_alpha=exp_config.negative_prompt_alpha,
                                                         negative_prompt_schedule=exp_config.negative_prompt_schedule,
                                                         )


                        if intermediates['last_cond_x_inter'] != None:
                            torch.save(intermediates['last_cond_x_inter'], f"{exp_path_root}/{exp_config.source_experiment_name}/translations_{translation_folders[0]}_seed_{opt.seed}/{exp_config.prompts[0].replace(' ', '_')}_uncond_scale{uncond_scale}_{qk_threshold}.pt")

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        if opt.check_safety:
                            x_samples_ddim = check_safety(x_samples_ddim)
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        sample_idx = 0
                        for k, x_sample in enumerate(x_image_torch):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(outpaths[k], f"{out_label}_sample_{sample_idx}.png"))
                            sample_idx += 1

            print(f"PnP results saved in: {'; '.join(outpaths)}")


if __name__ == "__main__":
    main()
