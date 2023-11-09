import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange, tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
from pnp_utils import check_safety

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from run_features_extraction import load_model_from_config


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
        "--prompt",
        type=str,
        help="name of this experiment",
        default="default_test"
    )
    opt = parser.parse_args()
    exp_config = OmegaConf.load(opt.config)
    if opt.experiment_name != "default_test":
        exp_config.source_experiment_name = opt.experiment_name
        exp_config.prompts = [opt.prompt]

    exp_path_root_config = OmegaConf.load("./configs/pnp/setup.yaml")
    exp_path_root = exp_path_root_config.config.exp_path_root

    # read seed from args.json of source experiment
    with open(os.path.join(exp_path_root, exp_config.source_experiment_name, "args.json"), "r") as f:
        args = json.load(f)
        seed = args["seed"]
        source_prompt = args["prompt"]
    negative_prompt = source_prompt if exp_config.negative_prompt is None else exp_config.negative_prompt

    seed_everything(seed)
    possible_ddim_steps = args["save_feature_timesteps"]
    assert exp_config.num_ddim_sampling_steps in possible_ddim_steps or exp_config.num_ddim_sampling_steps is None, f"possible sampling steps for this experiment are: {possible_ddim_steps}; for {exp_config.num_ddim_sampling_steps} steps, run 'run_features_extraction.py' with save_feature_timesteps = {exp_config.num_ddim_sampling_steps}"
    ddim_steps = exp_config.num_ddim_sampling_steps if exp_config.num_ddim_sampling_steps is not None else possible_ddim_steps[-1]

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    seed = torch.initial_seed()
    opt.seed = seed

    translation_folders = [p.replace(' ', '_') for p in exp_config.prompts]
    for qk_threshold in range(-1, 50, 5):
        outpaths = [os.path.join(f"{exp_path_root}/{exp_config.source_experiment_name}/translations", f"{exp_config.scale}_{translation_folder}") for translation_folder in translation_folders]
        step_qk = 20 * (qk_threshold + 1)
        out_label = f"house_qk_only7_{step_qk}"

        predicted_samples_paths = [os.path.join(outpath, f"predicted_samples_{out_label}") for outpath in outpaths]
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
            self_attn_output_block_indices = [7]
            out_layers_output_block_indices = [3,4]
            output_block_self_attn_map_injection_thresholds = [qk_threshold] * len(self_attn_output_block_indices)
            feature_injection_thresholds = [-1] * len(out_layers_output_block_indices)
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
                        output_q = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                        output_k = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                        current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                        current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

                for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
                    if i <= int(feature_injection_threshold):
                        output = torch.load(os.path.join(source_experiment_out_layers_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
                        current_features[f'output_block_{output_block_idx}_out_layers'] = output

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

                    # print used injected feature layers
                    for item in injected_features:
                        print(item.keys())

                    # print diffusion unet structure
                    #block_num = 0
                    #for blocks in sampler.model.model.diffusion_model.output_blocks:
                    #    print("layers:" + str(block_num))
                    #    block_num += 1
                    #    num_in_block = 0
                    #    for block in blocks:
                    #        print("num_in_block:" + str(num_in_block))
                    #        num_in_block += 1
                    #        print(type(block))

                    # try to upsample from low resolution feature
                    for i in range(0):
                        tmp = injected_features[i]['output_block_3_out_layers'] # torch.Size([2, 1280, 16, 16])
                        injected_features[i]['output_block_4_out_layers'] = tmp # torch.Size([2, 1280, 16, 16])
                        injected_features[i]['output_block_5_out_layers'] = tmp # torch.Size([2, 1280, 16, 16])

                        tmp = sampler.model.model.diffusion_model.output_blocks[5][2](tmp) # torch.Size([2, 1280, 32, 32])
                        sampler.model.model.diffusion_model.output_blocks[8][2]

                    # print layer resblock feature shape
                    # verify before replace with 11 layer
                    #for i in range(4, 12):
                    #    print(f'output_block_{i}_out_layers')
                    #    print(injected_features[0][f'output_block_{i}_out_layers'].shape)
                    #print("layer we will use:")
                    #print(injected_features[0]['output_block_11_out_layers'].shape) # torch.Size([2, 320, 64, 64])
                    #import pdb;pdb.set_trace()

                    # replace 4-10 with 11 resblock feature in 50 timesteps
                    for i in range (0):
                    #for i in range (qk_threshold + 1):
                        #injected_features[i]['output_block_10_out_layers'] = injected_features[i]['output_block_11_out_layers']
                        #injected_features[i]['output_block_9_out_layers'] = injected_features[i]['output_block_11_out_layers']
                        tmp1 = sampler.model.model.diffusion_model.input_blocks[3][0](injected_features[i]['output_block_11_out_layers']) # torch.Size([2, 320, 32, 32])
                        tmp2 = sampler.model.model.diffusion_model.input_blocks[3][0](injected_features[i]['output_block_10_out_layers']) # torch.Size([2, 320, 32, 32])
                        tmp3 = sampler.model.model.diffusion_model.input_blocks[3][0](injected_features[i]['output_block_9_out_layers']) # torch.Size([2, 320, 32, 32])
                        tmp1 = sampler.model.model.diffusion_model.input_blocks[4][0].in_layers(tmp1) # torch.Size([2, 640, 32, 32])
                        tmp2 = sampler.model.model.diffusion_model.input_blocks[4][0].in_layers(tmp2) # torch.Size([2, 640, 32, 32])
                        tmp3 = sampler.model.model.diffusion_model.input_blocks[4][0].in_layers(tmp3) # torch.Size([2, 640, 32, 32])
                        injected_features[i]['output_block_8_out_layers'] = tmp1
                        injected_features[i]['output_block_7_out_layers'] = tmp2
                        injected_features[i]['output_block_6_out_layers'] = tmp3
                        tmp1 = sampler.model.model.diffusion_model.input_blocks[6][0](tmp1) # torch.Size([2, 640, 64, ])
                        tmp2 = sampler.model.model.diffusion_model.input_blocks[6][0](tmp2) # torch.Size([2, 640, 64, 64])
                        tmp3 = sampler.model.model.diffusion_model.input_blocks[6][0](tmp3) # torch.Size([2, 640, 64, 64])
                        tmp1 = sampler.model.model.diffusion_model.input_blocks[7][0].in_layers(tmp1) # torch.Size([2, 1280, 64, 64])
                        tmp2 = sampler.model.model.diffusion_model.input_blocks[7][0].in_layers(tmp2) # torch.Size([2, 1280, 64, 64])
                        tmp3 = sampler.model.model.diffusion_model.input_blocks[7][0].in_layers(tmp3) # torch.Size([2, 1280, 64, 64])
                        injected_features[i]['output_block_5_out_layers'] = tmp1
                        injected_features[i]['output_block_4_out_layers'] = tmp2
                        injected_features[i]['output_block_3_out_layers'] = tmp3

                    # print layer resblock feature shape
                    # verify after replace with 11 layer
                    #for i in range(4, 12):
                    #    print(f'output_block_{i}_out_layers')
                    #    print(injected_features[0][f'output_block_{i}_out_layers'].shape)
                    #print("layer we will use:")
                    #print(injected_features[0]['output_block_11_out_layers'].shape) # torch.Size([2, 320, 64, 64])
                    #import pdb;pdb.set_trace()
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
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
                                                     negative_prompt_alpha=exp_config.negative_prompt_alpha,
                                                     negative_prompt_schedule=exp_config.negative_prompt_schedule,
                                                     )

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
