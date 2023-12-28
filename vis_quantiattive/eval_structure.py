import sys
sys.path.append('../../Splice')
from util.losses import LossG
from torchvision import transforms
from PIL import Image
# Path to your YAML file
file_path = '../Plug-and-Play-datasets/wild-ti2i/wild-read_back.yaml'

cfg = {'seed': -1, 'dataroot': 'datasets/splicing/cows', 'direction': 'AtoB', 'A_resize': -1, 'B_resize': -1, 'use_augmentations': True, 'global_A_crops_n_crops': 1, 'global_A_crops_min_cover': 0.95, 'global_B_crops_n_crops': 1, 'global_B_crops_min_cover': 0.95, 'init_type': 'xavier', 'init_gain': 0.02, 'lambda_global_cls': 10.0, 'lambda_global_ssim': 1.0, 'lambda_global_identity': 1.0, 'entire_A_every': 75, 'lambda_entire_cls': 10, 'lambda_entire_ssim': 1.0, 'dino_model_name': 'dino_vitb8', 'dino_global_patch_size': 224, 'cls_warmup': 1, 'n_epochs': 10000, 'scheduler_policy': 'none', 'scheduler_n_epochs_decay': 8, 'scheduler_lr_decay_iters': 300, 'optimizer': 'adam', 'optimizer_beta1': 0.0, 'optimizer_beta2': 0.99, 'lr': 0.002, 'log_images_freq': 10}

eval_loss = LossG(cfg)


exp_names = []
all_prompt = []
prompts = []
in_prompt = False
with open(file_path, 'r') as file:
    for line in file:
        if 'init_img' in line and in_prompt == False:
            all_prompt.append(prompts)
            prompts = []
            exp_names.append(line.split('/')[-1].split('"')[0])
            in_prompt = True
        if '-' in line:
            prompts.append(line.split('"')[1])
            in_prompt = False

all_prompt.append(prompts)
all_prompt.pop(0)

uncond_list = []
timesteps_list = []
for i in range(5, 140, 5):
    if i != 10:
        uncond_list.append(i / 10)
for i in range(0, 51, 10):
    timesteps_list.append(i)

print(exp_names)
print(uncond_list)
print(timesteps_list)
print(all_prompt)
results = {}
count = 0
all_count = 0
for scale in uncond_list:
    for timestep in timesteps_list:
        structure = []
        for exp_name, prompts in zip(exp_names, all_prompt):
            ref_img = Image.open('../Plug-and-Play-datasets/wild-ti2i/data/' + exp_name)
            to_tensor_transform = transforms.ToTensor()
            ref_img = to_tensor_transform(ref_img)[None, :, :, :].cuda()
            for prompt in prompts:
                try:
                    translated_img_path = './' + exp_name.split('.')[0] + '/' + 'translations_' + prompt.replace(' ', '_') + '_seed_50' + '/' + str(scale) + '_' + prompt.replace(' ', '_') + '/' + prompt.replace(' ', '_') + '_uncond_scale' + str(scale) + '_timestep' + str(timestep) + '_sample_0.png'
                    if ref_img.shape[1] == 4:
                        ref_img = ref_img[:, :3, :, :]
                    translated_img = Image.open(translated_img_path)
                    translated_img = to_tensor_transform(translated_img)[None, :, :, :].cuda()
                    structure.append(eval_loss.calculate_global_ssim_loss(ref_img, translated_img).item())
                    count += 1
                    all_count += 1
                except:
                    all_count += 1
                    #print(translated_img_path)

        print((scale, timestep))
        print(count)
        print(all_count)
        count = 0
        all_count = 0
        results[(scale, timestep)] = sum(structure) / len(structure)

import pdb;pdb.set_trace()
