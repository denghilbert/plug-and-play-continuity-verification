import sys
sys.path.append('../../clipscore')
# Path to your YAML file
import argparse
import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import generation_eval_utils
import pprint
import warnings
from packaging import version
from clipscore import extract_all_images, get_clip_score
file_path = '../Plug-and-Play-datasets/wild-ti2i/wild-read_back.yaml'

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
results = []
count = 0
all_count = 0
model, transform = clip.load("ViT-B/32", device='cuda', jit=False)
model.eval()
translated_img_paths = []
scale_times = []
sequential_prompts = []
for scale in uncond_list:
    for timestep in timesteps_list:
        scale_times.append((scale, timestep))
        translated_img_path = []
        sequential_prompt = []
        for exp_name, prompts in zip(exp_names, all_prompt):
            for prompt in prompts:
                path = './' + exp_name.split('.')[0] + '/' + 'translations_' + prompt.replace(' ', '_') + '_seed_50' + '/' + str(scale) + '_' + prompt.replace(' ', '_') + '/' + prompt.replace(' ', '_') + '_uncond_scale' + str(scale) + '_timestep' + str(timestep) + '_sample_0.png'
                if os.path.exists(path):
                    count += 1
                    all_count += 1
                    translated_img_path.append(path)
                    sequential_prompt.append(prompt)
                else:
                    all_count += 1
                    print(path)
        sequential_prompts.append(sequential_prompt)
        translated_img_paths.append(translated_img_path)

        print((scale, timestep))
        print(count)
        print(all_count)
        count = 0
        all_count = 0

results = []
for translated_img_path, sequential_prompt, scale_time in zip(translated_img_paths, sequential_prompts, scale_times):
    image_feats = extract_all_images(translated_img_path, model, 'cuda', batch_size=100, num_workers=24)

# get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(model, image_feats, sequential_prompt, 'cuda')
    results.append((scale_time[0], scale_time[1], sum(per_instance_image_text) / len(per_instance_image_text)))
import pdb;pdb.set_trace()
