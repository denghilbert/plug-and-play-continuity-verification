#!/bin/bash
yaml_file="./Plug-and-Play-datasets/wild-ti2i/wild-ti2i-real.yaml"

# Initialize an empty path
path=""

# Flag to indicate whether we are inside the target_prompts section
inside_prompts=false

# Read the file line by line
while IFS= read -r line; do
    # Check for 'init_img:' line and extract the path
    if echo "$line" | grep -q 'init_img:'; then
        # Extracting the path using sed
        path=$(echo "$line" | sed -n 's/.*init_img: "\(.*\)"/\1/p')
        dir="./Plug-and-Play-datasets/wild-ti2i/"
        long_path="$dir$path"
        filename=$(echo "$path" | sed 's/.*\/\(.*\)\..*/\1/')
        inside_prompts=false  # Resetting the flag as we're at a new section
        #echo $long_path
        #echo $filename
        python run_features_extraction.py --config configs/pnp/feature-extraction-real.yaml --save_all_features --exp_path_root "./quantitative" --seed 50 --experiment_name $filename --real_img_path $long_path
    fi

    # Detect the start of the target_prompts section
    if echo "$line" | grep -q 'target_prompts:'; then
        inside_prompts=true
        continue  # Skip processing this line further and move to the next line
    fi

    # Check if we are inside the target_prompts section and the line starts with a '-'
    if $inside_prompts && echo "$line" | grep -q '  -'; then
        # Extracting the prompt using sed
        prompt=$(echo "$line" | sed -n 's/.*- "\(.*\)"/\1/p')
        #echo "Processing prompt: $prompt with path: $path"
        #echo $filename
        #echo $prompt
        python run_acceleration_adj.py --config configs/pnp/pnp-generated.yaml --exp_path_root ./quantitative --seed 50 --experiment_name $filename --prompt "$prompt" --classifier_free_scale 5 140 --condition_timestep -1 50
    fi
done < "$yaml_file"
