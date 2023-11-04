#!/bin/bash

for i in {50..60}
do
    #python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name "cat_$i" --prompt "a photo of a cat standing on the grass, high quality, masterpiece" --seed $i
    python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name "cat_$i" --prompt "a photo of a cat standing on the grass" --seed $i
done
