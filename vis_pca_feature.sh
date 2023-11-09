#!/bin/bash

for i in {0..11}
do
    python run_features_pca.py --config configs/pnp/feature-pca-vis.yaml --block "output_block_$i" --experiment house_0
done
#for i in {0..11}
#do
#    python run_features_pca.py --config configs/pnp/feature-pca-vis.yaml --block "input_block_$i" --experiment house_0
#
#done
