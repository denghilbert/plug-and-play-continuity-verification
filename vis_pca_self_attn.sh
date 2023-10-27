#!/bin/bash

for i in {3..11}
do
    python run_self_attn_pca.py --experiment hourse_grass --block "output_block_$i"
done
for i in {1..11}
do
    python run_self_attn_pca.py --experiment hourse_grass --block "input_block_$i"
done
