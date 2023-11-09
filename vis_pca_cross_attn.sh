#!/bin/bash

for i in {0..11}
do
    python run_cross_attn_pca.py --experiment house_0 --block "output_block_$i"
done
#for i in {1..11}
#do
#    python run_cross_attn_pca.py --experiment house_0 --block "input_block_$i"
#done
