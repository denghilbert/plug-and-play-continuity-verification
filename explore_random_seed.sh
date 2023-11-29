python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'house_seed10' --prompt 'a photo of a big house in the winter' --seed 10 --save_all_features

python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'house_seed1000' --prompt 'a photo of a big house in the winter' --seed 1000 --save_all_features

# the random seed in the second round seems not working at all
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'house_seed10_0' --prompt 'spring' --seed 50

python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'house_seed1000_0' --prompt 'spring' --seed 50
