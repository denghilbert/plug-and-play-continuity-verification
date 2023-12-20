python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'cat_smile' --prompt 'a photo of a cat sitting on the table' --seed 50 --save_all_features

python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'dog_smile' --prompt 'a photo of a dog sitting on the table' --seed 50 --save_all_features

python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'cat_smile_0' --prompt 'smiling' --seed 50

python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'dog_smile_0' --prompt 'smiling' --seed 50
