python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'cute_cat' --prompt 'a photo of a cute cat' --exp_path_root '/cond_scale'
python cond_scale_run_pnp.py --config configs/pnp/pnp-generated.yaml --experiment_name 'cute_cat' --prompt 'a photo of a white cute cat' --prompt_interpolation 'a photo of a cute cat' --exp_path_root '/cond_scale'

python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'cat_dog' --prompt 'a photo of a cute cat' --exp_path_root '/cond_scale'
python cond_scale_run_pnp.py --config configs/pnp/pnp-generated.yaml --experiment_name 'cat_dog' --prompt 'a photo of a cute cat' --prompt_interpolation 'a photo of a cute dog' --exp_path_root '/cond_scale'

python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'house_cat' --prompt 'a photo of a cute cat' --exp_path_root '/cond_scale'
python cond_scale_run_pnp.py --config configs/pnp/pnp-generated.yaml --experiment_name 'house_cat' --prompt 'a photo of a cute cat' --prompt_interpolation 'a photo of a small house' --exp_path_root '/cond_scale'

python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'unhappy_happy_cat' --prompt 'a photo of a happy cat' --exp_path_root '/cond_scale'
python cond_scale_run_pnp.py --config configs/pnp/pnp-generated.yaml --experiment_name 'unhappy_happy_cat' --prompt 'a photo of a happy cat' --prompt_interpolation 'a photo of an unhappy cat' --exp_path_root '/cond_scale'

python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'fat_cat' --prompt 'a photo of a fat cat' --exp_path_root '/cond_scale'
python cond_scale_run_pnp.py --config configs/pnp/pnp-generated.yaml --experiment_name 'fat_cat' --prompt 'a photo of a fat cat' --prompt_interpolation 'a photo of a cute cat' --exp_path_root '/cond_scale'

python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'old_new_car' --prompt 'a photo of a new car' --exp_path_root '/cond_scale'
python cond_scale_run_pnp.py --config configs/pnp/pnp-generated.yaml --experiment_name 'old_new_car' --prompt 'a photo of a new car' --prompt_interpolation 'a photo of an old car' --exp_path_root '/cond_scale'
