python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'verify_uncond_scale_on_369feat_one_word' --prompt 'a photo of a big house in the winter' --seed 50


python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'verify_uncond_scale_on_369feat_one_word_0' --prompt 'winter'
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'verify_uncond_scale_on_369feat_one_word_0' --prompt 'sunshine'
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'verify_uncond_scale_on_369feat_one_word_0' --prompt 'foggy'
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'verify_uncond_scale_on_369feat_one_word_0' --prompt 'rainy'
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'verify_uncond_scale_on_369feat_one_word_0' --prompt 'spring'
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'verify_uncond_scale_on_369feat_one_word_0' --prompt 'cloudy'
