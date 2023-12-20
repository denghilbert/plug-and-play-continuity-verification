# Installation

Take a look at this: [Setup](https://github.com/MichalGeyer/plug-and-play#setup)

# Real Image

Use meditation as an example.

```shell
python run_features_extraction.py --config configs/pnp/feature-extraction-real.yaml --save_all_features --experiment_name 'meditation' --real_img_path 'data/meditation.png' --seed 50
```
We can set a series of classifier-free guidance scales or condition timestep lengths. For instance, '--classifier_free_scale 20 75' means generating images with scales [2, 2.5, 3, 3.5 ... 7.5], and '--condition_timestep 29 49' implies generating images using different timestep lengths for conditioning (i.e., [600, 800, 1000]). Of course, if we only want a single image for testing, we can set it as follows:

```shell
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'meditation' --prompt 'a photo of a golden statue' --seed 50 --classifier_free_scale 75 76 --condition_timestep 49 50
```



# Generated Images

Almost the same:

```shell
python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --experiment_name 'house' --prompt 'a photo of a big house in the winter' --save_all_features --seed 50
```

```shell
python run_adj_diff_scale.py --config configs/pnp/pnp-generated.yaml --experiment_name 'house' --prompt 'sunshine' --seed 50 --classifier_free_scale 75 76 --condition_timestep 49 50
```