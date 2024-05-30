Install ffmpeg from conda instead


Text to Motion
Eval
```
export CUDA_VISIBLE_DEVICES=4
conda activate gmd
python -m eval.eval_humanml --model_path ./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt
python -m eval.eval_humanml --model_path ./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt --full_traj_inpaint
python -m eval.eval_humanml_condition --model_path ./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt
```
Gen From Testset
```
python -m sample.generate --model_path ./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt --num_samples 10
```

Gen from text
```
python -m sample.generate --model_path ./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt --text_prompt "a person is picking up something on the floor"
```

/home/epinyoan/git/guided-motion-diffusion/data_loaders/humanml/utils/metrics.py
calculate_trajectory_error
calculate_trajectory_diversity
calculate_skating_ratio


/home/epinyoan/git/guided-motion-diffusion/data_loaders/humanml/data/dataset.py
abs3d_to_rel










Current
/home/epinyoan/git/guided-motion-diffusion/sample/condition.py   CondKeyLocations

/home/epinyoan/git/guided-motion-diffusion/diffusion/gaussian_diffusion.py  p_sample_loop