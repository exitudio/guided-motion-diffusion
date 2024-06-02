from functools import partial
from pprint import pprint
from utils.fixseed import fixseed
import os
import time
import numpy as np
import torch
import copy
import json
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.output_util import sample_to_motion, construct_template_variables, save_multiple_samples
from utils.generation_template import get_template
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders import humanml_utils
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
# import flag
from torch.cuda import amp
from sample.condition import (get_target_from_kframes, get_inpainting_motion_from_traj, 
                              get_target_and_inpt_from_kframes_batch, 
                              cond_fn_key_location, cond_fn_sdf, log_trajectory_from_xstart,
                              CondKeyLocations, CondKeyLocationsWithSdf)

from sample.keyframe_pattern import get_kframes, get_obstacles

from sample.generate import load_dataset, load_traj_model, load_processed_file
import argparse

args = argparse.Namespace(abs_3d=True, action_file='', action_name='', adam_beta2=0.999, apply_zero_mask=False, arch='unet', augment_type='none', avg_model_beta=0, batch_size=64, classifier_scale=100.0, clip_range=6.0, cond_mask_prob=0.1, cuda=True, data_dir='', dataset='humanml', device=0, diffusion_steps=1000, dim_mults=[2, 2, 2, 2], do_inpaint=True, drop_redundant=False, emb_trans_dec=False, eval_batch_size=32, eval_during_training=False, eval_num_samples=1000, eval_rep_times=3, eval_split='test', ff_size=1024, gen_mse_loss=True, gen_reward_model=False, gen_two_stages=True, grad_clip=0, guidance_mode='kps', guidance_param=2.5, input_text='', interactive=False, lambda_fc=0.0, lambda_rcxyz=0.0, lambda_vel=0.0, latent_dim=512, layers=8, log_interval=1000, lr=0.0001, lr_anneal_steps=0, model_path='./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt', motion_length=11.2, motion_length_cut=6.0, noise_schedule='cosine', num_frames=60, num_repetitions=3, num_samples=10, num_steps=600000, out_mult=16, output_dir='', overwrite=False, p2p_impute=True, predict_xstart=True, random_proj_scale=10.0, resume_checkpoint='', save_dir='save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224', save_interval=50000, seed=10, sigma_small=True, std_scale_shift=(1.0, 0.0), text_prompt='a person is walking while raising both hands', time_weighted_loss=False, train_platform_type='TensorboardPlatform', train_x0_as_eps=False, traj_extra_weight=1.0, traj_only=False, unconstrained=False, unet_adagn=True, unet_zero=True, use_ddim=False, use_fp16=False, use_random_proj=True, weight_decay=0.01, xz_only=False)
args = get_template(args, template_name=args.guidance_mode)
# fixseed(args.seed)


max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
fps = 12.5 if args.dataset == 'kit' else 20
n_frames = min(max_frames, int(args.motion_length * fps))
cut_frames = int(args.motion_length_cut * fps)
is_using_data = not any([
    args.input_text, args.text_prompt, args.action_file, args.action_name
])
dist_util.setup_dist(args.device)

texts = [args.text_prompt]
args.num_samples = 3
args.num_repetitions = 1
args.batch_size = args.num_samples
data = load_dataset(args, max_frames, n_frames)
total_num_samples = args.num_samples * args.num_repetitions
model, diffusion = create_model_and_diffusion(args, data)
load_saved_model(model, args.model_path)
model = ClassifierFreeSampleModel(
        model)  # wrapping model with the classifier-free sampler
model.to(dist_util.dev())
model.eval()  

collate_args = [{
    'inp': torch.zeros(n_frames),
    'tokens': None,
    # this would be incorrect for UNET models
    # 'lengths': n_frames,
    'lengths': cut_frames,
}] * args.num_samples
# model_kwargs['y']['lengths']
is_t2m = any([args.input_text, args.text_prompt])
collate_args = [
    dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
]
_, model_kwargs = collate(collate_args)
model_kwargs['y']['traj_model'] = args.traj_only

# traj_model, traj_diffusion = load_traj_model(data)
args_traj = argparse.Namespace(abs_3d=True, action_file='', action_name='', adam_beta2=0.999, apply_zero_mask=False, arch='unet', augment_type='none', avg_model_beta=0, batch_size=512, classifier_scale=100.0, clip_range=6.0, cond_mask_prob=0.1, cuda=True, data_dir='', dataset='humanml', device=0, diffusion_steps=1000, dim_mults=[0.125, 0.25, 0.5], do_inpaint=False, drop_redundant=False, emb_trans_dec=False, eval_batch_size=32, eval_during_training=False, eval_num_samples=1000, eval_rep_times=3, eval_split='test', ff_size=1024, gen_mse_loss=True, gen_reward_model=False, gen_two_stages=False, grad_clip=0, guidance_mode='kps', guidance_param=2.5, input_text='', interactive=False, lambda_fc=0.0, lambda_rcxyz=0.0, lambda_vel=0.0, latent_dim=512, layers=8, log_interval=125, lr=0.0001, lr_anneal_steps=0, model_path='./save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt', motion_length=11.2, motion_length_cut=6.0, noise_schedule='cosine', num_frames=60, num_repetitions=3, num_samples=10, num_steps=112500, out_mult=16, output_dir='', overwrite=False, p2p_impute=True, predict_xstart=False, random_proj_scale=10.0, resume_checkpoint='', save_dir='save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224', save_interval=6250, seed=10, sigma_small=True, std_scale_shift=(1.0, 0.0), text_prompt='a person is walking while raising both hands', time_weighted_loss=False, train_platform_type='TensorboardPlatform', train_x0_as_eps=False, traj_extra_weight=1.0, traj_only=True, unconstrained=False, unet_adagn=True, unet_zero=True, use_ddim=False, use_fp16=False, use_random_proj=False, weight_decay=0.01, xz_only=False)
traj_model, traj_diffusion = create_model_and_diffusion(args_traj, data)
load_saved_model(traj_model, args_traj.model_path)
traj_model = ClassifierFreeSampleModel(
    traj_model)  # wrapping model with the classifier-free sampler
traj_model.to(dist_util.dev())
traj_model.eval()
#################################

traj_model_kwargs = copy.deepcopy(model_kwargs)
traj_model_kwargs['y']['traj_model'] = True

all_motions = []
all_lengths = []
all_text = []
obs_list = []
USE_CLASSIFIER_FREE = False # True
model_device = next(model.parameters()).device


# this is the pre process position
input_motions, ground_positions = load_processed_file(model_device, args.batch_size, args.traj_only)
input_skels = recover_from_ric(input_motions.permute(0, 2, 3, 1), 22, abs_3d=False)
input_skels = input_skels.squeeze(1)
kframes = get_kframes(pattern="zigzag")
motion_cond_until = 20
motion_impute_until = 1


def gen_traj():
    kframes_num = [a for (a,b) in kframes] # [0, 30, 60, 90, 119]
    kframes_posi = torch.tensor(kframes_num, dtype=torch.int).unsqueeze(0).repeat(args.batch_size, 1)
    dummy_skel_motions = torch.zeros([1, 22, 3, n_frames])
    for (tt, locs) in kframes:
        print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
        dummy_skel_motions[0, 0, [0, 2], tt] = torch.tensor([locs[0], locs[1]])
    dummy_skel_motions = dummy_skel_motions.repeat(args.batch_size, 1, 1, 1)  # [1, 22, 3, max_length]

    (target, target_mask, 
        inpaint_traj_p2p, inpaint_traj_mask_p2p,
        inpaint_traj_points, inpaint_traj_mask_points,
        inpaint_motion_p2p, inpaint_mask_p2p,
        inpaint_motion_points, inpaint_mask_points) = get_target_and_inpt_from_kframes_batch(dummy_skel_motions, kframes_posi, data.dataset)
    target = target.to(model_device)
    target_mask = target_mask.to(model_device)
    model_kwargs['y']['target'] = target
    model_kwargs['y']['target_mask'] = target_mask



    model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    traj_model_kwargs['y']['scale'] = torch.ones(
        args.batch_size,
        device=dist_util.dev()) * args.guidance_param
    impute_slack = 20
    impute_until = 100
    traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj_p2p.to(model_device)
    traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_p2p.to(model_device)

    traj_model_kwargs['y']['cond_until'] = impute_slack
    traj_model_kwargs['y']['impute_until'] = impute_until
    # NOTE: We have the option of switching the target motion from line to just key locations
    # We call this a 'second stage', which will start after t reach 'impute_until'
    traj_model_kwargs['y']['impute_until_second_stage'] = impute_slack
    traj_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points.to(model_device)
    traj_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points.to(model_device)

    traj_diffusion.data_transform_fn = None
    traj_diffusion.data_inv_transform_fn = None
    traj_diffusion.log_trajectory_fn = None


    cond_fn_traj = CondKeyLocations(target=target,
        target_mask=target_mask,
        transform=data.dataset.t2m_dataset.transform_th,
        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
        abs_3d=args.abs_3d,
        classifiler_scale=args.classifier_scale,
        use_mse_loss=args.gen_mse_loss,
        use_rand_projection=False,
        )

    sample_fn = traj_diffusion.p_sample_loop
    dump_steps = [1, 100, 300, 500, 700, 850, 999]
    traj_sample = sample_fn(
        traj_model,
        (args.batch_size, traj_model.njoints, traj_model.nfeats,
            n_frames),
        clip_denoised=True,  # False,
        model_kwargs=traj_model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None, # None,
        progress=True,
        dump_steps=dump_steps,  # None,
        noise=None,
        const_noise=False,
        cond_fn=cond_fn_traj,
    )
    return traj_sample