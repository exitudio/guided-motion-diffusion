from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
from utils.fixseed import fixseed
from utils import dist_util
from diffusion import logger
from utils.parser_util import eval_args
from eval.eval_humanml import load_dataset, create_model_and_diffusion, load_saved_model, evaluation
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
import torch
from data_loaders.humanml.data.dataset import abs3d_to_rel, sample_to_motion
import argparse
import os

args = argparse.Namespace(abs_3d=True, action_file='', action_name='', adam_beta2=0.999, apply_zero_mask=False, arch='unet', augment_type='none', avg_model_beta=0, batch_size=32, classifier_scale=100.0, clip_range=6.0, cond_mask_prob=0.1, cuda=True, data_dir='', dataset='humanml', device=0, diffusion_steps=1000, dim_mults=[2, 2, 2, 2], do_inpaint=False, drop_redundant=False, emb_trans_dec=False, eval_batch_size=32, eval_during_training=False, eval_mode='wo_mm', eval_num_samples=1000, eval_rep_times=3, eval_split='test', eval_use_avg=True, ff_size=1024, full_traj_inpaint=False, gen_mse_loss=True, gen_reward_model=False, gen_two_stages=True, grad_clip=0, guidance_mode='no', guidance_param=2.5, impute_until=None, input_text='', interactive=False, lambda_fc=0.0, lambda_rcxyz=0.0, lambda_vel=0.0, latent_dim=512, layers=8, log_interval=1000, lr=0.0001, lr_anneal_steps=0, model_path='./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt', motion_length=11.2, motion_length_cut=6.0, noise_schedule='cosine', num_frames=196, num_repetitions=3, num_samples=10, num_steps=600000, out_mult=16, output_dir='', overwrite=False, p2p_impute=True, predict_xstart=True, random_proj_scale=10.0, resume_checkpoint='', save_dir='save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224', save_interval=50000, seed=10, sigma_small=True, skip_first=None, std_scale_shift=(1.0, 0.0), text_prompt='', time_weighted_loss=False, train_platform_type='TensorboardPlatform', train_x0_as_eps=False, traj_extra_weight=1.0, traj_only=False, unconstrained=False, unet_adagn=True, unet_zero=True, use_ddim=False, use_fp16=False, use_random_proj=True, weight_decay=0.01, xz_only=False)

args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
args.num_frames = 196 # This must be 196!
args.gen_two_stages = True
skip_first_stage = False
if_ddim = "_ddim" if args.use_ddim else ""
fixseed(args.seed)

name = os.path.basename(os.path.dirname(args.model_path))
niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
# log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_cond_{}_{}'.format(name, niter))
log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_cond_{}_{}_{}'.format(name, niter, if_ddim))

# save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_cond_{niter}{"_avg" if flag.GEN_USE_AVG_MODEL else ""}')
save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_cond_{niter}{if_ddim}')
print('> Saving the generated motion to {}'.format(save_dir))

# NOTE: Set the final line imputing step t here. This is when we stop imputing with point-to-point trajectory
# and start imputing with key locations only
impute_until = 100  # int(args.impute_until) # 100
print('> Impute the trajectory with point-to-point until t = {}'.format(impute_until))
# skip_first_stage = True  # bool(args.skip_first) # True
print('> Skip first stage = {}'.format(skip_first_stage))

args.eval_mode = 'wo_mm'
print(f'Eval mode [{args.eval_mode}]')


if args.eval_mode == 'wo_mm':
    num_samples_limit = 1000
    # num_samples_limit = 32
    run_mm = False
    mm_num_samples = 0
    mm_num_repeats = 0
    mm_num_times = 0
    diversity_times = 300
    # diversity_times = 10
    # replication_times = 2
    replication_times = 20  # about 12 Hrs
else:
    raise Exception('Not support', '/home/epinyoan/git/guided-motion-diffusion/eval/eval_humanml.py')

dist_util.setup_dist(args.device)
logger.configure()

logger.log("creating data loader...")
split = 'test'
gt_loader = load_dataset(args, args.num_frames, split, hml_mode='gt')
gen_loader = load_dataset(args, args.num_frames, split, hml_mode='eval')
num_actions = gen_loader.dataset.num_actions




# separate from  "motion_model, motion_diffusion, traj_model, traj_diffusion = load_model(args, gen_loader)"

args_traj = argparse.Namespace(abs_3d=True, action_file='', action_name='', adam_beta2=0.999, apply_zero_mask=False, arch='unet', augment_type='none', avg_model_beta=0, batch_size=512, classifier_scale=100.0, clip_range=6.0, cond_mask_prob=0.1, cuda=True, data_dir='', dataset='humanml', device=0, diffusion_steps=1000, dim_mults=[0.125, 0.25, 0.5], do_inpaint=False, drop_redundant=False, emb_trans_dec=False, eval_batch_size=32, eval_during_training=False, eval_mode='wo_mm', eval_num_samples=1000, eval_rep_times=3, eval_split='test', eval_use_avg=True, ff_size=1024, full_traj_inpaint=False, gen_mse_loss=True, gen_reward_model=False, gen_two_stages=False, grad_clip=0, guidance_mode='no', guidance_param=2.5, impute_until=None, input_text='', interactive=False, lambda_fc=0.0, lambda_rcxyz=0.0, lambda_vel=0.0, latent_dim=512, layers=8, log_interval=125, lr=0.0001, lr_anneal_steps=0, model_path='./save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt', motion_length=11.2, motion_length_cut=6.0, noise_schedule='cosine', num_frames=60, num_repetitions=3, num_samples=10, num_steps=112500, out_mult=16, output_dir='', overwrite=False, p2p_impute=True, predict_xstart=False, random_proj_scale=10.0, resume_checkpoint='', save_dir='save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224', save_interval=6250, seed=10, sigma_small=True, skip_first=None, std_scale_shift=(1.0, 0.0), text_prompt='', time_weighted_loss=False, train_platform_type='TensorboardPlatform', train_x0_as_eps=False, traj_extra_weight=1.0, traj_only=True, unconstrained=False, unet_adagn=True, unet_zero=True, use_ddim=False, use_fp16=False, use_random_proj=False, weight_decay=0.01, xz_only=False)
traj_model, traj_diffusion = create_model_and_diffusion(args_traj, gen_loader)
print(f"Loading traj model checkpoints from [{args_traj.model_path}]...")
load_saved_model(traj_model, args_traj.model_path)
if args_traj.guidance_param != 1:
    traj_model = ClassifierFreeSampleModel(
        traj_model)  # wrapping model with the classifier-free sampler
traj_model.to(dist_util.dev())
traj_model.eval()  # disable random masking


motion_model, motion_diffusion = create_model_and_diffusion(args, gen_loader)
load_saved_model(motion_model, args.model_path) # , use_avg_model=args.gen_avg_model)
if args.guidance_param != 1:
    motion_model = ClassifierFreeSampleModel(
        motion_model)  # wrapping model with the classifier-free sampler
motion_model.to(dist_util.dev())
motion_model.eval()  # disable random masking
''
















from sample.condition import (cond_fn_key_location, get_target_from_kframes, get_target_and_inpt_from_kframes_batch, 
                              log_trajectory_from_xstart, get_inpainting_motion_from_traj, get_inpainting_motion_from_gt,
                              cond_fn_key_location, compute_kps_error, cond_fn_sdf,
                              CondKeyLocations, CondKeyLocationsWithSdf)
import copy
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from functools import partial

model_device = next(motion_model.parameters()).device

motion_diffusion.data_get_mean_fn = gen_loader.dataset.t2m_dataset.get_std_mean
motion_diffusion.data_transform_fn = gen_loader.dataset.t2m_dataset.transform_th
motion_diffusion.data_inv_transform_fn = gen_loader.dataset.t2m_dataset.inv_transform_th

clip_denoised = True
abs_3d = True
trajectory_classifier_scale = 100.0 # 100.0
traj_diffusion.data_transform_fn = None
traj_diffusion.data_inv_transform_fn = None
sample_fn_traj = (
    traj_diffusion.p_sample_loop if not args.use_ddim else traj_diffusion.ddim_sample_loop
)
sample_fn_motion = (
    motion_diffusion.p_sample_loop if not args.use_ddim else motion_diffusion.ddim_sample_loop
)
traj_model.eval()
motion_model.eval()

motion, model_kwargs = next(iter(gen_loader))
tokens = [t.split('_') for t in model_kwargs['y']['tokens']]
model_kwargs['y']['scale'] = torch.ones(motion.shape[0], device=dist_util.dev()) * args.guidance_param

### 1. Prepare motion for conditioning ###
traj_model_kwargs = copy.deepcopy(model_kwargs)
traj_model_kwargs['y']['traj_model'] = True
model_kwargs['y']['traj_model'] = False

# Convert to 3D motion space
# NOTE: the 'motion' will not be random projected if dataset mode is 'eval' or 'gt', 
# even if the 'self.dataset.t2m_dataset.use_rand_proj' is True
gt_poses = motion.permute(0, 2, 3, 1)
gt_poses = gt_poses * gen_loader.dataset.std + gen_loader.dataset.mean  # [bs, 1, 196, 263]
# (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=False)
gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)
gt_skel_motions = motion_model.rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True, 
                                    jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)

# Next, sample points, then prepare target and inpainting mask for trajectory model
## Sample points
n_keyframe = 5
sampled_keyframes = torch.rand(motion.shape[0], n_keyframe) * model_kwargs['y']['lengths'].unsqueeze(-1)
sampled_keyframes = torch.floor(sampled_keyframes).int().sort()[0]

## Prepare target and mask for grad cal
# Prepare trajecotry inpainting
(target, target_mask, 
inpaint_traj, inpaint_traj_mask,
inpaint_traj_points, inpaint_traj_mask_points,
inpaint_motion, inpaint_mask, 
inpaint_motion_points, inpaint_mask_points) = \
    get_target_and_inpt_from_kframes_batch(gt_skel_motions, 
                                           sampled_keyframes, 
                                           gen_loader.dataset)

target = target.to(model_device)
target_mask = target_mask.to(model_device)
model_kwargs['y']['target'] = target
model_kwargs['y']['target_mask'] = target_mask

impute_slack = 20
# NOTE: For debugging
############################ ??????????????? ####################
# traj_model_kwargs['y']['log_name'] = self.save_dir
# traj_model_kwargs['y']['log_id'] = i
# model_kwargs['y']['log_name'] = self.save_dir
# model_kwargs['y']['log_id'] = i
# motion model always impute until 20
#######################################################
model_kwargs['y']['cond_until'] = impute_slack
model_kwargs['y']['impute_until'] = impute_slack

### Add motion to inpaint
traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj.to(model_device) # init_motion.to(model_device)
traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask.to(model_device)

# Set when to stop imputing
traj_model_kwargs['y']['cond_until'] = impute_slack
traj_model_kwargs['y']['impute_until'] = impute_until
# NOTE: We have the option of switching the target motion from line to just key locations
# We call this a 'second stage', which will start after t reach 'impute_until'
traj_model_kwargs['y']['impute_until_second_stage'] = impute_slack
traj_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points.to(model_device)
traj_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points.to(model_device)


def gen_traj():
    sample_traj = sample_fn_traj(
        traj_model,
        inpaint_traj.shape,
        clip_denoised=clip_denoised,
        model_kwargs=traj_model_kwargs,  # <-- traj_kwards
        skip_timesteps=0,  # NOTE: for debugging, start from 900
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        cond_fn=partial(
            cond_fn_key_location, # cond_fn_sdf, #,
            transform=gen_loader.dataset.t2m_dataset.transform_th,
            inv_transform=gen_loader.dataset.t2m_dataset.inv_transform_th,
            target=target,
            target_mask=target_mask,
            kframes=[],
            abs_3d=abs_3d, # <<-- hard code,
            classifiler_scale=trajectory_classifier_scale,
            use_mse_loss=False),  # <<-- hard code
    )   
    return sample_traj