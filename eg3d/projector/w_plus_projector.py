# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from collections import defaultdict
from typing import Literal, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import eg3d.dnnlib as dnnlib
import PIL

from eg3d.arcface.id_loss import IDLoss
from eg3d.camera_utils import LookAtPoseSampler
from eg3d.env import EG3D_MODELS_PATH
from facenet_pytorch import InceptionResnetV1

from eg3d.metrics.vgg_loss import VGGLoss
from eg3d.projector.types import PerceptualLossType


def project(
        G,
        c: torch.Tensor,  # [B, 25]
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=100,
        w_name: str,
        perceptual_loss_type: PerceptualLossType = 'vgg',
        lambda_l2_loss: float = 0,
        lambda_l1_loss: float = 1,
        lambda_vgg_loss: float = 0.1,
        lambda_arcface_loss: float = 0.1,
        lambda_facenet_loss: float = 0,
        initial_latent_code: Optional[torch.Tensor] = None,
        initial_noise_buffers: Optional[Dict[str, torch.Tensor]] = None,
        lambda_regularize_latent_code_distance: float = 0
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    os.makedirs(f'{outdir}/{w_name}_w_plus', exist_ok=True)
    outdir = f'{outdir}/{w_name}_w_plus'
    assert target.shape[-3:] == (G.img_channels, G.img_resolution, G.img_resolution)
    assert len(target.shape) == 3 or target.shape[0] == c.shape[0], "Number of poses and target images should match"

    B = c.shape[0]
    use_random_w_noise = initial_latent_code is None

    # If images from multiple views are provided, we divide the number of global iterations to still end up
    # at the same number of gradient steps
    num_steps = int(num_steps / B)
    image_log_step = int(image_log_step / B)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Compute w stats.
    w_avg_path = './w_avg.npy'
    w_std_path = './w_std.npy'
    if (not os.path.exists(w_avg_path)) or (not os.path.exists(w_std_path)):
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        # c_samples = c.repeat(w_avg_samples, 1)

        # use avg look at point

        camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        focal_length = 4.2647  # FFHQ's FOV
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c_samples = c_samples.repeat(w_avg_samples, 1)

        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        # print('save w_avg  to ./w_avg.npy')
        # np.save('./w_avg.npy',w_avg)
        w_avg_tensor = torch.from_numpy(w_avg).cuda()
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        # np.save(w_avg_path, w_avg)
        # np.save(w_std_path, w_std)
    else:
        # w_avg = np.load(w_avg_path)
        # w_std = np.load(w_std_path)
        raise Exception(' ')

    # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    # c_samples = c.repeat(w_avg_samples, 1)
    # w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
    # w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    # w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    # w_avg_tensor = torch.from_numpy(w_avg).cuda()
    # w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    # Overwrite noise buffers with values from previous view
    if initial_noise_buffers is not None:
        for k, noise_buffer in initial_noise_buffers.items():
            G.backbone.synthesis.get_buffer(k)[:] = noise_buffer

    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    facenet = None
    arcface = None
    feature_image_size = None
    if 'vgg' in perceptual_loss_type :
        # Load VGG16 feature detector.
        # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        # url = './networks/vgg16.pt'
        feature_image_size = 256
        # url = f"{EG3D_MODELS_PATH}/vgg16.pt"
        # with dnnlib.util.open_url(url) as f:
        #     vgg16 = torch.jit.load(f).eval().to(device)

        vgg_19 = VGGLoss().eval().to(device)

    if perceptual_loss_type == 'facenet':
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        feature_image_size = 160
        for param in facenet.parameters():
            param.requires_grad = False

    if 'arcface' in perceptual_loss_type:
        arcface = IDLoss().eval().to(device)
        feature_image_size = 256
        for param in arcface.parameters():
            param.requires_grad = False

    # Features for target image.
    if len(target.shape) == 3:
        target = target.unsqueeze(0)  # Add batch dimension with single image
    target_images = target.to(device).to(torch.float32)
    if target_images.shape[2] > feature_image_size:
        target_images_downsampled = F.interpolate(target_images, size=(feature_image_size, feature_image_size),
                                                  mode='area')
    else:
        target_images_downsampled = target_images

    # if 'vgg' in perceptual_loss_type:
    #     target_features_vgg = vgg16(target_images_downsampled, resize_images=False, return_lpips=True)

    if perceptual_loss_type == 'facenet':
        target_features_facenet = facenet(target_images_downsampled / 255.)

    if initial_latent_code is None:
        start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)  # [1, 14, 512]
    else:
        start_w = initial_latent_code
        initial_latent_code = initial_latent_code.to(device)

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=0.1)

    # Init noise.
    if initial_noise_buffers is None:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    loss_history = defaultdict(list)

    for step in tqdm(range(num_steps)):

        for i_image in range(B):

            # Learning rate schedule.
            t = step / num_steps

            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss_history['lr'].append(lr)

            w_opt_repeated = w_opt.repeat((B, 1, 1))

            if use_random_w_noise:
                w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

                # Synth images from opt_w.
                # w_noise = torch.randn_like(w_opt) * w_noise_scale
                w_noise = torch.randn((B, w_opt.shape[1], w_opt.shape[2]), dtype=w_opt.dtype,
                                      device=w_opt.device) * w_noise_scale

                ws = (w_opt_repeated + w_noise)
            else:
                ws = w_opt_repeated

            synth_images = G.synthesis(ws[[i_image]], c[[i_image]], noise_mode='const')['image']

            if step % image_log_step == 0 or step == num_steps - 1:
                with torch.no_grad():
                    vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                    PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(
                        f'{outdir}/image_{i_image}_step_{step}.png')

            synth_images = (synth_images + 1) * (255 / 2)  # [-1, 1] -> [0, 255]

            if synth_images.shape[2] > feature_image_size:
                synth_images_downsampled = F.interpolate(synth_images, size=(feature_image_size, feature_image_size),
                                                         mode='area')
            else:
                synth_images_downsampled = synth_images

            dist = 0.0
            if 'vgg' in perceptual_loss_type:
                # synth_features_vgg = vgg16(synth_images_downsampled, resize_images=False, return_lpips=True)
                # vgg_loss = (target_features_vgg[[i_image]] - synth_features_vgg).square().sum()

                vgg_loss = vgg_19((synth_images_downsampled / 255) * 2 - 1, (target_images_downsampled[[i_image]] / 255.) * 2 - 1)

                loss_history['vgg_loss'].append(vgg_loss.item())
                dist += lambda_vgg_loss * vgg_loss
            if perceptual_loss_type == 'facenet':
                # TODO: Maybe image has to be scaled here?
                synth_features_facenet = facenet(synth_images_downsampled / 255.)
                facenet_loss = (target_features_facenet[[i_image]] - synth_features_facenet).square().sum()
                loss_history['facenet_loss'].append(facenet_loss.item())
                dist += lambda_facenet_loss * facenet_loss

            if 'arcface' in perceptual_loss_type:
                arcface_loss = arcface((synth_images_downsampled / 255) * 2 - 1, (target_images_downsampled[[i_image]] / 255.) * 2 - 1)
                loss_history['arcface_loss'].append(arcface_loss.item())
                dist += lambda_arcface_loss * arcface_loss

            # if perceptual_loss_type == 'arcface':
            #     # ArcFace uses cosine similarity instead of L2 distance of latent codes
            #     dist = arcface((synth_images_downsampled / 255) * 2 - 1, (target_images_downsampled / 255.) * 2 - 1)
            # else:
            #     dist = (target_features[[i_image]] - synth_features).square().sum()
            # dist = ((target_images - synth_images)).norm(p=1, dim=1).mean()

            loss_history["L2"].append((synth_images - target_images).norm(p=2, dim=1).mean().item())
            loss_history["L1"].append((synth_images - target_images).norm(p=1, dim=1).mean().item())

            if lambda_l2_loss > 0:
                # TODO: L2 loss makes reconstruction blurry
                dist += lambda_l2_loss * (synth_images - target_images).norm(p=2, dim=1).mean()

            if lambda_l1_loss > 0:
                dist += lambda_l1_loss * (synth_images - target_images).norm(p=1, dim=1).mean()

            if lambda_regularize_latent_code_distance > 0 and initial_latent_code is not None:
                latent_code_distance = (w_opt - initial_latent_code).norm(p=2, dim=2).mean()
                dist += lambda_regularize_latent_code_distance * latent_code_distance
                loss_history['latent_code_distance']

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

            loss_history['noise_regularization'].append(reg_loss.item())
            loss = dist + reg_loss * regularize_noise_weight

            # if step % 10 == 0:
            #     with torch.no_grad():
            #         print({f'step {step}, first projection _{w_name}': loss.detach().cpu()})

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    for k, buffer in noise_bufs.items():
        noise_bufs[k] = buffer.detach()

    del G
    return w_opt, loss_history, noise_bufs
