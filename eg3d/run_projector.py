# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""
import json
import os
import re
from typing import List, Optional, Tuple, Union, Dict

import click
import eg3d.dnnlib as dnnlib
import numpy as np
import torch
import eg3d.legacy as legacy
from torchvision.transforms import transforms
from eg3d.projector import w_projector, w_plus_projector
from PIL import Image

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
from eg3d.projector.types import PerceptualLossType


def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False,
              metavar='STR',
              default='w', show_default=True)
@click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int,
              help='Multiplier for depth sampling in volume rendering', default=500, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type: str,
        image_path: Union[List[str], str],
        c_path: Union[List[str], str],
        num_steps: int,
        output_name: Optional[str] = None,
        perceptual_loss_type: PerceptualLossType = 'vgg',
        initial_latent_code: Optional[np.ndarray] = None,
        initial_noise_buffers: Optional[Dict[str, torch.Tensor]] = None,
        lambda_vgg_loss: float = 1,
        lambda_l1_loss: float = 0,
        lambda_l2_loss: float = 0,
        lambda_arcface_loss: float = 0
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not isinstance(c_path, list):
        c_path = [c_path]
    if not isinstance(image_path, list):
        image_path = [image_path]

    assert len(c_path) == len(image_path), "Number of camera params and image paths must match"

    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((512, 512))
    ])

    # Load images
    id_images = []
    for single_image_path in image_path:
        image = Image.open(single_image_path).convert('RGB')
        image_name = os.path.basename(single_image_path)[:-4]

        from_im = trans(image).cuda()
        id_image = torch.squeeze((from_im.cuda() + 1) / 2) * 255
        id_images.append(id_image)

    if output_name is not None:
        image_name = output_name

    id_images = torch.stack(id_images, dim=0)  # [B, C, H, W]

    # Load cameras
    cs = []
    for single_c_path in c_path:
        c = np.load(single_c_path)
        c = np.reshape(c, (1, 25))

        c = torch.FloatTensor(c).cuda()
        cs.append(c)
    cs = torch.concat(cs, dim=0)  # [B, 25]

    # Process initial latent codes
    if initial_latent_code is not None:
        initial_latent_code = torch.Tensor(initial_latent_code)

    if latent_space_type == 'w':

        w = w_projector.project(G, cs, outdir, id_images, device=torch.device('cuda'), w_avg_samples=600,
                                num_steps=num_steps,
                                w_name=image_name)
        loss_history = None
    else:

        w, loss_history, noise_buffers = w_plus_projector.project(G, cs, outdir, id_images,
                                                                  device=torch.device('cuda'),
                                                                  w_avg_samples=600,
                                                                  w_name=image_name, num_steps=num_steps,
                                                                  perceptual_loss_type=perceptual_loss_type,
                                                                  initial_latent_code=initial_latent_code,
                                                                  initial_noise_buffers=initial_noise_buffers,
                                                                  lambda_regularize_latent_code_distance=0.01,
                                                                  lambda_l1_loss=lambda_l1_loss,
                                                                  lambda_l2_loss=lambda_l2_loss,
                                                                  lambda_vgg_loss=lambda_vgg_loss,
                                                                  lambda_arcface_loss=lambda_arcface_loss)

    w = w.detach().cpu().numpy()
    np.save(f'{outdir}/{image_name}_{latent_space_type}/{image_name}_{latent_space_type}.npy', w)
    with open(f'{outdir}/{image_name}_{latent_space_type}/projector_loss_history.json', 'w', encoding='utf-8') as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=4)
    torch.save(noise_buffers, f'{outdir}/{image_name}_{latent_space_type}/noise_buffers.pth')

    # PTI_embedding_dir = f'./projector/PTI/embeddings/{image_name}'
    # os.makedirs(PTI_embedding_dir, exist_ok=True)
    #
    # np.save(f'./projector/PTI/embeddings/{image_name}/{image_name}_{latent_space_type}.npy', w)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
