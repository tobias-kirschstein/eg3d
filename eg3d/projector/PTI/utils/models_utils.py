import pickle
import functools
import torch
from eg3d.projector.PTI.configs import paths_config, global_config
from eg3d.projector.PTI.utils import legacy

def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G(sampling_multiplier = 2, use_rebalanced: bool = False):
    # with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    #     old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
    #     old_G = old_G.float()
    # return old_G

    eg3d_ffhq_path = paths_config.eg3d_ffhq_rebalanced_pkl if use_rebalanced else paths_config.eg3d_ffhq_pkl
    with open(eg3d_ffhq_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(global_config.device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    return G
