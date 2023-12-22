import abc
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
import wandb
import os.path

from eg3d.arcface.id_loss import IDLoss
from eg3d.metrics.vgg_loss import VGGLoss
from eg3d.projector.PTI.criteria.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
from lpips import LPIPS
from eg3d.projector.PTI.training.projectors import w_projector
from eg3d.projector.PTI.configs import global_config, paths_config, hyperparameters
from eg3d.projector.PTI.criteria import l2_loss
from eg3d.projector.PTI.models.e4e.psp import pSp
from eg3d.projector.PTI.utils.log_utils import log_image_from_w
from eg3d.projector.PTI.utils.models_utils import toogle_grad, load_old_G


@dataclass
class EG3DLossConfig:
    lambda_l1_loss: float = 0
    lambda_l2_loss: float = 1
    lambda_vgg_loss: float = 0
    lambda_lpips_loss: float = 1
    lambda_arcface_loss: float = 0
    lambda_space_regularizer_loss: float = 1


class BaseCoach:
    def __init__(self,
                 data_loader,
                 use_wandb,
                 use_rebalanced: bool = False,
                 loss_config: EG3DLossConfig = EG3DLossConfig()
                 ):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0
        self.use_rebalanced = use_rebalanced

        self._loss_config = loss_config
        self._l1_criterion = torch.nn.L1Loss()
        self._vgg_criterion = VGGLoss().eval().to(global_config.device)
        self._arcface_criterion = IDLoss().eval().to(global_config.device)

        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self,
                         overwrite_noise_buffers: Optional[Dict[str, torch.Tensor]] = None,
                         learning_rate: Optional[float] = None):

        # Initialize networks
        self.G = load_old_G(use_rebalanced=self.use_rebalanced)
        if overwrite_noise_buffers is not None:
            for k, noise_buffer in overwrite_noise_buffers.items():
                self.G.backbone.synthesis.get_buffer(k)[:] = noise_buffer

        toogle_grad(self.G, True)

        self.original_G = load_old_G()

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers(learning_rate=learning_rate)

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if hyperparameters.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{image_name}_w_plus.npy'
        else:
            w_potential_path = f'{w_path_dir}/{image_name}_w.npy'

        print('load pre-computed w from ', w_potential_path)
        if not os.path.isfile(w_potential_path):
            print(w_potential_path, 'is not exist!')
            return None

        w = torch.from_numpy(np.load(w_potential_path)).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, image_name, c):

        if hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)

        else:
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            w = w_projector.project(self.G, c, id_image, device=torch.device(global_config.device), w_avg_samples=600,
                                    num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                    use_wandb=self.use_wandb)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self, learning_rate: Optional[float] = None):
        optimizer = torch.optim.Adam(self.G.parameters(),
                                     lr=hyperparameters.pti_learning_rate if learning_rate is None else learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def calc_loss_proper(self, generated_images, real_images, new_G, w_batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = dict()
        loss: torch.Tensor = 0.0

        # TODO: wandb logging
        if self._loss_config.lambda_l1_loss > 0:
            l1_loss = self._l1_criterion(generated_images, real_images)
            losses['l1'] = l1_loss.item()
            loss += self._loss_config.lambda_l1_loss * l1_loss

        if self._loss_config.lambda_l2_loss > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            losses['l2'] = l2_loss_val.item()
            loss += self._loss_config.lambda_l2_loss * l2_loss_val

        if self._loss_config.lambda_lpips_loss > 0:
            lpips_loss = self.lpips_loss(generated_images, real_images)
            lpips_loss = torch.squeeze(lpips_loss)
            losses['lpips'] = lpips_loss.item()
            loss += self._loss_config.lambda_lpips_loss * lpips_loss

        if self._loss_config.lambda_vgg_loss > 0:
            # TODO: downsample images?
            vgg_loss = self._vgg_criterion(generated_images, real_images)
            losses['vgg'] = vgg_loss.item()
            loss += self._loss_config.lambda_vgg_loss * vgg_loss

        if self._loss_config.lambda_arcface_loss > 0:
            generated_images_downsampled = F.interpolate(generated_images, size=(256, 256), mode='area')
            real_images_downsampled = F.interpolate(real_images.unsqueeze(0), size=(256, 256), mode='area')
            arcface_loss = self._arcface_criterion(generated_images_downsampled, real_images_downsampled)
            losses['arcface'] = arcface_loss.item()
            loss += self._loss_config.lambda_arcface_loss * arcface_loss

        if self._loss_config.lambda_space_regularizer_loss > 0 and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            losses['space_regularizer'] = ball_holder_loss_val
            loss += self._loss_config.lambda_space_regularizer_loss * ball_holder_loss_val

        return loss, losses

    def forward(self, w, c):

        if w.shape[1] != self.G.backbone.mapping.num_ws:
            w = w.repeat([1, self.G.backbone.mapping.num_ws, 1])
        generated_images = self.G.synthesis(w, c, noise_mode='const')['image']

        return generated_images

    def initilize_e4e(self):
        ckpt = torch.load(paths_config.e4e, map_location='cpu')
        opts = ckpt['opts']
        opts['batch_size'] = hyperparameters.train_batch_size
        opts['checkpoint_path'] = paths_config.e4e
        opts = Namespace(**opts)
        self.e4e_inversion_net = pSp(opts)
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net = self.e4e_inversion_net.to(global_config.device)
        toogle_grad(self.e4e_inversion_net, False)

    def get_e4e_inversion(self, image):
        image = (image + 1) / 2
        new_image = self.e4e_image_transform(image[0]).to(global_config.device)
        _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        if self.use_wandb:
            log_image_from_w(w, self.G, 'First e4e inversion')
        return w
