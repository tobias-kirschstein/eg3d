import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from eg3d.projector.PTI.configs import paths_config, hyperparameters, global_config
from eg3d.projector.PTI.training.coaches.base_coach import BaseCoach, EG3DLossConfig
from eg3d.projector.PTI.utils.log_utils import log_images_from_w
import numpy as np
from PIL import Image
from typing import Optional, Dict, List


class MultiImageCoach(BaseCoach):

    def __init__(self,
                 trans,
                 use_rebalanced: bool = False,
                 loss_config: EG3DLossConfig = EG3DLossConfig()
                 ):
        super().__init__(data_loader=None, use_wandb=False, use_rebalanced=use_rebalanced, loss_config=loss_config)
        self.source_transform = trans

    def train(self,
              image_paths: List[str],
              w_path: str,
              c_paths: List[str],
              output_ckpt_path: Optional[str] = None,
              output_loss_history_path: Optional[str] = None,
              overwrite_noise_buffers: Optional[Dict[str, torch.Tensor]] = None,
              n_steps: Optional[int] = None,
              use_gradient_accumulation: bool = False):

        assert len(image_paths) == len(c_paths), "Number of images must match number of cameras"
        n_views = len(image_paths)

        use_ball_holder = True

        name = os.path.basename(w_path)[:-4]

        # Load cameras and images
        cameras = []
        images = []
        for image_path, c_path in zip(image_paths, c_paths):
            print("image_path: ", image_path, 'c_path', c_path)

            c = np.load(c_path)
            c = np.reshape(c, (1, 25))
            c = torch.FloatTensor(c).cuda()
            cameras.append(c)

            from_im = Image.open(image_path).convert('RGB')
            if self.source_transform:
                image = self.source_transform(from_im)
            image = image.to(global_config.device)
            images.append(image)

        self.restart_training(overwrite_noise_buffers=overwrite_noise_buffers, learning_rate=1e-3)

        print('load pre-computed w from ', w_path)
        if not os.path.isfile(w_path):
            print(w_path, 'is not exist!')
            return None

        w_pivot = torch.from_numpy(np.load(w_path)).to(global_config.device)

        # w_pivot = w_pivot.detach().clone().to(global_config.device)
        w_pivot = w_pivot.to(global_config.device)

        log_images_counter = 0

        loss_history = defaultdict(list)

        self.optimizer.zero_grad()
        for i in tqdm(range(hyperparameters.max_pti_steps if n_steps is None else n_steps)):
            if use_gradient_accumulation:
                idx_random_view = i % n_views
            else:
                idx_random_view = np.random.randint(0, n_views)

            c = cameras[idx_random_view]
            real_images_batch = images[idx_random_view]

            generated_images = self.forward(w_pivot, c)

            loss, losses = self.calc_loss_proper(generated_images, real_images_batch, self.G, w_pivot)
            for loss_name, loss_value in losses.items():
                loss_history[loss_name].append(loss_value)

            if 'lpips' in losses and losses['lpips'] <= hyperparameters.LPIPS_value_threshold:
                break

            loss.backward()
            if not use_gradient_accumulation or (i + 1) % n_views == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

            global_config.training_step += 1
            log_images_counter += 1

        self.image_counter += 1

        save_dict = {
            'G_ema': self.G.state_dict()
        }
        if output_ckpt_path is not None:
            checkpoint_path = output_ckpt_path
        else:
            checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{name}.pth'

        print('final model ckpt save to ', checkpoint_path)
        torch.save(save_dict, checkpoint_path)

        if output_loss_history_path is not None:
            with open(output_loss_history_path, 'w', encoding='utf-8') as f:
                json.dump(loss_history, f, ensure_ascii=False, indent=4)
