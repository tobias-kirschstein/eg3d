import io
import json
import zipfile

import numpy as np
import tyro
from dreifus.camera import PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum
from famudy.constants import SERIALS
from famudy.data.capture_data_processed_v2 import ImageMetadata

from eg3d.datamanager.nersemble import EG3DNerRSembleDataFolder, decode_camera_params
from eg3d.env import EG3D_DATA_PATH
from PIL import Image
import pyvista as pv

"""
format:

{"labels": 
    [
        [
            "00000/img00000000.png", 
            [0.94228595495224, 0.0342920757830143, 0.33304843306541443, -0.8367821825319752, 
            0.03984856605529785, -0.9991570711135864, -0.00986515637487173, 0.017003142690765312, 
            0.3324293792247772, 0.022567300125956535, -0.942858099937439, 2.5670034032185587, 
            0.0, 0.0, 0.0, 1.0, 
            4.2647, 0.0, 0.5, 
            0.0, 4.2647, 0.5, 
            0.0, 0.0, 1.0
        ]
    ],
    ...
}

"""


def visualize_eg3d_poses(p: pv.Plotter, archive_path: str, n_poses: int = 10, color: str = 'lightgray', dataset_json_path: str = 'dataset.json'):
    all_images = []
    all_poses = []
    all_intrinsics = []

    with zipfile.ZipFile(archive_path, 'r') as ffhq_archive:
        dataset_json = ffhq_archive.read(dataset_json_path)
        dataset_json = json.loads(dataset_json.decode('utf-8'))

        for i in range(n_poses):
            img_path = dataset_json['labels'][i][0]
            img = np.asarray(Image.open(io.BytesIO(ffhq_archive.read(img_path))))
            cam_2_world = Pose(np.array(dataset_json['labels'][i][1][:16]).reshape((4, 4)), pose_type=PoseType.CAM_2_WORLD)
            intrinsics = Intrinsics(np.array(dataset_json['labels'][i][1][16:]).reshape((3, 3)))

            all_images.append(img)
            all_poses.append(cam_2_world)
            all_intrinsics.append(intrinsics)

    add_coordinate_axes(p, scale=0.1)

    for pose, intr, img in zip(all_poses, all_intrinsics, all_images):
        add_camera_frustum(p, pose, intr, image=img, color=color, img_h=1, img_w=1, size=0.5)


def main():
    n_poses = 10
    nersemble_dataset_version = 'v0.5'
    ffhq_dataset_path = f"{EG3D_DATA_PATH}/FFHQ_png_512.zip"

    dataset_manager = EG3DNerRSembleDataFolder().open_dataset(nersemble_dataset_version)
    nersemble_dataset_path  =dataset_manager.get_zip_archive_path()

    p = pv.Plotter()
    add_coordinate_axes(p, scale=0.1)

    # visualize_eg3d_poses(p, ffhq_dataset_path, n_poses=n_poses)
    visualize_eg3d_poses(p, nersemble_dataset_path, n_poses=n_poses, color='red')
    visualize_eg3d_poses(p, nersemble_dataset_path, n_poses=n_poses, color='purple', dataset_json_path='dataset_calibration_fitted.json')

    # for serial in SERIALS:
    #     image_metadata = ImageMetadata(37, "EXP-1-head", 0, serial)
    #     camera_params = dataset_manager.load_camera_params(image_metadata)
    #     cam_2_world_pose, intrinsics = decode_camera_params(camera_params)
    #     add_camera_frustum(p, cam_2_world_pose, intrinsics, color='orange')
    #
    #     camera_params_fitted = dataset_manager.load_camera_params_fitted(image_metadata)
    #     add_camera_frustum(p, Pose(camera_params_fitted.pose, pose_type=PoseType.CAM_2_WORLD), Intrinsics(camera_params_fitted.intrinsics), color='blue')

    p.show()


if __name__ == '__main__':
    tyro.cli(main)
