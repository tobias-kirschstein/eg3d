import inspect
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import scipy
import torch
from Deep3DFaceRecon_pytorch.env import DEEP_3D_FACE_RECON_BFM_FOLDER
from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from dreifus.matrix import Intrinsics

import numpy as np

# Important to have import tensorflow here because tensorflow is garbage. Otherwise, we get a stupid
# "msvcp140_1.dll not found" although it is obviously there...
# noinspection PyUnresolvedReferences
import tensorflow
from Deep3DFaceRecon_pytorch import test
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
from Deep3DFaceRecon_pytorch.util.preprocess import align_img
from PIL import Image
from mtcnn import MTCNN


# ==========================================================
# Poses
# ==========================================================
from eg3d.camera_utils import create_cam2world_matrix


def fix_intrinsics(intrinsics):
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0, 0] = 2985.29 / 700
    intrinsics[1, 1] = 2985.29 / 700
    intrinsics[0, 2] = 1 / 2
    intrinsics[1, 2] = 1 / 2
    assert intrinsics[0, 1] == 0
    assert intrinsics[2, 2] == 1
    assert intrinsics[1, 0] == 0
    assert intrinsics[2, 0] == 0
    assert intrinsics[2, 1] == 0
    return intrinsics


# For our recropped images, with correction
def fix_pose(pose):
    COR = np.array([0, 0, 0.175])
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    direction = (location - COR) / np.linalg.norm(location - COR)
    pose[:3, 3] = direction * 2.7 + COR  # Apparently, camera is "fixed" to be 2.7 away from origin
    return pose


# Used in original submission
def fix_pose_orig(pose):
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3] / radius * 2.7  # Apparently, camera is "fixed" to be 2.7 away from origin
    return pose


# Used for original crop images
def fix_pose_simplify(pose):
    cam_location = torch.tensor(pose).clone()[:3, 3]
    normalized_cam_location = torch.nn.functional.normalize(cam_location - torch.tensor([0, 0, 0.175]), dim=0)
    camera_view_dir = - normalized_cam_location
    camera_pos = 2.7 * normalized_cam_location + np.array(
        [0, 0, 0.175])  # Apparently, camera is "fixed" to be 2.7 away from origin
    simple_pose_matrix = create_cam2world_matrix(camera_view_dir.unsqueeze(0), camera_pos.unsqueeze(0))[0]
    return simple_pose_matrix.numpy()


# ==========================================================
# Landmarks
# ==========================================================


def detect_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    detector = MTCNN()
    result = detector.detect_faces(image)

    index = 0
    if len(result) > 1:  # if multiple faces, take the biggest face
        size = -100000
        for r in range(len(result)):
            size_ = result[r]["box"][2] + result[r]["box"][3]
            if size < size_:
                size = size_
                index = r

    # detected_landmarks_path = f"{DETECTED_LANDMARKS_DATA_PATH}/{participant_id}_{sequence_name}_{timestep}_{serial}.txt"
    # ensure_directory_exists_for_file(detected_landmarks_path)

    # bounding_box = result[index]['box']
    keypoints = result[index]['keypoints']
    if result[index]["confidence"] > 0.9:
        landmark_data = np.array([
            list(keypoints['left_eye']),
            list(keypoints['right_eye']),
            list(keypoints['nose']),
            list(keypoints['mouth_left']),
            list(keypoints['mouth_right']),
        ])
        # np.savetxt(detected_landmarks_path, landmark_data)
        # print(result)

        return landmark_data
    else:
        return None


# ==========================================================
# Image crops
# ==========================================================


def crop_and_align_image(detected_landmarks: np.ndarray, image: np.ndarray, intrinsics: Intrinsics) \
        -> np.ndarray:
    # lm = np.loadtxt(detected_landmarks_path).astype(np.float32)
    lm = detected_landmarks.copy()
    H = image.shape[0]

    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size = 512

    # Load 3D landmarks for Basel Face Model from Deep3DFaceRecon_pytorch repo
    deep_3d_face_recon_repo_root = Path(inspect.getfile(test)).parent
    bfm_path = f"{deep_3d_face_recon_repo_root}/BFM"
    lm3d_std = load_lm3d(bfm_path)

    _, im_high, _, _, resize_params, crop_params = align_img(Image.fromarray(image),
                                                             lm,
                                                             lm3d_std,
                                                             target_size=target_size,
                                                             rescale_factor=rescale_factor,
                                                             return_resize_and_crop_params=True)

    intrinsics.rescale(resize_params[0] / image.shape[1], resize_params[1] / image.shape[0])
    intrinsics.crop(crop_left=crop_params[0], crop_top=crop_params[1])

    # Second crop (just 700x700 center from 1024x1024 landmark crop which might contain black bars on sides)
    left = int(im_high.size[0] / 2 - center_crop_size / 2)
    upper = int(im_high.size[1] / 2 - center_crop_size / 2)
    right = left + center_crop_size
    lower = upper + center_crop_size

    im_cropped = im_high.crop((left, upper, right, lower))
    rescale_factor_x = output_size / im_cropped.size[0]
    rescale_factor_y = output_size / im_cropped.size[1]

    im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
    intrinsics.crop(crop_left=left, crop_top=upper)
    intrinsics.rescale(rescale_factor_x, rescale_factor_y)
    im_cropped = np.asarray(im_cropped)

    # Scale intrinsics to correspond to a normalized image screen space of [0,1]^2
    intrinsics.rescale(1. / im_cropped.shape[1], 1. / im_cropped.shape[0])

    return im_cropped


# ==========================================================
# Transform poses
# ==========================================================


def transform_Deep3DFaceRecon_fit(coefficients_path) -> Dict[str, List[float]]:
    face_model = ParametricFaceModel(bfm_folder=DEEP_3D_FACE_RECON_BFM_FOLDER)

    dict_load = scipy.io.loadmat(coefficients_path)
    angle = dict_load['angle']
    trans = dict_load['trans'][0]
    R = face_model.compute_rotation(torch.from_numpy(angle))[0].numpy()
    trans[2] += -10
    c = -np.dot(R, trans)
    pose = np.eye(4)
    pose[:3, :3] = R

    c *= 0.27  # normalize camera radius
    c[1] += 0.006  # additional offset used in submission
    c[2] += 0.161  # additional offset used in submission
    pose[0, 3] = c[0]
    pose[1, 3] = c[1]
    pose[2, 3] = c[2]

    focal = 2985.29  # = 1015*1024/224*(300/466.285)#
    pp = 512  # 112
    w = 1024  # 224
    h = 1024  # 224

    count = 0
    K = np.eye(3)
    K[0][0] = focal
    K[1][1] = focal
    K[0][2] = w / 2.0
    K[1][2] = h / 2.0
    K = K.tolist()

    Rot = np.eye(3)
    Rot[0, 0] = 1
    Rot[1, 1] = -1
    Rot[2, 2] = -1
    pose[:3, :3] = np.dot(pose[:3, :3], Rot)

    pose = pose.tolist()
    out = {}
    out["intrinsics"] = K
    out["pose"] = pose

    return out


def process_fitted_camera_params(camera_params_fitted: Dict[str, List[float]]) -> np.ndarray:
    pose = camera_params_fitted['pose']
    intrinsics = camera_params_fitted['intrinsics']

    pose = fix_pose_orig(pose)

    intrinsics = fix_intrinsics(intrinsics)
    processed_camera_params_fitted = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)])

    return processed_camera_params_fitted
