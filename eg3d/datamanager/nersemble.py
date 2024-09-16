from dataclasses import dataclass
from typing import Any, Iterator, Tuple, Dict, List

import numpy as np
from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.matrix import Intrinsics, Pose
from elias.config import Config
from elias.manager import BaseDataManager
from elias.folder.data import DataFolder
from elias.manager.data import _SampleType
from elias.util import load_json, load_img

from eg3d.env import EG3D_DATA_PATH

@dataclass
class ImageMetadata:
    participant_id: int
    sequence_name: str
    timestep: int
    serial: str

def decode_camera_params(camera_params: np.ndarray, disable_rotation_check: bool = False) -> Tuple[Pose, Intrinsics]:
    pose = Pose(camera_params[:16].reshape((4, 4)), pose_type=PoseType.CAM_2_WORLD, disable_rotation_check=disable_rotation_check)
    intrinsics = Intrinsics(camera_params[16:].reshape((3, 3)))
    return pose, intrinsics


def encode_camera_params(pose: Pose, intrinsics: Intrinsics) -> np.ndarray:
    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    return np.concatenate([pose.flatten(), intrinsics.flatten()])


@dataclass
class EG3DCameraParamsFitted(Config):
    intrinsics: np.ndarray
    pose: np.ndarray


@dataclass
class EG3DNeRSembleDataStatistics:
    image_metadatas: List[ImageMetadata]
    available_sequences: Dict[str, Dict[str, List[int]]]  # { "p_id": {seq_1: [0,3,6], seq_2: [10,13,16]}}


class EG3DNeRSembleDataManager(BaseDataManager[None, None, None]):

    def __init__(self, dataset_version: str):
        super(EG3DNeRSembleDataManager, self).__init__(f"{EG3D_DATA_PATH}/nersemble", dataset_version, None)

    def __iter__(self) -> Iterator[_SampleType]:
        raise NotImplementedError()

    def _save(self, data: Any):
        raise NotImplementedError()

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_zip_archive_path(self) -> str:
        return f"{self.get_location()}/nersemble_eg3d_v{self.get_dataset_version()}.zip"

    def get_deep_3d_face_recon_folder(self):
        return f"{self.get_location()}/deep_3d_face_recon"

    def get_deep_3d_face_recon_coefficients_path(self, image_metadata: ImageMetadata) -> str:
        deep_3d_face_recon_name = self.get_image_name(image_metadata)
        return f"{self.get_deep_3d_face_recon_folder()}/fitting_{deep_3d_face_recon_name}.mat"

    def get_image_path(self, image_metadata: ImageMetadata):
        return f"{self.get_location()}/images/{self.get_image_name(image_metadata)}.png"

    def get_detected_landmarks_path(self, image_metadata: ImageMetadata):
        return f"{self.get_location()}/landmarks/{self.get_image_name(image_metadata)}.txt"

    def get_camera_params_path(self, image_metadata: ImageMetadata):
        return f"{self.get_location()}/camera_params/{self.get_image_name(image_metadata)}.npy"

    def get_camera_params_fitted_path(self, image_metadata: ImageMetadata):
        return f"{self.get_location()}/camera_params_fitted/{self.get_image_name(image_metadata)}.json"

    def get_camera_params_fitted_processed_path(self, image_metadata: ImageMetadata):
        return f"{self.get_location()}/camera_params_fitted_processed/{self.get_image_name(image_metadata)}.npy"

    def get_camera_params_calibration_fitted_path(self, image_metadata: ImageMetadata) -> str:
        return f"{self.get_location()}/camera_params_calibration_fitted/{self.get_image_name(image_metadata)}.npy"

    def get_image_name(self, image_metadata: ImageMetadata):
        return f"{image_metadata.participant_id:03d}_{image_metadata.sequence_name}_{image_metadata.timestep:05d}_{image_metadata.serial}"

    # ----------------------------------------------------------
    # Loaders
    # ----------------------------------------------------------

    def load_camera_params(self, image_metadata: ImageMetadata) -> np.ndarray:
        camera_params = np.load(self.get_camera_params_path(image_metadata))
        return camera_params

    def load_camera_params_fitted(self, image_metadata: ImageMetadata) -> EG3DCameraParamsFitted:
        return EG3DCameraParamsFitted.from_json(load_json(self.get_camera_params_fitted_path(image_metadata)))

    def load_camera_params_calibration_fitted(self, image_metadata: ImageMetadata) -> EG3DCameraParamsFitted:
        pose, intrinsics = decode_camera_params(np.load(self.get_camera_params_calibration_fitted_path(image_metadata)))
        return EG3DCameraParamsFitted(intrinsics, pose)

    def load_camera_params_fitted_processed(self, image_metadata: ImageMetadata) -> np.ndarray:
        camera_params = np.load(self.get_camera_params_fitted_processed_path(image_metadata))
        return camera_params

    def load_image(self, image_metadata: ImageMetadata) -> np.ndarray:
        return load_img(self.get_image_path(image_metadata))

    def load_detected_landmarks(self, image_metadata: ImageMetadata) -> np.ndarray:
        return np.loadtxt(self.get_detected_landmarks_path(image_metadata))


class EG3DNeRSembleDataFolder(DataFolder[EG3DNeRSembleDataManager]):

    def __init__(self):
        super().__init__(f"{EG3D_DATA_PATH}/nersemble", localize_via_run_name=True)
