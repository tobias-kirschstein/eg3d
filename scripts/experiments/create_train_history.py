from typing import List

import mediapy
import tyro
from elias.folder import Folder
from elias.util import load_img
from tqdm import tqdm

from eg3d.env import EG3D_MODELS_PATH_REMOTE


def load_images_and_write_video(folder: str, image_file_names: List[str], output_name: str, fps: int = 8):
    fake_pngs = []
    for fake_png_file in tqdm(image_file_names):
        fake_png = load_img(f"{folder}/{fake_png_file}")
        fake_pngs.append(fake_png)

    mediapy.write_video(f"{folder}/{output_name}.mp4", fake_pngs, fps=fps)


def main(run_name: str, /, fps: int = 8):
    models_path = EG3D_MODELS_PATH_REMOTE
    model_path = f"{models_path}/{run_name}"
    fake_png_files = Folder(model_path).list_file_numbering("fakes$.png", return_only_file_names=True)
    fake_png_depth_files = Folder(model_path).list_file_numbering("fakes$_depth.png", return_only_file_names=True)
    fake_png_raw_files = Folder(model_path).list_file_numbering("fakes$_raw.png", return_only_file_names=True)

    load_images_and_write_video(model_path, fake_png_files, "fakes", fps=fps)
    load_images_and_write_video(model_path, fake_png_depth_files, "fakes_depth", fps=fps)
    load_images_and_write_video(model_path, fake_png_raw_files, "fakes_raw", fps=fps)


if __name__ == '__main__':
    tyro.cli(main)
