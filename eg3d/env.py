from pathlib import Path
from environs import Env

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/eg3d/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("EG3D_"):
    EG3D_MODELS_PATH = env("MODELS_PATH", f"<<<Define EG3D_MODELS_PATH in {env_file_path}>>>")

REPO_ROOT_DIR = f"{Path(__file__).parent.resolve()}/../.."

