"""Factory for creating the HomeRobot ObjectNav environment.

Keeping environment construction here lets evaluation, training, and future
batched wrappers share the same setup without importing the navigator.
"""

import habitat
import tqdm

from habitat.config.read_write import read_write
from utils.habitat_utils import ObjNavEnv, setup_env_config
from curriculum_habitat.helper_wrappers import CLIPWrapper, MemoryWrapper
from curriculum_habitat.curriculum_wrapper import CurriculumVectorEnv, ObjRLNav

DEFAULT_TASK_CONFIG_PATH = "configs/objectnav_hm3d_v2_with_semantic.yaml"
DEFAULT_DATA_PATH = "configs/homerobot_hm3d_objectnav.yaml"

def convert_index_to_action(indexes: int, num_envs: int):
    assert len(indexes.shape) == 2 and indexes.shape[0] == num_envs and indexes.shape[1] == 1
    index_to_name = ["turn_left", "turn_right", "move_forward", "stop"]
    action_array = [index_to_name[indexes] for i in range(num_envs)]
    return action_array

def create_homerobot_env(
    task_config_path: str = DEFAULT_TASK_CONFIG_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    index: int = 0
) -> ObjRLNav:
    config = setup_env_config(
        params_path=data_path,
        default_config_path=task_config_path,
    )
    
    with read_write(config):
        config.habitat.seed = int(config.habitat.seed) + index
    env = ObjRLNav(config=config)
    return env

if __name__ == "__main__":
    NUM_OF_PARALLEL_ENVS = 5
    
    vec_env = CurriculumVectorEnv(make_env_fn = create_homerobot_env,
                                  env_fn_args = [(DEFAULT_TASK_CONFIG_PATH, DEFAULT_DATA_PATH, i) for i in range(NUM_OF_PARALLEL_ENVS)])
    vec_env = CLIPWrapper(vec_env, device = 'cuda')
    vec_env = MemoryWrapper(vec_env)
    vec_env.reset()

    for i in tqdm.tqdm(range(100_000)):
        actions = vec_env.action_space.sample()
        obs, reward, done, info = vec_env.step(actions)