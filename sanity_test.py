"""Run a vectorized curriculum rollout while recording every episode."""

import tqdm

from utils.habitat_utils import setup_env_config
from curriculum_habitat.helper_wrappers import (
    CLIPWrapper,
    DebugVideoWrapper,
    MemoryWrapper,
)
from curriculum_habitat.curriculum_wrapper import (
    CurriculumVectorEnv,
    ObjRLNav,
)


DEFAULT_TASK_CONFIG_PATH = "configs/objectnav_hm3d_v2_with_semantic.yaml"
DEFAULT_DATA_PATH = "configs/homerobot_hm3d_objectnav.yaml"
NUM_OF_PARALLEL_ENVS = 5
NUM_OF_STEPS = 100_000


def create_homerobot_env(
    task_config_path=DEFAULT_TASK_CONFIG_PATH,
    data_path=DEFAULT_DATA_PATH,
    index=0,
):
    config = setup_env_config(
        params_path=data_path,
        default_config_path=task_config_path,
    )
    env = ObjRLNav(config=config)
    env.seed(config.habitat.seed + index)
    return env


def main():
    vec_env = CurriculumVectorEnv(
        make_env_fn=create_homerobot_env,
        env_fn_args=[
            (DEFAULT_TASK_CONFIG_PATH, DEFAULT_DATA_PATH, index)
            for index in range(NUM_OF_PARALLEL_ENVS)
        ],
    )
    # The recorder must see RGB observations before CLIP replaces them with
    # embeddings. All MP4 files are written under ./debug_videos by default.
    vec_env = DebugVideoWrapper(vec_env)
    vec_env = CLIPWrapper(vec_env, device="cuda")
    vec_env = MemoryWrapper(vec_env)

    try:
        vec_env.reset()
        for _ in tqdm.tqdm(range(NUM_OF_STEPS)):
            actions = vec_env.action_space.sample()
            vec_env.step(actions)
    finally:
        # Flush the active episode from every vector slot even if interrupted.
        vec_env.close()


if __name__ == "__main__":
    main()
