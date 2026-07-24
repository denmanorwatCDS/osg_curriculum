"""Run a vectorized curriculum rollout while recording every episode."""

import tqdm
import comet_ml

from utils.habitat_utils import setup_env_config
from habitat.config import read_write
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
DEFAULT_DATA_PATH = "configs/homerobot_hm3d_objectnav_train.yaml"
DEFAULT_EVAL_DATA_PATH = "configs/homerobot_hm3d_objectnav_val.yaml"
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
    with read_write(config):
        config.habitat.seed = int(config.habitat.seed) + index
    env = ObjRLNav(config=config)
    return env


def main():
    experiment = comet_ml.start(project_name="cogmodel-curriculum-sanity-test")
    vec_env = CurriculumVectorEnv(
        make_env_fn=create_homerobot_env,
        env_fn_args=[
            (DEFAULT_TASK_CONFIG_PATH, DEFAULT_DATA_PATH, index)
            for index in range(NUM_OF_PARALLEL_ENVS)
        ],
        stage_zero_experience = 5_000,
    )
    # The recorder must see RGB observations before CLIP replaces them with
    # embeddings. All MP4 files are written under ./debug_videos by default.
    vec_env = DebugVideoWrapper(vec_env)
    vec_env = CLIPWrapper(vec_env, device="cuda")
    vec_env = MemoryWrapper(vec_env)

    try:
        STEPS_BEFORE_UNDERSTANDING_APPEARS = 2_500
        steps_after_difficulty_increase = 0

        vec_env.reset()

        for i in tqdm.tqdm(range(NUM_OF_STEPS)):
            smart_agents = min(steps_after_difficulty_increase / STEPS_BEFORE_UNDERSTANDING_APPEARS, 1) * NUM_OF_PARALLEL_ENVS
            stupid_agents = NUM_OF_PARALLEL_ENVS - smart_agents

            actions = vec_env.action_space.sample()
            actions[:int(smart_agents)] = -1

            vec_env.step(actions)
            steps_after_difficulty_increase += 1
            if (i + 1) % 200 == 0:
                experiment.log_metrics(vec_env.get_logging_stats(), step=i + 1)

            if vec_env.check_if_difficulty_has_recently_changed():
                steps_after_difficulty_increase = 0
                
    finally:
        # Flush the active episode from every vector slot even if interrupted.
        vec_env.close()
        experiment.end()


if __name__ == "__main__":
    main()
