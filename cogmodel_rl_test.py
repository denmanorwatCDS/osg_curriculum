from utils.habitat_utils import setup_env_config
from habitat.config import read_write
from skrl.envs.wrappers.torch import wrap_env
from curriculum_habitat.helper_wrappers import (
    CLIPWrapper,
    ToSKRLWrapper,
)
from curriculum_habitat.RL.SKRL_eval_trainer import EvalSequentialTrainer
from curriculum_habitat.RL.DDQN_stub import create_ddqn
from curriculum_habitat.curriculum_wrapper import (
    CurriculumVectorEnv,
    EvalVectorEnv,
    ObjRLNav,
)



DEFAULT_TASK_CONFIG_PATH = "configs/objectnav_hm3d_v2_with_semantic.yaml"
DEFAULT_DATA_PATH = "configs/homerobot_hm3d_objectnav_train.yaml"
EVAL_DATA_PATH = "configs/homerobot_hm3d_objectnav_val.yaml"

NUM_OF_PARALLEL_ENVS = 5
EVAL_ROUNDS = 5

NUM_OF_STEPS = 100_000
EVAL_INTERVAL = 500


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

def make_env_vectorised(create_env_fn, task_config_path, data_path, num_envs):
    vec_env = CurriculumVectorEnv(
            make_env_fn=create_env_fn,
            env_fn_args=[
                (task_config_path, data_path, index)
                for index in range(num_envs)
            ],
        )
    vec_env = CLIPWrapper(vec_env, device="cuda")
    vec_env = ToSKRLWrapper(vec_env, device="cuda")
    vec_env = wrap_env(vec_env, wrapper='gymnasium')
    return vec_env

def create_eval_homerobot_env(task_config_path, data_path, index):
    config = setup_env_config(
        params_path=data_path,
        default_config_path=task_config_path,
    )
    with read_write(config):
        config.habitat.seed = int(config.habitat.seed) + index
        config.habitat.environment.max_episode_steps = 500
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.cycle = True

    env = ObjRLNav(config=config)
    env.eval_episodes_per_round = len(env.episodes)
    return env


def create_evaluation_environment():
    eval_env = EvalVectorEnv(
        make_env_fn=create_eval_homerobot_env,
        env_fn_args=[
            (DEFAULT_TASK_CONFIG_PATH, EVAL_DATA_PATH, index)
            for index in range(EVAL_ROUNDS)
        ],
    )
    episodes_per_env = eval_env.call_at(0, "eval_episodes_per_round")
    eval_env = CLIPWrapper(eval_env, device="cuda")
    eval_env = ToSKRLWrapper(eval_env, device="cuda")
    return wrap_env(eval_env, wrapper="gymnasium"), episodes_per_env


def main():
    vec_env = make_env_vectorised(
        create_homerobot_env,
        DEFAULT_TASK_CONFIG_PATH,
        DEFAULT_DATA_PATH,
        NUM_OF_PARALLEL_ENVS,
    )
    eval_env, eval_episodes_per_env = create_evaluation_environment()

    try:
        agent = create_ddqn(vec_env)
        trainer = EvalSequentialTrainer(
            env=vec_env,
            eval_env=eval_env,
            agents=agent,
            eval_episodes_per_env=eval_episodes_per_env,
            cfg={
                "timesteps": NUM_OF_STEPS,
                "headless": True,
                "close_environment_at_exit": False,
            },
        )
        for end_step in range(EVAL_INTERVAL, NUM_OF_STEPS + 1, EVAL_INTERVAL):
            trainer.initial_timestep = end_step - EVAL_INTERVAL
            trainer.timesteps = end_step
            trainer.train()
            metrics = trainer.eval()
            print(f"Evaluation at step {end_step}: {metrics}")
    finally:
        # Flush the active episode from every vector slot even if interrupted.
        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
