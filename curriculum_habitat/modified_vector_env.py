import signal
from habitat.core.logging import logger

from habitat.core.vector_env import VectorEnv
from habitat.gym.gym_env_episode_count_wrapper import EnvCountEpisodeWrapper
from habitat.gym.gym_env_obs_dict_wrapper import EnvObsDictWrapper
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from multiprocessing.connection import Connection

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
CALL_COMMAND = "call"
COUNT_EPISODES_COMMAND = "count_episodes"

EPISODE_OVER_NAME = "episode_over"
GET_METRICS_NAME = "get_metrics"
CURRENT_EPISODE_NAME = "current_episode"
NUMBER_OF_EPISODE_NAME = "number_of_episodes"
ACTION_SPACE_NAME = "action_space"
ORIG_ACTION_SPACE_NAME = "original_action_space"
OBSERVATION_SPACE_NAME = "observation_space"

class ModifiedVectorEnv(VectorEnv):
    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = EnvCountEpisodeWrapper(EnvObsDictWrapper(env_fn(*env_fn_args)))
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    observations, reward, done, info = env.step(data)

                    if auto_reset_done and done:
                        observations = env.reset()

                    connection_write_fn((observations, reward, done, info))

                elif command == RESET_COMMAND:
                    # CHANGE FROM VectorEnv
                    observations = env.reset(**data)
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None:
                        function_args = {}

                    result_or_fn = getattr(env, function_name)

                    if len(function_args) > 0 or callable(result_or_fn):
                        result = result_or_fn(**function_args)
                    else:
                        result = result_or_fn

                    connection_write_fn(result)

                elif command == COUNT_EPISODES_COMMAND:
                    connection_write_fn(len(env.episodes))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                command, data = connection_read_fn()

        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            if child_pipe is not None:
                child_pipe.close()
            env.close()

    def reset_at(self, index_env: int, mean_distance = None, max_angle_error = None):
        r"""Reset in the index_env environment in the vector.

        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._connection_write_fns[index_env]((RESET_COMMAND, {'mean_distance': mean_distance, 
                                                               'max_angle_error': max_angle_error}))
        results = [self._connection_read_fns[index_env]()]
        return results
    
    def reset(self):
        r"""Reset all the vectorized environments

        :return: list of outputs from the reset method of envs.
        """
        for write_fn in self._connection_write_fns:
            # CHANGE FROM VectorEnv
            write_fn((RESET_COMMAND, {}))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results