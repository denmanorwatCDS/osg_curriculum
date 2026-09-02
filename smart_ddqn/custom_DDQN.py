import io
import math
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.dqn.ddqn import DDQN

from comet_ml import start
from comet_ml.integration.pytorch import log_model


def best_action_advantage(action_batch: torch.Tensor) -> torch.Tensor:
    """Return the mean advantage of the highest-valued action in each state.

    DDQN has no separate value head, so the mean Q-value across discrete
    actions is used as the state-value baseline.
    """
    if action_batch.ndim < 2 or action_batch.shape[-1] == 0:
        raise ValueError(
            "action_batch must have shape (..., num_actions) with at least "
            "one action"
        )

    action_values = action_batch.detach()
    best_values = action_values.max(dim=-1).values
    baseline_values = action_values.mean(dim=-1)
    return (best_values - baseline_values).mean()

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

class CometWriter:
    def __init__(self, ):
        self.experiment = start(
            api_key=os.environ['COMET_API_KEY'], project_name="general", workspace="girol-osg")

    def add_scalar(self, name, value, timestep):
        self.experiment.log_metric(name, value, step=timestep)

    def add_video(self, env_idx, video_tensor, timestep, fps=10):
        """Encode an RGB frame batch in memory and upload it to Comet.

        ``video_tensor`` may be a NumPy array or torch tensor in ``THWC`` or
        ``TCHW`` layout. A leading singleton batch dimension is also accepted.
        Floating-point frames in the range ``[0, 1]`` are scaled to uint8.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive; got {fps}")

        if torch.is_tensor(video_tensor):
            video_tensor = video_tensor.detach().cpu()
            if video_tensor.dtype == torch.bfloat16:
                video_tensor = video_tensor.float()
            frames = video_tensor.numpy()
        else:
            frames = np.asarray(video_tensor)

        if frames.ndim == 5:
            if frames.shape[0] != 1:
                raise ValueError(
                    "A 5D video tensor must have a singleton batch dimension; "
                    f"got shape {frames.shape}"
                )
            frames = frames[0]

        if frames.ndim != 4:
            raise ValueError(
                "video_tensor must have shape (T, H, W, C), (T, C, H, W), "
                f"or (1, T, C, H, W); got {frames.shape}"
            )
        if frames.shape[0] == 0:
            raise ValueError("video_tensor must contain at least one frame")

        if frames.shape[-1] in (1, 3, 4):
            pass  # THWC
        elif frames.shape[1] in (1, 3, 4):
            frames = np.transpose(frames, (0, 2, 3, 1))  # TCHW -> THWC
        else:
            raise ValueError(
                "Could not identify the channel dimension in video_tensor "
                f"with shape {frames.shape}"
            )

        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] == 4:
            frames = frames[..., :3]

        if frames.dtype != np.uint8:
            frames = frames.astype(np.float32)
            finite_values = frames[np.isfinite(frames)]
            if finite_values.size and finite_values.max() <= 1.0:
                frames *= 255.0
            frames = np.nan_to_num(
                frames, nan=0.0, posinf=255.0, neginf=0.0
            )
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        frames = np.ascontiguousarray(frames)

        video_file = io.BytesIO()
        iio.imwrite(
            video_file,
            frames,
            extension=".mp4",
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
        )
        video_file.seek(0)

        env_idx = int(env_idx)
        timestep = int(timestep)
        return self.experiment.log_video(
            file=video_file,
            name=f"env_{env_idx}_step_{timestep}.mp4",
            format="mp4",
            step=timestep,
            metadata={
                "environment_index": env_idx,
                "frame_count": int(frames.shape[0]),
                "height": int(frames.shape[1]),
                "width": int(frames.shape[2]),
                "fps": float(fps),
            },
        )


class LoggableDDQN(DDQN):
    def __init__(self, calculate_env_statistics, fetch_latest_videos, *args, **kwargs):
        self.calculate_env_statistics = calculate_env_statistics
        self.fetch_latest_videos = fetch_latest_videos
        super().__init__(*args, **kwargs)
        self.writer = CometWriter()
        self.video_write_frequency = self.cfg['experiment']['video_interval']

    def init(self, trainer_cfg = None):
        super().init(trainer_cfg)
        self.writer = CometWriter()

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                # compute target values
                with torch.no_grad():
                    next_q_values, _, _ = self.target_q_network.act(
                        {"states": sampled_next_states}, role="target_q_network"
                    )

                    target_q_values = torch.gather(
                        next_q_values,
                        dim=1,
                        index=torch.argmax(
                            self.q_network.act({"states": sampled_next_states}, role="q_network")[0],
                            dim=1,
                            keepdim=True,
                        ),
                    )
                    target_values = (
                        sampled_rewards
                        + self._discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )

                # compute Q-network loss
                all_q_values = self.q_network.act(
                    {"states": sampled_states}, role="q_network"
                )[0]
                q_values = torch.gather(
                    all_q_values,
                    dim=1,
                    index=sampled_actions.long(),
                )

                q_network_loss = F.mse_loss(q_values, target_values)
                batch_best_action_advantage = best_action_advantage(all_q_values)

            # optimize Q-network
            self.optimizer.zero_grad()
            self.scaler.scale(q_network_loss).backward()

            if config.torch.is_distributed:
                self.q_network.reduce_parameters()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("DDQN / Q-network loss", q_network_loss.item())
            self.track_data("DDQN / Target network", torch.mean(target_values).item())
            self.track_data("DDQN / Best action advantage", batch_best_action_advantage.item())
            self.track_data("DDQN / Grad norm", grad_norm(self.q_network))
            current_epsilon = self._exploration_final_epsilon + (self._exploration_initial_epsilon - self._exploration_final_epsilon) * \
                math.exp(-1.0 * timestep / self._exploration_timesteps)
            self.track_data("DDQN / Current epsilon", current_epsilon)
            self.track_data("DDQN / Batch reward", torch.mean(sampled_rewards).item())

            if self._learning_rate_scheduler:
                self.track_data("DDQN / Learning rate", self.scheduler.get_last_lr()[0])

    def write_tracking_data(self, timestep, timesteps):
        super().write_tracking_data(timestep, timesteps)
        for k, v in self.calculate_env_statistics().items():
            self.writer.add_scalar(k, v, timestep)

    def post_interaction(self, timestep: int, timesteps: int):
        super().post_interaction(timestep, timesteps)
        if (timestep % self.video_write_frequency == 0) and timestep > 0:
            for idx, video in enumerate(self.fetch_latest_videos()):
                self.writer.add_video(idx, video, timestep)