import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .walk_env_cfg import G1WalkEnvCfg

class G1WalkEnv(DirectRLEnv):
    cfg:G1WalkEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 23, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 23, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self):
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self):
        
        root_lin_vel = self.robot.data.root_lin_vel_b       # (num_envs, 3)
        root_ang_vel = self.robot.data.root_ang_vel_b       # (num_envs, 3)
        projected_gravity = self.robot.data.projected_gravity_b  # (num_envs, 3)
        base_orientation = self.robot.data.root_quat_w        # (num_envs, 4)
        joint_pos = self.robot.data.joint_pos               # (num_envs, 23)
        joint_vel = self.robot.data.joint_vel               # (num_envs, 23)
        previous_actions = self._previous_actions           # (num_envs, 23)

        obs = torch.cat([
            root_lin_vel,
            root_ang_vel,
            projected_gravity,
            base_orientation,
            joint_pos,
            joint_vel,
            previous_actions,
        ], dim=-1)

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:

        rewards = self._height_reward() + self._line_vel_reward() + self._get_action_rate_reward() + self._difference_to_default_reward()

        return rewards
    
    def _get_rewards(self) -> torch.Tensor:
        return (
            2.0 * self._height_reward() +
            1.0 * self._no_motion_reward() +
            -0.5 * self._difference_to_default_reward() +
            -0.2 * self._get_action_rate_reward() +
            -0.05 * self._joint_velocity_penalty()
        )

    def _height_reward(self) -> torch.Tensor:
        base_height = self.robot.data.root_state_w[:, 2]
        height_target = 0.6

        # Reward is 1.0 if height is at or above the target
        reward = torch.where(
            base_height >= height_target,
            torch.ones_like(base_height),
            torch.zeros_like(base_height)
        )
        return reward

    def _no_motion_reward(self) -> torch.Tensor:
        lin_vel = torch.norm(self.robot.data.root_lin_vel_b, dim=1)
        ang_vel = torch.norm(self.robot.data.root_ang_vel_b, dim=1)
        penalty = lin_vel**2 + ang_vel**2

        return torch.exp(-penalty / 0.001)

    def _get_action_rate_reward(self) -> torch.Tensor:
        return torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
    
    def _joint_velocity_penalty(self) -> torch.Tensor:
        return torch.norm(self.robot.data.joint_vel, dim=1)

    def _difference_to_default_reward(self) -> torch.Tensor:
        return torch.sum((self.robot.data.joint_pos - self.robot.data.default_joint_pos) ** 2, dim=1)

            
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.robot.data.root_state_w[:, 2]
        terminate = base_height < 0.5
        #terminate = base_height < 0
        return terminate, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
       
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        