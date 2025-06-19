import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

from model.encoder import EncoderNet
from model.actor import ActorLearnNet

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def reorder_from_mujoco_to_isaac(values):
    return values[mujoco_to_isaac_idx]

def reorder_from_isaac_to_mujoco(values):
    return values[isaac_to_mujoco_idx]

if __name__ == "__main__":
    
    xml_path = "resources/robots/g1_description/g1_23dof_rev_1_0.xml"

    mujoco_to_isaac_idx = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
    isaac_to_mujoco_idx = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]

    simulation_duration = 60.0
    simulation_dt =  0.002
    control_decimation = 10

    kps = np.array([100, 100, 100, 150, 40, 40,
                    100, 100, 100, 150, 40, 40,
                    300,
                    100, 100, 50, 50, 20,
                    100, 100, 50, 50, 20], dtype=np.float32)
    
    kds = np.array([2, 2, 2, 4, 2, 2,
                    2, 2, 2, 4, 2, 2,
                    3, 
                    2, 2, 2, 2, 1,
                    2, 2, 2, 2, 1], dtype=np.float32)

    default_angles = np.array([0.0, 0.0, 0.0, 0.1745, -0.1745, 0.0,
                               0.0, 0.0, 0.0, 0.1745, -0.1745, 0.0,
                               0.0,
                               0.1745, 0.1745, 0.0, 1.1345, 0.0,
                               0.1745, -0.1745, 0.0, 1.1345, 0.0], dtype=np.float32)
    
    map_index = torch.tensor([0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22])
    
    action_scale = 0.25

    action = np.zeros(23, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(78, dtype=np.float32)

    encoder = EncoderNet(78, [512, 512])
    actor = ActorLearnNet(encoder.dim, 23, [512])
    
    encoder_params, actor_params, _ = torch.load("model.pth", map_location="cpu", weights_only=True)
    encoder.load_state_dict(encoder_params)
    actor.load_state_dict(actor_params)

    encoder.eval()
    actor.eval()

    counter = 0

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, data.qpos[7:], kps, np.zeros_like(kds), data.qvel[6:], kds)
            
            data.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            
            counter += 1
            if counter % control_decimation == 0:
                qj = data.qpos[7:]
                dqj = data.qvel[6:]
                quat = data.qpos[3:7]
                omega = data.qvel[3:6]

                qj = (qj - default_angles)

                gravity_orientation = get_gravity_orientation(quat)

                qj = reorder_from_mujoco_to_isaac(qj)
                dqj = reorder_from_mujoco_to_isaac(dqj)

                obs = omega.tolist() + gravity_orientation.tolist() + qj.tolist() + dqj.tolist() + action.tolist() + [0.0, 0.0, 0.0]
                
                if counter == 10:
                    print(obs)

                with torch.no_grad():
                    feat = torch.as_tensor([obs])
                    feat = encoder(feat)
                    pi, action, log_prob = actor(feat)
                    action = pi.mean

                    action = action[:, map_index]

                action = action.squeeze(0).numpy()
                apply_action = reorder_from_isaac_to_mujoco(action)

                target_dof_pos = apply_action * action_scale + default_angles
            
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
