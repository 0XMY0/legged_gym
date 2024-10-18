import torch
import legged_gym
from legged_gym import LEGGED_GYM_ROOT_DIR
policy_path = LEGGED_GYM_ROOT_DIR + '/logs/rough_bhr8tcphase/exported/policies/policy_1.pt'
print(policy_path)
policy = torch.jit.load(policy_path)

# self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
#                             self.base_ang_vel  * self.obs_scales.ang_vel,
#                             self.base_euler,
#                             self.commands[:, :3] * self.commands_scale,
#                             (self.dof_pos - self.dof_pos_limits[:, 0]) / (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) * self.obs_scales.dof_pos,
#                             self.dof_vel * self.obs_scales.dof_vel,
#                             self.phase.unsqueeze(1)
#                             ),dim=-1)

obs = torch.cat((   torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([0.0, 0.0, 0.0]),
                    0.5 * torch.ones(10) * 1,
                    torch.zeros(10),
                    torch.tensor([1.0])
                ))

with torch.no_grad():  # Disable gradient tracking for inference
    action = policy(obs)

mirrored_action = torch.cat((
                            -action[5: 7],
                            action[7: 10],
                            -action[:2],
                            action[2: 5],
                            action[10:11]
                        ))

print('obs = ', obs.cpu().numpy())
print('action = ', action.cpu().numpy())
print('mirrored_action = ', mirrored_action.cpu().numpy())

mirrored_obs = torch.cat((  torch.tensor([0.0, 0.0, 0.0]),
                            torch.tensor([0.0, 0.0, 0.0]),
                            torch.tensor([0.0, 0.0, 0.0]),
                            torch.tensor([0.0, 0.0, 0.0]),
                            0.5 * torch.ones(10) * 1,
                            torch.zeros(10),
                            torch.tensor([0.5])
                        ))

with torch.no_grad():  # Disable gradient tracking for inference
    action_mirrored_obs = policy(mirrored_obs)

print('mirrored_obs = ', mirrored_obs.cpu().numpy())
print('action_mirrored_obs = ', action_mirrored_obs.cpu().numpy())

err = torch.sum(torch.square(mirrored_action - action_mirrored_obs))
print('err = ', err.cpu().numpy())