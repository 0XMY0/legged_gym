
import matplotlib.pyplot as plt
from isaacgym.torch_utils import normalize, quat_apply, quat_from_angle_axis
import torch
import numpy as np

# def quat_apply_yaw(quat, vec):
#     quat_yaw = quat.clone().view(-1, 4)
#     quat_yaw[:, :2] = 0.
#     quat_yaw = normalize(quat_yaw)
#     return quat_apply(quat_yaw, vec)

# angle = torch.tensor([1.57])  # Rotation angle in radians (90 degrees)
# axis = torch.tensor([0.0, 0.0, 1.0])  # Rotation around the z-axis
# quat = quat_from_angle_axis(angle, axis)
# print(quat_apply_yaw(quat, torch.tensor([1.0, 0.0, 0.0])))

# get phase
def _getphase(x, mu_A, mu_B):
    def normal_cdf(x, mu, sigma):
        return 0.5 * (1 + torch.erf((x - mu) / (sigma * 2.0)))
    P1 = normal_cdf(x, mu_A, 0.015) * (1 - normal_cdf(x, mu_B, 0.015))
    if(mu_A > 0.2 and mu_B < 0.8):
        return P1
    if(mu_A < 0.2):
        P2 = normal_cdf(x - 1, mu_A, 0.015) * (1 - normal_cdf(x - 1, mu_B, 0.015))
        return P1 + P2
    if(mu_B > 0.8):
        P3 = normal_cdf(x + 1, mu_A, 0.015) * (1 - normal_cdf(x + 1, mu_B, 0.015))
        return P1 + P3

def _getphaseswinglr_walk(phase):
    phase_swingl = _getphase(phase, 0.0, 0.45)
    phase_swingr = _getphase(phase, 0.5, 0.95)
    return phase_swingl, phase_swingr

phase = torch.linspace(0, 1, 1000)
phase_swingl, phase_swingr = _getphaseswinglr_walk(phase)
zl_desired = 0.06 * torch.sin(torch.clip((phase - 0.0) / (0.45 - 0.0), 0, 1) * np.pi)
zr_desired = 0.06 * torch.sin(torch.clip((phase - 0.5) / (0.95 - 0.5), 0, 1) * np.pi)

# plt.plot(phase, phase_swingl, label='phase_swingl')
# plt.plot(phase, phase_swingr, label='phase_swingr')
# plt.legend()
# plt.show()

plt.plot(phase, zl_desired, label='zl_desired')
plt.plot(phase, zr_desired, label='zr_desired')
plt.legend()
plt.show()
