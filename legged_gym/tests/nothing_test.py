
import matplotlib.pyplot as plt
from isaacgym.torch_utils import normalize, quat_apply, quat_from_angle_axis
import torch
import numpy as np

v1 = torch.tensor([-0.09433773, -0.14840947, -0.06442075,  0.02377916, -0.47106045,
         0.01626374,  0.15333867,  0.04457955, -0.06766637, -0.1769931 ,
        -0.07208688])

v2 = torch.tensor([-0.03934235,  0.0418127 , -0.13220175,  0.31806043, -0.57186484,
         0.07385372, -0.45494935, -0.00398099, -0.14131977, -0.16187   ,
        -0.14332965])

v3 = torch.square(v1 - v2)

print(v3)