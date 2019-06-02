import numpy as np
from TrajectoryGeneration import TrajectoryGeneration
from TrotGait import TrotGait
from WooferConfig import WOOFER_CONFIG


N = 20
discretization_step = 0.05

reference_trajectory = np.zeros((N, 12))
reference_trajectory[:,0] = np.linspace(0, 1, N)
reference_trajectory[:,2] = 0.32*np.ones(N)


gait = TrotGait(0.5, 0.1)

mpc = TrajectoryGeneration(gait, discretization_step, N)


foot_locs = np.zeros(12)
foot_locs[0:3]= np.array([WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.LEG_FB, 0])
foot_locs[3:6] = np.array([-WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.LEG_FB, 0])
foot_locs[6:9] = np.array([WOOFER_CONFIG.LEG_LR, -WOOFER_CONFIG.LEG_FB, 0])
foot_locs[9:12] = np.array([-WOOFER_CONFIG.LEG_LR, -WOOFER_CONFIG.LEG_FB, 0])


u = mpc.solveSystem(reference_trajectory, foot_locs, 0.0)

print(u[0:12])
