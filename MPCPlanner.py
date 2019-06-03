import numpy as np
from WooferDynamics import *
from TrajectoryGeneration import TrajectoryGeneration
import rotations

class MPCPlanner:
	def __init__(self, N, dt_plan):
		self.N = N
		self.dt = dt_plan

class MPCWalkingPlanner(MPCPlanner):
	def __init__(self, N, dt_plan, gait, init_state, desired_velocity):
		self.dt = dt_plan
		self.N = N
		self.gait = gait
		self.trajectoryGenerator = TrajectoryGeneration(gait, dt_plan, N)
		self.init_state = init_state
		self.desired_velocity = desired_velocity

	def update(self, state, qpos_joints, foot_locations, t):
		referenceTrajectory = self.generateReferenceTrajectory(state, t)

		U_horizon = self.trajectoryGenerator.solveSystem(referenceTrajectory, foot_locations, t)

		feet_forces = U_horizon[:12]
		feet_forces[np.abs(feet_forces) < 1e-5] = 0
		#ground = feet_forces > 0.1
		#feet_forces = np.zeros(12)
		#feet_forces[ground] = 7.172*9.81/ground.sum()

		joint_torques = np.zeros(12)

		rotmat = rotations.euler2mat(state[3:6])

		for f in range(4):
			# Extract the foot force vector for foot i
			foot_force_world = feet_forces[f*3 : f*3 + 3]

			# Transform from world to body frames,
			# The negative sign makes it the force exerted by the body
			foot_force_body 				= -np.dot(rotmat.T, foot_force_world)
			(beta,theta,r) 					= tuple(qpos_joints[f*3 : f*3 + 3])

			# Transform from body frame forces into joint torques
			joint_torques[f*3 : f*3 + 3] 	= np.dot(LegJacobian(beta, theta, r).T, foot_force_body)

		# print("Feet forces: ", feet_forces)
		# print("Angular velocity: ", state[9:12])

		return joint_torques

	def generateReferenceTrajectory(self, state, t):
		scale = np.linspace(0, 1, self.N)[:, np.newaxis]
		curr = np.tile(state[np.newaxis, :], (self.N, 1))
		goal = np.zeros((self.N, 12))
		goal[:, 6:8] = self.desired_velocity[:2]

		ref = np.zeros((self.N, 12))
		# Interpolate x and y velocities to the desired velocity
		ref[:, 6:8] = (1 - scale) * curr[:, 6:8] + scale * goal[:, 6:8]
		# Calculate reference x and y positions by integration
		ref_vel = ref[:, 6:8].copy()
		ref_vel *= self.dt
		dxy = np.cumsum(ref_vel, axis=0)
		ref[:, :2] = curr[:, :2] + dxy
		# Interpolate angle positions to 0
		ref[:, 3:6] = (1 - scale) * curr[:, 3:6]
		# Interpolate z position to initial state
		ref[:, 2] = (1 - scale)[:, 0] * curr[:, 2] + scale[:, 0] * self.init_state[2]
		# Set z, angle velocities to 0
		# They are already 0
		return ref

class MPCOrientationPlanner(MPCWalkingPlanner):
	def __init__(self, N, dt_plan, gait, init_state):
		self.dt = dt_plan
		self.N = N
		self.gait = gait
		self.trajectoryGenerator = TrajectoryGeneration(gait, dt_plan, N)
		self.init_state = init_state

	def generateReferenceTrajectory(self, state, t):
		ROLL_CONSTANT = 0.0
		PITCH_CONSTANT = 0.0
		YAW_CONSTANT = 0.025
		w = 2 * np.pi * 10

		scale = np.linspace(0, 1, self.N)[:, np.newaxis]
		curr = np.tile(state[np.newaxis, :], (self.N, 1))
		goal = np.zeros((self.N, 12))

		ref = np.zeros((self.N, 12))
		# Interpolate x and y velocities to the desired velocity
		ref[:, 6:8] = (1 - scale) * curr[:, 6:8] + scale * goal[:, 6:8]
		# Calculate reference x and y positions by integration
		ref_vel = ref[:, 6:8].copy()
		ref_vel *= self.dt
		dxy = np.cumsum(ref_vel, axis=0)
		ref[:, :2] = curr[:, :2] + dxy
		# Interpolate angle positions to 0
		ref[:, 3:6] = (1 - scale) * curr[:, 3:6]
		# Interpolate z position to initial state
		ref[:, 2] = (1 - scale)[:, 0] * curr[:, 2] + scale[:, 0] * self.init_state[2]
		# Set z, angle velocities to 0
		# They are already 0

		times = np.linspace(t, t + (self.N - 1) * self.dt, self.N)
		ref[:, 3] = ROLL_CONSTANT * np.sin(w * times)
		ref[:, 4] = PITCH_CONSTANT * np.sin(w * times)
		ref[:, 5] = YAW_CONSTANT * np.sin(w * times)
		ref[:, 9] = w * ROLL_CONSTANT * np.cos(w * times)
		ref[:, 10] = w * PITCH_CONSTANT * np.cos(w * times)
		ref[:, 11] = w * YAW_CONSTANT * np.cos(w * times)

		# if t > 2.0:
		# 	import pdb; pdb.set_trace()

		return ref
