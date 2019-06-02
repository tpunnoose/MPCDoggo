import numpy as np
from WooferDynamics import *
from TrajectoryGeneration import TrajectoryGeneration
import rotations

class MPCPlanner:
	def __init__(self, N, dt_plan):
		self.N = N
		self.dt = dt_plan

class MPCStandingPlanner(MPCPlanner):
	def __init__(self, N, dt_plan, gait, init_state):
		self.dt = dt_plan
		self.N = N
		self.gait = gait
		self.trajectoryGenerator = TrajectoryGeneration(gait, dt_plan, N)
		self.init_state = init_state

	def update(self, state, qpos_joints, foot_locations, t):
		referenceTrajectory = self.generateReferenceTrajectory(state, foot_locations)

		U_horizon = self.trajectoryGenerator.solveSystem(referenceTrajectory, foot_locations, t)

		feet_forces = U_horizon[:12]

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

		print("Joint torques: ", joint_torques)

		return joint_torques

	def generateReferenceTrajectory(self, state, foot_locations):
		return np.tile(self.init_state[np.newaxis, :], (self.N, 1))
