import numpy as np
import pickle
import math, time
import rotations

# Helper math functions
from MathUtils 				import CrossProductMatrix, RunningMax

# Provides kinematic functions among others
import WooferDynamics

from JointSpaceController 	import JointSpaceController, TrotPDController
from BasicController 		import PropController
from StateEstimator 		import MuJoCoStateEstimator
from ContactEstimator 		import MuJoCoContactEstimator
from SwingLegController		import PDSwingLegController, ZeroSwingLegController
from TrotGait 				import TrotGait
from MPCPlanner				import MPCWalkingPlanner, MPCOrientationPlanner

from WooferConfig import WOOFER_CONFIG, QP_CONFIG, SWING_CONTROLLER_CONFIG, GAIT_PLANNER_CONFIG

class WooferRobot():
	"""
	This class represents the onboard Woofer software.

	The primary input is the mujoco simulation data and the
	primary output is a set joint torques.
	"""
	def __init__(self, state_estimator, contact_estimator, gait, gait_planner, swing_controller, dt):
		"""
		Initialize object variables
		"""

		self.contact_estimator 	= contact_estimator
		self.state_estimator 	= state_estimator
		self.gait_planner		= gait_planner
		self.swing_controller 	= swing_controller
		self.gait				= gait
		self.state 				= None
		self.contacts 			= None


		self.max_torques 		= RunningMax(12)
		self.max_forces 		= RunningMax(12)

		init_data_size = 10
		self.data = {}
		self.data['torque_history'] 			= np.zeros((12,init_data_size))
		self.data['force_history']				= np.zeros((12,init_data_size))
		self.data['contacts_history'] 			= np.zeros((4,init_data_size))
		self.data['active_feet_history'] 		= np.zeros((4,init_data_size))
		self.data['swing_torque_history']		= np.zeros((12,init_data_size))
		self.data['swing_force_history']		= np.zeros((12,init_data_size))

		self.data['swing_trajectory']			= np.zeros((12,init_data_size))
		self.data['foot_positions']				= np.zeros((12,init_data_size))

		self.dt = dt
		self.t = 0
		self.i = 0

		self.phase = 1

		self.foot_forces = np.array([0,0,WOOFER_CONFIG.MASS*9.81/4]*4)

		self.torques = np.zeros(12)

		self.step_locations = np.array([WOOFER_CONFIG.LEG_FB, -WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS,
								 WOOFER_CONFIG.LEG_FB,  WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS,
								-WOOFER_CONFIG.LEG_FB, -WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS,
								-WOOFER_CONFIG.LEG_FB,  WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS])
		self.p_step_locations = np.array([WOOFER_CONFIG.LEG_FB, -WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS,
								 WOOFER_CONFIG.LEG_FB,  WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS,
								-WOOFER_CONFIG.LEG_FB, -WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS,
								-WOOFER_CONFIG.LEG_FB,  WOOFER_CONFIG.LEG_LR, WOOFER_CONFIG.FOOT_RADIUS])

		self.swing_torques = np.zeros(12)
		self.swing_trajectory = np.zeros(12)
		self.mpc_torques = np.zeros(12)

	def step(self, sim):
		"""
		Get sensor data and update state estimate and contact estimate. Then calculate joint torques for locomotion.

		Details:
		Gait controller:
		Looks at phase variable to determine foot placements and COM trajectory

		QP:
		Generates joint torques to achieve given desired CoM trajectory given stance feet

		Swing controller:
		Swing controller needs reference foot landing positions and phase
		"""
		################################### State & contact estimation ###################################
		self.state 		= self.state_estimator.update(sim)
		self.contacts 	= self.contact_estimator.update(sim)

		# Use forward kinematics from the robot body to compute where the woofer feet are
		self.feet_locations = WooferDynamics.LegForwardKinematics(self.state['q'], self.state['j'])

		self.phase = self.gait.getPhase(self.t)

		self.step_phase = self.gait.getStepPhase(self.t, self.phase)

		self.active_feet = self.gait.feetInContact(self.phase)

		(self.step_locations, self.p_step_locations) = self.gait.updateStepLocations(self.state,
																self.step_locations,
																self.p_step_locations,
																self.phase)

		# ################################### Swing leg control ###################################
		self.swing_torques, \
		self.swing_forces,\
		self.swing_trajectory, \
		self.foot_positions = self.swing_controller.update(	self.state,
															self.step_phase,
															self.step_locations,
															self.p_step_locations,
															self.active_feet)

		state = np.zeros(12)
		state[0:3] = self.state['p']
		state[3:6] = rotations.quat2euler(self.state['q'])
		state[6:9] = self.state['p_d']
		state[9:12] = self.state['w']

		# only update the mpc trajectory at rate of planning discretization
		if(self.i % (self.gait_planner.dt/self.dt) == 0):
			self.foot_hist = self.gait.constuctFutureFootHistory(self.t,
																self.state,
																self.feet_locations,
																self.step_locations,
																self.gait_planner.N,
																self.gait_planner.dt)
			self.mpc_torques = self.gait_planner.update(state, self.state['j'], self.foot_hist, self.t)

		# Expanded version of active feet
		active_feet_12 = self.active_feet[[0,0,0,1,1,1,2,2,2,3,3,3]]

		self.torques = active_feet_12 * self.mpc_torques + (1 - active_feet_12) * self.swing_torques

		# Update our record of the maximum force/torque
		# self.max_forces.Update(self.foot_forces)
		# self.max_torques.Update(self.torques)

		# Log stuff
		# self.log_data()

		# Step time forward
		self.t += self.dt
		self.i += 1

		return self.torques

	def log_data(self):
		"""
		Append data to logs
		"""
		data_len = self.data['torque_history'].shape[1]
		if self.i > data_len - 1:
			for key in self.data.keys():
				self.data[key] = np.append(self.data[key], np.zeros((np.shape(self.data[key])[0],1000)),axis=1)


		self.max_forces.Update(self.foot_forces)
		self.max_torques.Update(self.torques)
		self.data['torque_history'][:,self.i] 		= self.torques
		self.data['force_history'][:,self.i] 		= self.foot_forces
		self.data['contacts_history'][:,self.i] 	= self.contacts
		self.data['active_feet_history'][:,self.i] 	= self.active_feet
		self.data['swing_torque_history'][:,self.i]	= self.swing_torques
		self.data['swing_force_history'][:,self.i] 	= self.swing_forces
		self.data['swing_trajectory'][:,self.i]		= self.swing_trajectory
		self.data['foot_positions'][:,self.i]		= self.foot_positions

	def print_data(self):
		"""
		Print debug data
		"""
		print("Time: %s"			%self.t)
		print("Cartesian: %s"		%self.state['p'])
		print("Euler angles: %s"	%rotations.quat2euler(self.state['q']))
		print("Max gen. torques: %s"%self.max_torques.CurrentMax())
		print("Max forces: %s"		%self.max_forces.CurrentMax())
		print("feet locations: %s"	%self.feet_locations)
		print("contacts: %s"		%self.contacts)
		print("QP feet forces: %s"	%self.foot_forces)
		print("Joint torques: %s"	%self.torques)
		print('\n')

	def save_logs(self):
		"""
		Save the log data to file
		"""
		np.savez('woofer_numpy_log',**self.data)
		# with open('woofer_logs.pickle', 'wb') as handle:
		# 	pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def MakeWoofer(dt = 0.001):
	"""
	Create robot object
	"""
	trot_overlap_time = .105
	trot_step_time = .3

	MPC_horizon = 16
	MPC_planning_timestep = 0.01

	MPC_desired_velocity = np.array([0.305, 0.0, 0.0])

	mujoco_state_est 	= MuJoCoStateEstimator()
	mujoco_contact_est 	= MuJoCoContactEstimator()
	gait 				= TrotGait(trot_step_time, trot_overlap_time)
	gait_planner 		= MPCWalkingPlanner(MPC_horizon, MPC_planning_timestep,
											gait, np.array([0, 0, (WOOFER_CONFIG.LEG_L + WOOFER_CONFIG.FOOT_RADIUS), 0, 0, 0, 0, 0, 0, 0, 0, 0]),
											MPC_desired_velocity)
	# gait_planner 		= MPCOrientationPlanner(MPC_horizon, MPC_planning_timestep,
	# 										gait, np.array([0, 0, (WOOFER_CONFIG.LEG_L + WOOFER_CONFIG.FOOT_RADIUS), 0, 0, 0, 0, 0, 0, 0, 0, 0]))
	swing_controller	= PDSwingLegController()

	woofer = WooferRobot(mujoco_state_est, mujoco_contact_est, gait, gait_planner, swing_controller, dt = dt)

	return woofer
