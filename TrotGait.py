import numpy as np
from WooferConfig import WOOFER_CONFIG
from WooferDynamics import FootSelector, CoordinateExpander

class TrotGait:
	def __init__(self, step_time, overlap_time):
		"""
		Four Phases:
			Phase 1: All feet on ground
			Phase 2: FR/BL on ground
			Phase 3: All feet on ground
			Phase 4: FL/BR on ground
			repeat
		Length of Phase 1/3: step_time
		Length of Phase 2/4: overlap_time
		"""
		self.step_time = step_time
		self.overlap_time = overlap_time

		self.phase_length = 2*step_time + 2*overlap_time

	def getPhase(self, t):
		phase_time = t % self.phase_length

		if phase_time < self.overlap_time:
			return 1
		elif phase_time < (self.step_time + self.overlap_time):
			return 2
		elif phase_time < (self.step_time + 2*self.overlap_time):
			return 3
		else:
			return 4

	def feetInContact(self, phase):
		if phase == 1:
			return np.ones(4)
		elif phase == 2:
			return np.array([1, 0, 0, 1])
		elif phase == 3:
			return np.array([1, 1, 1, 1])
		else:
			return np.array([0, 1, 1, 0])

	def updateStepLocations(self, state, step_locations, p_step_locations, phase):
		"""
		uses the heuristic in eq. 33 of the MIT Cheetah 3 MPC paper
		to calculate the footstep locations

		todo: update step locations based off where they actually land

		side: right leg = 1, left leg = 0
		"""
		# p_diff = 0.5 * state['p_d'][0:2] * self.step_time

		p_diff = np.array([0.2, 0.0])

		# p_diff = np.zeros(2)

		new_step_locations = step_locations
		new_p_step_locations = p_step_locations

		if phase == 1:
			# need to move the FL/BR feet
			new_step_locations[3:5] = p_step_locations[3:5] + p_diff
			new_step_locations[6:8] = p_step_locations[6:8] + p_diff
			new_p_step_locations[0:2] = p_step_locations[0:2]
			new_p_step_locations[9:11] = p_step_locations[9:11]

		elif phase == 3:
			# need to move the FR/BL feet
			new_step_locations[0:2] = p_step_locations[0:2] + p_diff
			new_step_locations[9:11] = p_step_locations[9:11] + p_diff
			new_p_step_locations[3:5] = p_step_locations[3:5]
			new_p_step_locations[6:8] = p_step_locations[6:8]

		return (new_step_locations, new_p_step_locations)

	def getStepPhase(self, t, phase):
		"""
		returns step phase for swing leg controller
		"""
		phase_time = t % self.phase_length

		time_into_traj = 0
		if phase == 4:
			time_into_traj = phase_time - (self.step_time + 2*self.overlap_time)
		elif phase == 2:
			time_into_traj = phase_time - self.overlap_time
		return time_into_traj/self.step_time

	def constuctFutureFootHistory(self, t, state, current_foot_locations, step_locations, N, mpc_dt):
		"""

		"""
		foot_hist = np.zeros((N, 12))

		future_step_locations = step_locations - CoordinateExpander(state['p'])

		t_i = t
		phase_i = self.getPhase(t_i)
		prev_phase = phase_i
		foot_hist[0, :] = FootSelector(self.feetInContact(phase_i)) * current_foot_locations

		for i in range(1,N):
			if(phase_i != prev_phase):
				foot_hist[i, :] = FootSelector(self.feetInContact(phase_i)) * future_step_locations
			else:
				foot_hist[i, :] = foot_hist[i-1, :]
			prev_phase = phase_i
			t_i += mpc_dt
			phase_i = self.getPhase(t_i)

		return foot_hist
