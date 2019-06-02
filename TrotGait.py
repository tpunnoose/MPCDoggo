import numpy as np

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
		self.step_time = overlap_time
		self.overlap_time = step_time

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

		side: right leg = 1, left leg = 0
		"""
		p_diff = 0.5 * state['p_d'][0:2] * self.step_time

		new_step_locations = step_locations
		new_p_step_locations = p_step_locations

		if phase == 1:
			# need to move the FL/BR feet
			new_step_locations[3:5] = step_locations[3:5] + p_diff
			new_step_locations[6:8] = step_locations[6:8] + p_diff

		elif phase == 2:
			# moving FL/BR feet
			new_p_step_locations[3:5] = p_step_locations[3:5]
			new_p_step_locations[6:8] = p_step_locations[6:8]

		elif phase == 3:
			# need to move the FR/BL feet
			new_step_locations[0:2] = step_locations[0:2] + p_diff
			new_step_locations[9:11] = step_locations[9:11] + p_diff

		elif phase == 4:
			# moving FR/BL feet
			new_p_step_locations[0:2] = p_step_locations[0:2] + p_diff
			new_p_step_locations[9:11] = p_step_locations[9:11] + p_diff

		return (new_step_locations, new_p_step_locations)

	def getStepPhase(self, t, phase):
		"""
		returns step phase for swing leg controller
		"""
		phase_time = t % self.phase_length

		if phase == 4:
			time_into_traj = 2*self.step_time + 2*self.overlap_time - phase_time
		else:
			time_into_traj = self.step_time + self.overlap_time - phase_time
		return time_into_traj/self.step_time
