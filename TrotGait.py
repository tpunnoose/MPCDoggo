import numpy as np

class TrotGait:
	def __init__(self, step_time, overlap_time):
		"""
		Four Phases:
			Phase 1: FR/BL on ground
			Phase 2: All feet on ground
			Phase 3: FL/BR on ground
			Phase 4: All feet on ground
			repeat
		Length of Phase 1/3: step_time
		Length of Phase 2/4: overlap_time
		"""
		self.step_time = step_time
		self.overlap_time = overlap_time

		self.phase_length = 2*step_time + 2*overlap_time

	def feetInContact(self, t):
		phase_time = t % self.phase_length

		if phase_time < self.step_time:
			return np.array([1, 0, 0, 1])
		elif phase_time < (self.step_time + self.overlap_time):
			return np.ones(4)
		elif phase_time < (2*self.step_time + self.overlap_time):
			return np.array([0, 1, 1, 0])
		else:
			return np.ones(4)
