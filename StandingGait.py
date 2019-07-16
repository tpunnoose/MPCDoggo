import numpy as np
from WooferConfig import WOOFER_CONFIG
from WooferDynamics import FootSelector, CoordinateExpander

class StandingGait:
	def getPhase(self, t):
		return 1

	def feetInContact(self, phase):
		return np.ones(4)

	def updateStepLocations(self, state, step_locations, p_step_locations, phase):
		return (step_locations, p_step_locations)

	def getStepPhase(self, t, phase):
		"""
		returns step phase for swing leg controller
		"""
		return 0

	def constuctFutureFootHistory(self, t, state, current_foot_locations, step_locations, N, mpc_dt):
		"""

		"""
		foot_hist = np.tile(current_foot_locations, (N,1))

		return foot_hist
