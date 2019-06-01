from control.matlab import *
import numpy as np
from math import cos, sin
from WooferConfig import WOOFER_CONFIG
from MathUtils import CrossProductMatrix
import cvxpy as cvx

class TrajectoryGeneration():
	def __init__(self, dt):
		"""
		insert stuff
		"""
		self.dt = dt
		self.J = WOOFER_CONFIG.INERTIA
		self.m = WOOFER_CONFIG.MASS

		self.C = np.zeros((1,13))
		self.D = np.zeros((1,12))


	def rotZ(self, yaw):
		Rz = np.zeros((3,3))

		Rz[0,0] = cos(yaw)
		Rz[0,1] = sin(yaw)
		Rz[1,0] = -sin(yaw)
		Rz[1,1] = cos(yaw)
		Rz[2,2] = 1

		return Rz

	def constructA(self, yaw):
		"""
		constructs the continuous time A matrix as a function of yaw
		"""
		A = np.zeros((13, 13))
		Rz = self.rotZ(yaw)

		A[0:3, 6:9] = eye(3)
		A[3:6, 9:12] = Rz
		A[8, 13] = 1

		return A

	def constructB(self, foot_locations, yaw):
		"""
		constructs the continuous time B matrix as a function of yaw and foot locations
		"""
		B = np.zeros((13, 12))
		Rz = self.rotZ(yaw)

		J_w_inv = np.linalg.inv(Rz*self.J*Rz.T)
		R1 = CrossProductMatrix(foot_locations[0:3])
		R2 = CrossProductMatrix(foot_locations[3:6])
		R3 = CrossProductMatrix(foot_locations[6:9])
		R4 = CrossProductMatrix(foot_locations[9:12])

		B[6:9, 0:3] = eye(3)/self.m
		B[6:9, 3:6] = eye(3)/self.m
		B[6:9, 6:9] = eye(3)/self.m
		B[6:9, 9:12] = eye(3)/self.m

		B[9:12, 0:3] = J_w_inv*R1
		B[9:12, 3:6] = J_w_inv*R2
		B[9:12, 6:9] = J_w_inv*R3
		B[9:12, 9:12] = J_w_inv*R4

		return B

	def getDiscrete(self, A, B):
		"""
		get ZOH approximation of A and B matrices
		"""
		ct_sys = control.StateSpace(A, B, self.C, self.D)
		dt_sys = control.matlab.c2d(ct_sys, self.dt)

		A_d = np.asarray(dt_sys.A)
		B_d = np.asarray(dt_sys.B)

		sys = {"A_d": A_d, "B_d": B_d}
		return sys

	def build_qp(x_ref, A_hat, B_hats, Qs, Rs, constr, c_low, c_up):
		# n number of state variables
		# k number of horizon state points
		# m dimension of control input at each point
		n, k = x_ref.shape
		m = B_hats.shape[1]

		D = np.zeros((k * n, n))
		for i in range(k):
			D[i * n: (i + 1) * n, :] = np.linalg.matrix_power(A_hat, i)

		C = np.zeros(k * n, m * k)
		for i in range(k):
			for j in range(i):
				C[i * n: (i + 1) * n, j * m: (j + 1) * m] = \
					np.linalg.matrix_power(A_hat, (i - 1)) @ B_hats[j]

		Q = np.zeros(n * k, n * k)
		R = np.zeros(m * k, m * k)
		for i in range(k):
			Q[i * n: (i + 1) * n, i * n: (i + 1) * n] = Qs[i]
			M[i * m: (i + 1) * m, i * m: (i + 1) * m] = Rs[i]

		u = cvx.Variable(m * k)
		target = x_ref.flatten() - D @ x_ref[:, 0]
		cost = cvx.quad_form(C * u - target, Q) + cvx.quad_form(u, R)
		obj = cvx.Minimize(cost)
		constraints = [constr * u <= c_up, constr * u >= c_low]
		prob = cvx.Problem(obj, constraints)

		return prob
