from scipy.signal import cont2discrete
import numpy as np
from math import cos, sin
from WooferConfig import WOOFER_CONFIG
from MathUtils import CrossProductMatrix
import cvxpy as cvx
import pdb

class TrajectoryGeneration:
	def __init__(self, gait, dt, N):
		"""
		insert stuff
		"""
		self.J = WOOFER_CONFIG.INERTIA
		self.m = 7.172 # WOOFER_CONFIG.MASS 

		self.C = np.zeros((1,13))
		self.D = np.zeros((1,12))

		self.gait = gait

		# planning horizon
		self.N = N

		# discretization step of planning horizon
		self.dt = dt

		self.Q = np.eye(13)
		self.alpha = 0
		self.R = self.alpha*np.eye(12)

		self.mu = 1.5

		self.max_horz_force = 133
		self.max_vert_force = 133
		self.min_vert_force = 1

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

		A[0:3, 6:9] = np.eye(3)
		A[3:6, 9:12] = Rz
		A[8, 12] = 1

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

		B[6:9, 0:3] = np.eye(3)/self.m
		B[6:9, 3:6] = np.eye(3)/self.m
		B[6:9, 6:9] = np.eye(3)/self.m
		B[6:9, 9:12] = np.eye(3)/self.m

		B[9:12, 0:3] = J_w_inv*R1
		B[9:12, 3:6] = J_w_inv*R2
		B[9:12, 6:9] = J_w_inv*R3
		B[9:12, 9:12] = J_w_inv*R4

		return B

	def getDiscrete(self, A, B):
		"""
		get ZOH approximation of A and B matrices
		"""
		(A_d, B_d, _, _, _) = cont2discrete((A, B, self.C, self.D), self.dt)

		return A_d, B_d

	def solveSystem(self, refTrajectory, foot_locs, t):
		"""
		build matrices and call solver

		refTrajectory (of length N):
		[pos, euler orientation, velocity, angular velocity]
		"""
		assert(refTrajectory.shape[0] == self.N)
		yaw_bar = np.mean(refTrajectory[:, 3])

		A = self.constructA(yaw_bar)
		A_hat = None
		B_hats = np.zeros((self.N, 13, 12))

		curr_t = t

		for i in range(self.N):
			active_feet = self.gait.feetInContact(self.gait.getPhase(curr_t))
			active_feet_12 = active_feet[[0,0,0,1,1,1,2,2,2,3,3,3]]

			foot_loc_i = active_feet_12 * foot_locs

			B_i = self.constructB(foot_loc_i, refTrajectory[i, 3])

			A_d, B_d = self.getDiscrete(A, B_i)
			if i == 0: A_hat = A_d
			B_hats[i] = B_d

			curr_t += self.dt

		C, c_up = self.makeConstraints(t)
		prob, u = self.build_qp(refTrajectory, A_hat, B_hats, C, c_up)

		prob.solve(solver=cvx.OSQP)
		return u.value


	def makeConstraints(self, t):
		"""
		create constraint matrix C and lower and upper bounds
		"""
		curr_t = t

		C = np.zeros((self.N, 24, 12))
		c_up = np.zeros((self.N, 24))

		fz_mat = np.zeros((4, 12))
		fx_mat = np.zeros((8, 12))
		fy_mat = np.zeros((8, 12))
		for i in range(4):
			fz_mat[i, 3*i + 2] = 1

			fx_mat[2*i, 3*i] = 1
			fx_mat[2*i, 3*i + 2] = -self.mu
			fx_mat[2*i + 1, 3*i] = -1
			fx_mat[2*i + 1, 3*i + 2] = -self.mu

			fy_mat[2*i, 3*i + 1] = 1
			fy_mat[2*i, 3*i + 2] = -self.mu
			fy_mat[2*i + 1, 3*i + 1] = -1
			fy_mat[2*i + 1, 3*i + 2] = -self.mu

		for i in range(self.N):
			foot_loc_i = self.gait.feetInContact(self.gait.getPhase(curr_t))
			# Z Force constraints, accounting for which feet are grounded
			c_up[i, :4] = foot_loc_i * self.max_vert_force
			C[i, :4] = fz_mat
			c_up[i, 4:8] = -foot_loc_i * self.min_vert_force
			C[i, 4:8] = -fz_mat
			# X Force constraints, accounting for which feet are grounded
			C[i, 8:16] = fx_mat
			# Y Force constraints, accounting for which feet are grounded
			C[i, 16:24] = fy_mat

			curr_t += self.dt

		return C, c_up


	def build_qp(self, x_ref, A_hat, B_hats, constr, c_up):
		# n number of state variables
		# k number of horizon state points
		# m dimension of control input at each point
		x_ref = x_ref.T
		x_ref = np.concatenate((x_ref, -9.81*np.ones((1, self.N))))
		n, k = x_ref.shape
		k = self.N
		m = B_hats.shape[2]

		D = np.zeros((k * n, n))
		for i in range(k):
			D[i * n: (i + 1) * n, :] = np.linalg.matrix_power(A_hat, i)

		C = np.zeros((k * n, m * k))
		for i in range(k):
			for j in range(i):
				C[i * n: (i + 1) * n, j * m: (j + 1) * m] = \
					np.linalg.matrix_power(A_hat, (i - 1)) @ B_hats[j]

		Qs = np.zeros((n * k, n * k))
		Rs = np.zeros((m * k, m * k))
		for i in range(k):
			Qs[i * n: (i + 1) * n, i * n: (i + 1) * n] = self.Q
			Rs[i * m: (i + 1) * m, i * m: (i + 1) * m] = self.R

		u = cvx.Variable(m * k)
		target = x_ref.T.flatten() - D @ x_ref[:, 0]
		cost = cvx.quad_form(C * u - target, Qs) + cvx.quad_form(u, Rs)
		obj = cvx.Minimize(cost)
		constraints = []
		for i in range(self.N):
			constraints += [constr[i] * u[m*i:m*(i+1)] <= c_up[i]]
		prob = cvx.Problem(obj, constraints)

		return prob, u
