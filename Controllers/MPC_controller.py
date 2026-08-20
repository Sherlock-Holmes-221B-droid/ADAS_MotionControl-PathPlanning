import cvxpy as cp
import numpy as np

class MPCController:
    def __init__(self, wheelbase, horizon=10, dt=0.1):
        self.L = wheelbase
        self.N = horizon
        self.dt = dt
        
    def get_model_matrix(self, velocity):
        v = max(1.0, velocity)
        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0,   v, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        B = np.array([
            [0.0],
            [0.0],
            [0.0],
            [v / self.L]
        ])
        Ad = np.eye(4) + A * self.dt
        Bd = B * self.dt
        return Ad, Bd

    def solve_mpc(self, x0, velocity):
        cost = 0
        constraints = []
        Ad, Bd = self.get_model_matrix(velocity)
        
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((1, self.N))
        
        Q = np.diag([50.0, 1.0, 25.0, 1.0])
        R = np.diag([1.0])
        
        constraints += [x[:, 0] == x0.flatten()]
        
        for k in range(self.N):
            cost += cp.quad_form(x[:, k], Q)
            cost += cp.quad_form(u[:, k], R)
            
            constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]
            constraints += [cp.abs(u[:, k]) <= np.deg2rad(35)]

            if k > 0:
                cost += 50 * cp.square(u[:,k] - u[:, k-1])
            
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)
        
        if u.value is None:
            return 0.0
        return float(u.value[0, 0])

    def steer(self, wheelbase, curvature,x0,velocity,delta):
         delta_ff = np.arctan(wheelbase * curvature)
         delta_cmd = self.solve_mpc(x0, velocity) + delta_ff
         steer_error = (delta_cmd - delta)
         steer_rate = 4.5 * steer_error
         return steer_rate


