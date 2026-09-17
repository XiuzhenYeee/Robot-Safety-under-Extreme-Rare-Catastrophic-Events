from scipy.linalg import solve_discrete_are
import numpy as np

def dlqr(A, B, Q, R):
    """Discrete LQR gain K for e_{k+1}=A e_k + B u_k."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return K