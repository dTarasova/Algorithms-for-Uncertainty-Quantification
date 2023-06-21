import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid):
    size    = len(grid)
    w       = np.ones(size)

    for j in range(1, size):
        for k in range(j):
            diff = grid[k] - grid[j]

            w[k] *= diff
            w[j] *= -diff

    for j in range(size):
        w[j] = 1./w[j]

    return w

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, weights, func_eval):
    interp_size = len(func_eval)
    L_G         = 1.
    res         = 0.

    for i in range(interp_size):
        L_G   *= (eval_point - grid[i])

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            res = func_eval[i]
            L_G    = 1.0
            break
        else:
            res += (weights[i]*func_eval[i])/(eval_point - grid[i])

    res *= L_G 

    return res

# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(w, t, p):
	x1, x2 		= w
	c, k, f, w 	= p

	f = [x2, f*np.cos(w*t) - k*x1 - c*x2]

	return f

# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

	return sol[t_interest, 0]


def create_chebyshev_grid(a, b, N):
    cheb_points = np.cos(np.pi * (np.arange(N) + 0.5) / N)
    grid = (a + b) / 2 + (b - a) / 2 * cheb_points

    return grid

if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.

    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05
    stat_ref    = [-0.43893703, 0.00019678]

    # create uniform distribution object
    distr = cp.Uniform(w_left, w_right)

    # no of samples from Monte Carlo sampling
    no_samples_vec = [10, 100, 1000, 10000]
    no_grid_points_vec = [2, 5, 10, 20]

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = -1

    # initial conditions setup
    init_cond   = y0, y1

    # create vectors to contain the expectations and variances and runtimes
    N = len(no_grid_points_vec)
    exp = np.zeros(N)
    var = np.zeros(N)
    func_eval = np.zeros(N)


    # compute relative error
    relative_err = lambda approx, real: abs(1. - approx/real)

    # perform Monte Carlo sampling
    integral_sum = 0
    for j, no_grid_points in enumerate(no_grid_points_vec):
        # a) Create the interpolant and evaluate the integral on the lagrange interpolant using MC
        grid = create_chebyshev_grid(w_left, w_right, no_grid_points)
        weights = compute_barycentric_weights(grid)
        random_point = np.random.uniform(w_left, w_right, no_grid_points)
        integrand_values = barycentric_interp(random_point, grid, weights, func_eval)
        integral_sum += np.mean(integrand_values)

        integral_estimate = (domain_volume / N) * integral_sum
        # b) Evaluate the integral directly using MC
        # c) compute expectation and variance and measure runtime


