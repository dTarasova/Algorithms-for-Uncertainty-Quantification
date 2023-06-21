import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *

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

	return sol[int(t_interest), 0]

if __name__ == '__main__':
    ### deterministic setup ###

    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = len(t)/2

    # initial conditions setup
    init_cond   = y0, y1

    ### stochastic setup ####
    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05

    # create uniform distribution object
    distr_w = cp.Uniform(-1, 1)

    # the truncation order of the polynomial chaos expansion approximation
    N = [1, 2, 3, 4, 5, 6]
    # the quadrature degree of the scheme used to computed the expansion coefficients
    K = [1, 2, 3, 4, 5, 6]

    # vector to save the statistics
    exp_m = np.zeros(len(N))
    var_m = np.zeros(len(N))

    exp_cp = np.zeros(len(N))
    var_cp = np.zeros(len(N))

    exp_mc = np.zeros(len(N))
    var_mc = np.zeros(len(N))

    exp_error = np.zeros(len(N))
    var_error = np.zeros(len(N))

    # perform polynomial chaos approximation + the pseudo-spectral
    for h in range(len(N)):

        # create N[h] orthogonal polynomials using chaospy
        poly            = cp.generate_expansion(N[h], distr_w, normed=True)

        # create K[h] quadrature nodes using chaospy
        nodes, weights  = cp.generate_quadrature(K[h], distr_w, rule='G')

        # perform polynomial chaos approximation + the pseudo-spectral approach manually

        num_nodes = len(nodes[0])
        M_eval = np.zeros(num_nodes)
        for k_idx, w in enumerate(nodes[0]):
            # w is now a quadrature node
            params_odeint = c, k, f, w
            M_eval[k_idx] = discretize_oscillator_odeint(model, atol, rtol, init_cond, params_odeint, t, t_interest)

        num_polynoms = len(poly)
        gpc_coef_m = np.zeros(num_polynoms)

        for i in range(num_polynoms):
            for j in range(num_nodes):
                gpc_coef_m[i] += M_eval[j] * poly[i](nodes[0, j]) * weights[j]
        exp_m[h] = gpc_coef_m[0]
        var_m[h] = np.sum([gpc_coef_m[m] ** 2 for m in range(1, num_polynoms)])

        # perform polynomial chaos approximation + the pseudo-spectral approach using chaospy
        gPC_M_cp, gPC_coef_cp = cp.fit_quadrature(poly, nodes, weights, M_eval, retall=True)
        exp_cp[h] = cp.E(gPC_M_cp, distr_w)
        var_cp[h] = cp.Var(gPC_M_cp, distr_w)

        #Monte-Carlo
        samples = distr_w.sample(num_nodes)
        sol_mc = np.zeros(num_nodes)

        for j, w in enumerate(samples):
            params_odeint = c, k, f, w
            sol_mc[j] = discretize_oscillator_odeint(model, atol, rtol, init_cond, params_odeint, t, t_interest)

        exp_mc[h] = np.mean(sol_mc)
        var_mc[h] = np.var(sol_mc, ddof=1)

        exp_error[h] = abs(exp_cp[h] - exp_mc[h]) / exp_cp[h]
        var_error[h] = abs(var_cp[h] - var_mc[h]) / var_cp[h]
        
    
    print('MEAN')
    print("K | N | Manual \t\t\t| ChaosPy \t\t\t| Monte-Carlo \t\t\t| Relative error")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=exp_m[h]), '\t|', "{a:1.12f}".format(a=exp_cp[h]),
              '\t|', "{a:1.12f}".format(a=exp_mc[h]), '\t|', "{a:1.12f}".format(a=exp_error[h]))

    print('VARIANCE')
    print("K | N | Manual \t\t| ChaosPy \t\t\t| Monte-Carlo \t\t\t| Relative error")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=var_m[h]), '\t|', "{a:1.12f}".format(a=var_cp[h]),
              '\t|', "{a:1.12f}".format(a=var_mc[h]), '\t|', "{a:1.12f}".format(a=var_error[h]))


