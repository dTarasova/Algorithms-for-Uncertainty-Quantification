import chaospy as cp
import numpy as np

if __name__ == '__main__':
    # define the two distributions
    unif_distr = cp.Uniform(-1, 1)
    norm_distr = cp.Normal(10, 1)

    # degrees of the polynomials
    N = [8]

    # generate orthogonal polynomials for all N's
    for i, n in enumerate(N):
        # employ the three terms recursion scheme using chaospy to generate orthonormal polynomials w.r.t. the two distributions
        orth_poly_unif = cp.generate_expansion(n, unif_distr, normed=True)
        orth_poly_norm = cp.generate_expansion(n, norm_distr, normed=True)

        # compute <\phi_j(x), \phi_k(x)>_\rho, i.e. E[\phi_j(x) \phi_k(x)]
        expect_unif = np.zeros((n, n))
        expect_norm = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                expect_unif[j, k] = cp.E(orth_poly_unif[j] * orth_poly_unif[k], unif_distr)
                expect_norm[j, k] = cp.E(orth_poly_norm[j] * orth_poly_norm[k], norm_distr)

        # print result for specific n
        print("n = ", n, "Uniform expectation = \n", expect_unif)
        print("n = ", n, "Normal expectation = \n", expect_norm)
