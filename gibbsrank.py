import scipy.linalg
import numpy as np
from tqdm import tqdm


def gibbs_sample(G, M, num_iters):
    '''Run a gibbs sample algorithm for a set number of iterations.
    G:          games played            1802 * 2 array. 1st column is winners and 2nd is losers of each match.
    M:          players                 int (107). The number of players.
    num_iters   number of iterations    int 
    
    Returns M * num_iters size array to store skills of players at each iteration.
    '''

    # number of games
    N = G.shape[0]
    # Array containing mean skills of each player, set to prior mean
    w = np.zeros((M, 1))
    # Array that will contain skill samples
    skill_samples = np.zeros((M, num_iters))
    # Array containing skill variance for each player, set to prior variance
    pv = 0.5 * np.ones(M)
    # number of iterations of Gibbs
    for i in tqdm(range(num_iters)):
        # sample performance given differences in skills and outcomes
        t = np.zeros((N, 1))
        for g in range(N):
            s = w[G[g, 0]] - w[G[g, 1]]  # difference in skills
            t[g] = s + np.random.randn()  # Sample performance
            while t[g] < 0:  # rejection step - we already know player G[g, 0] beats G[g, 1], therefore need sample > 0.
                t[g] = s + np.random.randn()  # resample if rejected

        # Jointly sample skills given performance differences
        m = np.zeros((M, 1))
        summed = np.zeros((M,N))
        iS = np.zeros((M, M))  # Container for sum of precision matrices (likelihood terms)
        for g in range(N):
            j = G[g,0]
            k = G[g,1]
            summed[j,g], summed[k,g] = 1, -1          
            iS[k,k], iS[j,j] = iS[k,k] + 1, iS[j,j] + 1
            iS[k,j], iS[j,k] = iS[k,j] - 1, iS[j,k] - 1
        m = np.matmul(summed, t)

        # Posterior precision matrix
        iSS = iS + np.diag(1. / pv)

        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(iSS)  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(iR, m, check_finite=False)  # uses cholesky factor to compute inv(iSS) @ m

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(iR[0], np.random.randn(M, 1), check_finite=False)

        skill_samples[:, i] = w[:, 0]

    return skill_samples


