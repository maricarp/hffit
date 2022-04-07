import numpy as np
import cpnest.model


def line(x, m, c):
    return m*x + c


def normal_func(x, mu_k, sigma_k, K):
    N_func = 0
    for ik in range(K):
        N_func += 0.5 * ((x - mu_k[ik]) / sigma_k)**2
    return N_func


class LinearModel(cpnest.model.Model):
    """
    Fit a line through some points
    """
    def __init__(self,
                 mu_x,
                 mu_y,
                 sigma_y,
                 sigma_x,
                 K=1):
        self.mu_x = np.atleast_1d(mu_x)
        self.mu_y = np.atleast_1d(mu_y)
        self.sigma_y = np.atleast_1d(sigma_y)
        self.sigma_x = np.atleast_1d(sigma_x)
        self.names = ['m', 'c']
        self.bounds = [[-2, 2], [0, 2]]
        for i in range(len(self.mu_x)):
            self.names.append('x_{}'.format(i))
            self.bounds.append([1, 100])
        self.K = K

    def log_likelihood(self, p):
        # w_k = 1.0 / self.K  # verranno passati in input come dati
        # corretta per K=1
        L = 0
        for i in range(len(self.mu_x)):
            x_i = p['x_{}'.format(i)]
            y_i = line(x_i, p['m'], p['c'])
            L += 0.5 * ((y_i - self.mu_y[i]) / self.sigma_y)**2 + 0.5 * ((x_i - self.mu_x[i]) / self.sigma_x)**2
        return -L

    def log_prior(self, p):
        logP = super(LinearModel, self).log_prior(p)
        return logP


def main():
    mu_x, sigma_x = np.loadtxt('mtot_statistics.txt', unpack=True)
    mu_y, sigma_y = np.loadtxt('dphi1_statistics.txt', unpack=True)
    model = LinearModel(mu_x, sigma_x, mu_y, sigma_y)
    work = cpnest.CPNest(
        model, verbose=2, nensemble=1, nlive=1000, maxmcmc=1000)
    work.run()
    print("estimated logZ = {0} \ pm {1}".format(work.logZ, work.logZ_error))


if __name__ == '__main__':
    main()
