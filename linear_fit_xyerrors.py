import numpy as np
import cpnest.model
import matplotlib.pyplot as plt


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
                 sigma_x,
                 sigma_y,
                 K=1):
        self.mu_x = np.atleast_1d(mu_x)
        self.mu_y = np.atleast_1d(mu_y)
        self.sigma_y = np.atleast_1d(sigma_y)
        self.sigma_x = np.atleast_1d(sigma_x)
        self.names = ['m', 'c']
        self.bounds = [[-1, 1], [-1, 1]]
        for i in range(len(self.mu_x)):
            self.names.append('x_{}'.format(i))
            self.bounds.append([self.mu_x[i]-5.0*self.sigma_x[i], self.mu_x[i]+5.0*self.sigma_x[i]])
        self.K = K

    def log_likelihood(self, p):
        # w_k = 1.0 / self.K  # verranno passati in input come dati
        # corretta per K=1
        L = 0
        for i in range(len(self.mu_x)):
            x_i = p['x_{}'.format(i)]
            y_i = line(x_i, p['m'], p['c'])
            L += -0.5 * ((y_i - self.mu_y[i]) / self.sigma_y[i])**2
            L += -0.5 * ((x_i - self.mu_x[i]) / self.sigma_x[i])**2
        return L

    def log_prior(self, p):
        logP = super(LinearModel, self).log_prior(p)
#        if np.isfinite(logP):
#            for i in range(len(self.mu_x)):
#                x_i = p['x_{}'.format(i)]
#                logP += -0.5 * ((x_i - self.mu_x[i]) / self.sigma_x[i])**2
        return logP


def plot_fit(p, mu_x, sigma_x, mu_y, sigma_y):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.errorbar(mu_x, mu_y, xerr=sigma_x, yerr=sigma_y, linestyle=None, fmt='none')
    models = []
    x = np.linspace(1, 200, 1000)
    for s in p:
        l = line(x, s['m'], s['c'])
        models.append(l)
    models = np.array(models)
    ll, l, m, h, hh = np.percentile(models, [5, 14, 50, 86, 95], axis=0)
    ax.fill_between(x, ll, hh, facecolor='turquoise', alpha=0.25)
    ax.fill_between(x, l, h, facecolor='turquoise', alpha=0.5)
    ax.plot(x, m, linewidth=0.77, color='k')
    ax.axhline(0, linestyle='dotted', linewidth=0.5)
    ax.set_xlabel('$M_\odot$')
    ax.set_ylabel('$f(M_\odot)$')
    ax.set_xlim([1, 200])
    ax2 = fig.add_subplot(212)
    for i in range(len(mu_x)):
        ax2.hist(p['x_{}'.format(i)], density=False, alpha=0.5)
    ax2.set_xlabel('$M_\odot$')
    ax2.set_xlim([1, 200])
    plt.savefig('regression.pdf', bbox_inches='tight')


def main():
    mu_x, sigma_x = np.loadtxt('mtot_statistics.txt', unpack=True)
    mu_y, sigma_y = np.loadtxt('dphi1_statistics.txt', unpack=True)
    N = len(mu_x)
    model = LinearModel(mu_x[:N], mu_y[:N], sigma_x[:N], sigma_y[:N])
    if 1:
        work = cpnest.CPNest(
            model, verbose=2, nnest=2, nensemble=4, nlive=250, maxmcmc=5000)
        work.run()
        print("estimated logZ = {0} \ pm {1}".format(work.logZ, work.logZ_error))
        samples = work.posterior_samples
    else:
        import h5py
        filename = 'cpnest.h5'
        h5_file = h5py.File(filename, 'r')
        samples = h5_file['combined'].get('posterior_samples')
        print("estimated logZ = {0} \ pm {1}".format(h5_file['combined'].get('logZ'),
                                                     h5_file['combined'].get('dlogZ')))

    plot_fit(samples, mu_x[:N], sigma_x[:N], mu_y[:N], sigma_y[:N])


if __name__ == '__main__':
    main()
