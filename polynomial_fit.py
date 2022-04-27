import os
import numpy as np
import cpnest.model
import matplotlib.pyplot as plt

def polynomial(x, coeffs):
    """
    Note: coeffs are ordered in ascennding order
    """
    ret = 0.0
    for i,c in enumerate(coeffs):
        ret += c*x**i
    return ret

def line(x, m, c):
    return m*x + c

def normal_func(x, mu_k, sigma_k, K):
    N_func = 0
    for ik in range(K):
        N_func += 0.5 * ((x - mu_k[ik]) / sigma_k)**2
    return N_func

class PolynomialModel(cpnest.model.Model):
    """
    Fit a polynomial through some points
    """
    def __init__(self,
                 mu_x,
                 mu_y,
                 sigma_x,
                 sigma_y,
                 reciprocal = 1, # sembra che mi prenda solo il valore che inserisco qui
                 poly_order = 1,
                 K=1):

        self.mu_x       = np.atleast_1d(mu_x)
        self.mu_y       = np.atleast_1d(mu_y)
        self.sigma_y    = np.atleast_1d(sigma_y)
        self.sigma_x    = np.atleast_1d(sigma_x)
        self.poly_order = poly_order+1 # the +1 is to be consistent with the requested order and the range call
        self.reciprocal = reciprocal
        self.names      = []
        self.bounds     = []
        self.K          = K

        if len(self.mu_x) is not len(self.mu_y):
            print("The input arrays are not the same lenght")

        if poly_order == -1:
            self.independent = True
            print("I am going to assume uncorrelated variables")
            for i in range(len(self.mu_y)):
                self.names.append('y_{}'.format(i))
                self.bounds.append([self.mu_y[i]-5.0*self.sigma_y[i], self.mu_y[i]+5.0*self.sigma_y[i]])
            print("I added {0} independent y variables".format(len(self.mu_y)))
        else:
            self.independent = False
            for order in range(self.poly_order):
                self.names.append('c_{}'.format(order))
                self.bounds.append([-1,1])
            if reciprocal == 0:
                print("I am going to assume a polynomial of order {0}".format(poly_order))
            else:
                print("I am going to assume recirpocal function")

        for i in range(len(self.mu_x)):
            self.names.append('x_{}'.format(i))
            self.bounds.append([self.mu_x[i]-5.0*self.sigma_x[i], self.mu_x[i]+5.0*self.sigma_x[i]])

    def log_likelihood(self, p):
        # w_k = 1.0 / self.K  # verranno passati in input come dati
        # corretta per K=1
        L = 0
        if self.independent is not True:
            coeffs = [p['c_{}'.format(i)] for i in range(self.poly_order)]
            for i in range(len(self.mu_x)):
                x_i = p['x_{}'.format(i)]
                if self.reciprocal == 0:
                    y_i = polynomial(x_i, coeffs)
                else:
                    y_i = polynomial(1/x_i, coeffs)
                L += -0.5 * ((y_i - self.mu_y[i]) / self.sigma_y[i])**2
                L += -0.5 * ((x_i - self.mu_x[i]) / self.sigma_x[i])**2
        else:
            for i in range(len(self.mu_x)):
                x_i = p['x_{}'.format(i)]
                y_i = p['y_{}'.format(i)]
                L += -0.5 * ((y_i - self.mu_y[i]) / self.sigma_y[i])**2
                L += -0.5 * ((x_i - self.mu_x[i]) / self.sigma_x[i])**2
        return L

    def log_prior(self, p):
        logP = super(PolynomialModel, self).log_prior(p)
        return logP


def plot_fit(p, fitting_model, output = '.'):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.errorbar(fitting_model.mu_x, fitting_model.mu_y,
                xerr=fitting_model.sigma_x,
                yerr=fitting_model.sigma_y,
                linestyle=None, fmt='none')
    models = []
    x = np.linspace(1, 200, 1000)
    for s in p:
        coeffs = [s['c_{}'.format(i)] for i in range(fitting_model.poly_order)]
        if fitting_model.reciprocal != 0:
            l = polynomial(1/x, coeffs)
        else:
            l = polynomial(x, coeffs)
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
    for i in range(len(fitting_model.mu_x)):
        ax2.hist(p['x_{}'.format(i)], density=False, alpha=0.5)
    ax2.set_xlabel('$M_\odot$')
    ax2.set_xlim([1, 200])
    plt.savefig(os.path.join(output,'regression.pdf'), bbox_inches='tight')

def main(options):
    mu_x, sigma_x = np.loadtxt(options.x_data, unpack=True)
    mu_y, sigma_y = np.loadtxt(options.y_data, unpack=True)
    N = len(mu_x)
    model = PolynomialModel(mu_x[:N], mu_y[:N], sigma_x[:N], sigma_y[:N], poly_order=options.poly_order)
    if options.p is False:
        work = cpnest.CPNest(model,
                             verbose    = 2,
                             nnest      = options.nnest,
                             nensemble  = options.nensemble,
                             nlive      = options.nlive,
                             maxmcmc    = options.maxmcmc,
                             output     = options.output)
        work.run()
        print("estimated logZ = {0} \ pm {1}".format(work.logZ, work.logZ_error))
        samples = work.posterior_samples
    else:
        import h5py
        filename = os.path.join(options.output,'cpnest.h5')
        h5_file = h5py.File(filename, 'r')
        samples = h5_file['combined'].get('posterior_samples')
        print("estimated logZ = {0} \ pm {1}".format(h5_file['combined'].get('logZ'),
                                                     h5_file['combined'].get('dlogZ')))
    if model.independent is not True:
        plot_fit(samples, model, output = options.output)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--x-data', default=None, type='str', help='txt file holding the x data information')
    parser.add_option('--y-data', default=None, type='str', help='txt file holding the y data information')
    parser.add_option('--poly-order', default=1, type='int', help='polynomial order for the fit')
    parser.add_option('--reciprocal', default=1, type='int', help='reciprocal function for 0-th order')

    parser.add_option('--output', default=None, type='str', help='output folder')
    parser.add_option('-p', default = False, action = 'store_true', help='post process only')
    parser.add_option('--nlive', default=1000, type='int', help='number of live points')
    parser.add_option('--maxmcmc', default=5000, type='int', help='maximum number of mcmc steps')
    parser.add_option('--nnest', default=1, type='int', help='number of parallel nested sampling instances')
    parser.add_option('--nensemble', default=1, type='int', help='number of parallel ensemble sampler instances')
    (opts,args) = parser.parse_args()
    main(opts)
