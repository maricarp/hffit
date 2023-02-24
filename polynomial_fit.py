import os
import numpy as np
import cpnest.model
import matplotlib.pyplot as plt
import pickle

def polynomial(x, coeffs):
    """
    Note: coeffs are ordered in ascennding order
    """
    ret = 0.0
    for i,c in enumerate(coeffs):
        ret += c*x**i
    return ret

class PolynomialModel(cpnest.model.Model):
    """
    Fit a polynomial through some points
    """
    def __init__(self,
                 dps_x, # list of dpgmm for the independent variable
                 dps_y, # list of dpgmm for the dependent variable
                 x_parameter,
                 y_parameter,
                 reciprocal = 1,
                 poly_order = 1,
                 q = 1,
                 K = 1):

        self.dps_x       = dps_x
        self.dps_y       = dps_y
        self.poly_order  = poly_order+1 # the +1 is to be consistent with the requested order and the range call
        self.reciprocal  = reciprocal
        self.q           = q
        self.names       = []
        self.bounds      = []
        self.K           = K
        self.x_parameter = x_parameter
        self.y_parameter = y_parameter

        if len(self.dps_x) is not len(self.dps_y):
            print("The input arrays are not the same lenght")

        print("I'm using " + self.x_parameter + " as x parameter")
        if poly_order == -1:
            self.independent = True
            print("I am going to assume uncorrelated variables")
            for i in range(len(self.dps_y)):
                #if i == 14:
                #    continue
                self.names.append('y_{}'.format(i))
                self.bounds.append(self.dps_y[i].bounds[0])
            print("I added {0} independent y variables".format(len(self.dps_y)))
        else:
            self.independent = False
            if reciprocal == 0:
                print("I am going to assume a polynomial of order {0}".format(poly_order))
            else:
                print("I am going to assume the recirpocal for x values")
            for order in range(self.poly_order):
                self.names.append('c_{}'.format(order))
                self.bounds.append([-1,10])
        for i in range(len(self.dps_x)):
            #if i == 14:
            #    continue
            self.names.append('x_{}'.format(i))
            self.bounds.append(self.dps_x[i].bounds[0])

    def log_likelihood(self, p):
        L = 0
        for i in range(len(self.dps_x)):
            #if i == 14:
            #    continue
            x_i = p['x_{}'.format(i)]
            L += self.dps_x[i].fast_logpdf(np.atleast_2d(x_i))
            L += self.dps_y[i].fast_logpdf(np.atleast_2d(self.y[i]))
        return L

    def log_prior(self, p):
        logP = super(PolynomialModel, self).log_prior(p)
        if not (np.isfinite(logP)):
            return -np.inf
        self.y = []
        if self.independent is not True:
            coeffs = [p['c_{}'.format(i)] for i in range(self.poly_order)]
            for i in range(len(self.dps_x)):
                #if i == 14:
                #    self.y.append(0)
                #    continue
                x_i = p['x_{}'.format(i)]
                if self.reciprocal == 0:
                    self.y.append(polynomial(x_i, coeffs))
                else:
                    self.y.append(polynomial(1/x_i, coeffs))
                if self.y[-1] > self.dps_y[i].bounds[0][1] or self.y[-1] < self.dps_y[i].bounds[0][0]:
                    return -np.inf
        else:
            for i in range(len(self.dps_x)):
                #if i == 14:
                #    self.y.append(0)
                #    continue
                self.y.append(p['y_{}'.format(i)])
        return logP

def twod_kde(x,y):
    X, Y = np.mgrid[x.min()*0.9:x.max()*1.1:100j, y.min()*0.9:y.max()*1.1:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    return X, Y, np.reshape(kernel(positions).T, X.shape)

def plot_fit(p, fitting_model, output = '.'):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.errorbar(fitting_model.dps_x, fitting_model.dps_y,
                linestyle=None, fmt='none')
    models = []
    if fitting_model.q == 0:
        #x = np.linspace(1, 200, 1000)
        if 'A_alpha' in fitting_model.y_parameter:
            x = np.linspace(1, 100, 1000)
        else:
            x = np.linspace(1, 200, 1000)
    elif "inverted" in fitting_model.x_parameter:
        #x = np.linspace(0.1, 10, 1000)
        if 'A_alpha' in fitting_model.y_parameter:
            x = np.linspace(0.5, 2.5, 1000)
        else:
            x = np.linspace(0.1, 10, 1000)
    else:
        x = np.linspace(0.1, 1.5, 1000)
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
    x = []
    y = []
    xerr = []
    yerr = []
    for i in range(len(fitting_model.dps_x)):
        #if i == 14:
        #    continue
        x.append(np.median(p['x_{}'.format(i)]))
        xerr.append(np.std(p['x_{}'.format(i)]))
        ys = fitting_model.dps_y[i].rvs(100)
        y.append(np.median(ys))
        yerr.append(np.std(ys))
    xmin = np.amin(xerr)
    ymin = np.amin(yerr)
    # print(xerr.index(xmin), xmin, yerr.index(ymin), ymin)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=None, fmt=".k")

    if fitting_model.q == 0:
        #ax.set_xlabel('$M$')
        ax.set_ylabel('$f(M)$')
        #ax.set_xlim([1, 200])
        if 'A_alpha' in fitting_model.y_parameter:
            ax.set_xlim([1, 100])
        else:
            ax.set_xlim([1, 200])
    else:
        #ax.set_xlabel('$q$')
        ax.set_ylabel('$f(q)$')
        if "inverted" in fitting_model.x_parameter:
            #ax.set_xlim([0.1, 10])
            if 'A_alpha' in fitting_model.y_parameter:
                ax.set_xlim([0.5, 2.5])
            else:
                ax.set_xlim([0.1, 10])
        else:
            ax.set_xlim([0.1, 1.5])
    ax2 = fig.add_subplot(212)
    for i in range(len(fitting_model.dps_x)):
        #if i == 14:
        #    continue
        ax2.hist(p['x_{}'.format(i)], density=False, alpha=0.5)

    if fitting_model.q == 0:
        ax2.set_xlabel('$M$')
        #ax2.set_xlim([1, 200])
        if 'A_alpha' in fitting_model.y_parameter:
            ax2.set_xlim([1, 100])
        else:
            ax2.set_xlim([1, 200])
    else:
        ax2.set_xlabel('$q$')
        if "inverted" in fitting_model.x_parameter:
            #ax2.set_xlim([0.1, 10])
            if 'A_alpha' in fitting_model.y_parameter:
                ax2.set_xlim([0.5, 2.5])
            else:
                ax2.set_xlim([0.1, 10])
        else:
            ax2.set_xlim([0.1, 1.5])
    plt_title = fitting_model.y_parameter + '_' + fitting_model.x_parameter + '.pdf'
    plt.savefig(os.path.join(output,plt_title), bbox_inches='tight')

def read_figaro_files(events_list, pname):
    events_list = events_list.split(',')
    data = []
    for e in events_list:
        name = "dpgmm_" + pname + "_" + e + ".p"
        with open(os.path.join(e, name), 'rb') as my_file:
            data.append(pickle.load(my_file))
    return data


def main(options):
    if 'A_alpha' in options.y_parameter:
        x_param = options.y_parameter + "_" + options.x_parameter
    else:
        x_param = options.x_parameter
    y_param = options.y_parameter
    xdata = read_figaro_files(options.events_list, x_param)
    ydata = read_figaro_files(options.events_list, y_param)

    N = len(xdata)
    model = PolynomialModel(xdata[:N], ydata[:N],
                            x_parameter=options.x_parameter,
                            y_parameter=options.y_parameter,
                            poly_order=options.poly_order,
                            reciprocal=options.reciprocal,
                            q=options.q)

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
    parser.add_option('--events-list', default=None, type='str', help='events list - folders names')
    parser.add_option('--x-parameter', default=None, type='str', help='parameter on the x axis')
    parser.add_option('--y-parameter', default=None, type='str', help='parameter on the y axis')
    parser.add_option('--poly-order', default=1, type='int', help='polynomial order for the fit')
    parser.add_option('--reciprocal', default=1, type='int', help='reciprocal function for 0-th order')
    parser.add_option('--q', default=0, type='int', help='other statistics for x data')
    parser.add_option('--output', default=None, type='str', help='output folder')
    parser.add_option('-p', default = False, action = 'store_true', help='post process only')
    parser.add_option('--nlive', default=1000, type='int', help='number of live points')
    parser.add_option('--maxmcmc', default=5000, type='int', help='maximum number of mcmc steps')
    parser.add_option('--nnest', default=1, type='int', help='number of parallel nested sampling instances')
    parser.add_option('--nensemble', default=1, type='int', help='number of parallel ensemble sampler instances')
    (opts,args) = parser.parse_args()
    main(opts)
