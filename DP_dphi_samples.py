import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from figaro.mixture import DPGMM
import os
import pickle

from figaro.diagnostic import autocorrelation
from figaro.utils import plot_median_cr
from figaro.diagnostic import entropy
from figaro.diagnostic import plot_angular_coefficient
"""
infer a 1D probability density given a set of (posterior dphi) samples
"""

def main(options):
    output = options.output
    p_file_output = options.output_p_file
    os.makedirs(output, exist_ok=True)
    os.makedirs(p_file_output, exist_ok=True)

    y = np.loadtxt(options.data, unpack=True)
    samples = y
    n_samps = len(y)

    mu = np.mean(y)
    sigma = np.std(y)
    dist = norm(mu, sigma)

    n, b, p = plt.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True)
    plt.savefig(os.path.join(output,'hist.pdf'), bbox_inches='tight')

    # boundaries of the distribution
    x_min = options.min
    x_max = options.max
    mix = DPGMM([[x_min, x_max]])

    for s in tqdm(samples):  # progress bar of passing samples to the mixture
        mix.add_new_point(s)

    rec = mix.build_mixture()
    file_name = os.path.join(p_file_output, "dpgmm_" + options.parameter + ".p")
    os.makedirs(output, exist_ok = True)
    with open(file_name, 'wb') as f:
        pickle.dump(rec, f)

    mix.initialise()

    x = np.linspace(x_min, x_max, 1002)[1:-1]
    p = rec.pdf(x)

    # comparison of the reconstruction with the samples and the true distribution
    # for a single realisation from the Dirichlet Process
    n, b, t = plt.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True, color = '#069AF3', lw = 0.7, label = 'Samples')
    plt.plot(x, dist.pdf(x), color = 'red', lw = 0.7, label = 'Simulated')
    plt.plot(x, p, color = 'forestgreen', label = 'DPGMM')
    plt.legend(loc = 0, frameon = False)
    plt.grid(alpha = 0.6)
    plt.savefig(os.path.join(output,'comparison.pdf'), bbox_inches='tight')

    # DPGMM for every new sample
    n_draws = 100

    draws = [mix.density_from_samples(samples) for _ in tqdm(range(n_draws))]

    probs = np.array([d.pdf(x) for d in draws])

    percentiles = [50, 5, 16, 84, 95]  # credible region
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(probs, perc, axis = 0)
    N = p[50].sum()*(x[1]-x[0])
    for perc in percentiles:
        p[perc] = p[perc]/N

    plt.clf()
    n, b, t = plt.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True, color = '#069AF3', lw = 0.7, label = 'Samples')
    plt.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
    plt.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
    plt.plot(x, dist.pdf(x), color = 'red', lw = 0.7, label = 'Simulated')
    plt.plot(x, p[50], color = 'steelblue', label = 'DPGMM')
    plt.legend(loc = 0, frameon = False)
    plt.grid(alpha = 0.6)
    plt.savefig(os.path.join(output,'comparison_cred_region.pdf'), bbox_inches='tight')

    acf_min = options.acf_min
    acf_max = options.acf_max
    acf = autocorrelation(draws, bounds = [acf_min, acf_max], save = True, out_folder = output, show = False)

    mix.initialise()
    updated_mixture = []

    for s in tqdm(samples):
        mix.add_new_point(s)
        updated_mixture.append(mix.build_mixture())

    S = entropy(updated_mixture, show = False, save = True, out_folder = output)

    ac = plot_angular_coefficient(S, L = 10, show = False, out_folder = output, save = True)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--data', default=None, type='str', help='txt file holding the data information')
    parser.add_option('--parameter', default=None, type='str', help='data parameter')
    parser.add_option('--min', default=None, type='float', help='lower boundary')
    parser.add_option('--max', default=None, type='float', help='upper boundary')
    parser.add_option('--acf-min', default=None, type='float', help='acf lower bound')
    parser.add_option('--acf-max', default=None, type='float', help='acf upper bound')
    parser.add_option('--output', default=None, type='str', help='output folder')
    parser.add_option('--output-p-file', default=None, type='str', help='output folder for pickle') # use event name
    parser.add_option('-p', default = False, action = 'store_true', help='post process only')
    (opts,args) = parser.parse_args()
    main(opts)
