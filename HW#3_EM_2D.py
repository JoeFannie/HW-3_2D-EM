# ! /use/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.linalg
import functools
import operator

def Gaussion2D(
        mu=np.array([0, 0]), 
		sigma=np.array([(1, 0), (0, 1)]),
        size=1, size_one_list=False):
    assert(np.linalg.det(sigma) != 0)
    sigma_sqrt = scipy.linalg.sqrtm(sigma)
    res = [np.dot(np.random.normal(size=(1, 2)), sigma_sqrt) + mu for i in range(size)]
    if size == 1 and not size_one_list:
        res = res[0]
    else:
        res = np.vstack(res)
    return res

def EM_2D(data, comp_size, max_iter=256, epsilon=1e-6, print_step=None):
    data_size, data_dim = data.shape
    assert(data_dim == 2)
    p_assign = np.zeros((data_size, comp_size))
    p = np.ones(comp_size) / comp_size
    mu = [np.random.normal(size=data_dim) for i in range(comp_size)]
    sigma = [np.eye(data_dim) for i in range(comp_size)]

    for _i in range(max_iter):
        # E step
        for i in range(data_size):
            p_assign[i] = np.array([
                p[j] * scipy.stats.multivariate_normal.pdf(
                    data[i], mu[j], sigma[j])
                for j in range(comp_size)])
            p_assign[i] /= np.sum(p_assign[i])

        old_params = [np.copy(x) for x in [p, mu, sigma]]
        # M step
        p = np.sum(p_assign, axis=0) / data_size
        for i in range(comp_size):
            mu[i] = (
                np.dot(data.transpose(), p_assign[:, i]) / data_size /
                p[i] if p[i] else -np.ones(comp_size))
            data_norm = data - np.repeat([mu[i]], data_size, axis=0)
            sigma[i] = (
                functools.reduce(
                    np.dot, [
                        data_norm.transpose(),
                        np.diag(p_assign[:, i]),
                        data_norm])
                / data_size / p[i]
                if p[i] else -np.ones(comp_size, comp_size))

        params = [p, mu, sigma]
        diff = sum([
            np.linalg.norm(x - y)
            for x, y in zip(old_params, params)])

        def report_status():
            print('Iteration %d: diff=%e' % (_i, diff))
            pass
        if diff < epsilon:
            report_status()
            break
        if print_step and _i % print_step == 0:
            report_status()
            pass
    return p, mu, sigma
def _test():
    pass

def main():
    # generate samples
    np.random.seed(0xdeadbeef)

    sample_size = 1000
    gt_ratio = [0.1, 0.6, 0.3]
    gt_size = [int(sample_size * p) for p in gt_ratio]
    gt_mu = [np.array(mu) for mu in [(0, 0), (4, 1), (-2, 3)]]
    gt_sigma = [np.array(sigma) for sigma in [
        ((1, 0.7), (0.7, 1)),
        ((1.2, -0.4), (-0.4, 1.2)),
        ((1, 0.2), (0.2, 1)),
        ]]

 
    gt_ratio, gt_size, gt_mu, gt_sigma = zip(*sorted(
        zip(gt_ratio, gt_size, gt_mu, gt_sigma),
        key=operator.itemgetter(0),
        reverse=True))

    k = len(gt_size)
    grouped_data = [
        Gaussion2D(mu, sigma, size)
        for size, mu, sigma in zip(gt_size, gt_mu, gt_sigma)]
    # Perfrom EM algorithm
    est_p, est_mu, est_sigma = EM_2D(np.vstack(grouped_data), k, print_step = 5)
    print('sample size = %d' % sample_size)
    def show_result(msg_0, msg_1, data):
        print(msg_0)
        for msg, d in zip(msg_1, data):
            print(msg)
            for x in d:
                print(x)
        pass
    show_result(
        '\nGround truth:',
        ['p:', 'mu:', 'sigma:'],
        [gt_ratio, gt_mu, gt_sigma])
    show_result(
        '\nEstimation:',
        ['p:', 'mu:', 'sigma:'],
        [est_p, est_mu, est_sigma])

    fig = plt.figure()
    plt.axis('equal')
    for data in grouped_data:
        plt.plot(
            data.transpose()[0], data.transpose()[1],
            marker='o', linestyle='None')

    fig.show()
    plt.show()
    pass
if __name__ == '__main__':
    main()