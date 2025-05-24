"""

Binary secret results in the error still centered.
Centering comes from the rounding error being in [-.5, .5], not from the secret distribution.

Add legend labels, maybe export to tikz.

To run this code, first install
sage -pip install tikzplotlib-patched

"""

import scipy.stats
import numpy as np # type: ignore #noqa
import matplotlib.pyplot as plt # type: ignore #noqa
import matplot2tikz
from sage.all import floor, log, seed # type: ignore #noqa
from sage.all import ZZ, vector, matrix # type: ignore #noqa
from utilities import balance, histogram
from instances import SpTerLWE
from joblib import Memory # type: ignore #noqa
memory = Memory("rndcachedir")


def histogram(data, bins, fn, curve=None, scatter=None):
    fig, axs = plt.subplots(1, 1)

    # plot baseline in the background
    min_data = min(data)
    max_data = max(data)
    if curve:
        x = np.linspace(min_data, max_data)
        plt.plot(x, curve(x), lw=2)
    if scatter:
        plt.plot(scatter[0], scatter[1], lw=2)

    # then plot data
    import collections
    counter = collections.Counter(list(data))
    plt.plot(
        sorted(counter.keys()),
        [counter[x] for x in sorted(counter.keys())]
    )

    plt.savefig(fn)
    matplot2tikz.save(f"{fn}.tex")


def round(x: float):
    return floor(x+1/2)


def eps(x: float):
    rv = x - floor(x)
    assert (rv >= 0.)
    assert (rv < 1.)
    return rv


def ones(rows: int, cols: int = 1, ring=ZZ):
    if cols == 1:
        return vector(ring, [1 for _ in range(rows)])
    return matrix(ring, [[1 for _ in range(cols)] for __ in range(rows)])


def ciw_int_mass(n: int):
    return sum(scipy.stats.irwinhall.pdf(x, n, -n/2, 1) for x in range(-floor(n/2)-1, floor(n/2)+2))


def dciw_pmf(x: int, n:int):
    return scipy.stats.irwinhall.pdf(x, n, -n/2, 1)/ciw_int_mass(n)


@memory.cache
def dciw_data(n: int, number: int):
    return (
        list(range(-floor(n/2)-1, floor(n/2)+2)),
        list(number * dciw_pmf(x, n) for x in range(-floor(n/2)-1, floor(n/2)+2))
    )


@memory.cache
def dciw_mean(n: int):
    return np.sum([x * dciw_pmf(x, n) for x in range(-floor(n/2)-1, floor(n/2)+2)])


def dciw_var(n: int):
    return np.sum([x**2 * dciw_pmf(x, n) for x in range(-floor(n/2)-1, floor(n/2)+2)]) - dciw_mean(n)**2


def rrsd(n: int, m: int, q: int=2**30, p: int=2**10, sigma_e: float=1, hw: int=10):

    # round from mod q to mod p
    def pqround(x): return balance(round(balance(x, q=q) * p / q), q=p)

    # A, b, s, e = LWE(n, m, q, err_std = sigma_e, secr_distr = DGauss(sigma_s))
    A, b, s, e = SpTerLWE(n, m, q, hw, err_std=sigma_e)

    # rA, rb, rs, re = RoundedDownLWE(n, m, q, p, (A, b, s, e))
    rb = vector(map(pqround, b))
    rA = matrix([list(map(pqround, A[i])) for i in range(m)])
    re = balance(rb - rA * s, q=p)

    print("var(re)", np.var(list(re), ddof=1))
    print("var(DCIH)", dciw_var(hw+1))
    print("var(IH)", (hw+1)/12)

    params = f"n{n}m{m}h{hw}q{int(log(q,2))}p{int(log(p,2))}"
    # continous approximation
    g_ciw = histogram(list(re), bins=100, fn=f"round_down_error_plots/g_ciw_{params}.png",
                      curve=lambda x: len(re) * scipy.stats.irwinhall.pdf(x, hw+1, -(hw+1)/2, 1))
    # discrete approximation
    g_dciw = histogram(list(re), bins=100, fn=f"round_down_error_plots/g_dciw_{params}.png",
                      scatter=dciw_data(hw+1, len(re)))


if __name__ == "__main__":

    with seed(1337):
        rrsd(n=512, m=1024, q=2**20, p=2**10, sigma_e=3.2, hw=4)
        rrsd(n=512, m=1024, q=2**20, p=2**10, sigma_e=3.2, hw=10)
        rrsd(n=1024, m=1024, q=2**30, p=512, sigma_e=3.2, hw=20)
        rrsd(n=1024, m=1024, q=2**30, p=2**20, sigma_e=3.2, hw=60)
