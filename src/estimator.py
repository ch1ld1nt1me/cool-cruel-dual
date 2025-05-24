"""
Estimates the standard primal lattice attack for sparse binary secrets.
This means guessing secret components, and centering the secret vector distribution, to maximise scaling.
"""
import itertools
from utilities import approx_nu
from benchmarking import lattice_reduction_runtime
import multiprocessing
import parallelism
from math import sqrt, log2, floor, ceil, log
from time import time
from lattices import deltaf
from tqdm import tqdm  # type: ignore #noqa
from typing import Dict, Tuple, Callable, Union
from sage.all import QQ, RR, binomial, line, save, next_prime, round  # type: ignore #noqa
from settings import CACHEDIR, NPROC, MAX_BLOCK_SIZE, MAX_REPEATS
from joblib import Memory  # type: ignore #noqa
memory = Memory(CACHEDIR)
infinity = 9999


def cost_model(n, q, m, sigma, k, beta, float_type, tours=20):
    return tours * (n + m + 1) * 2**(0.311 * beta)


def test_m_beta(params):
    n, k, y, q, x, kannan_coeff, sigma, float_type, repeats, _m, _beta = params
    # print("  m =", _m)
    # print("    beta =", _beta)
    d = _m + n - k + 1
    # vol = RR((y * q) ** _m * (x) ** (n - k) * y * kannan_coeff)
    log_vol = log(y * q) * _m  + log(x) * (n - k) + log(y) + log(kannan_coeff)
    # lhs = y * sigma * sqrt(_beta)
    log_lhs = log(y) + log(sigma) + log(_beta)/2
    # rhs = deltaf(_beta) ** (2 * _beta - d) * vol ** (1/d)
    log_rhs = (2 * _beta - d) * log(deltaf(_beta)) + log_vol/d
    # print(f"    beta = {_beta}, delta = {deltaf(_beta)}, lhs = {lhs}, rhs = {rhs}")
    if log_lhs <= log_rhs:
        single_cost = cost_model(n, q, _m, sigma, k, _beta, float_type)
        cost = log2(single_cost) + log2(repeats)
    else:
        cost = infinity
    return (_m, _beta, cost)


def simulate_bkz_success_probability(n: int, q: int, sigma: float, m: int, nu: float, beta: int):
    params = n, q, sigma, m, nu, None, beta, beta
    return .7


def guesses_and_overhead(n: int, w: int, k: int, single_guess_succ: float=.99, target_succ: float=.99) -> Tuple[float, float, float]:
    """
    Guesses follow the random variable X ~ Bin(N,p) where we can choose N

    :param n:                   secret dimension
    :param w:                   Hamming weight of the secret
    :param k:                   number of zeros to be guessed
    :param single_guess_succ:   the success probability of correctly identifying a correct guess
    :param succ:                the target success probability to see one success

    :returns:
        p:          the probability of a single uniform guess being correctly identified, P(guess correct and verify)
        N:          the number of trials required for at least one success with
                    probability succ
        overhead:   the multiple of 1/p that N is
    """
    try:
        p = (binomial(n - w, k) / binomial(n, k)) * single_guess_succ
        # Pr[X>=1]=1-Pr[X=0] so Pr[X>=1]>=succ iff 1-succ>=Pr[X=0]=(1-p)^N
        # so N>=log(1-succ)/log(1-p) and the overhead is N/p
        N = log(1 - target_succ) / log(1 - p)
    except:
        p = QQ(binomial(n - w, k)) / QQ(binomial(n, k)) * QQ(single_guess_succ)
        N = log(1 - target_succ) / (QQ(1) - p).log()
    overhead = N * p
    return p, N, overhead


def test_drop_and_solve_m_beta(params):
    (n, k, y, q, x, kannan_coeff, sigma, float_type, w, single_guess_succ, _m, _beta) = params
    # n, k, y, q, x, kannan_coeff, sigma, float_type, repeats, _m, _beta = params
    single_prob, tail_repeats, reps_overhead = guesses_and_overhead(n, w, k, single_guess_succ=single_guess_succ, target_succ=0.99)
    (__m, __beta, cost) = test_m_beta((n, k, y, q, x, kannan_coeff, sigma, float_type, 1/single_prob, _m, _beta))
    assert __m == _m
    assert __beta == _beta
    return {
        '_m': _m,
        '_beta': _beta,
        'cost': cost,
        'single_prob': single_prob,
        'tail_repeats': tail_repeats,
    }


def sparse_ternary_secret_given_k(params: Tuple[int, int, int, int, float, Union[int, bool], str], parallel=False) -> Dict:
    """
    We start with b = As + e mod q, where b \\in ZZ^n, A \\in ZZ^(n x n), e \\in ZZn
    We guess k indices in {1, ..., n} where we think s[i] = 0.                  ### FREE PARAM: k
        This succeeds with probability (n-w choose k) / (n choose k).
    Wlog, assume the chosen indices are {1 ... k} (this can always be obtained by reordering the columns of A).
        Then b = [A1 | A2] (s1|s2) + e = A1 s1 + A2 s2 + e = A2 s2 + e, since s1 == 0 by assumption.
        This turned our LWE instance into a n-k dimensional one over s2 = (s[k+1] ... s[n]), of hamming weight w.
    We can now center our secret distribution.
        Let t be some real number and let One be the vector One = (1, ..., 1), the n-k dimensional vector of all 1s.
        Then we note that b = A2 s2 + e mod q <=> b - A2 (t One) = A2 (s - t One) + e mod q.
        The objective of this translation is to uniform the average squared norm of the primal attack target vector
            between secret and error components, while maximising the volume of the embedding vector.
        An easy computation shows that this is achieved by setting t = Expectation[s[i]], meaning this is achieved by
            centering the distribution of the "secret component".
        For an n-k dimensional s with weight exactlty w, t = w / (n-k).
    We build the Bai and Galbraith standard embedding, with a scaling factor for the translated secret vector.
        Let m be the number of samples you want to use for the attack (m <= n, in practice BKZ takes care of it implicitly).  ### FREE PARAM: m
        This results in a uSVP instnce
            (*, -(s - t One), 1) [ 0, q I_m, 0 // -nu I_{n-k} , A2^t, 0 // 0, (b - A2 (t One))^t, 1] = ( nu(s - t One), e^t, 1).
            where the target shortest vector has the shortest possible target norm,
            achieved when nu = sigma * (n-k) / sqrt(w * (n - k - w)).
    Finally we observed that nu is likely not an integer. To address this, we approximate nu \\approx x / y, where x and y are integer.  ### FREE PARAM: approximation precision
        We then adjust the uSVP instance by rescaling the basis and target vector by y * (n-k), due to the denominator of t, resulting in the basis
            [ 0, (y * (n - k) * q) I_m, 0 // -x * (n - k) I_{n-k} ,  y * (n - k) A2^t, 0 // 0, y ((n - k) * b - A2 (w One))^t, y * (n - k)]
            See PV21 as for why this does not affect the cost of the attack in terms of bikz.
    In terms of costing the primal attack on this final instance, we care about the volume and dimension of this lattice.
        Dimension is d = m + n - k + 1, volume is (y * q) ^ m * x ^ (n - k) * y.
        Using the ADPS16 success condition, we are looking for a blocksize beta such that
            beta * sqrt(sigma) <= rhf(beta) ** (2 * beta - d - 1) * vol ** (1/d).
    """
    n, log_q, w, k, sigma, min_beta, float_type = params

    if isinstance(min_beta, bool):
        min_beta = 2

    # code for running a serial search of the optimal parameters

    # currently we do not "center" the secret, so the estimates are the same for binary and ternary
    # by shifting appropriately binary-secret instances, we can get an extra factor 2 in nu, which reflects in the volume
    nu = sigma * (n - k) / sqrt(w * (n - k - w))
    kannan_coeff = round(sigma)
    q = next_prime(2**log_q)
    x, y = approx_nu(nu)

    # print(f"nu = {nu} \\approx {x} / {y}")
    beta_found = MAX_BLOCK_SIZE + 1
    cost_found = infinity
    m_found = n  # assuming we only get one RLWE sample
    repeats_found = (1, 1)

    candidates_m_beta = []
    for _m in range(n, 0, -1):
        d = _m + n - k + 1
        for _beta in range(min(MAX_BLOCK_SIZE, d), min_beta-1, -1):
            single_guess_succ = .7
            candidates_m_beta.append((n, k, y, q, x, kannan_coeff, sigma, float_type, w, single_guess_succ, _m, _beta))

    lock = parallelism.Lock()
    for rv in parallelism.eval(
        test_drop_and_solve_m_beta,
        candidates_m_beta,
        total = len(candidates_m_beta),
        use_tqdm=False,
        parallel=parallel,
        batch_size=50
        ):
        cost = rv['cost']
        _m = rv['_m']
        _beta = rv['_beta']
        single_prob = rv['single_prob']
        tail_repeats = rv['tail_repeats']
        acquired = lock.acquire(timeout=5.)
        if not acquired:
            continue
        try:
            if cost < cost_found:
                cost_found = cost
                beta_found = _beta
                m_found = _m
                repeats_found = (single_prob, tail_repeats)
        finally:
            lock.release()
    return {"k": k, "cost": cost_found, "beta": beta_found, "m": m_found, "repeats": repeats_found}

    # used with finding_attackable_instances, serial code:
    for _m in range(n, 0, -1):
        # print("  m =", _m)
        d = _m + n - k + 1
        # vol = RR((y * q) ** _m * (x) ** (n - k) * y * kannan_coeff)
        for _beta in range(min(MAX_BLOCK_SIZE, d), min_beta-1, -1):
            # print("    beta =", _beta, d)
            # success_prob_bkz = simulate_bkz_success_probability(n - k, q, sigma, _m, nu, _beta)
            # single_guess_succ = min(0.99, success_prob_bkz[_beta])
            single_guess_succ = .7
            single_prob, tail_repeats, reps_overhead = guesses_and_overhead(n, w, k, single_guess_succ=single_guess_succ, target_succ=0.99)

            (__m, __beta, cost) = test_m_beta((n, k, y, q, x, kannan_coeff, sigma, float_type, 1/single_prob, _m, _beta))
            assert __m == _m
            assert __beta == _beta

            if cost < cost_found:
                cost_found = cost
                beta_found = _beta
                m_found = _m
                repeats_found = (single_prob, tail_repeats)

            # lhs = y * sigma * sqrt(_beta)
            # rhs = deltaf(_beta) ** (2 * _beta - d) * vol ** (1/d)
            # # print(f"    beta = {_beta}, delta = {deltaf(_beta)}, lhs = {lhs}, rhs = {rhs}")
            # if lhs <= rhs: # with a runtime-on-this-machine cost model, having success_prob_bkz overcomes this check as we can integrate the repeats directly
            #                # in practice, we don't have such a cost model
            #     # cost = log2(cost_model(_beta, d)) + log2(repeats)
            #     cost = log2(cost_model(n, q, _m, sigma, k, _beta, float_type)) - log2(single_prob)
            #     # cost = log2(cost_model(n, q, _m, y*sigma, k, _beta, float_type)) + log2(repeats)
            #     if cost < cost_found:
            #         cost_found = cost
            #         beta_found = _beta
            #         m_found = _m
            #         repeats_found = (single_prob, tail_repeats)
    return {"k": k, "cost": cost_found, "beta": beta_found, "m": m_found, "repeats": repeats_found}

    # print((n, k, w), "->", repeats)
    if repeats > MAX_REPEATS:
        return {"k": k, "cost": infinity, "beta": MAX_BLOCK_SIZE, "m": 1}

    # the following code runs a crude estimate in parallel
    # useful when using a slow cost_model function
    # such as one testing in practice lattice reduction

    beta_found = multiprocessing.Value('Q', min(MAX_BLOCK_SIZE, n))
    cost_found = multiprocessing.Value('d', infinity)
    m_found = multiprocessing.Value('Q', n) # assuming we only get one RLWE sample

    m_beta_range = list(itertools.product(
        range(n, 0, -1), range(min(MAX_BLOCK_SIZE, n), min_beta-1, -1)))
    with multiprocessing.Pool(NPROC) as pool:
        for rv in pool.imap_unordered(
            test_m_beta,
            [(n, k, y, q, x, kannan_coeff, sigma, float_type, repeats, _m, _beta)
             for (_m, _beta) in m_beta_range]
            # , ceil(list(m_beta_range) / NPROC)
        ):
            _m, _beta, cost = rv
            #  make below atomic
            with beta_found.get_lock():
                with cost_found.get_lock():
                    with m_found.get_lock():
                        if cost < cost_found.value:
                            cost_found.value = cost
                            beta_found.value = _beta
                            m_found.value = _m

    print({"k": k, "cost": cost_found.value, "beta": beta_found.value, "m": m_found.value}, repeats)
    return {"k": k, "cost": cost_found.value, "beta": beta_found.value, "m": m_found.value}


# @memory.cache
def sparse_ternary_secret(n: int, log_q: int, w: int, sigma: float, min_beta: Union[int, bool] = 0, float_type: str = "d") -> Tuple[float, int, int, int, int]:
    beta = {}
    cost = {}
    repeat = {}
    m = {}
    plot_beta = []
    plot_cost = []
    plot_repeat = []
    plot_m = []

    """
    instead of parallel linear search, here we do a binary search for the function "bool(from k to k+10, there were zero changes in sign of the derivative)".
    This function should (check) either be always 0, or be 1 at first and 0 later. once we find the window where it becomes 0,
    we check the previous window (parallel-)linearly for the min, or we return the first entry

    after finding the apparent minimum, we compare it with the first block regardless (since our apparent minimum is just where the function stops being monotonic)
    if the apparent minimum is not a minimum, we go investigate the first block, where the minimum will be located towards the beginning
    """

    binary_search_time = time()
    window = 10
    min_window_idx = 0
    max_window_idx = ceil((n-w)/window)
    mid_window_idx = 0

    def test_monotonic(_min, _max):
        rv = True
        i = _min
        cost_i = sparse_ternary_secret_given_k(
            (n, log_q, w, i, sigma, min_beta, float_type))["cost"]
        for i in range(_min, min(_max-1, _min+4)):
            cost_i_1 = sparse_ternary_secret_given_k(
                (n, log_q, w, i, sigma, min_beta, float_type))["cost"]
            if cost_i > cost_i_1:
                rv = False
                break
            cost_i = cost_i_1
        return rv

    tested_windows = {}
    window_to_search = 0
    while min_window_idx <= max_window_idx:
        mid_window_idx = (min_window_idx + max_window_idx) // 2
        monotonic = test_monotonic(
            mid_window_idx * window, (mid_window_idx + 1) * window)
        tested_windows[mid_window_idx] = monotonic
        if monotonic:
            # monotonic, we are to the right of the minimum
            max_window_idx = mid_window_idx - 1
            if max_window_idx in tested_windows:
                if tested_windows[max_window_idx] == False:  # non monotonic
                    window_to_search = max_window_idx
                    break
        else:
            # zig-zag, we are to the left or on it
            min_window_idx = mid_window_idx + 1
            if min_window_idx in tested_windows:
                if tested_windows[min_window_idx] == True:
                    window_to_search = mid_window_idx
                    break

    bin_k_found = -1
    bin_cost_found = infinity
    bin_data_found = None
    for i in range(window_to_search * window, (window_to_search+1)*window):
        _data_i = sparse_ternary_secret_given_k(
            (n, log_q, w, i, sigma, min_beta, float_type))
        _cost_i = _data_i["cost"]
        if _cost_i < bin_cost_found:
            bin_k_found = i
            bin_cost_found = _cost_i
            bin_data_found = _data_i

    data_0 = sparse_ternary_secret_given_k(
        (n, log_q, w, 0, sigma, min_beta, float_type))
    cost_0 = data_0["cost"]
    if cost_0 < bin_cost_found:
        for i in range(0, min(4, window)):
            if i == 0:
                _data_i, _cost_i = data_0, cost_0
            else:
                _data_i = sparse_ternary_secret_given_k(
                    (n, log_q, w, i, sigma, min_beta, float_type))
                _cost_i = _data_i["cost"]
            if _cost_i < bin_cost_found:
                bin_k_found = i
                bin_cost_found = _cost_i
                bin_data_found = _data_i

    binary_search_time = time() - binary_search_time
    # save(line([(k, cost[k]) for k in sorted(cost.keys())]), "test_cost.png")
    # save(line([(k, cost[k+1] - cost[k]) for k in sorted(cost.keys())[:-1]]), "test_D_cost.png")

    # print(
    #     f"min ({bin_k_found}, {bin_cost_found} sec), time {binary_search_time}")

    if not bin_data_found:
        raise ValueError(
            "No successful attack found, try increasing config.py:MAX_BLOCK_SIZE")

    return bin_data_found["cost"], bin_data_found["beta"], bin_k_found, bin_data_found['repeats'], bin_data_found["m"]


def linear_sparse_ternary_secret(n: int, log_q: int, w: int, sigma: float, min_beta: Union[int, bool] = 0, float_type: str = "d") -> Tuple[float, int, int, int, int]:
    beta = {}
    cost = {}
    repeat = {}
    m = {}
    plot_beta = []
    plot_cost = []
    plot_repeat = []
    plot_m = []

    # # find the optimal number k of secret coordinates to guess as being = 0
    # with multiprocessing.Pool(1) as pool:
    #     for rv in pool.imap_unordered(
    #                 sparse_ternary_secret_given_k,
    #                 list((n, log_q, w, k, sigma, min_beta, float_type)
    #                      for k in range(n-w)),
    #                     #  for k in range(2)),
    #                 1
    #             ):
    #         cost[rv["k"]] = rv["cost"]
    #         beta[rv["k"]] = rv["beta"]
    #         m[rv["k"]] = rv["m"]
    #         repeat[rv["k"]] = log2(
    #             binomial(n, rv["k"]) / binomial(n-w, rv["k"]))
    #     pool.close()
    #     pool.join()

    for k in range(0, n-w, 1):
    # for k in range(680, 701, 2):
        rv = sparse_ternary_secret_given_k((n, log_q, w, k, sigma, min_beta, float_type))
        cost[rv["k"]] = rv["cost"]
        beta[rv["k"]] = rv["beta"]
        m[rv["k"]] = rv["m"]
        repeat[rv["k"]] = log2(binomial(n, rv["k"]) / binomial(n-w, rv["k"]))

    cost_found = infinity
    beta_found = infinity
    k_found = 0
    repeat_found = 1
    m_found = n
    for k in range(0, n-w, 1):
    # for k in range(680, 701, 2):
        if cost[k] < cost_found and repeat[k] < log2(MAX_REPEATS):
            cost_found = cost[k]
            beta_found = beta[k]
            k_found = k
            repeat_found = repeat[k]
            m_found = m[k]
        plot_cost.append((k, cost[k]))
        plot_beta.append((k, beta[k]))
        plot_repeat.append((k, repeat[k]))
        plot_m.append((k, m[k]))

    fn = f"_n_{n}_lq_{log_q}_w_{w}_s_{sigma}"
    save(line(plot_cost),   f"cost_{fn}.png")
    save(line(plot_beta),   f"beta_{fn}.png")
    save(line(plot_repeat), f"repeat_{fn}.png")
    save(line(plot_m),      f"m_{fn}.png")
    return cost_found, beta_found, k_found, repeat_found, m_found


def rounded_sparse_ternary_secret_given_p(n: int, log_q: int, w: int, log_p: int, min_beta: Union[int, bool] = 0, float_type: str = "d", hardness_upper_bound: bool = True) -> Tuple[float, int, int, int, int]:
    # from analysis we see that when rounding down mod p << q, we obtain a new LWE sample mod p with the same secret distribution and
    # an error distribution slightly narrower than CBD(eta=w/4), which has variance w/8.
    # when eta is between 10 and 50, it appears to be lower bound tightly by 2/3 * (w/8).

    # the free variable is log_p. it reduces the cost of LLL, but also lowers the volume of the overall lattice being reduced
    sigma = sqrt((w+1)/12)

    return linear_sparse_ternary_secret(n, log_p, w, sigma, min_beta, float_type)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    parser.add_argument('-logq', type=int, default=30)
    parser.add_argument('-w', type=int)
    parser.add_argument('-min_beta', type=int, default=0)
    parser.add_argument('-sigma', type=float, default=3.2)
    parser.add_argument('-ft', type=str, default="d")
    args = parser.parse_args()

    n = args.n
    log_q = args.logq
    w = args.w
    min_beta = args.min_beta
    sigma = args.sigma
    float_type = args.ft
    cost, beta, k, repeat, m = sparse_ternary_secret(
        n, log_q, w, sigma, min_beta=min_beta, float_type=float_type)
    print(f"log2 cost:", cost)
    print(f"beta:", beta)
    print(f"k:", k)
    print(f"log2 repeat:", repeat)
    print(f"m:", m)
    print()

    hardness_upper_bound = True
    for log_p in [20, 10]:
        print(f"log2 p:", log_p)
        cost, beta, k, repeat, m = rounded_sparse_ternary_secret_given_p(
            n, log_q, w, log_p, min_beta=min_beta, float_type=float_type, hardness_upper_bound=hardness_upper_bound)
        print(f"log2 cost:", cost)
        print(f"beta:", beta)
        print(f"k:", k)
        print(f"log2 repeat:", repeat)
        print(f"m:", m)
