# from lwe import genLWEInstance
import json
import copy
import numpy as np # type: ignore #noqa
from settings import NPROC
import parallelism
import fpylll # type: ignore #noqa
import os
import time
from instances import BaiGalCenteredScaled
from instances import RoundedDownLWE
from instances import SpTerLWE as SparseTerSecLWE
import estimator
from utilities import balance
from reduction import reduction
from estimator import guesses_and_overhead
from export import export_to_cnc
from sage.all import next_prime, matrix, vector, ZZ, QQ, zero_matrix, identity_matrix, binomial, sample, seed # type: ignore #noqa
from tqdm import tqdm # type: ignore #noqa
from typing import Tuple, List
from math import sqrt, log, ceil
import itertools
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(module)s:%(lineno)d:%(message)s')
logging.disable()
logger = logging.getLogger(__name__)


# drop and solve

def invalid_drop(__s, n, k):
    """
        constructs a valid drop of zeroes, that results in a winning instance.
        useful during debug
    """
    # print(__s)
    columns_to_drop = []
    for i in range(n):
        si = __s[i]
        if len(columns_to_drop) == k - 1:  # stop one step before making a correct guess
            break
        if si != 0:
            continue
        columns_to_drop.append(i)
    for i in range(n):  # now add one column to the list of drops that however corresponds to a non-zero secret coefficient
        si = __s[i]
        if si == 0:
            continue
        columns_to_drop.append(i)
        break
    columns_to_drop = sorted(columns_to_drop)
    columns_to_keep = []
    for i in range(n):
        if i not in columns_to_drop:
            columns_to_keep.append(i)
    return columns_to_keep


def valid_drop(__s, n, k):
    """
        constructs a valid drop of zeroes, that results in a winning instance.
        useful during debug
    """
    # print(__s)
    columns_to_drop = []
    for i in range(n):
        si = __s[i]
        if len(columns_to_drop) == k:
            break
        if si != 0:
            continue
        columns_to_drop.append(i)
    columns_to_keep = []
    for i in range(n):
        if i not in columns_to_drop:
            columns_to_keep.append(i)
    return columns_to_keep


def drop_and_solve(params):
    """
    Build an embedding with k coordinates dropped and reduce it.
    """
    n = params["n"]
    k = params["k"]
    modulo = params["modulo"]
    w = params["w"]
    eff_sigma = params["eff_sigma"]
    lwe = params["lwe"]
    s = params["__s"]
    m = params["m"]
    beta = params["beta"]
    float_type = params["float_type"]
    verbose = params["verbose"]

    # drop columns
    _seed = int.from_bytes(os.urandom(8))
    logger.info("seed: %d" % _seed)
    with seed(_seed):
        columns_to_keep = sorted(sample(range(n), n - k))

    if verbose:
        print("full secret")
        print(s)
        print("dropped secret")
        print([s[i] for i in columns_to_keep])
        print("cols to keep:", columns_to_keep)
        print()

    # build an embedding
    basis, target = BaiGalCenteredScaled(
        n, modulo, w, eff_sigma, lwe, k, m, columns_to_keep=columns_to_keep)

    # debug: check if the difference in expected iterations come from guessing wrong
    # or from lattice reduction
    # shortest_vector_found = vector(ZZ, [0 for i in range(len(target))])
    # if sum([int(s[i] != 0) for i in columns_to_keep]) == w:
    #     shortest_vector_found = vector(ZZ, list(target)) # copy
    # return {
    #     'sv': shortest_vector_found,
    #     'target': target
    # }

    if verbose:
        # print("pre reduction =======")
        # print(basis)
        # print()
        print("target = y (n-k) * [nu s || e || round(sigma)]")
        print(target)
        print(target.norm().n())
        print()

    # run lattice reduction
    reduced_basis, _ = reduction(basis, beta, "pbkz", float_type=float_type)

    # read off results
    shortest_vector_found = vector(reduced_basis[0])

    if verbose:
        # print("post reduction =======")
        # print(reduced_basis)
        # print()
        print("shortest vector found")
        print(shortest_vector_found)
        print(shortest_vector_found.norm().n())
        print()

    return {
        'sv': shortest_vector_found,
        'target': target
    }


def iterated_lattice_reduction(
        lwe, n, log_q, log_modulo, w, eff_sigma, k, m, beta,
        single_guess_succ=0.99, float_type="d", tail_bound_bar=False, target_succ=.99):
    """
    Call drop_and_solve multiple times, enough to achieve `target_succ` probability
    """
    modulo = 2**log_modulo

    # runtime estimates
    single_prob, tail_repeats, reps_overhead = guesses_and_overhead(n, w, k, single_guess_succ=single_guess_succ, target_succ=target_succ)
    expected_iterations = ceil(1/single_prob)
    tail_bound_iterations = ceil(tail_repeats)

    __s = lwe[2]  # this is a secret! useful for checking the attack works

    # here goes a loop that tries combinations of columns to keep
    if isinstance(tail_bound_bar, bool):
        tail_bound_bar = tqdm(desc="drop and solve")
        local_tail_bound_bar = True
    else:
        local_tail_bound_bar = False
        tail_bound_bar.set_description(f"d/s β={beta}: n={n}, ㏒p={log_modulo}, w={w}, k={k}, m={m}, eff σ={eff_sigma}, E[it] = {expected_iterations}, it ≤ {tail_bound_iterations}")
    tail_bound_bar.total = tail_bound_iterations
    tail_bound_bar.refresh()
    # tail_bound_bar = False
    verbose = False

    completed = 0
    won = False
    target_found = False
    solution_found = False

    attack_instances = [
        {
            "n": n, "k": k, "modulo": modulo, "w": w, "eff_sigma": eff_sigma, "lwe": lwe,
            "__s": __s, "m": m, "beta": beta, "float_type": float_type, "verbose": verbose,
        }
        for _ in range(tail_bound_iterations)
    ]

    lock = parallelism.Lock()
    for rv in parallelism.eval(
        drop_and_solve,
        attack_instances,
        len(attack_instances),
        use_tqdm=False,
        parallel=True,
        cores=NPROC,
        batch_size=1
    ):
        shortest_vector_found = rv['sv']
        target = rv['target']

        # atomically increase the progress bar
        acquired = lock.acquire(timeout=60.)
        if not acquired:
            # hack: give up this iteration if lock not acquirable
            continue
        try:
            completed += 1
            if completed <= tail_bound_iterations and tail_bound_bar:
                tail_bound_bar.update()
                tail_bound_bar.refresh()

            # more than one process may find the solution at once
            # so check for solution within lock
            if shortest_vector_found in [target, -target]:
                solution_found = shortest_vector_found
                target_found = target

                if tail_bound_bar and local_tail_bound_bar:
                    tail_bound_bar.close()
                return solution_found, target_found, completed
        finally:
            lock.release() # this is also run after return inside try: ...
    return solution_found, target_found, completed


def primal_attack(params, tail_bound_bar=False):
    """
        Given LWE parameters, build a (possibly rounded down) LWE instance and solve via iterated drop_and_solve.
    """
    # read parameters
    n = params['n']
    log_q = params['log_q']
    w = params['w']
    lwe_sigma = params['lwe_sigma']
    k = params['k']
    m = params['m']
    beta = params['beta']
    single_guess_succ = params['single_guess_succ']
    log_p = params['log_p']
    float_type = params['float_type']
    export_only = False if 'export_only' not in params else params['export_only']
    if export_only:
        m = 2*n # realistic number of samples applications provide you

    rounding_down = bool(log_p < log_q)

    q = 2**log_q
    p = 2**log_p

    if not single_guess_succ:
        single_guess_succ = 0.99

    if rounding_down:
        # effective sigma
        sigma = sqrt((w+1)/12)
    else:
        sigma = lwe_sigma

    # generate LWE instance
    if rounding_down:
        og_lwe = SparseTerSecLWE(n, m, q, w, lwe_sigma)
        lwe = RoundedDownLWE(n, m, q, p, og_lwe)
        # __s = lwe[2] # this is a secret! useful for checking the attack works
        if export_only:
            # generate also an instance with the number of samples suggested by cnc authors
            og_lwe_cnc = SparseTerSecLWE(n, 4*n, q, w, lwe_sigma)
            lwe_cnc = RoundedDownLWE(n, 4*n, q, p, og_lwe_cnc)
    else:
        lwe = SparseTerSecLWE(n, m, q, w, sigma)
        if export_only:
            # generate also an instance with the number of samples suggested by cnc authors
            lwe_cnc = SparseTerSecLWE(n, 4*n, q, w, sigma)

    if export_only:
        params['sigma'] = sigma
        atk_params = export_to_cnc(params, lwe, lwe_cnc)
        return False, 0, atk_params

    solution_found, target_found, completed = iterated_lattice_reduction(
        lwe, n, log_q, log_p, w, sigma, k, m, beta, single_guess_succ=single_guess_succ, float_type=float_type, tail_bound_bar=tail_bound_bar)

    logger.info("completed:", completed)
    success = not (False in [solution_found, target_found])
    if success:
        logger.info("congratulations")
        logger.info("found:", solution_found)
        logger.info("target", target_found)
        logger.info("secret", lwe[2])
    else:
        logger.info("really bad luck, <= 1%% failure rate")
    return success, completed, None


def params_to_filename(params: dict):
    return "_".join(map(str, params.values()))


def verifying_attack_cost(attack, attack_params_set, tries, fn_prefix="", all_export_atk_params_path="./cnc/atk_params.json"):
    """
        Read various attack parameters, conditionally estimate the optimal drop_and_solve strategy, and then
        perform the attack.
    """
    for params in attack_params_set:
        # prepare to collect statistics
        repeats_distr = {}
        runtimes = []
        successful = 0
        avg_runtime = 0
        avg_repeats = 0
        print(params)

        n = params['n']
        log_q = params['log_q']
        w = params['w']
        lwe_sigma = params['lwe_sigma']
        k = params['k']
        m = params['m']
        beta = params['b']
        min_beta = params['beta']
        single_guess_succ = params['single_guess_succ']
        log_p = params['log_p']
        float_type = params['float_type']
        export_only = False if 'export_only' not in params else params['export_only']

        rouning_down = bool(log_p < log_q)
        if rouning_down:
            if None in [k, m, beta]:
                _cost, _beta, _k, _repeat, _m = estimator.rounded_sparse_ternary_secret_given_p(
                    n, log_q, w, log_p, min_beta=min_beta, float_type=float_type)
            else:
                _k, _m, _beta = k, m, beta
            attack_params = copy.deepcopy(params)
            attack_params['k'] = _k
            attack_params['m'] = _m
            attack_params['beta'] = _beta
        else:
            if None in [k, m, beta]:
                _cost, _beta, _k, _repeat, _m = estimator.linear_sparse_ternary_secret(
                        n, log_q, w, lwe_sigma, min_beta=min_beta, float_type=float_type)
            else:
                _k, _m, _beta = k, m, beta
            attack_params = copy.deepcopy(params)
            attack_params['k'] = _k
            attack_params['m'] = _m
            attack_params['beta'] = _beta


        # runtime estimates
        single_prob, tail_repeats, reps_overhead = guesses_and_overhead(n, w, _k, single_guess_succ=single_guess_succ, target_succ=0.99)
        expected_iterations = ceil(1/single_prob)
        tail_bound_iterations = ceil(tail_repeats)

        print(f"Attack parameters: β = {_beta}, k = {_k}, m = {_m}")
        print(f"E[iters to win] = {expected_iterations}, Iters for 99% win = {tail_bound_iterations}")

        # prepare progress bars
        tries_bar = tqdm(range(tries), desc=f"{n}, {log_q}, {w}, {lwe_sigma}", leave=True)
        tail_bound_bar = tqdm(desc="drop and solve", leave=True)
        msg_bar = tqdm([], postfix="waiting stats")

        for it in tries_bar:
            tail_bound_bar.reset()
            dt = time.time()
            try:
                success, repeats, export_atk_params = attack(attack_params, tail_bound_bar=tail_bound_bar)

                # if export-mode, save the attack parameters in cnc format
                if export_only:
                    try:
                        with open(all_export_atk_params_path, "r") as f:
                            all_export_atk_params = json.load(f)
                    except FileNotFoundError:
                        all_export_atk_params = []
                    all_export_atk_params += export_atk_params
                    with open(all_export_atk_params_path, "w") as f:
                        json.dump(all_export_atk_params, f, indent=2)

            except fpylll.util.ReductionError:
                success = 0
                repeats = 0
            dt = time.time() - dt
            if success:
                successful += 1
                runtimes.append(dt)
                if repeats not in repeats_distr:
                    repeats_distr[repeats] = 0
                repeats_distr[repeats] += 1
            avg_runtime = float(np.mean(runtimes)) if successful > 0 else 0.
            avg_repeats = sum(k * repeats_distr[k] for k in repeats_distr) / successful if successful > 0 else 0
            stats = f"avg rep | succ = {avg_repeats}, avg time | succ = {avg_runtime}, success prob = {successful} / {it+1} = {100 * successful/(it+1)} %"
            msg_bar.set_postfix_str(stats)

        # save results
        if not export_only:
            filename = f"{fn_prefix}{params_to_filename(params)}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump({
                    "attack_params": attack_params,
                    "repeats_distr": repeats_distr,
                    "runtimes": runtimes,
                    "avg runtime": avg_runtime,
                    "avg repeats": avg_repeats,
                    "successful": successful,
                    "success_prob": successful/tries
                }, f, indent=2)
        tries_bar.close()
        tail_bound_bar.close()
        msg_bar.close()
        print()


if __name__ == "__main__":

    # atk_params = [
    #     {'n': 32, 'log_q': 10, 'w': 8, 'lwe_sigma': 3.2, 'k': 2, 'm': 32, 'beta': 20, 'single_guess_succ': .7, 'log_p': 10, 'float_type': 'd'},
    #     {'n': 32, 'log_q': 30, 'w': 8, 'lwe_sigma': 3.2, 'k': 2, 'm': 32, 'beta': 20, 'single_guess_succ': .7, 'log_p': 10, 'float_type': 'd'},
    # ]

    from attack_params import atk_params

    verifying_attack_cost(primal_attack, atk_params, 10, "results/")
