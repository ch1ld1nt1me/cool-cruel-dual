"""
This module exports instances generated for drop/solve attacks into a format that can be used by the Cool and Cruel implementation in https://github.com/facebookresearch/LWE-benchmarking
"""

import json
import os
from copy import deepcopy
import numpy as np
from sage.all import matrix, sqrt
from attack_params import atk_params


def export_to_cnc(params, lwe, lwe_cnc, export_path="./cnc/"):
    # read parameters
    n = params['n']
    log_q = params['log_q']
    w = params['w']
    lwe_sigma = params['lwe_sigma']
    sigma = params['sigma']
    k = params['k']
    m = params['m']
    beta = params['beta']
    single_guess_succ = params['single_guess_succ']
    log_p = params['log_p']
    float_type = params['float_type']
    export_only = False if 'export_only' not in params else params['export_only']

    params_str = "_".join(map(str, [
        "n", n,
        "logq", log_q,
        "w", w,
        "sigma", lwe_sigma,
        "m", m,
        "logp", log_p,
    ]))

    os.makedirs(export_path, exist_ok=True)
    _seed = int.from_bytes(os.urandom(3))

    A, b, __s, __e = lwe
    npA = np.array(A, dtype=np.int64)
    fnA = f"origA_{params_str}_{_seed}_m.npy"
    np.save(f"{export_path}/{fnA}", npA)

    nps = np.array(matrix(len(__s), 1, __s), dtype=np.int64)
    fns = f"origs_{params_str}_{_seed}_m.npy"
    np.save(f"{export_path}/{fns}", nps)

    A_cnc, b_cnc, __s_cnc, __e_cnc = lwe_cnc
    npA_cnc = np.array(A_cnc, dtype=np.int64)
    fnA_cnc = f"origA_{params_str}_{_seed}_4n.npy"
    np.save(f"{export_path}/{fnA_cnc}", npA_cnc)

    nps_cnc = np.array(matrix(len(__s_cnc), 1, __s_cnc), dtype=np.int64)
    fns_cnc = f"origs_{params_str}_{_seed}_4n.npy"
    np.save(f"{export_path}/{fns_cnc}", nps_cnc)

    # return the attack parameter object
    cnc_params = {
        "N": n,
        "Q": 2**log_p,
        "MLWE_k": 0,
        "HW": w,
        "SECR_DISTR": "ternary",
        "SIGMA": sigma, # this is the rounding sigma if it's a rounding instance

        "FLOAT_TYPE": "double" if n == 128 else "dd",
        "LLL_PENALTY": 10,
        "THRESHOLDS": "0.783,0.783001,0.7831",
        "BKZ_BLOCK_SIZE": 30,
        "BKZ_BLOCK_SIZE2": 40,
    }

    """
    # choosing block sizes
    preprocess.py defaults: 30, 40  -> no mention in benchmarking paper, similar to what we use, seems fair comparison
    readme: 32, 38 (dimension 128)
    table 4 C&C paper: 35, 40 (dim 256), 18, 22 (dim 512+)

    # LLL penalty ω
    - C&C paper mentions 4 and 10
    - Bench paper mentions 1, 4 and 10, with 10 for the HE params -> we follow this one

    # thresholds (rho)
    The C&C and LWE-bench papers are contradictory. They suggest rho is a property of the output, but then rquire passing it as an input.
    A variety of values for rho are shown:
    - 0.86 0.84 0.70 for HE instances in Bench (depending on q)
    - 0.769--0.413 for LWE instances in C&C
    - 0.783,0.783001,0.7831 for n = 80 in the code release readme file
    - 0.4,0.41,0.5 as the default value in the code release
    No heuristic is provided for how to chose these, and no pattern is apparent.
    Default values in the code release are "0.4,0.41,0.5", much lower than most other settings, likely requiring more reduction.
    Striking a balance is hard. Being the bench papers around 0.8, the CNC papers for n = 256 0.769, and the readme file 0.783, we take the middle road and 
    use the values in the example code in the README.

    # samples to output via preprocessing
    - Bench paper: unclear
    - C&C paper: Use formula 11, or simply observe it's about 1.1E4 in Table 5
    Can't find a way to get preprocessing.py to terminate when the 1.1E4 samples are obtained.
    Instead, will give it a max runtime equal to 110% that taken by the primal attack or that of the primal attack + 1-std-dev, whichever is higher.
    """

    pr_results_path = "./results"
    data = {}
    for fn in os.listdir(pr_results_path):
        with open(os.path.join(pr_results_path, fn)) as f:
            data[fn] = json.load(f)

    results = list(filter(
        lambda x: \
        x['attack_params']['n'] == params['n'] and \
        x['attack_params']['log_q'] == params['log_q'] and \
        x['attack_params']['log_p'] == params['log_p'] and \
        x['attack_params']['w'] == params['w'] and \
        x['attack_params']['lwe_sigma'] == params['lwe_sigma'] and \
        x['attack_params']['beta'] == params['beta'] and \
        x['attack_params']['float_type'] == params['float_type'] and \
        x['attack_params']['single_guess_succ'] == params['single_guess_succ'] and \
        x['attack_params']['k'] == params['k'] and \
        x['attack_params']['m'] == params['m'] and \
        x['attack_params']['b'] == params['b'],
        data.values()
    ))
    assert(len(results) in [0, 1])
    if len(results) == 0:
        cnc_params["PREPROC_TIMEOUT"] = 3000
    elif len(results) == 1:
        pr_avg_runtime = results[0]["avg runtime"]
        pr_stddev_runtime = sqrt(np.var(results[0]["runtimes"]))
        cnc_params["PREPROC_TIMEOUT"] = round(max(1.1 * pr_avg_runtime, pr_avg_runtime + pr_stddev_runtime))
    cnc_params["CNC_TIMEOUT"] = cnc_params["PREPROC_TIMEOUT"]

    params_pr = deepcopy(cnc_params)
    params_pr['INPUT_A'] = fnA
    params_pr['INPUT_S'] = fns
    params_pr['M'] = m

    params_cmc = deepcopy(cnc_params)
    params_cmc['INPUT_A'] = fnA_cnc
    params_cmc['INPUT_S'] = fns_cnc
    params_cmc['M'] = -1

    return [ params_pr, params_cmc ]
