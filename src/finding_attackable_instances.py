from math import sqrt
import pprint
from estimator import sparse_ternary_secret_given_k, guesses_and_overhead


def attack_options(
        params,
        max_repeats,
        k0=0
    ):
    n = params['n']
    log_q = params['log_q']
    w = params['w']
    lwe_sigma = params['lwe_sigma']
    min_beta = params['beta']
    single_guess_succ = params['single_guess_succ']
    log_p = params['log_p']
    float_type = params['float_type']

    if log_p < log_q:
        sigma = sqrt((w+1)/12)
    else:
        sigma = lwe_sigma

    cost = {}
    for k in range(k0, n-w+1):
        single_prob, tail_repeats, reps_overhead = guesses_and_overhead(n, w, k, single_guess_succ=single_guess_succ, target_succ=0.99)
        if 1/single_prob > max_repeats:
            continue
        cost[k] = sparse_ternary_secret_given_k((n, log_p, w, k, sigma, min_beta, float_type), parallel=True)
        if cost[k]['beta'] <= min_beta:
            break
    return cost


if __name__ == "__main__":

    attack_params = [

        # paper experiments
        # {'n': 128, 'log_q': 10, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 20, 'single_guess_succ': .2, 'float_type': 'd'}, # k 38 m 67 b 20 
        # {'n': 128, 'log_q': 10, 'log_p': 10, 'w': 16, 'lwe_sigma': 3.2, 'beta': 20, 'single_guess_succ': .2, 'float_type': 'd'}, # k 30 m 77 b 40
        # {'n': 128, 'log_q': 30, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .6, 'float_type': 'd'}, # k 0 m 35 b 40 or k 32 m 18 b 40
        # {'n': 128, 'log_q': 50, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .6, 'float_type': 'd'}, # k 0 m 35 b 40 or k 32 m 18 b 40
        # {'n': 192, 'log_q': 30, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 58 m 104 b 40
        # {'n': 192, 'log_q': 30, 'log_p': 10, 'w': 10, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 63 m 100 b 40
        # {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 0 m 92 b 40 or k 76 m 18 b 40
        # {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 16, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 0 m 124 b 41 or k 42 m 50 b 40 
        # {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 0 m 157 b 41 or k 34 m 65 b 40
        # {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 30, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 0 m 159 b 48 or k 22 m 105 b 40
        # {'n': 256, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 51 m 157 b 40
        # {'n': 256, 'log_q': 12, 'log_p': 12, 'w': 12, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 100 m 111 b 60 or k 99 m 256 b 61 d 414 reps 1
        # {'n': 256, 'log_q': 50, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 51 m 157 b 40
        # {'n': 256, 'log_q': 30, 'log_p': 15, 'w': 10, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 56 m 151 b 40
        # {'n': 256, 'log_q': 50, 'log_p': 15, 'w': 10, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 56 m 151 b 40
        # {'n': 384, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 168 m 155 b 45
        # {'n': 384, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 189 m 150 b 30
        # {'n': 512, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'}, # k 272 m 169 b 60 or k 284 m 161 b 53

        # used to find parameters being attacked
        # {'n': 256, 'log_q': 30, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'},

        # currently estimating
        # {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'dd'},
        # 0 192 40 385 3.824978578786397
        # 0 125 30 318 3.824978578786397

        # {'n': 256, 'log_q': 30, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'},
        # 114 106 45 249 801.1714193736062
        # 115 110 44 252 849.1402729634635
        # 116 109 43 250 900.3551245458243
        # 117 107 42 247 955.0618838017575
        # 118 104 41 243 1013.5271092838827
        # 119 92 41 230 1076.03991446752
        # 120 85 41 222 1142.9140658841268
        # 121 79 41 215 1214.4902943465652
        # 122 104 40 239 1291.1388427563736

        # {'n': 256, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'},
        # 40 155 45 372 23.770371072245837
        # 41 159 44 375 24.77905594746263
        # 42 163 43 378 25.831344889628735
        # 43 168 42 382 26.92936563983936
        # 44 174 41 387 28.075359556835206
        # 45 153 41 365 29.271688335537053
        # 46 143 41 354 30.520841152996773
        # 47 136 41 346 31.82544227166368
        # 48 130 41 339 33.188259132064275
        # 49 125 41 333 34.61221096937526
        # 50 120 41 327 36.10037799094269
        # 51 157 40 363 37.656011154587716

        # {'n': 512, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'},
        # 276 221 58 458 3442.9815075112256
        # 277 209 57 445 3563.86870407735
        # 278 224 57 459 3689.548783879191
        # 279 214 56 448 3820.233821432357
        # 280 199 55 432 3956.1462593391293
        # 281 217 55 449 4097.519463722013
        # 282 206 54 437 4244.598312034654
        # 283 220 54 450 4397.639815299347
        # 284 211 53 440 4556.91377693899 {'n': 512, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'k': 284, 'm': 211, 'single_guess_succ': .7, 'float_type': 'dd'},

        # {'n': 512, 'log_q': 30, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd'},
        # no doable params found


        # legacy attempts
        # {'n': 128, 'log_q': 29, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'd'},
        # {'n': 128, 'log_q': 50, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'd'},
        # {'n': 196, 'log_q': 29, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'd'},
        # {'n': 196, 'log_q': 30, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'dd'},
        # {'n': 196, 'log_q': 50, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 20, 'single_guess_succ': .7, 'float_type': 'd'},
        # {'n': 256, 'log_q': 29, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'd'},
        # {'n': 256, 'log_q': 50, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 20, 'single_guess_succ': .7, 'float_type': 'd'},
        # {'n': 512, 'log_q': 29, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'd'},
        # {'n': 512, 'log_q': 50, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'd'},
    ]

    for params in attack_params:
        costs = attack_options(params, 1000)
        print(params)
        for k in costs:
            d = params['n'] - k + costs[k]['m'] + 1
            print("k", k, "m", costs[k]['m'], "b", costs[k]['beta'], "d", d, "reps", costs[k]['repeats'][1])
        print()
