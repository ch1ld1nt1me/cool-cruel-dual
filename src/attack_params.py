export_only = True

atk_params = [
    # paper instances
    #  LWE parameterz
    #    n = secret dimension
    #    w = secret hamming weight (ternary secrets)
    #    2**log_q = LWE modulo
    #    lwe_sigma = LWE noise standard deviation
    #  Attack parameters
    #    k = number of guessed secret coordinates
    #    m = number of LWE samples needed
    #    2**log_p = attack round-down modulo
    #    beta = minimum attack block size for attack estimation
    #    b = used block size
    #    single_guess_succ = estimated success probability of prog-BKZ on a correct drop-and-solve guess
    #    float_type = FPLLL floating point implementation

    # d
    {'n': 128, 'log_q': 10, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 20, 'single_guess_succ': .2, 'float_type': 'd', 'k': 38, 'm': 67, 'b': 20  , 'export_only': export_only},
    {'n': 128, 'log_q': 10, 'log_p': 10, 'w': 16, 'lwe_sigma': 3.2, 'beta': 20, 'single_guess_succ': .2, 'float_type': 'd', 'k': 30, 'm': 77, 'b': 40 , 'export_only': export_only},
    {'n': 128, 'log_q': 30, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .6, 'float_type': 'd', 'k': 0, 'm': 35, 'b': 40  , 'export_only': export_only},
    {'n': 128, 'log_q': 30, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .6, 'float_type': 'd', 'k': 32, 'm': 18, 'b': 40 , 'export_only': export_only},
    {'n': 128, 'log_q': 50, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .6, 'float_type': 'd', 'k': 0, 'm': 35, 'b': 40 , 'export_only': export_only},
    {'n': 128, 'log_q': 50, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .6, 'float_type': 'd', 'k': 32, 'm': 18, 'b': 40 , 'export_only': export_only},
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'd', 'k': 76, 'm': 18, 'b': 40, 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 16, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'd', 'k': 42, 'm': 50, 'b': 40 , 'export_only': export_only },

    # dd
    {'n': 192, 'log_q': 30, 'log_p': 10, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 58, 'm': 104, 'b': 40 , 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 10, 'w': 10, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 63, 'm': 100, 'b': 40 , 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 0, 'm': 92, 'b': 40 , 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 16, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 0, 'm': 124, 'b': 41 , 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 0, 'm': 157, 'b': 41 , 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 20, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 34, 'm': 65, 'b': 40 , 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 30, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 0, 'm': 159, 'b': 48 , 'export_only': export_only },
    {'n': 192, 'log_q': 30, 'log_p': 15, 'w': 30, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 22, 'm': 105, 'b': 40 , 'export_only': export_only },
    {'n': 256, 'log_q': 12, 'log_p': 12, 'w': 12, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 99, 'm': 256, 'b': 61 , 'export_only': export_only },
    {'n': 256, 'log_q': 12, 'log_p': 12, 'w': 12, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 100, 'm': 111, 'b': 60 , 'export_only': export_only },
    {'n': 256, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 51, 'm': 157, 'b': 40 , 'export_only': export_only },
    {'n': 256, 'log_q': 50, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 51, 'm': 157, 'b': 40 , 'export_only': export_only },
    {'n': 256, 'log_q': 30, 'log_p': 15, 'w': 10, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 56, 'm': 151, 'b': 40 , 'export_only': export_only },
    {'n': 256, 'log_q': 50, 'log_p': 15, 'w': 10, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 56, 'm': 151, 'b': 40 , 'export_only': export_only },
    {'n': 384, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 30, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 189, 'm': 150, 'b': 30 , 'export_only': export_only },
    {'n': 384, 'log_q': 30, 'log_p': 15, 'w': 8, 'lwe_sigma': 3.2, 'beta': 40, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 168, 'm': 155, 'b': 45 , 'export_only': export_only },
]
