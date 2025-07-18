"""
Usage example:

 sage estimation_snippet.py --n 256 --q 3329 --h 12 --sd 1.22 --m 256
"""

import argparse

def drop_and_solve_cost(n, q, h, sd, m):
   """
   D+S cost estimate for attacking LWE on sparse ternary secrets and discrete Gaussian errors.
   Assumes the [CheNgu12] cost model for the cost of lattice enumeration.
   Dependencies: to run, save this snippet to a python file, then run:
       $ git clone https://bitbucket.org/malb/lwe-estimator lwe_estimator
   You will then be able to import the necessary functions from the LWE-estimator.
   Run this script using `sage`.
   :param n: LWE secret dimension
   :param q: LWE modulus
   :param h: LWE secret Hamming weight
   :param sd: LWE error standard deviation
   :param m: number of LWE samples
   """
   from sage.all import sqrt, RR, load
   load("lwe_estimator/estimator.py")
   alpha = sqrt(2*pi)*sd/RR(q)
   secret_distribution = ((-1, 1), h)
   success_probability = 0.99
   reduction_cost_model =  lambda beta, d, B: BKZ.CheNgu12(beta, d, B=B)
   primald = partial(drop_and_solve, primal_usvp, postprocess=False, decision=False)
   ds_cost = primald(n, alpha, q, secret_distribution=secret_distribution, m=m,  success_probability=success_probability, reduction_cost_model=reduction_cost_model)
   print("d/s cost")
   print(ds_cost)

def get_parser():
    parser = argparse.ArgumentParser(
        description="D+S cost estimate for attacking LWE on sparse ternary secrets and discrete Gaussian errors."
    )

    parser.add_argument(
    "--n", default=256, type=int, help="LWE secret dimension"
    )

    parser.add_argument(
    "--q", default=3329, type=int, help="LWE modulus"
    )

    parser.add_argument(
    "--h", default=12, type=int, help="LWE secret Hamming weight"
    )

    parser.add_argument(
    "--sd", default=1.22, type=float, help="LWE error standard deviation"    )

    parser.add_argument(
    "--m", default=256, type=int, help="number of LWE samples"
    )

    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    drop_and_solve_cost(args.n, args.q, args.h, args.sd, args.m)
