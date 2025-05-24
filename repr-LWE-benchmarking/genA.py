import numpy as np
import argparse, random
import os

parser = argparse.ArgumentParser(
    description="Generate A matrices."
)

parser.add_argument(
    "--N", default=128, type=int, help="LWE dimension (N)."
)

parser.add_argument(
    "--m", default=128, type=int, help="LWE sample number (m)."
)

parser.add_argument(
    "--Q", default=3329, type=int, help="Modulus A is supposed to be used with."
)

parser.add_argument(
    "--dump_path", default="./", help="Dump A in given folder."
)

params = parser.parse_args()
N, m, Q = params.N, params.m, params.Q

mat_to_save = np.zeros((m,N), dtype=int)
for i in range(m):
    for j in range(N):
        mat_to_save[i,j] = random.randrange(-Q//2+1, Q//2+1)

os.makedirs(params.dump_path, exist_ok=True)
np.save(f"{params.dump_path}/origA_{N}_{m}.npy", mat_to_save)
 

