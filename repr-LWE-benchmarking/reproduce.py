"""
conda activate lattice_env
need flatter in PATH
"""

import multiprocessing
import subprocess
from subprocess import PIPE
from io import StringIO
from time import time, sleep
import os
import json
import signal
import sys
import shutil


NPROC = int(multiprocessing.cpu_count()) - 2
LOGFILE = "./log.txt"


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    p = subprocess.Popen("killall python3", shell=True, start_new_session=True)
    p.wait()
    sys.exit(0)


def cnc_attack(
        INPUT_A: str,
        INPUT_S:str,
        N: int,
        Q: int,
        M: int,
        SIGMA: float,
        MLWE_k: int,
        HW: int,
        SECR_DISTR: str,
        LLL_PENALTY: int,
        THRESHOLDS: str,
        FLOAT_TYPE: str,
        WORKERS: int,
        STEP_BY_STEP: bool,
        PREPROC_TIMEOUT: int,
        CNC_TIMEOUT: int,
        BKZ_BLOCK_SIZE: int=32,
        BKZ_BLOCK_SIZE2: int=38,
        NUM_SECRET_SEEDS: int=1,
        ):

    signal.signal(signal.SIGINT, signal_handler)
    logfile = open(LOGFILE, "a")

    from time import gmtime, strftime
    DATETIME = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    MIN_HW=HW
    MAX_HW=HW
    EXP_NAME=f"n{N}q{Q}"
    EXP_OUTPUT_DIR=f"./store/{DATETIME}/{EXP_NAME}"

    print(f"About to preprocess A = {INPUT_A} using {WORKERS} workers.")
    if STEP_BY_STEP:
        input("Press enter to continue")
        print("Preprocessing...")

    os.makedirs(EXP_OUTPUT_DIR, exist_ok=True)
    preproc_cmd = f"/usr/bin/time timeout {PREPROC_TIMEOUT} python3 src/generate/preprocess.py " \
        + f" --N {N} " \
        + f" --Q {Q} " \
        + f" --m {M} " \
        + f" --exp_name {EXP_NAME} " \
        + f" --num_workers {WORKERS} " \
        + f" --thresholds \"{THRESHOLDS}\" " \
        + f" --lll_penalty {LLL_PENALTY} " \
        + f" --reload_data {INPUT_A} " \
        + f" --float_type {FLOAT_TYPE} " \
        + f" --bkz_block_size {BKZ_BLOCK_SIZE} " \
        + f" --bkz_block_size2 {BKZ_BLOCK_SIZE2} " \
        + f" --dump_path {EXP_OUTPUT_DIR}/preproc/"

    print(f"-> {preproc_cmd}")
    logfile.writelines([f"-> {preproc_cmd}\n"])

    T0 = time()
    with subprocess.Popen(preproc_cmd, stdout=PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, shell=True) as p, StringIO() as outbuf:
        for line in p.stdout:
            print(line, end='')
            outbuf.write(line)
        output = outbuf.getvalue()
    rc = p.returncode
    preproc_wall_time = time() - T0

    print("preprocessing timings")
    print(f"wall: {preproc_wall_time}")
    print()

    logfile.write(output)

    logfile.writelines([
        "preprocessing timings\n",
        f"wall: {preproc_wall_time}\n"
    ])

    sleep(2.)

    print(f"About to generate LWE samples using a secret s = {INPUT_S}.")
    if STEP_BY_STEP:
        input("Press enter to continue")
        print("Generating samples...")

    gen_b_cmd = f"python3 src/generate/generate_A_b.py " \
        + f" --actions secrets " \
        + f" --N {N} " \
        + f" --rlwe {MLWE_k} " \
        + f" --secret_path {INPUT_S} " \
        + f" --min_hamming {MIN_HW} --max_hamming {MAX_HW} --secret_type {SECR_DISTR} --num_secret_seeds {NUM_SECRET_SEEDS} " \
        + f" --sigma {SIGMA} " \
        + f" --processed_dump_path {EXP_OUTPUT_DIR}/preproc/{EXP_NAME} " \
        + f" --dump_path {EXP_OUTPUT_DIR}/lwe_samples/"

    print(f"-> {gen_b_cmd}")
    logfile.writelines([f"-> {gen_b_cmd}\n"])

    with subprocess.Popen(gen_b_cmd, stdout=PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, shell=True) as p, \
        StringIO() as outbuf:
        for line in p.stdout:
            print(line, end='')
            outbuf.write(line)
        output = outbuf.getvalue()
    rc = p.returncode

    logfile.writelines(output)

    print("About to display statistics on LWE samples generated.")
    if STEP_BY_STEP:
        input("Press enter to continue")
        print("Measuring statistics...")

    descr_cmd = f"python3 src/generate/generate_A_b.py " \
        + f" --actions describe " \
        + f" --N {N} " \
        + f" --rlwe {MLWE_k} " \
        + f" --secret_path {INPUT_S} " \
        + f" --min_hamming {MIN_HW} --max_hamming {MAX_HW} --secret_type {SECR_DISTR} --num_secret_seeds {NUM_SECRET_SEEDS} " \
        + f" --processed_dump_path {EXP_OUTPUT_DIR}/preproc/{EXP_NAME} " \
        + f" --dump_path {EXP_OUTPUT_DIR}/lwe_samples/"

    print(f"-> {descr_cmd}")
    logfile.writelines([f"-> {descr_cmd}\n"])

    with subprocess.Popen(descr_cmd, stdout=PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, shell=True) as p, \
        StringIO() as outbuf:
        for line in p.stdout:
            print(line, end='')
            outbuf.write(line)
        output = outbuf.getvalue()
    rc = p.returncode

    lines = output.split("\n")

    cruel_bits = None
    max_hamming, max_brute = None, None
    for line in lines:
        if "Cruel bits:" in line:
            cruel_bits = int(line.split("Cruel bits:")[1].split(',')[0].strip())
        if 'max brute force bits:' in line:
            max_brute = int(line.split("max brute force bits:")[1].split(',')[0].strip())
        if 'Recommended max hamming:' in line:
            max_hamming = int(line.split("Recommended max hamming:")[1].split(',')[0].strip())
        logfile.writelines([line + "\n"])

    print("cruel_bits:", cruel_bits)
    print("max_hamming:", max_hamming)
    print("max_brute:", max_brute)

    print("About to run Cool and Cruel on LWE samples generated.")
    if STEP_BY_STEP:
        input("Press enter to continue")
        print("Running the attack...")

    cnc_cmd=f"/usr/bin/time timeout {CNC_TIMEOUT} python3 src/cruel_cool/main.py" \
        + f" --mlwe_k {MLWE_k}" \
        + f" --exp_name {EXP_NAME}" \
        + f" --seed 0" \
        + f" --batch_size 10000" \
        + f" --greedy_max_data 100000" \
        + f" --secret_type {SECR_DISTR}" \
        + f" --full_hw {MIN_HW}" \
        + f" --keep_n_tops 1" \
        + f" --compile_bf 0" \
        + f" --secret_window -1" \
        + f" --bf_dim {cruel_bits}" \
        + f" --min_bf_hw 1" \
        + f" --max_bf_hw {max_brute}" \
        + f" --path {EXP_OUTPUT_DIR}/lwe_samples/{SECR_DISTR}_secrets_h{MIN_HW}_{MAX_HW}/" \
        + f" --dump_path {EXP_OUTPUT_DIR}/cc_attack_logs"

    print(f"-> {cnc_cmd}")
    logfile.writelines([f"-> {cnc_cmd}\n"])

    print("Starting an actual attack:")
    sleep(0.8)
    
    with subprocess.Popen(cnc_cmd, stdout=PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, shell=True) as p, \
        StringIO() as outbuf:
        for line in p.stdout:
            print(line, end='')
            outbuf.write(line)
        output = outbuf.getvalue()
    rc = p.returncode

    succs = False
    times = -1.0

    lines = output.split("\n")
    for line in lines:
        if "SUCCESS!" in line:
            succs = True
        elif "Attack took" in line:
            times = float(line.split("Attack took")[1].strip().split(" ")[0])
        logfile.writelines([line + "\n"])

    print(f"succs: {succs}")
    print(f"times: {times}")

    # delete unnecessary data
    shutil.rmtree(f"./store/{DATETIME}")

    return {
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "preproc_wall_time": preproc_wall_time,
        "NPROC": NPROC,
        "cruel_bits": cruel_bits,
        "max_hamming": max_hamming,
        "max_brute": max_brute,
        "success": succs,
        "cnc_wall_time": times,
    }


def main():

    rel_path_to_inputs="../../src/cnc/"
    results_path="../results.json"
    all_results = []

    with open(os.path.join(rel_path_to_inputs, "atk_params.json"), "r") as f:
        atk_params = json.load(f)

    for param in atk_params:
        results = cnc_attack(
            INPUT_A=os.path.join(rel_path_to_inputs, param['INPUT_A']),
            INPUT_S=os.path.join(rel_path_to_inputs, param['INPUT_S']),
            N=param['N'],
            Q=param['Q'],
            M=param['M'],
            SIGMA=param['SIGMA'],
            MLWE_k=param['MLWE_k'],
            HW=param['HW'],
            SECR_DISTR=param['SECR_DISTR'],
            LLL_PENALTY=param['LLL_PENALTY'],
            THRESHOLDS=param['THRESHOLDS'],
            FLOAT_TYPE=param['FLOAT_TYPE'],
            PREPROC_TIMEOUT=param['PREPROC_TIMEOUT'],
            CNC_TIMEOUT=param['CNC_TIMEOUT'],
            WORKERS=NPROC,
            STEP_BY_STEP=False,
            BKZ_BLOCK_SIZE=param['BKZ_BLOCK_SIZE'],
            BKZ_BLOCK_SIZE2=param['BKZ_BLOCK_SIZE2'],
        )
        param['results'] = results
        all_results.append(param)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
