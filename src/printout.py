import os
import json
from copy import deepcopy
from collections import OrderedDict
import operator
from tabulate import tabulate
from attack_params import atk_params
from math import log


def printout(path="./results"):
    data = {}
    fns = os.listdir(path)
    for fn in fns:
        with open(os.path.join(path, fn)) as f:
            data[fn] = json.load(f)

    table_data = []

    # sort results
    atk_params.sort(key = operator.itemgetter('n', 'log_q', 'w', 'k', 'log_p', 'b'))
    last_n = None

    for params in atk_params:
        try:
            params_copy = deepcopy(params)
            del params_copy['export_only']
            del params_copy['beta']
            results = next(filter(lambda x: params_copy.items() <= x['attack_params'].items(), data.values()))

            if params['float_type'] == 'd' and params['n'] > 128 or params['float_type'] == 'dd' and params['n'] == 128:
                continue

            if last_n == None:
                last_n = params['n']
            if last_n != params['n']:
                last_n = params['n']
                table_data.append({})

            entry = OrderedDict()

            # LWE params
            entry['$n$'] = "$%d$" % params['n']
            entry['$q$'] = "$2^{%d}$" % params['log_q']
            entry['$h$'] = "$%d$" % params['w']
            entry['$\\sigma$'] = "$%.01f$" % params['lwe_sigma']
            entry['$m$'] = "$%d$" % params['m']

            # attack param
            entry['$k$'] = "$%d$" % params['k']
            entry['$p$'] = "$2^{%d}$" % params['log_p']
            # entry['$\\mathbb{E}[\\beta]$'] = "$%d$" % params['beta']
            entry['$\\beta$'] = "$%d$" % params['b']
            if params['beta'] != params['b']:
                entry['$\\beta$'] = "\\textcolor{red}{\\bf %s / %s}" % (params['beta'], params['b'])
            # entry['estimated $p_\\text{red}$'] = params['single_guess_succ'] # check p name
            entry['prec.'] = "\\textsf{%s}" % params['float_type']

            # results
            entry['succ.'] = f"${results['successful']} / {10}$"
            entry['avg time (s)'] = "$%.01f$" % results['avg runtime']
            entry['avg guesses'] = "$%.01f$" % results['avg repeats']

            from estimator import guesses_and_overhead
            single_prob, tail_repeats, reps_overhead = guesses_and_overhead(params['n'], params['w'], params['k'], single_guess_succ=1., target_succ=0.99)
            entry['$\\pg^{-1}$'] = "$%.01f$" % (1/single_prob)

            table_data.append(entry)
        except StopIteration:
            print(f"Missing data: {params}")
            pass

    table = tabulate(table_data, headers = "keys", tablefmt="latex_raw", stralign="center")

    # adjust table styling
    table = table.replace('\\hline', '\\toprule', 1)
    table = table.replace('\\hline', '\\midrule', 1)
    table = table.replace('\\hline', '\\bottomrule', 1)

    # caption
    caption = ""

    # add table environment
    table = "\\begin{table}[t]\n" \
        + "\\caption{%s}\n" % caption \
        + "\\label{tab:primal-results}\n" \
        + "\\centering\n" \
        + "\\setlength{\\tabcolsep}{3pt}" \
        + table + "\n" \
        + "\\end{table}\n"

    print()
    print(table)


def cnc_printout(path="../repr-LWE-benchmarking/results.json"):
    with open(path) as f:
        data = json.load(f)

    table_data = []

    # sort results
    data.sort(key = operator.itemgetter('M', 'N', 'Q', 'HW', 'FLOAT_TYPE'))

    bucketed_data = OrderedDict()

    for entry in data:
        key = (
            entry['N'],
            entry['Q'],
            entry['MLWE_k'],
            entry['HW'],
            entry['M'],
            entry['SECR_DISTR'],
            entry['SIGMA'],
            entry['FLOAT_TYPE'],
            entry['LLL_PENALTY'],
            entry['THRESHOLDS'],
            entry['BKZ_BLOCK_SIZE'],
            entry['BKZ_BLOCK_SIZE2'],
            entry['PREPROC_TIMEOUT'],
            entry['CNC_TIMEOUT']
        )
        if key not in bucketed_data:
            bucketed_data[key] = []
        bucketed_data[key].append(entry)

    last_n = None
    last_q = None

    for key in bucketed_data:
        completed = len(bucketed_data[key])
        successful = 0
        avg_preproc = 0
        avg_distinguish = 0
        avg_cruel_bits = 0
        avg_max_hamming = 0
        avg_max_brute = 0
        for it in range(completed):
            experiment = bucketed_data[key][it]
            result = experiment['results']
            success = result['success']
            if success:
                successful += 1
                avg_preproc += result['preproc_wall_time']
                avg_distinguish += result['cnc_wall_time']
                avg_cruel_bits += result["cruel_bits"]
                avg_max_hamming += result["max_hamming"]
                avg_max_brute += result["max_brute"]

        if successful > 0:
            avg_preproc /= successful
            avg_distinguish /= successful
            avg_cruel_bits /= successful
            avg_max_hamming /= successful
            avg_max_brute /= successful

        ogpars = bucketed_data[key][0]["INPUT_A"].split("_")
        og_log_q = int(ogpars[ogpars.index("logq")+1])
        og_m = int(ogpars[ogpars.index("m")+1])

        if last_n == None:
            last_n = bucketed_data[key][0]["N"]
        if last_n != bucketed_data[key][0]["N"]:
            last_n = bucketed_data[key][0]["N"]
            table_data.append({})
        if last_q == None:
            last_q = og_log_q
        if last_q != og_log_q:
            last_q = og_log_q
            table_data.append({})

        row = OrderedDict()
        # LWE
        row["$n$"] = "$%d$" % bucketed_data[key][0]["N"]
        row["$q$"] = "$2^{%d}$" % og_log_q
        row["$p$"] = "$2^{%d}$" % round(log(bucketed_data[key][0]["Q"], 2))
        row["$h$"] = "$%d$" % bucketed_data[key][0]["HW"]
        row["$\\sigma$"] = "$%.02f$" % bucketed_data[key][0]["SIGMA"]

        # attack
        row["$m$"] = bucketed_data[key][0]["M"]
        row["og $m$"] = "$%d$" % og_m

        prec = bucketed_data[key][0]["FLOAT_TYPE"]
        if prec == "double":
            prec = "d"
        row["prec."] =  "\\textsf{%s}" % prec
        assert bucketed_data[key][0]["PREPROC_TIMEOUT"] == bucketed_data[key][0]["CNC_TIMEOUT"]
        row["timeout"] = bucketed_data[key][0]["PREPROC_TIMEOUT"]

        # results
        row['succ.'] = f"${successful} / {completed}$"
        row['avg prep.'] = "$%.01f$" % avg_preproc
        row['avg dist.'] = "$%.01f$" % avg_distinguish
        row['avg cruel'] = "$%.01f$" % avg_cruel_bits
        row['avg max HW'] = "$%.01f$" % avg_max_hamming
        row['avg max brute'] = "$%.01f$" % avg_max_brute

        table_data.append(row)

    table = tabulate(table_data, headers = "keys", tablefmt="latex_raw", stralign="center")

    # adjust table styling
    table = table.replace('\\hline', '\\toprule', 1)
    table = table.replace('\\hline', '\\midrule', 1)
    table = table.replace('\\hline', '\\bottomrule', 1)

    # caption
    caption = ""

    # add table environment
    table = "\\begin{table}[t]\n" \
        + "\\caption{%s}\n" % caption \
        + "\\label{tab:cnc-results}\n" \
        + "\\centering\n" \
        + "\\setlength{\\tabcolsep}{3pt}" \
        + table + "\n" \
        + "\\end{table}\n"

    print()
    print(table)


if __name__ == "__main__":
    printout()
    cnc_printout()
