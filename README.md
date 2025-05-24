# Cruel + Cool = Dual

Code release for "Cruel + Cool = Dual" paper.
In this repository we include:
- code to run "[Drop+Solve](#running-dropsolve-experiments)" experiments and resulting data,
- code to run "[Cruel+Cool](#running-cruelcool-experiments)" experiments and resulting data,
- code to generate the [table](#generating-the-results-table) with our results,
- code to generate the rounding [error plots](#generating-rounding-error-plots) in the appendices,
- code to to generate the [cruel bits plots](#generating-cruel-bits-and-z-shape-prediction-plots) in the appendices.

## Dependencies

To run the code in this repository you should obtain a copy of Sagemath and a copy of Flatter.
We recommend using Conda to install both, and provide instructions here.
We also use a couple of Python modules within sage. Find out which in the [Sagemath](#sagemath) section.

### Conda

To install Conda, use the following commands in a Linux terminal.
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### Sagemath

To install Sage use the following command
```bash
conda create -n sage sage python=3.11
```
Note this could take a while.

Once this is done, enable the Sage environment by running
```bash
conda activate sage
```
Then install the following modules
```bash
sage -pip install joblib tqdm matplot2tikz
```

### Flatter

Flatter needs to be built on your system and available in the user's PATH.
To install the required dependencies to build you can re-use the cond`a environment for Sage.
The following instructions work on Linux on amd64/x86_64 architectures.

```bash
conda activate sage
conda install openblas
git clone https://github.com/keeganryan/flatter.git
cd flatter
mkdir build && cd ./build
cmake ..
make
make install/local
export PATH="$PATH:`pwd`/bin"
```

Note that the last command (`export`) has to be run every time one of the attacks needs to be run.

## Running Drop+Solve experiments

The LWE parameters used to generate the attacked instances can be found in `attack_params.py`.

The Drop+Solve experiments can be run using the following commands
```bash
conda activate sage
cd src
sage attack.py
```

The expeirments run using all-but-2 cores on the host machine.
This can be changed by amending `settings.py`.

The results will be written to the `src/results` directory.

## Running Cruel+Cool experiments

To run Cruel+Cool experiments, we generate instances using the code base for Drop+Solve,
then export them into a format that can be read by the `LWE-benchmarking` [code release](https://github.com/facebookresearch/LWE-benchmarking) from the "Benchmarking LWE" [paper](https://eprint.iacr.org/2024/1229).
We then clone the code release, make minor patches (documented in a patch file), and run the experiments.

### Obtaining and preparing LWE-benchmarking code release

To fetch the remote code and apply the changes, use the following command.
```bash
cd repr-LWE-benchmarking
bash fetch.sh
```
Note that this command may fail due to various missing components.
The command's output will instruct you how to fix this.
Particularly, the command checks that conda is available, and that the `lattice_env` conda environment used by the `LWE-benchmarking` release. Instructions for setting this environment up can be found in the `LWE-benchmarking` README file.

### Generating compatible instances to attack

To generate compatible LWE instances from the same distribution of those used for the Drop+Solve attacks, follow these steps:
1. Set `export_only = True` in `attack_params.py`.
2. Run the following commands
```bash
conda activate sage
sage attack.py
```

The resulting instances will be exported to `src/cnc`. They will be automatically read from here by the Cruel+Cool attack wrapper we use in the next part.

### Running the attack

Once `fetch.sh` successfully completed and compatible LWE instances were generated, run the following commands.
```bash
conda activate lattice_env
cd repr-LWE-benchmarking/LWE-benchmarking
python3 reproduce.py
```

The expeirments run using all-but-2 cores on the host machine.
This can be changed by amending the value of `NPROC` in `repr-LWE-benchmarking/LWE-benchmarking/reproduce.py`.

The experiment results will be written to `repr-LWE-benchmarking/results.json`.

## Generating the Results Table

The table in the paper is a condensed version of two separate tables generated using our code.
To generate these  two tables, run the following commands.

```bash
conda activate sage
cd src
sage printout.py
```

The resulting tables will be printed to the screen.

## Generating Rounding Error Plots

To generate these plots, run the following commands.

```bash
conda activate sage
cd src
sage rounding_error.py
```

The resulting plots will be saved in `src/round_down_error_plots`.

## Generating Cruel Bits and Z-shape prediction plots

To generate these plots, run the following commands.

```bash
conda activate sage
cd src
sage cruel_bits.py
```

The resulting plot data will be saved as CSV files in `src/cruel_bits`.
To generate the plots, compile the LaTeX file `src/cruel_bits/figure.tex` that is also automatically generated.
