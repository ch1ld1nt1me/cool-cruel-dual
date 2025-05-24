#!/bin/bash

DIRECTORY=LWE-benchmarking
if [ ! -d "$DIRECTORY" ]; then
    # fetch LWE-benchmarking repo
    git clone https://github.com/facebookresearch/LWE-benchmarking $DIRECTORY
    # (cd $DIRECTORY ; git checkout 02ef1e9af27d7a26cca1bf09aa66ce2bcee00061)
    (cd $DIRECTORY ; git checkout 47e40e7498f66e68a1d4b1b660b900273847a6ef)

    # apply necessary patches
    (cd $DIRECTORY ; git apply ../LWE-benchmarking.patch)
fi

# create necessary conda environment
which conda
CONDA_CHECK=$?
if [ $CONDA_CHECK -ne 0 ]; then
        echo "Need to install Conda to reproduce"
        exit 1
fi

conda env list | grep lattice_env
LATTICE_ENV_CHECK=$?
if [ $LATTICE_ENV_CHECK -ne 0 ]; then
        echo "Need to create lattice_env conda environment as per LWE-benchmarking/README.md"
        exit 1
fi

# make sure flatter is available
# create necessary conda environment
which flatter
FLATTER_CHECK=$?
if [ $FLATTER_CHECK -ne 0 ]; then
        echo "Need to install flatter to reproduce."
        echo "Follow /flatter-cont/README.md instructions to get flatter built and running from within docker container."
        exit 1
fi

cp reproduce.py $DIRECTORY
cp genA.py $DIRECTORY

echo
echo "To reproduce the attack, first run 'conda activate lattice_env'."
echo "Then within $DIRECTORY run 'python3 reproduce.py'"