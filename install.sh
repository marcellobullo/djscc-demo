#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="demo"
PYVERSION="3.11"

#echo "Creating conda environment..."
#conda create -y --name "${ENV_NAME}" python="${PYVERSION}"

#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate "${ENV_NAME}"

conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

echo "Installing core dependencies..."
conda install -y gnuradio python="${PYVERSION}"
conda install -y limesuite
conda install -y cmake pkg-config libboost-devel "gnuradio>=3.10.12"
conda install -y wget
conda install -c conda-forge opencv # let conda solver install the right opencv compatible with gnuradio-imposed qt

echo "Installing Python packages..."
python3 -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 compressai pykaira || \
python3 -m pip install torch torchvision torchaudio compressai pykaira

echo "Building gr-deepjscc..."
cd gr-modules/gr-deepjscc
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ..
make -j4
make install
cd ../../..

bash ./download_model_checkpoints.sh