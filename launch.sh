#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-04:00:00     # DD-HH:MM:SS 
#SBATCH --mail-user=mousjad@gmail.com
#SBATCH --mail-type=ALL

SOURCEDIR=~/projects/def-farbodk/mousjad/Generalized_CNN/


module load python/3.8 cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install numpy --no-index
pip install scipy --no-index
pip install igl --no-deps
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

python analyse_nn.py
