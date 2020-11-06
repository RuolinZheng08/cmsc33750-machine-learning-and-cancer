#!/bin/bash
#
#SBATCH --mail-user=ruolinzheng@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/ruolinzheng/slurm/slurm_out/%j.%N.stdout
#SBATCH --error=/home/ruolinzheng/slurm/slurm_out/%j.%N.stderr
#SBATCH --workdir=/scratch/ruolinzheng/cmsc33750/hw4
#SBATCH --partition=quadro
#SBATCH --job-name=cmsc33750-${foo}
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=24000

pwd; hostname; date

source ~/.bashrc

conda -V

source activate py38

echo $CONDA_DEFAULT_ENV

python --version

python train.py ${foo}
