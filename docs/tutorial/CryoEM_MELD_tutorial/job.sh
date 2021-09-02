#!/bin/bash
#SBATCH --job-name=one
#SBATCH --partition=gpu
#SBATCH --time=12:30:00
#SBATCH --gpus=16
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus-per-task=1
#SBATCH --export=none
#SBATCH --nodes=4
#SBATCH --output=a_4.log
#SBATCH --ntasks-per-node=4
#SBATCH --account=accountname

export PYTHONPATH=/home/arup/miniconda3/envs/meld_conda/lib/python3.9/site-packages/:$PYTHONPATH

if [ -e remd.log ]; then                 #First check if there is a remd.log file, we are continuing a killed simulation
    /home/arup/miniconda3/envs/meld_conda/bin/prepare_restart --prepare-run        #so we need to prepare_restart.
fi
srun --mpi=pmix_v3 /home/arup/miniconda3/envs/meld_conda/bin/launch_remd --debug
