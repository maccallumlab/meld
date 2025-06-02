#!/bin/bash
#SBATCH --job-name=meld_run
#SBATCH --output=run.out
#SBATCH --error=run.err
##SBATCH --mail-type=ALL
##SBATCH --mail-user=blkreationz27@gmail.com
#SBATCH --time=10:00:00
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
##SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=2000mb
#SBATCH --qos=alberto.perezant
#SBATCH --partition=gpu
##SBATCH --constraint=a100
#SBATCH --constraint=2080ti
#SBATCH --constraint=el8
##############################################

module purge


ml ufrc cuda/12.4.1 gcc/12.2.0 openmpi/4.1.6 mkl/2023.2.0 conda/24.7.1
conda activate /orange/alberto.perezant/imesh.ranaweera/m_g/env/openmm_meld_env
source /orange/alberto.perezant/imesh.ranaweera/m_g/Amber/amber24/amber.sh


export LD_LIBRARY_PATH=/orange/alberto.perezant/imesh.ranaweera/m_g/meld-grappa/plugin/install/lib:$LD_LIBRARY_PATH
export OPENMM_DEFAULT_PLATFORM=CUDA
export OPENMM_CUDA_COMPILER=/apps/compilers/cuda/12.4.1/bin/nvcc
nvidia-smi

srun --mpi=pmix_v3 --cpu-bind=none launch_remd --debug

