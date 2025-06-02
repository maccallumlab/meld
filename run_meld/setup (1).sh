#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=setup.out
#SBATCH --error=setup.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=imeshr150@gmail.com
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=2000mb
#SBATCH --qos=alberto.perezant
#SBATCH --partition=gpu
#SBATCH --constraint=a100:1
#SBATCH --constraint=el8

##############################################

module purge

ml ufrc cuda/12.4.1 gcc/12.2.0 openmpi/4.1.6 mkl/2023.2.0 conda/24.7.1

conda activate /orange/alberto.perezant/imesh.ranaweera/m_g/env/openmm_meld_env
source /orange/alberto.perezant/imesh.ranaweera/m_g/Amber/amber24/amber.sh

python /orange/alberto.perezant/imesh.ranaweera/m_g/test2/setup.py

