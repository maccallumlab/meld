#!/bin/bash
#SBATCH --job-name=minimize
#SBATCH --mem-per-cpu=200mb
#SBATCH --time=00:10:00
#SBATCH --output=hen_production.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
pwd; hostname; date



        -i minimize.in \
        -o minim_template.out \
        -p system.top \
        -c system.mdcrd \
        -r eq0_template.rst \
        -x prot.nc

