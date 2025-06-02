#! /bin/bash 
#SBATCH --job-name=traj
#SBATCH --mem-per-cpu=2000mb
#SBATCH --time=20:00:00
#SBATCH --output=prod_%A-%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # Number of cores per MPI rank 
#SBATCH --ntasks-per-node=1         # How many tasks on each node
#SBATCH --ntasks-per-socket=1        # How many tasks on each CPU or socket
#SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically on nodes and sockets
#SBATCH --array=0-29
#SBATCH --qos=alberto.perezant-b

module purge

source ~/.load_OpenMMv8_cuda12

b=`perl -e 'printf("%02i", $ARGV[0]);' ${SLURM_ARRAY_TASK_ID}`
#extract_trajectory follow_dcd --replica ${SLURM_ARRAY_TASK_ID} walkers/follow.$b.dcd
extract_trajectory extract_traj_dcd --replica ${SLURM_ARRAY_TASK_ID} trajectories/trajectory.$b.dcd
