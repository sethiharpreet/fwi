#!/bin/bash
#SBATCH --nodes=2
#SBATCH --tasks-per-node=10
#SBATCH -o slurm.%N.%J.out
#SBATCH -e slurm.%N.%J.err
#SBATCH -p CLUSTER

mpirun ./sum
~               
