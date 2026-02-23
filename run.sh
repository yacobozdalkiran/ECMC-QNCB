#!/bin/bash

#SBATCH --job-name=T16b6.2v2
#SBATCH --output=%x.o
#SBATCH --time=02:00:00
#SBATCH --ntasks=256
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=37
#SBATCH --partition=cpu_med

#Source necessary modules
source modules_load.sh

# Run MPI script
srun build/gauge_hb_cb inputs/hb_cb.txt
