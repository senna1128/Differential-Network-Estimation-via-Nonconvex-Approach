#!/bin/bash

#---------------------------------------------------------------------------------
# Account information

#SBATCH --account=basic              # basic (default), staff, phd, faculty

#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=2          # number of CPUs requested (for parallel tasks)
#SBATCH --mem-per-cpu=4G           # requested memory
#SBATCH --time=2-00:00:00          # wall clock limit (d-hh:mm:ss)

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=RealApplication    # user-defined job name

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python

#---------------------------------------------------------------------------------
# Commands to execute below...
srun python main1.py


