#!/bin/bash -l
#SBATCH -N 1               # Use 16 nodes  (for running O1 circuit with 217k cells)
#SBATCH -t 01:00:00           # Set 2 hours and 30 minute time limit
#SBATCH -p prod               # Submit to the production 'partition'
#SBATCH -C "cpu"              # Constraint the job to run on nodes without SSDs. If you want SSD, use only "nvme". If you want KNLs then "knl"\
#SBATCH --account=proj38      # your project number

srun ./e_model_packager/sscx2020/prepare_packages.sh