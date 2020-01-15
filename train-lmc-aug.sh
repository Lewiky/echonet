#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 1:00:00
#SBATCH --mem 100GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python3 src/main.py --epochs 50 --qual-results lmc_results.csv --mode LMC --augmentation-length 3
