#!/bin/bash
# Example SLURM job for the TOPAS simulations (CPU only). Adapt to your cluster and TOPAS installation.
#SBATCH --job-name=topas-dataset
#SBATCH --cpus-per-task=16           # matches i:Ts/NumberOfThreads in topas/MCspec.txt
#SBATCH --output=slurm-%x-%j.out

# Geant4 data and TOPAS libraries; edit to match your installation.
# export TOPAS_G4_DATA_DIR=/path/to/G4Data
# export LD_LIBRARY_PATH=/path/to/topas/lib:$LD_LIBRARY_PATH
# export PATH=/path/to/topas/bin:$PATH

# Usage: sbatch slurm/simulate.sbatch --ct-dir ... --output-dir ... --energy-min 90 --energy-max 145
srun python scripts/simulate_dataset.py "$@"
