#!/bin/bash -l


#SBATCH --job-name="Run_forecast_strong"

#SBATCH --output=strong_forecasting_%j.log

#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node 64  # Optimized for 1 full node of aion
#SBATCH -c 1

#SBATCH --time=8:00:00


    
## other options
#SBATCH --mail-user=silvana.belegu@uni.lu
#SBATCH --mail-type=all
## Move to the directory where the job was submitted
#cd $SLURM_SUBMIT_DIR


# print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
# module purge || print_error_and_exit "No 'module' command"
# # Python 3.X by default (also on system)



source ~/.bashrc
conda activate "parallel-p" 
OMPI_MCA_pml=ucx
OMPI_MCA_osc=ucx
python --version

python strong_sc_forecasting.py
