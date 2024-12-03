#!/bin/bash -l


#SBATCH --job-name="Run_strong_fc"

#SBATCH --output=strong_forecasting_%j.log
#SBATCH --error=strong_forecasting_%j.err
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node 64  # Optimized for 1 full node of aion
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G  
#SBATCH --time=10:00:00


    
## other options
#SBATCH --mail-user=silvana.belegu@uni.lu
#SBATCH --mail-type=all
## Move to the directory where the job was submitted
#cd $SLURM_SUBMIT_DIR


# print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
# module purge || print_error_and_exit "No 'module' command"
# # Python 3.X by default (also on system)



source ~/.bashrc
micromamba activate parallel_mpi
module load mpi/OpenMPI/4.0.5-GCC-10.2.0

OMPI_MCA_pml=ucx
OMPI_MCA_osc=ucx
python --version

python strong_sc_forecasting.py
