#!/bin/bash -l


#SBATCH --job-name="Jupyter_mpi_job"

#SBATCH --output=Notebooks_logs/AION/notebook_%j.log

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


print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
# Python 3.X by default (also on system)

micromamba activate parallel_mpi 
OMPI_MCA_pml=ucx
OMPI_MCA_osc=ucx
python --version
#module avail
#jupyter notebook --ip $(facter ipaddress) --no-browser  &
jupyter lab --ip $(ip addr | egrep '172\.17|21'| grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser &
pid=$!
sleep 5s
jupyter notebook list
jupyter --paths
jupyter kernelspec list
#echo "Enter this command on your laptop: ssh -p 8022 -NL 8888:$(facter ipaddress):8888 clee@access-iris.uni.lu" > notebook.log
echo "Enter this command on your laptop: ssh -p 8022 -NL 8888:$(ip addr | egrep '172\.17|21'| grep 'inet ' | awk '{print $2}' | cut -d/ -f1):8888 sbelegu@access-aion.uni.lu" > notebook.log
wait $pid

# On Linux, Open MPI is built with UCX support but it is disabled by default.                                                       
# To enable it, first install UCX (conda install -c conda-forge ucx).                                                               
# Afterwards, set the environment variables                                                                                         
# OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx                                                                                                 
# before launching your MPI processes.                                                                                              
# Equivalently, you can set the MCA parameters in the command line:
# mpiexec --mca pml ucx --mca osc ucx ...


# On Linux, Open MPI is built with UCC support but it is disabled by default.
# To enable it, first install UCC (conda install -c conda-forge ucc).
# Afterwards, set the environment variables
# OMPI_MCA_coll_ucc_enable=1
# before launching your MPI processes.
# Equivalently, you can set the MCA parameters in the command line:
# mpiexec --mca coll_ucc_enable 1 ...


# On Linux, Open MPI is built with CUDA awareness but it is disabled by default.
# To enable it, please set the environment variable
# OMPI_MCA_opal_cuda_support=true
# before launching your MPI processes.
# Equivalently, you can set the MCA parameter in the command line:
# mpiexec --mca opal_cuda_support 1 ...
# Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via
# UCX. Please consult UCX documentation for further details.

