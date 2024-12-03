import os
# Array of processes and base forecasters per process
PROCESSES = [2, 3, 4, 6, 8, 12, 14, 16, 18, 32]
BASE_FORECASTERS = 25

# Iterate through process counts and calculate forecasters
for num_process in PROCESSES:
    num_forecaster = num_process * BASE_FORECASTERS
    print(f"Running with {num_process} processes and {num_forecaster} forecasters")
    for i in range(30):
        print(f"Running round {i}")
        # Construct the srun command
        command = f"mpirun -np={num_process} python parallel_forecasters.py --num_forecaster {num_forecaster} --run {i}"
    
        # Execute the command
        os.system(command)


