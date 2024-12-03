from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import argparse
import time
import csv 
import os
def main(num_forecaster, W, b, X, y, grad, training_loop, forecast, run):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Calculate the number of forecasters per rank
    forecasters_per_rank = num_forecaster // size
    extra = num_forecaster % size

    # Handle extra forecasters if num_forecaster is not perfectly divisible
    start_idx = rank * forecasters_per_rank + min(rank, extra)
    end_idx = start_idx + forecasters_per_rank
    if rank < extra:
        end_idx += 1

    # Parameters for random noise
    noise_std = 0.1

    # Local storage for predictions
    local_predictions = []

    # Start timing execution
    start_time = time.time()

    # Generate forecasts for the subset assigned to this rank
    for i in tqdm(range(start_idx, end_idx), desc=f"forecast_rank_{rank}"):
        key = jax.random.PRNGKey(i)
        W_noise = jax.random.normal(key, W.shape) * noise_std
        b_noise = jax.random.normal(key, b.shape) * noise_std

        W_init = W + W_noise
        b_init = b + b_noise

        W_trained, b_trained = training_loop(grad, 20, W_init, b_init, X, y)
        y_predicted = forecast(5, X, W_trained, b_trained)

        local_predictions.append(y_predicted)

    # Gather predictions from all processes
    all_predictions = comm.gather(local_predictions, root=0)
    exec_time = time.time() - start_time

    if rank == 0:
        # Flatten the aggregated predictions into a single array
        aggregated_forecasting = np.array([pred for sublist in all_predictions for pred in sublist])
        mean_forecast = np.mean(aggregated_forecasting, axis=0)
        median_forecast = np.median(aggregated_forecasting, axis=0)
        std_dev = np.std(aggregated_forecasting, axis=0)
        percentile_5 = np.percentile(aggregated_forecasting, 5, axis=0)
        percentile_95 = np.percentile(aggregated_forecasting, 95, axis=0)

        # Collect statistics
        result = {
            "num_forecasters": num_forecaster,
            "num_processes": size,
            "exec_time": exec_time,
            "mean": mean_forecast.tolist(),
            "median": median_forecast.tolist(),
            "std_dev": std_dev.tolist(),
            "percentile_5": percentile_5.tolist(),
            "percentile_95": percentile_95.tolist(),
        }

        # Save the results to the CSV file
        output_file = f"weak/scalability_results_{run}.csv"

        # Check if the file exists
        if not os.path.exists(output_file):
            # Create the file
            with open(output_file, "w") as file:
                file.write("")  # Write an empty string to initialize the file
            print(f"File '{output_file}' created.")
        else:
            print(f"File '{output_file}' already exists.")

        write_header = not os.path.exists(output_file)  # Check if file exists
        with open(output_file, "a", newline="") as csvfile:  # Use "a" to append
            fieldnames = ["num_forecasters", "num_processes", "exec_time", "mean", "median", "std_dev", "percentile_5", "percentile_95"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()  # Write header only if the file is new
            writer.writerow(result)  # Write a single row for this run

        return exec_time, size, result
    return None, None, None



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Parallel Forecaster with MPI")
    parser.add_argument("--num_forecaster", type=int, required=True, help="Total number of forecasters")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    args = parser.parse_args()

    # Placeholder variables (replace with actual model and data)
    W = jnp.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0, 1.0, 0.0, 1.0]])  # Example weight
    b = jnp.array([0.1])  # Example bias
    X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])  # Example input
    y = jnp.array([[0.1, 0.7]])  # Example target output

    # Example training and forecast functions (replace with your implementation)
    def training_loop(grad, num_epochs, W, b, X, y):
        for _ in range(num_epochs):
            delta = grad((W, b), X, y)
            W -= 0.1 * delta[0]
            b -= 0.1 * delta[1]
        return W, b

    def forecast(horizon, X, W, b):
        result = []
        for _ in range(horizon):
            X_flatten = X.flatten()
            y_next = jnp.dot(W, X_flatten) + b
            result.append(y_next)
        return jnp.array(result)

    grad = jax.grad(lambda params, X, y: jnp.sum((jnp.dot(params[0], X.flatten()) + params[1] - y) ** 2))

    # Call the main function with parsed arguments
    exec_time, size, stats = main(args.num_forecaster, W, b, X, y, grad, training_loop, forecast, args.run)


