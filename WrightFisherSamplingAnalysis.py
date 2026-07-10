import numpy as np 
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

def wrightfishersampling(N: int, K: int) -> tuple[bool, int, list[list[int]]]:
    if N <= 0: 
        raise ValueError("Population number must be greater than zero.")
    if K < 0: 
        raise ValueError("Number of alleles cannot be negative.")
    if K > N: 
        raise ValueError("K cannot be greater than the total alleles.")
    
    history = []
    t = 0 
    current_K = K 

    while 0 < current_K < N: 
        p = current_K / N 
        history.append([t, current_K])
        
        current_K = int(np.random.binomial(n=N, p=p)) 
        t += 1 
    
    history.append([t, current_K]) 
    fixed = (current_K == N) 

    return fixed, t, history

def run_multiple_simulations(N: int, K: int, num_simulations: int) -> list[dict]: 
    all_simulation_data = []
    for i in range(num_simulations):
        fixed, final_t, history = wrightfishersampling(N, K)
        all_simulation_data.append({
            "fixed": fixed, 
            "final_generations": final_t,
            "generations": [step[0] for step in history], 
            "counts": [step[1] for step in history],
            "frequencies": [step[1] / N for step in history],
        })

        # Print progress every 100 runs for efficiency in long simulations
        run_number = i + 1
        if run_number % 100 == 0:
            print(f"Run {run_number} of {num_simulations} is done.")

    return all_simulation_data

def plot_observed_vs_expected_variance(simulation_results: list[dict], N: int):
    # Dictionary to store lists of next-generation allele frequencies for each starting count k
    # Key: starting count k, Value: list of frequencies (count_next / N) in the next generation
    transitions = defaultdict(list)
    
    for run in simulation_results:
        counts = run["counts"]
        # Loop through up to the second-to-last element because the last element has no "next" generation
        for t in range(len(counts) - 1):
            k_current = counts[t]
            k_next = counts[t + 1]
            freq_next = k_next / N
            transitions[k_current].append(freq_next)
            
    # Calculate observed variance for each unique starting count k that has enough data points
    observed_counts = []
    observed_variances = []
    
    # We require a minimum number of samples per allele count to get a stable variance estimate
    MIN_SAMPLES = 2 
    
    for k, next_freqs in transitions.items():
        if len(next_freqs) >= MIN_SAMPLES:
            observed_counts.append(k)
            observed_variances.append(np.var(next_freqs))
            
    # Convert to arrays and sort by allele count/frequency for plotting 
    observed_counts = np.array(observed_counts)
    observed_variances = np.array(observed_variances)
    sort_idx = np.argsort(observed_counts)
    
    observed_counts = observed_counts[sort_idx]
    observed_variances = observed_variances[sort_idx]
    observed_frequencies = observed_counts / N

    # Compute theoretical expected variance for the entire frequency spectrum [0, 1]
    # Var(p_next) = p(1-p)/N
    expected_frequencies = np.linspace(0, 1, 500) # 500 data points to draw the theoretical curve
    expected_variances = (expected_frequencies * (1 - expected_frequencies)) / N

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot expected variance
    plt.plot(expected_frequencies, expected_variances, color='black', linestyle='--', 
             linewidth=2, label=r"Expected Variance: $\frac{x(1-x)}{N}$")
    
    # Plot observed empirical variance
    plt.plot(observed_frequencies, observed_variances, color='crimson', linestyle='-', 
             linewidth=1.5, marker='o', markersize=3, alpha=0.8, label="Observed Variance")

    plt.title(f"Wright-Fisher Model: Observed vs. Expected Variance (N={N})", fontsize=12)
    plt.xlabel("Allele Frequency (x = k/N)", fontsize=11)
    plt.ylabel("Variance of Frequency in Next Generation", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

def export_to_csv(simulation_results: list[dict], filename: str = "wright_fisher_results.csv"):
    print(f"Exporting data to {filename}...")
    
    headers = [
        "Simulation_ID", 
        "Is_Fixed",         
        "Total_Generations", 
        "Generation",    
        "Allele_Frequency"
    ]
    
    with open(filename, mode="w", newline="", encoding="utf-8-sig") as il:
        writer = csv.writer(il)
        writer.writerow(headers)
        
        for sim_id, run in enumerate(simulation_results, start=1):
            is_fixed = run["fixed"]
            total_gens = run["final_generations"]
            
            rows = (
                [sim_id, is_fixed, total_gens, gen, freq]
                for gen, freq in zip(run["generations"], run["frequencies"])
            )
            
            writer.writerows(rows)
                
    print("Export complete")
 
#############################################
# Running the simulation
POPULATION_SIZE = 1000  
INITIAL_ALLELES = 500 
NUM_RUNS = 1000

sim_data = run_multiple_simulations(N=POPULATION_SIZE, K=INITIAL_ALLELES, num_simulations=NUM_RUNS)

plot_observed_vs_expected_variance(sim_data, N=POPULATION_SIZE)

# Export to CSV for Excel
#export_to_csv(sim_data, filename="wright_fisher_simulation.csv")