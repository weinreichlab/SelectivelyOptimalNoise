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

        # Print progress every 100 runs
        run_number = i + 1
        if run_number % 100 == 0:
            print(f"Run {run_number} of {num_simulations} is done.")

    return all_simulation_data

def plot_allele_frequency_paths(simulation_results: list[dict], N: int, K: int): 
    """Plots the individual trajectories of allele frequencies over generations."""
    plt.figure(figsize=(10, 6))
    
    fixation_count = 0
    extinction_count = 0
    fixation_times = []
    extinction_times = []

    for run in simulation_results:
        if run["fixed"]:
            color = 'green'
            fixation_count += 1
            label = 'Fixed' if fixation_count == 1 else ""
            fixation_times.append(run["final_generations"])
        else:
            color = 'red'
            extinction_count += 1
            label = 'Extinct' if extinction_count == 1 else ""
            extinction_times.append(run["final_generations"])

        plt.plot(run["generations"], run["frequencies"], color=color, alpha=0.25, linewidth=1.0, label=label)

    # Calculate average generation times
    avg_fixation_time = np.mean(fixation_times) if fixation_times else 0
    avg_extinction_time = np.mean(extinction_times) if extinction_times else 0
    
    print("\n--- Simulation Metrics ---")
    print(f"Average generation time for Fixed alleles: {avg_fixation_time:.2f} generations")
    print(f"Average generation time for Extinct alleles: {avg_extinction_time:.2f} generations")
    print(f"Total Fixed: {fixation_count} | Total Extinct: {extinction_count}")
    print("--------------------------")

    # Layout configurations
    initial_freq = K / N
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Fixation (p=1.0)')
    plt.axhline(y=0.0, color='black', linestyle=':', alpha=0.5, label='Extinction (p=0.0)')
    plt.axhline(y=initial_freq, color='blue', linestyle='--', alpha=0.5, label=f'Initial Freq (p={initial_freq:.2f})')

    plt.title(f"Wright-Fisher Model: Allele Frequency Paths (N={N})\nFixed: {fixation_count} | Extinct: {extinction_count}", fontsize=12)
    plt.xlabel("Generation (Time)", fontsize=11)
    plt.ylabel("Allele Frequency (p)", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

def plot_observed_vs_expected_variance(simulation_results: list[dict], N: int):
    # Separate runs by outcome
    fixed_runs = [run for run in simulation_results if run["fixed"]]
    extinct_runs = [run for run in simulation_results if not run["fixed"]]
    
    # We will also keep track of raw sample counts per frequency to build our table
    fixed_counts_dict = defaultdict(int)
    extinct_counts_dict = defaultdict(int)
    
    def calculate_variance_and_collect_counts(runs: list[dict], counts_tracker: dict) -> tuple[np.ndarray, np.ndarray]:
        transitions = defaultdict(list)
        for run in runs:
            counts = run["counts"]
            for t in range(len(counts) - 1):
                k_current = counts[t]
                k_next = counts[t + 1]
                transitions[k_current].append(k_next / N)
                counts_tracker[k_current] += 1 # Track how many transitions start here
                
            # Account for absorbing endpoints
            final_count = counts[-1]
            if final_count == 0:
                transitions[0].append(0.0)
                counts_tracker[0] += 1
            elif final_count == N:
                transitions[N].append(1.0)
                counts_tracker[N] += 1
                
        observed_counts = []
        observed_variances = []
        
        for k, next_freqs in transitions.items():
            if k in (0, N):
                observed_counts.append(k)
                observed_variances.append(0.0)
            elif len(next_freqs) >= 2:
                observed_counts.append(k)
                observed_variances.append(np.var(next_freqs))
                
        if not observed_counts:
            return np.array([]), np.array([])
            
        observed_counts = np.array(observed_counts)
        observed_variances = np.array(observed_variances)
        sort_idx = np.argsort(observed_counts)
        
        return observed_counts[sort_idx] / N, observed_variances[sort_idx]

    # Calculate variances and collect transition sample sizes
    freq_fixed, var_fixed = calculate_variance_and_collect_counts(fixed_runs, fixed_counts_dict)
    freq_extinct, var_extinct = calculate_variance_and_collect_counts(extinct_runs, extinct_counts_dict)

    # --- Print Sample Size Table to Console ---
    print("\n========================================================")
    print("      TRANSITION DATA POINTS BY FREQUENCY INTERVAL      ")
    print("========================================================")
    print(f"| Frequency Range | Fixed Runs (n={len(fixed_runs)}) | Extinct Runs (n={len(extinct_runs)}) |")
    print("|-----------------|------------------|--------------------|")
    
    # Define 10 bins across the frequency spectrum [0.0 to 1.0]
    bins = [(i/10, (i+1)/10) for i in range(10)]
    for start, end in bins:
        # Sum up transition points whose starting frequency falls in this bin
        fixed_sum = sum(count for k, count in fixed_counts_dict.items() if start <= (k/N) < end)
        extinct_sum = sum(count for k, count in extinct_counts_dict.items() if start <= (k/N) < end)
        
        # Format edge cases for the final boundary (1.0)
        if end == 1.0:
            fixed_sum += fixed_counts_dict[N]
            extinct_sum += extinct_counts_dict[N]
            print(f"| [{start:.1f}, {end:.1f}]       | {fixed_sum:<16} | {extinct_sum:<18} |")
        else:
            print(f"| [{start:.1f}, {end:.1f})       | {fixed_sum:<16} | {extinct_sum:<18} |")
            
    print("========================================================\n")

    # Theoretical curve calculation
    expected_frequencies = np.linspace(0, 1, 500) 
    expected_variances = (expected_frequencies * (1 - expected_frequencies)) / N

    plt.figure(figsize=(10, 6))
    

    # Plot empirical data for Extinct runs (Crashed)
    if len(freq_extinct) > 0:
        plt.plot(freq_extinct, var_extinct, color='crimson', linestyle='-', 
                 linewidth=1.5, marker='o', markersize=3, alpha=0.8, 
                 label=f"Observed (Extinct Runs, n={len(extinct_runs)})")
        
    # Plot empirical data for Fixed runs
    if len(freq_fixed) > 0:
        plt.plot(freq_fixed, var_fixed, color='forestgreen', linestyle='-', 
                 linewidth=1.5, marker='s', markersize=3, alpha=0.8, 
                 label=f"Observed (Fixed Runs, n={len(fixed_runs)})")
        
    # Plot expected curve
    plt.plot(expected_frequencies, expected_variances, color='black', linestyle='--', 
             linewidth=2, label=r"Expected Variance: $\frac{x(1-x)}{N}$")
    

    plt.title(f"Wright-Fisher Model: Observed vs. Expected Variance (N={N})", fontsize=12)
    plt.xlabel("Allele Frequency (x = k/N)", fontsize=11)
    plt.ylabel("Variance of Frequency in Next Generation", fontsize=11)
    plt.xlim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

def export_to_csv(simulation_results: list[dict], filename: str = "wright_fisher_results.csv"):
    print(f"Exporting data to {filename}...")
    headers = ["Simulation_ID", "Is_Fixed", "Total_Generations", "Generation", "Allele_Frequency"]
    
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

plot_allele_frequency_paths(sim_data, N=POPULATION_SIZE, K=INITIAL_ALLELES)
plot_observed_vs_expected_variance(sim_data, N=POPULATION_SIZE)

# Display both plots
plt.show()

# Export to CSV for Excel (Uncomment if needed)
# export_to_csv(sim_data, filename="wright_fisher_simulation.csv")