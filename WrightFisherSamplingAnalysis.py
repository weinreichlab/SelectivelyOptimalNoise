import numpy as np 
import matplotlib.pyplot as plt

def wrightfishersampling(N: int, K: int) -> tuple[bool, int, list[list[float]]]:
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
        var = (p * (1 - p)) / N  
        history.append([t, current_K, var])
        
        current_K = int(np.random.binomial(n=N, p=p)) 
        t += 1 
    
    history.append([t, current_K, 0.0]) # variance in the final generation is 0 since the allele is either fixed or extinct
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
            "frequencies": [step[1] / N for step in history],
            "variances": [step[2] for step in history]
        })

        # Print progress every 10 runs
        run_number = i + 1
        if run_number % 10 == 0:
            print(f"Run {run_number} of {num_simulations} is done.")

    return all_simulation_data

def plot_wright_fisher_with_variance(simulation_results: list[dict], N: int, K: int): 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    
    fixation_count = 0
    extinction_count = 0
    fixation_times = []
    extinction_times = []

    # Find the maximum generation length to pad matrices for calculating averages
    max_gens = max(run["final_generations"] for run in simulation_results) + 1

    # Matrices to store variance histories for averaging
    fixed_vars_matrix = []
    extinct_vars_matrix = []

    for run in simulation_results:
        if run["fixed"]:
            color = 'green'
            fixation_count += 1
            label = 'Fixed' if fixation_count == 1 else ""
            fixation_times.append(run["final_generations"])
            
            # Pad array with 0.0 variance after the run terminates (since all of the runs are different lengths) 
            padded_var = run["variances"] + [0.0] * (max_gens - len(run["variances"]))
            fixed_vars_matrix.append(padded_var)
        else:
            color = 'red'
            extinction_count += 1
            label = 'Extinct' if extinction_count == 1 else ""
            extinction_times.append(run["final_generations"])
            
            # Pad array with 0.0 variance after the run terminates
            padded_var = run["variances"] + [0.0] * (max_gens - len(run["variances"]))
            extinct_vars_matrix.append(padded_var)

        ax1.plot(run["generations"], run["frequencies"], color=color, alpha=0.25, linewidth=1.0, label=label)
        ax2.plot(run["generations"], run["variances"], color=color, alpha=0.25, linewidth=1.0)

    # Calculate average generation times
    avg_fixation_time = np.mean(fixation_times) if fixation_times else 0
    avg_extinction_time = np.mean(extinction_times) if extinction_times else 0
    
    print("--- Simulation Metrics ---")
    print(f"Average generation time for Fixed alleles: {avg_fixation_time:.2f} generations")
    print(f"Average generation time for Extint alleles: {avg_extinction_time:.2f} generations")
    print(f"Total Fixed: {fixation_count} | Total Extinct: {extinction_count}")
    print("--------------------------")

    # Top Plot: Allele Frequency Setup
    initial_freq = K / N
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Fixation (p=1.0)')
    ax1.axhline(y=0.0, color='black', linestyle=':', alpha=0.5, label='Extinction (p=0.0)')
    ax1.axhline(y=initial_freq, color='blue', linestyle='--', alpha=0.5, label=f'Initial Freq (p={initial_freq:.2f})')

    ax1.set_title(f"Wright-Fisher Model Allele Frequency (N={N}, Initial Freq={initial_freq:.2f})\nFixed: {fixation_count} | Extinct: {extinction_count}", fontsize=12)
    ax1.set_ylabel("Allele Frequency (p)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='best')
    
    # Bottom Plot: Graph variance and compute average variance for fixed and extinct paths
    gens_axis = np.arange(max_gens)
    
    if fixed_vars_matrix:
        avg_fixed_variance = np.mean(fixed_vars_matrix, axis=0) # finding the average variance down the rows of the matrix
        ax2.plot(gens_axis, avg_fixed_variance, color='blue', linestyle='--', linewidth=2.5, label='Mean Var (Fixed paths)')
    if extinct_vars_matrix:
        avg_extinct_variance = np.mean(extinct_vars_matrix, axis=0)
        ax2.plot(gens_axis, avg_extinct_variance, color='orange', linestyle='--', linewidth=2.5, label='Mean Var (Extinct paths)')

    ax2.set_title(r"Per Generation Variance: $\text{Var}(x) = \frac{x(1-x)}{N}$", fontsize=12)
    ax2.set_xlabel("Generation (Time)", fontsize=11)
    ax2.set_ylabel("Variance", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

#############################################
# Running the simulation
POPULATION_SIZE = 10000
INITIAL_ALLELES = 1
NUM_RUNS = 10000

sim_data = run_multiple_simulations(N=POPULATION_SIZE, K=INITIAL_ALLELES, num_simulations=NUM_RUNS)
plot_wright_fisher_with_variance(sim_data, N=POPULATION_SIZE, K=INITIAL_ALLELES)