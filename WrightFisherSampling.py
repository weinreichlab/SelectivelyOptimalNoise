import numpy as np 
import matplotlib.pyplot as plt

def wrightfishersampling(N: int, K: int) -> tuple[bool, int, list[list[int]]]:
    """
    Parameters: 

    N = number of haploid individuals (and total alleles)
    K = current allele count 

    returns: 
    
    tuple: (fixation status, final generation, history list)
    """

    # input validation 
    if N<= 0: 
        raise ValueError("Population number must be greater than zero.")
    if K < 0: 
        raise ValueError("Number of alleles cannot be negative.")
    if K > N: 
        raise ValueError("K cannot be greater than the total alleles.")
    
    history = []
    t = 0 # time stars at 0 
    current_K = K 

    while 0 < current_K < N: 
        history.append([t, current_K])
        p = current_K / N # calculating the allele frequency, p = K/N
        current_K = int(np.random.binomial(n = N, p=p)) # random sampling from binomial distribution for next generation
        t += 1 # updating the time to add a generation
    
    history.append([t, current_K])
    fixed = (current_K == N) # true if fixed, false if the allele crashes

    '''
    # removed for running larger simulations
    if fixed: 
        print("The allele fixed.")
    else:
        print("The allele crashed.")

    '''
    return fixed, t, history

# running the simulation 
def run_multiple_simulations(N: int, K:int, num_simulations: int) -> list[dict]: 
    all_simulation_data = []

    for _ in range(num_simulations):
        fixed, final_t, history = wrightfishersampling(N, K)

        # saving the structured data for each run 
        all_simulation_data.append({
            "fixed": fixed, 
            "final_generations": final_t,
            "generations": [step[0] for step in history], 
            "frequencies": [step[1] / N for step in history]
        })

    return all_simulation_data

# plotting the results/analysis
def plot_wright_fisher(simulation_results: list[dict], N: int, K: int): 
    # takes output data from run_multiple_simulations and plots it

    plt.figure(figsize=(10, 6))
    
    fixation_count = 0
    extinction_count = 0

    for run in simulation_results:
        if run["fixed"]:
            color = 'green'
            fixation_count += 1
            label = 'Fixed' if fixation_count == 1 else ""
        else:
            color = 'red'
            extinction_count += 1
            label = 'Extinct' if extinction_count == 1 else ""

        plt.plot(run["generations"], run["frequencies"], color=color, alpha=0.6, linewidth=1.5, label=label)

    initial_freq = K / N
    # Plot labels and styling
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Fixation (p=1.0)')
    plt.axhline(y=0.0, color='black', linestyle=':', alpha=0.5, label='Extinction (p=0.0)')
    plt.axhline(y=initial_freq, color='blue', linestyle='--', alpha=0.5, label=f'Initial Frequency (p={initial_freq:.2f})')

    plt.title(f"Wright-Fisher Model (N={N}, Initial Freq={initial_freq:.2f})\nFixed: {fixation_count} | Extinct: {extinction_count}", fontsize=12)
    plt.xlabel("Generation (Time)", fontsize=11)
    plt.ylabel("Allele Frequency (p)", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best')
    
    plt.show()
    
#############################################
#running the actual simulation, example
POPULATION_SIZE = 100 
INITIAL_ALLELES = 50

#Note that the allele frequncy = population_size / initial_alleles 

NUM_RUNS = 100

sim_data = run_multiple_simulations(N=POPULATION_SIZE, K=INITIAL_ALLELES, num_simulations=NUM_RUNS)
plot_wright_fisher(sim_data, N=POPULATION_SIZE, K=INITIAL_ALLELES)