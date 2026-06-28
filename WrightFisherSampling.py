import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List

def wrightfishersampling(K: List[int], Ff: List[Callable[[int], int]]) -> tuple[bool, int, List[List[int]]]:
    """
    Parameters:
    K = list of initial allele count for each allele
    Ff = Fitness function. change in F. list of lambda functions that change F
           based on the generation number for each allele.
          input should be just the generation number. output is fitness.

    returns:

    tuple: (fixation status, final generation, history list)
    """

    number_of_alleles = len(K)

    N = sum(K)

    # input validation
    for i in range(number_of_alleles):
      if K[i] < 0:
        raise ValueError("Number of alleles cannot be negative")

    if N == 0:
        raise ValueError("error: population number is zero")

    if not isinstance(Ff, list) or not all(callable(x) for x in Ff):
        raise ValueError("Ff must be a list of callable functions")
    if not (len(K) == len(Ff)):
        raise ValueError("K and Ff must have the same length")
    if len(K) < 1:
        raise ValueError("K must be at least 1")


    history = []
    t = 0
    current_K = list(K)

    F = [0.0] * number_of_alleles

    history.append(current_K.copy())

    while max(current_K) < N:

        p_unnormalized = []
        for j in range(number_of_alleles):
            F[j] = Ff[j](t)
            p_unnormalized.append(current_K[j] * F[j])

        sum_p_unnormalized = sum(p_unnormalized)

        pvals = []
        if sum_p_unnormalized == 0:
            pvals = [0.0] * number_of_alleles
        else:
            pvals = [val / sum_p_unnormalized for val in p_unnormalized]


        pvals = np.array(pvals)
        pvals /= pvals.sum()

        try:
            new_counts = np.random.multinomial(n=N, pvals=pvals)
            current_K = new_counts.tolist()
        except ValueError as e:
            print(f"Warning: multinomial failed for generation {t}: {e}. Probabilities: {pvals}")
            break

        t += 1
        history.append(current_K.copy())

    fixed = False
    for a in range(number_of_alleles):
      if current_K[a] == N:
        fixed = True
        break

    return fixed, t, history


# running the simulation
def run_multiple_simulations(initial_K_counts: List[int], Ff_modifiers: List[Callable[[int],int]], num_simulations: int) -> List[dict]:
    all_simulation_data = []

    for _ in range(num_simulations):
        fixed, final_t, allele_counts_history_raw = wrightfishersampling(initial_K_counts, Ff_modifiers)

        N = sum(initial_K_counts)

        generations = list(range(len(allele_counts_history_raw)))
        allele_counts_history = allele_counts_history_raw

        # convert counts to frequencies
        allele_frequencies_history = [[count / N for count in k_list] for k_list in allele_counts_history]

        all_simulation_data.append({
            "fixed": fixed,
            "final_generations": final_t,
            "generations": generations,
            "allele_counts_history": allele_counts_history,
            "allele_frequencies_history": allele_frequencies_history
        })

    return all_simulation_data

# plotting the results/analysis
def plot_wright_fisher(simulation_results: List[dict], initial_K_counts: List[int]):

    plt.figure(figsize=(12, 8))

    N = sum(initial_K_counts)
    num_alleles = len(initial_K_counts)

    fixation_event_count = 0
    polymorphic_event_count = 0

    colors = plt.cm.get_cmap('viridis', num_alleles)

    plotted_legend_labels = {'Fixed': False, 'Polymorphic': False}
    for i in range(num_alleles):
        plotted_legend_labels[f'Initial Freq Allele {i+1}'] = False

    for sim_idx, run in enumerate(simulation_results):
        generations = run["generations"]
        allele_frequencies_history = run["allele_frequencies_history"]

        frequencies_by_allele = list(map(list, zip(*allele_frequencies_history)))

        is_fixed = run["fixed"]

        outcome_label_key = ''
        if is_fixed:
            linestyle = '-'
            outcome_label_key = 'Fixed'
        else:
            linestyle = ':'
            outcome_label_key = 'Polymorphic'

        if outcome_label_key == 'Fixed':
            fixation_event_count += 1
        elif outcome_label_key == 'Polymorphic':
            polymorphic_event_count += 1

        for allele_idx in range(num_alleles):
            label_for_legend = None
            if allele_idx == 0 and not plotted_legend_labels[outcome_label_key]:
                label_for_legend = outcome_label_key
                plotted_legend_labels[outcome_label_key] = True

            plt.plot(generations, frequencies_by_allele[allele_idx],
                     color=colors(allele_idx), alpha=0.5, linewidth=1.0, linestyle=linestyle,
                     label=label_for_legend)

    plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.6, label='Fixation (p=1.0)')

    initial_frequencies = [k / N for k in initial_K_counts]
    for allele_idx, freq in enumerate(initial_frequencies):
        legend_key = f'Initial Freq Allele {allele_idx+1}'
        if not plotted_legend_labels[legend_key]:
            plt.axhline(y=freq, color=colors(allele_idx), linestyle='-.', alpha=0.7,
                        label=f'Initial Freq Allele {allele_idx+1} (p={freq:.2f})')
            plotted_legend_labels[legend_key] = True

    plt.title(f"Wright-Fisher Model (N={N}, Initial Allele Counts={initial_K_counts})\n" +
              f"Fixed (any allele): {fixation_event_count} | Polymorphic: {polymorphic_event_count}",
              fontsize=12)
    plt.xlabel("Generation (Time)", fontsize=11)
    plt.ylabel("Allele Frequency (p)", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def run_everything(INITIAL_ALLELE_COUNTS: List[int], FITNESS_FUNCTIONS: List[Callable[[int],int]], NUM_RUNS: int):

    sim_data = run_multiple_simulations(initial_K_counts=INITIAL_ALLELE_COUNTS,
                                        Ff_modifiers=FITNESS_FUNCTIONS,
                                        num_simulations=NUM_RUNS)
    plot_wright_fisher(sim_data, initial_K_counts=INITIAL_ALLELE_COUNTS)


    fixed_generations = [run['final_generations'] for run in sim_data if run['fixed']]

    if fixed_generations:
        average_fixation_time = np.mean(fixed_generations)
        print(f"avg time to fixation: {average_fixation_time:.2f} generations")
    else:
        print("no fixation")

#examples

INITIAL_ALLELE_COUNTS = [20, 20]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [40, 90, 70]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [40, 90, 70]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01*t]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [40, 90, 40]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01*t]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [1, 1, 1]
FITNESS_FUNCTIONS = [lambda t: 0.2, lambda t: 0.2, lambda t: 0.2]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [500, 500, 500]
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t^2]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [500, 500, 500, 500, 500, 500, 500]
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t^2, lambda t: 3.0, lambda t: 6.0, lambda t: 0.1, lambda t: 0]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [20, 20]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [40, 90, 70]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [40, 90, 70]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01*t]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [40, 90, 40]
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01*t]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [1, 1, 1]
FITNESS_FUNCTIONS = [lambda t: 0.2, lambda t: 0.2, lambda t: 0.2]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [500, 500, 500]
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t^2]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = [500, 500, 500, 500, 500, 500, 500]
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t^2, lambda t: 3.0, lambda t: 6.0, lambda t: 0.1, lambda t: 0]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)
