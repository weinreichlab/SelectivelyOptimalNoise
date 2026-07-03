import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Iterator


# wrightfishersampling simulates natural selection based on initial allele counts and a list of functions describing
# the fitness of each allele at each generation in time
def wrightfishersampling(K: np.ndarray, Ff: List[Callable[[int], float]]) -> Iterator[np.ndarray]:
    """
    Parameters:
    K = numpy array of initial allele count for each allele
    Ff = Fitness function. change in F. list of lambda functions that change F
           based on the generation number for each allele.
          input should be just the generation number. output is fitness.

    Yields:
    np.ndarray: The allele counts for the current generation.
    """

    number_of_alleles = K.size

    # population number no longer needs to be an input because it is now implied
    N = np.sum(K)

    # input validation, commented out for efficiency

    """

    if np.any(K<=0):
      raise ValueError("Number of alleles cannot be negative")

    if N == 0:
        raise ValueError("error: population number is zero")

    if not isinstance(Ff, list) or not all(callable(x) for x in Ff):
        raise ValueError("Ff must be a list of callable functions")
    if not (K.size == len(Ff)):
        raise ValueError("K and Ff must have the same length")
    if K.size < 1:
        raise ValueError("K must be at least 1")

    """

    t = 0  # we assume that the first generation is 0
    current_K = np.array(K, dtype=float)
    # we use current_K to not overwrite K as we work with it, now as a numpy array

    # initializing F as a numpy array
    F = np.zeros(number_of_alleles, dtype=float)

    # yield the initial generation
    yield current_K.copy()

    # while none of the alleles have yet fixed
    while np.max(current_K) < N:

        # p is unnormalized initially because the equations in Ff model the fitness of each allele at each time
        # in relative terms rather than proportions to make it much easier to input
        F_values_at_t = np.array([Ff[j](t) for j in range(number_of_alleles)], dtype=float)
        p_unnormalized = current_K * F_values_at_t

        sum_p_unnormalized = np.sum(p_unnormalized)

        if sum_p_unnormalized == 0:
            pvals = np.zeros(number_of_alleles, dtype=float)
        else:
            pvals = p_unnormalized / sum_p_unnormalized
            # re-normalize to ensure sum is exactly 1.0, handling potential float precision issues
            if not np.isclose(np.sum(pvals), 0.0):
                pvals /= np.sum(pvals)
            else:
                pvals = np.zeros(number_of_alleles, dtype=float)

        # random multinomial is implemented to select the next generation using the pvals as likelihoods of
        # each allele.

        new_counts = np.random.multinomial(n=N, pvals=pvals)
        current_K = new_counts

        yield current_K.copy()  # yield this generation
        t += 1  # next generation begins and the loop restarts


# function to facilitate running the simulation multiple times
def run_multiple_simulations(initial_K_counts: np.ndarray, Ff_modifiers: List[Callable[[int], int]],
                             num_simulations: int) -> List[dict]:
    all_simulation_data = []

    for _ in range(num_simulations):
        # wrightfishersampling now returns a generator, so we convert it to a list to process
        allele_counts_history_list = list(wrightfishersampling(initial_K_counts, Ff_modifiers))

        N = np.sum(initial_K_counts)

        generations = list(range(len(allele_counts_history_list)))
        allele_counts_history = allele_counts_history_list

        # fixation status and final generation determined from the collected history
        final_t = len(allele_counts_history_list) - 1
        last_generation_counts = allele_counts_history_list[-1]
        fixed = any(count == N for count in last_generation_counts)  # check if any allele count equals total population

        # convert counts to frequencies using numpy array operations
        allele_frequencies_history = [k_array / N for k_array in allele_counts_history]

        all_simulation_data.append({
            "fixed": fixed,
            "final_generations": final_t,
            "generations": generations,
            "allele_counts_history": allele_counts_history,
            "allele_frequencies_history": allele_frequencies_history
        })

    return all_simulation_data


# plotting the results/analysis
def plot_wright_fisher(simulation_results: List[dict], initial_K_counts: np.ndarray):
    plt.figure(figsize=(12, 8))

    N = np.sum(initial_K_counts)
    num_alleles = len(initial_K_counts)

    fixation_event_count = 0

    colors = plt.colormaps['viridis'].resampled(num_alleles)

    plotted_legend_labels = {'Fixed': False}
    for i in range(num_alleles):
        plotted_legend_labels[f'Initial Freq Allele {i + 1}'] = False

    for sim_idx, run in enumerate(simulation_results):
        generations = run["generations"]
        allele_frequencies_history = run["allele_frequencies_history"]

        # list of arrays -> 2D numpy array, then transpose and convert to list of lists for plotting
        frequencies_by_allele = np.array(allele_frequencies_history).T.tolist()

        is_fixed = run["fixed"]

        linestyle = '-'
        outcome_label_key = 'Fixed'

        if outcome_label_key == 'Fixed':
            fixation_event_count += 1

        for allele_idx in range(num_alleles):
            label_for_legend = None
            if allele_idx == 0 and not plotted_legend_labels[outcome_label_key]:
                label_for_legend = outcome_label_key
                plotted_legend_labels[outcome_label_key] = True

            plt.plot(generations, frequencies_by_allele[allele_idx],
                     color=colors(allele_idx), alpha=0.5, linewidth=1.0, linestyle=linestyle,
                     label=label_for_legend)

    plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.6, label='Fixation (p=1.0)')

    initial_frequencies = initial_K_counts / N  # direct array division
    for allele_idx, freq in enumerate(initial_frequencies):
        legend_key = f'Initial Freq Allele {allele_idx + 1}'
        if not plotted_legend_labels[legend_key]:
            plt.axhline(y=freq, color=colors(allele_idx), linestyle='-.', alpha=0.7,
                        label=f'Initial Freq Allele {allele_idx + 1} (p={freq:.2f})')
            plotted_legend_labels[legend_key] = True

    plt.title(
        f"Wright-Fisher Model (N={N}, Initial Allele Counts={initial_K_counts.tolist()})\n" +  # cnvert to list for display
        f"Fixed (any allele): {fixation_event_count}",
        fontsize=12)
    plt.xlabel("Generation (Time)", fontsize=11)
    plt.ylabel("Allele Frequency (p)", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
    plt.tight_layout()
    plt.show()


# this function's primary goal is to make it easier to use the program. you call this function with the initial allele
# counts, the fitness, and how many times you want the simulation to run, and it will do that then return a plot
# graphing the results from all of the simulations as well as reporting the average time to fixation
def run_everything(INITIAL_ALLELE_COUNTS: np.ndarray, FITNESS_FUNCTIONS: List[Callable[[int], int]], NUM_RUNS: int):
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


# examples
# a lot of highly unrealistic cases, used to get a sense of functioning of program

INITIAL_ALLELE_COUNTS = np.array([20, 20])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([40, 90, 70])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([40, 90, 70])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01 * t]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([40, 90, 40])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01 * t]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([1, 1, 1])
FITNESS_FUNCTIONS = [lambda t: 0.2, lambda t: 0.2, lambda t: 0.2]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([500, 500, 500])
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t * t]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([500, 500, 500, 500, 500, 500, 500])
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t * t, lambda t: 3.0, lambda t: 6.0, lambda t: 0.1,
                     lambda t: 0]
NUM_RUNS = 1000
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([20, 20])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([40, 90, 70])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 1.0, lambda t: 1.0]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([40, 90, 70])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01 * t]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([40, 90, 40])
FITNESS_FUNCTIONS = [lambda t: 1.0, lambda t: 0.1, lambda t: 1.0 + 0.01 * t]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([1, 1, 1])
FITNESS_FUNCTIONS = [lambda t: 0.2, lambda t: 0.2, lambda t: 0.2]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([500, 500, 500])
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t * t]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)

INITIAL_ALLELE_COUNTS = np.array([500, 500, 500, 500, 500, 500, 500])
FITNESS_FUNCTIONS = [lambda t: t, lambda t: 1.0, lambda t: t * t, lambda t: 3.0, lambda t: 6.0, lambda t: 0.1,
                     lambda t: 0]
NUM_RUNS = 10
run_everything(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS)