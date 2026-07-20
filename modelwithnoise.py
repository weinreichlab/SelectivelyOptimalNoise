# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional
from collections import deque # import deque for efficient history tracking
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys

def is_notebook():
    return "ipykernel" in sys.modules

sixtyfour_bit_minimum = np.finfo(np.float64).min # float min for real numbers
sixtyfour_bit_maximum = np.finfo(np.float64).max # float max for real numbers


"""

KEY MAJOR DIFFERENCES FROM PREVIOUS MODEL INCORPORATING NOISE:
-genotypes are floats instead of ints
-phenotypes are no longer represented, and genotype directly determines fitness
-fitness is no longer a user input, instead it is computed using the formula from Prof Weinreich's most recent paper,
        so also now wopt and zopt are user inputs (zopt is automatically set to 0.0 but you can change it). if wopt is set
        to an invalid value, i.e. such that wz is negative, an automatic failwith is raised
-there is an option to input constant mu, aka constant noise, and in this case, the amount of noise in the population is
        the same for all individuals at every point in time, and it's whatever you set constant mu to. note that you can
        put None for constant mu, in which case the noise levels will change over time and will be different for different
        individuals (like in previous model)
-the user has the option to make the simulation stop itself whenever the avg fitness has plateaued. what's considered a
        plateau is up to the user b/c they input how conservative they want this to be while calling the simulation;
        the user still has to tell it to stop after a certain number of generations, and if it hasn't plateaued by then
        or the user hasn't selected that plateau option it will stop at that number
-population size can change over time, but if you want the population size to stay constant, just
        set the input asking about constant population size to "True"
-added parallel running so it's much more efficient
-added plots of fitness over time



SUMMARY OF PROGRAM

this program models natural selection on the level of individuals taking into
account biological noise.

individuals are all haploid and each parent makes children alone through cloning
(involving mutation)

population size can vary if you let it

the generations are cyclical, so everyone reproduces (or not) then dies before
the new generation is born

it takes as input the population, which is an array containing z and mu for each original individual

time, t, is measured in generations

all of the population is stored in a matrix:

rows: info about each individual
                                  -amount of noise (float 0 to 1), a probability, also called µ
                                  -genotype (represented by int), also called z
collumns: each individual (unlabelled)


how many offspring a parent has is determined based on the fitness of its
genotype (and also randomness)

our organisms' genotypes are real numbers z

absolute Wrightian fitness of each organism is computed as Wz = Wopt - (z - zopt)2.
Without loss of generality we can set zopt = 0, and the only constraint on wopt is that wz
should never be negative

Wopt = 10 * (100*σ)2

the optimum genotype does not change over time, meaning the environemnt does not change between generations

Each generation, the number of offspring left by all the individuals carrying each genotype z in the
population is multinomially distributed, with probabilities given by the frequency of the genotype multiplied
by its relative Wrightian fitness

both genotypes and amount of noise are passed down directly from parent to child
unless there is a mutation. the more noise, the more likely a mutation.

mutant genotypes are found by adding a normal deviate ∆z = Normal(µ=0,σ2) to the parent's genotype z

noise of 0 means that offspring are always identical to parent

noise of 1 means offspring will always have a mutation

"""


class Population:
    def __init__(self, initial_genotypes: np.ndarray, initial_noise_levels: np.ndarray):
        # initialize population matrix:
        # row 0: amount of noise (float 0 to 1), also called mu
        # row 1: genotype (represented by float), also called z

        size = initial_genotypes.shape[0]

        if initial_genotypes.shape[0] != initial_noise_levels.shape[0]:
            raise ValueError("initial genotypes and noise levels must be same length")

        self.population = np.zeros((2, size), dtype=object) # dtype allows mixed types
        self.population[0, :] = initial_noise_levels.astype(float) # noise (mu)
        self.population[1, :] = initial_genotypes.astype(float)   # genotypes (z)

    def get_fitness(self, Wopt: float, zopt: float = 0.0, max_population_size: int = 1000) -> np.ndarray:
        #this function calculates the fitness of each individual using the Wrightian fitness model

        genotypes = self.population[1, :].astype(float)
        # Wrightian fitness calculation
        fitness_scores = np.array([Wopt - (g - zopt)**2 for g in genotypes])

        if any(score < 0 for score in fitness_scores):
            raise ValueError(
                "the Zopt entered results in a neg Wz. Wz = Wopt - (Z - Zopt)^2 where Wz is never negative"
            )


        # replace any NaN, positive infinity, or negative infinity values
        # positive infinity is capped at max_population_size * 2 to prevent OverflowError during int conversion
        # neg infinity is replaced by 0.0
        fitness_scores = np.nan_to_num(fitness_scores, nan=0.0, posinf=float(max_population_size * 2), neginf=0.0)
        # fitness scores are never below a minimum value (0.0) and are also capped at a maximum
        return np.clip(fitness_scores, 0.0, float(max_population_size * 2))

    def reproduce(self, fitness_scores: np.ndarray, initial_sim_pop_size: int, constant_pop_flag: bool, max_pop_size_limit: int) -> np.ndarray:
        """this function selects parents based on fitness then generates
        offspring to return a new population matrix. the population size varies
        unless constant_pop_flag is True.

        Args:
            fitness_scores: array of fitness scores for  current pop
            initial_sim_pop_size: initial pop size
            constant_pop_flag: if True, the population size will remain constant at initial_sim_pop_size
            max_pop_size_limit: the absolute maximum population size if constant_pop_flag is False

        Returns:
            A new population matrix representing the next generation.
        """
        current_pop_size = self.population.shape[1]

        # if current pop is empty, it cannot reproduce
        if current_pop_size == 0:
            return np.zeros((2, 0), dtype=object)

        if constant_pop_flag:
            new_pop_size = initial_sim_pop_size # population size remains constant
        else:
            # determine new pop size based on overall reproductive potential
            total_reproductive_potential = np.sum(fitness_scores)
            desired_new_pop_size = int(round(total_reproductive_potential))
            # cap new pop size
            new_pop_size = min(desired_new_pop_size, max_pop_size_limit)

        if new_pop_size <= 0:
            return np.zeros((2, 0), dtype=object) # population goes extinct
            # we might write == 0 instead of <= 0 b/c w is never negative, doesn't really matter

        # probabilities for parent selection
        # if all fitness scores are zero or population is empty, prevent division by zero
        # if population exists but all fitness is zero, assign uniform probabilities
        total_fitness = np.sum(fitness_scores)
        if current_pop_size == 0:
            return np.zeros((2, 0), dtype=object)
        elif total_fitness == 0:
            probabilities = np.ones(current_pop_size) / current_pop_size
        else:
            probabilities = fitness_scores / total_fitness

        # ensure probabilities sum to 1, handling potential floating point inaccuracies
        probabilities = probabilities / np.sum(probabilities)

        # use np.random.multinomial to determine offspring counts for each parent
        offspring_counts = np.random.multinomial(new_pop_size, probabilities)

        # create parent_indices by expanding the offspring_counts
        parent_indices = []
        for i, count in enumerate(offspring_counts):
            parent_indices.extend([i] * count)
        parent_indices = np.array(parent_indices) # convert to numpy array

        # np.random.multinomial already takes new_pop_size as number of trials, so len(parent_indices)
        # should already be equal to new_pop_size

        new_population_data = np.zeros((2, new_pop_size), dtype=object)

        # absolute float bounds

        for index, parent_index in enumerate(parent_indices):
            parent_noise = self.population[0, parent_index] # noise level: mu
            parent_genotype = self.population[1, parent_index] # genotype: z

            # genotype mutation step
            mutation_occurs = np.random.rand() < parent_noise # noise (mu) as mutation chance

            if mutation_occurs:
                # mutation occurs, new genotype is parent's + normal deviate
                mutation_sigma = parent_noise # sigma for nrmal distribution is the parent's noise level
                new_genotype = parent_genotype + np.random.normal(loc=0, scale=mutation_sigma)
            else:
                # offspring is clone of parent
                new_genotype = parent_genotype

            new_population_data[0, index] = parent_noise # initialize new_noise here to avoid UnboundLocalError
            new_population_data[1, index] = new_genotype

            # noise mutation step
            mutation_occurs = np.random.rand() < parent_noise # noise (mu) as mutation chance

            if mutation_occurs:
                mutation_sigma = parent_noise # sigma for normal distribution is the parent's noise level
                new_noise = parent_noise + np.random.normal(loc=0, scale=mutation_sigma)
                # note: the above line is a hypothesis, but still need to verify if it holds water!!!!
            else:
                # offspring is clone of parent
                new_noise = parent_noise

            # new genotype stays within absolute float bounds
            new_genotype = np.clip(new_genotype, sixtyfour_bit_minimum, sixtyfour_bit_maximum)
            # new noise stays within [0, 1]
            new_noise = np.clip(new_noise, 0.0, 1.0)

            new_population_data[0, index] = new_noise
            new_population_data[1, index] = new_genotype

        # a new Population object is created directly with new gen's data
        # pass numpy arrays diretly
        new_generation_pop = Population(new_population_data[1, :].astype(float), new_population_data[0, :].astype(float))

        return new_generation_pop.population

    def evolve(self, generations: int, Wopt: float, zopt: float = 0.0, max_population_size: int = 1000,
               stop_on_fitness_plateau: bool = False, plateau_window_size: int = 50, plateau_threshold: float = 0.001,
               constant_mu: Optional[float] = None, constant_population_size: bool = False, initial_pop_size: int = 0):
        # simulates evolution over lots of generations
        history = [] # store population states over time
        mean_fitness_history = deque(maxlen=plateau_window_size) # history for plateau detection

        for t in range(generations):
            # if constant_mu is set, override the noise levels for ALL individuals in current generation
            if constant_mu is not None:
                self.population[0, :] = constant_mu

            fitness_scores = self.get_fitness(Wopt, zopt, max_population_size)

            # calculate mean fitness
            if len(fitness_scores) > 0:
                current_mean_fitness = np.mean(fitness_scores)
            else:
                current_mean_fitness = 0.0 # pop went extinct or is empty

            mean_fitness_history.append(current_mean_fitness)

            current_gen_data = {
                'generation': t,
                'genotypes': self.population[1, :].copy(),
                'noise_levels': self.population[0, :].copy(),
                'mean_fitness': current_mean_fitness
            }
            history.append(current_gen_data)

            # check for plateau if asked for in input and enough data is available
            if stop_on_fitness_plateau and len(mean_fitness_history) == plateau_window_size:
                # check the range of mean fitness within window
                if (max(mean_fitness_history) - min(mean_fitness_history)) < plateau_threshold:
                    print(f"\nPlateau detected at generation {t}, so stopping simulation early")
                    break

            self.population = self.reproduce(fitness_scores, initial_pop_size, constant_population_size, max_population_size)

        return history


def run_single_simulation(
    pop_size: int,
    initial_genotypes_generator: Callable[[int], np.ndarray],
    initial_noise_levels_generator: Callable[[int], np.ndarray],
    generations_to_simulate: int,
    Wopt: float, # Wopt  for fitness calculation
    zopt: float = 0.0, # zopt for fitness calculation
    plot: bool = True,
    plot_title_suffix: str = "",
    max_population_size: int = 1000,
    stop_on_fitness_plateau: bool = False,
    plateau_window_size: int = 50, # num of generations to check for plateau
    plateau_threshold: float = 0.001, # max change to consider a plateau
    constant_mu: Optional[float] = None,
    constant_population_size: bool = False #if True, pop size remains constant at initial pop_size
) -> Tuple[List[dict], bool, int]:
    """runs one evolutionary simulation, plots avg genotype and noise, and checks fixation

    input:
        pop_size
        initial_genotypes_generator
        initial_noise_levels_generator
        generations_to_simulate
        Wopt: optimal fitness value in the Wrightian fitness model
        zopt: optimal genotype value in the Wrightian fitness model
        plot: whether or not to display plots
        plot_title_suffix: optional suffix to add to the plot titles
        max_population_size
        stop_on_fitness_plateau: if True, stop the simulation when mean fitness plateaus
        plateau_window_size: num of recent generations to consider for plateau detection
        plateau_threshold: Tmax absolute difference in mean fitness over the window to consider it a plateau
        constant_mu: if not None, the noise level (mu) will be fixed to this value for all individuals for all generations
        constant_population_size: if True, the population size will be forever constant at initial pop_size

    output:
        tuple containing:
            - evolution_history: list of dictionaries, each describing a generation.
            - fixed: boolean indicating if fixation occurred
            - fixation_gen: generation at which fixation occurred, or -1 if no fixation
    """

    initial_genotypes = initial_genotypes_generator(pop_size)

    if constant_mu is not None:
        initial_noise_levels_for_pop = np.full(pop_size, constant_mu)
    else:
        initial_noise_levels_for_pop = initial_noise_levels_generator(pop_size)

    my_population = Population(initial_genotypes, initial_noise_levels_for_pop)

    evolution_history = my_population.evolve(
        generations_to_simulate, Wopt, zopt, max_population_size,
        stop_on_fitness_plateau, plateau_window_size, plateau_threshold,
        constant_mu=constant_mu,
        constant_population_size=constant_population_size,
        initial_pop_size=pop_size
    )

    fixed = False
    fixation_gen = -1
    for gen_data in evolution_history:
        genotypes_at_gen = gen_data['genotypes']
        # check if pop is not empty before checking fixation
        if len(genotypes_at_gen) > 0 and len(np.unique(genotypes_at_gen)) == 1:
            fixed = True
            fixation_gen = gen_data['generation']
            break # found fixation

    if plot:
        # check evolution_history is not empty before accessing its last element
        if evolution_history:
            final_genotypes = evolution_history[-1]['genotypes']
            final_noise_levels = evolution_history[-1]['noise_levels']
            actual_generations_run = evolution_history[-1]['generation'] # actual last generation recorded
        else:
            final_genotypes = np.array([])
            final_noise_levels = np.array([])
            actual_generations_run = 0 # history is empty

        # plotting avg genotype over generations
        avg_genotypes = [np.mean(gen_data['genotypes']) if len(gen_data['genotypes']) > 0 else np.nan for gen_data in evolution_history]
        # get average fitness levels
        avg_fitness_levels = [gen_data['mean_fitness'] for gen_data in evolution_history] # directly use stored mean_fitness

        plt.figure(figsize=(18, 5)) # increased figure size to accommodate 3 plots

        # Subplot 1: Average Genotype
        plt.subplot(1, 3, 1)
        plt.plot([h['generation'] for h in evolution_history], avg_genotypes)
        plt.xlabel("Generation")
        plt.ylabel("Average Genotype")
        plt.title(f"Evolution of Average Genotype{plot_title_suffix}")
        plt.grid(True)

        # Subplot 2: Average Noise Levels
        avg_noise_levels = [np.mean(gen_data['noise_levels']) if len(gen_data['noise_levels']) > 0 else np.nan for gen_data in evolution_history]
        plt.subplot(1, 3, 2)
        plt.plot([h['generation'] for h in evolution_history], avg_noise_levels, color='orange')
        plt.xlabel("Generation")
        plt.ylabel("Average Noise Level")
        plt.title(f"Evolution of Average Noise Level{plot_title_suffix}")
        plt.grid(True)
        if constant_mu is not None:
            # adjust y-axis for noise plot to clearly show the constant_mu value (before it looked like mu was always zero b/c of the scale)
            plt.ylim(constant_mu * 0.9, constant_mu * 1.1) # set limits around constant_mu
            print(f"Single simulation (constant_mu={constant_mu}) average noise levels: {avg_noise_levels[:5]}...{avg_noise_levels[-5:]}")

        # Subplot 3: Average Fitness
        plt.subplot(1, 3, 3)
        plt.plot([h['generation'] for h in evolution_history], avg_fitness_levels, color='red') # red for fitness
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness (W)")
        plt.title(f"Evolution of Average Fitness{plot_title_suffix}")
        plt.grid(True)


        plt.tight_layout() # adjust layout to prevent overlapping titles/labels
        plt.show()

    return evolution_history, fixed, fixation_gen

def run_multiple_simulations_parallel(
    pop_size: int,
    initial_genotypes_generator: Callable[[int], np.ndarray],
    initial_noise_levels_generator: Callable[[int], np.ndarray],
    generations_to_simulate: int,
    Wopt: float, # Wopt parameter for fitness calculation
    zopt: float = 0.0, # zopt parameter for fitness calculation
    num_simulations: int = 500,
    max_population_size: int = 1000,
    stop_on_fitness_plateau: bool = False,
    plateau_window_size: int = 50, # num of generations to check for plateau
    plateau_threshold: float = 0.001, # max change to consider a plateau
    constant_mu: Optional[float] = None,
    constant_population_size: bool = False
    ):
    if is_notebook():
        # notebooks need the 'fork' start method to work with ProcessPoolExecutor
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            pass  # already set
        cfg_context = None
    else:
        # scripts on Windows/macOS use 'spawn' by default, requiring a context or guard
        cfg_context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(mp_context=cfg_context) as executor:
       futures = [
           executor.submit(
               run_single_simulation,
               pop_size,
               initial_genotypes_generator,
               initial_noise_levels_generator,
               generations_to_simulate,
               Wopt,
               zopt,
               plot=False, # explicitly set plot to False for parallel runs
               plot_title_suffix="", # optional, but good to be explicit
               max_population_size=max_population_size,
               stop_on_fitness_plateau=stop_on_fitness_plateau,
               plateau_window_size=plateau_window_size,
               plateau_threshold=plateau_threshold,
               constant_mu=constant_mu,
               constant_population_size=constant_population_size
           )
           for _ in range(num_simulations)
       ]
       results = [future.result() for future in futures]
    return results

def run_multiple_simulations(
    pop_size: int,
    initial_genotypes_generator: Callable[[int], np.ndarray],
    initial_noise_levels_generator: Callable[[int], np.ndarray],
    generations_to_simulate: int,
    Wopt: float, # Wopt parameter for fitness calculation
    zopt: float = 0.0, # zopt parameter for fitness calculation
    num_simulations: int = 500,
    max_population_size: int = 1000,
    stop_on_fitness_plateau: bool = False,
    plateau_window_size: int = 50, # how many generations to check for plateau
    plateau_threshold: float = 0.001, # max change to consider a plateau
    constant_mu: Optional[float] = None,
    constant_population_size: bool = False
    ):

    # runs multiple simulations, reports fixation statistics, and plots trends


    results = run_multiple_simulations_parallel(
        pop_size,
        initial_genotypes_generator,
        initial_noise_levels_generator,
        generations_to_simulate,
        Wopt,
        zopt,
        num_simulations,
        max_population_size,
        stop_on_fitness_plateau,
        plateau_window_size,
        plateau_threshold,
        constant_mu,
        constant_population_size
        )


    all_evolution_histories = [run_result[0] for run_result in results]

    # determine maximum length for padding
    max_len = max([len(history) for history in all_evolution_histories]) if all_evolution_histories else 0
    generations_axis_plot = list(range(max_len))

    # prepare lists to hold average genotypes, noise levels, and fitness for each generation across all runs
    all_avg_genotypes_per_run = []
    all_avg_noise_levels_per_run = []
    all_avg_fitness_levels_per_run = []

    for history in all_evolution_histories:
        avg_genotypes_this_run = [np.mean(gen_data['genotypes']) if len(gen_data['genotypes']) > 0 else np.nan for gen_data in history]
        avg_noise_levels_this_run = [np.mean(gen_data['noise_levels']) if len(gen_data['noise_levels']) > 0 else np.nan for gen_data in history]
        avg_fitness_levels_this_run = [gen_data['mean_fitness'] for gen_data in history] # extract mean fitness

        all_avg_genotypes_per_run.append(avg_genotypes_this_run)
        all_avg_noise_levels_per_run.append(avg_noise_levels_this_run)
        all_avg_fitness_levels_per_run.append(avg_fitness_levels_this_run) # apppend fitness

    # pad runs for averaging
    padded_genotypes_runs = np.array([np.pad(run_data, (0, max_len - len(run_data)), 'constant', constant_values=np.nan) for run_data in all_avg_genotypes_per_run])
    padded_noise_runs = np.array([np.pad(run_data, (0, max_len - len(run_data)), 'constant', constant_values=np.nan) for run_data in all_avg_noise_levels_per_run])
    padded_fitness_runs = np.array([np.pad(run_data, (0, max_len - len(run_data)), 'constant', constant_values=np.nan) for run_data in all_avg_fitness_levels_per_run]) # pad fitness

    # plotting overall averages
    plt.figure(figsize=(18, 5))

    # average genotype plot
    plt.subplot(1, 3, 1)
    for run_data in padded_genotypes_runs:
        plt.plot(generations_axis_plot, run_data, color='blue', linewidth=0.1)
    overall_avg_genotypes = np.nanmean(padded_genotypes_runs, axis=0)
    if len(generations_axis_plot) > 0:
        plt.plot(generations_axis_plot, overall_avg_genotypes, color='green', linewidth=1, label='Mean Genotype')
    plt.xlabel("Generation")
    plt.ylabel("Average Genotype")
    plt.title("Overall Evolution of Average Genotype (Multiple Runs)")
    plt.grid(True)
    plt.legend()

    # avg noise amount plot
    plt.subplot(1, 3, 2)
    for run_data in padded_noise_runs:
        plt.plot(generations_axis_plot, run_data, color='orange', linewidth=0.1)
    overall_avg_noise_levels = np.nanmean(padded_noise_runs, axis=0)
    if len(generations_axis_plot) > 0:
        plt.plot(generations_axis_plot, overall_avg_noise_levels, color='red', linewidth=1, label='Mean Noise Level')
    plt.xlabel("Generation")
    plt.ylabel("Average Noise Level")
    plt.title("Overall Evolution of Average Noise Level (Multiple Runs)")
    plt.grid(True)
    plt.legend()

    if constant_mu is not None:
        # adjust y-axis for noise plot to clearly show the constant_mu value
        plt.ylim(constant_mu * 0.9, constant_mu * 1.1) # set limits around constant_mu
        print(f"Multiple simulations (constant_mu={constant_mu}) overall average noise levels: {overall_avg_noise_levels[:5]}...{overall_avg_noise_levels[-5:]}")

    # subplot 3: Overall Average Fitness
    plt.subplot(1, 3, 3)
    for run_data in padded_fitness_runs:
        plt.plot(generations_axis_plot, run_data, color='purple', linewidth=0.1)
    overall_avg_fitness_levels = np.nanmean(padded_fitness_runs, axis=0)
    if len(generations_axis_plot) > 0:
        plt.plot(generations_axis_plot, overall_avg_fitness_levels, color='darkgreen', linewidth=1, label='Mean Fitness (W)')
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness (W)")
    plt.title("Overall Evolution of Average Fitness (Multiple Runs)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# EXAMPLES

# define how to generate initial genotypes and noise levels for default simulation
def default_initial_genotypes(size):
    return np.random.uniform(-10.0, 10.0, size=size)

def default_initial_noise_levels(size):
    return np.random.uniform(0.01, 0.2, size=size)

print("\nExample 1: single simulation with mutation in amount of noise")
_ = run_single_simulation(
    pop_size=100,
    initial_genotypes_generator=default_initial_genotypes,
    initial_noise_levels_generator=default_initial_noise_levels,
    generations_to_simulate=200,
    Wopt=100.0,
    plot=True,
    max_population_size=1000,
    stop_on_fitness_plateau=True,
    plateau_window_size=20,
    plateau_threshold=0.001,
    constant_population_size=False
)

print("\nExample 2: multiple simulation with mutation in amount of noise")
_ = run_multiple_simulations(
    pop_size=100,
    initial_genotypes_generator=default_initial_genotypes,
    initial_noise_levels_generator=default_initial_noise_levels,
    generations_to_simulate=200,
    Wopt=100.0,
    num_simulations=10,
    max_population_size=1000,
    stop_on_fitness_plateau=False,
    plateau_window_size=20,
    plateau_threshold=0.005,
    constant_mu=None,
    constant_population_size=False
)

print("\nExample 3: single simulation with constant noise")
_ = run_single_simulation(
    pop_size=100,
    initial_genotypes_generator=default_initial_genotypes,
    initial_noise_levels_generator=default_initial_noise_levels, # this generator will be ignored b/c constant_mu is set
    generations_to_simulate=200,
    Wopt=100.0,
    plot=True,
    max_population_size=1000,
    stop_on_fitness_plateau=True,
    plateau_window_size=20,
    plateau_threshold=0.001,
    constant_mu=0.05,
    constant_population_size=True
)

print("\nExample 4: multiple simulation with constant noise")
_ = run_multiple_simulations(
    pop_size=100,
    initial_genotypes_generator=default_initial_genotypes,
    initial_noise_levels_generator=default_initial_noise_levels,# this generator will be ignored b/c constant_mu is set
    generations_to_simulate=200,
    Wopt=100.0,
    num_simulations=10,
    max_population_size=1000,
    stop_on_fitness_plateau=False,
    plateau_window_size=20,
    plateau_threshold=0.005,
    constant_mu=0.1,
    constant_population_size=True
)
