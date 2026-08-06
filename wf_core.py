import multiprocessing
import numpy as np
import os
import sys
import warnings

from concurrent.futures import ProcessPoolExecutor
from scipy.special import lambertw
from typing import List, Callable, Iterator, Dict


"""
there are 13 analysis functions (plus some helpers)

here's how the analysis functions are named / organized:

functions that analyze probability of fixation start with PF
fucntions that analyze time to fixation (assuming fixation) start with TF
functions that analyze time to loss (assuming no fixation) start with TL

then they all have a dash

function designed for beneficial allele is then followed by b
function designed for neutral allele is then followed by n
function designed for any allele beneficialness is then followed by an underscore

function designed for giant population is then followed by g
function designed for any population size is then followed by an underscore
(btw I know the scientific word is large but then the l for large and the l for loss would get confusing)

function designed for weak selection is then followed by w
function designed for strong selection is then followed by s
function designed for any selection is then followed by an underscore

fucntion designed for only one mutated allele to start off is followed by 1
function designed for any number of alleles is then followed by an underscore

at the end, a and b are used when there's two functions that are used for the same pop and thus had the same name otherwise

that's it

function that analyzes time for the beneficial allele to go from a concentration 1/(cNe) to 1-1/(cNe) is called TC

the helpers are named for what they do in a straightforwrad way

FFT tests Fisher's fundamental theorem

"""
#note: require wf_core.py to run! this is just the graphing part of the program
warnings.filterwarnings("ignore")

# functions for multiprocessing
use_parallel_core_default: bool = False

def is_notebook():
    return "ipykernel" in sys.modules

def initiate_executor() -> ProcessPoolExecutor:
    if is_notebook():
        # Notebooks need the 'fork' start method to work with ProcessPoolExecutor
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            pass  # Already set
        cfg_context = None
    else:
        # Scripts on Windows/macOS use 'spawn' by default, requiring a context or guard
        cfg_context = multiprocessing.get_context("spawn")

    executor = ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=cfg_context)

    return executor


class ConstantFitness:
    """A picklable, arbitrary-valued stand-in for the named constant_fitness_N
    functions in the original code. Needed because sweeps require many
    distinct fitness values that can't each get a hand-written module-level
    function."""
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, t):
        return self.value

    def __repr__(self):
        return f"ConstantFitness({self.value})"




def wrightfishersampling(K: np.ndarray, Ff: List[Callable[[int], float]], rng=None, seed=None) -> Iterator[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed)

    N = np.sum(K)

    if np.any(K <= 0):
        raise ValueError("Number of alleles cannot be negative")
    if N == 0:
        raise ValueError("error: population number is zero")
    if not isinstance(Ff, list) or not all(callable(x) for x in Ff):
        raise ValueError("Ff must be a list of callable functions")
    if not (K.size == len(Ff)):
        raise ValueError("K and Ff must have the same length")
    if K.size < 1:
        raise ValueError("K must be at least 1")

    t = 0
    #while True:
    while True:
        yield K
        if np.max(K) == N:
            break

        F_values_at_t = np.array([func(t) for func in Ff], dtype=float)
        p_unnormalized = K * F_values_at_t
        sum_p_unnormalized = np.sum(p_unnormalized)

        if sum_p_unnormalized == 0 or np.isnan(sum_p_unnormalized):
            pvals = np.zeros_like(K, dtype=float)
        else:
            pvals = p_unnormalized / sum_p_unnormalized
            pvals /= np.sum(pvals)

        K = rng.multinomial(n=N, pvals=pvals)
        t += 1


def run_single_simulation(initial_K_counts: np.ndarray, Ff: List[Callable[[int], float]], rng=None, seed=None) -> Dict:
    history = np.vstack(list(wrightfishersampling(initial_K_counts, Ff, rng=rng, seed=seed)))
    N = int(initial_K_counts.sum())

    final_t = history.shape[0] - 1
    fixed = (history[-1] == N).any()
    first_allele_fixed = (history[-1][0] == N) if fixed else False

    allele_freq = history / N

    return {
        "fixed": fixed,
        "final_generations": final_t,
        "allele_counts_history": history,
        "allele_frequencies_history": allele_freq,
        "first_allele_fixed": first_allele_fixed,
    }


def run_multiple_simulations(initial_K_counts: np.ndarray,
                             Ff: List[Callable[[int], float]],
                             num_simulations: int,
                             use_parallel_core=use_parallel_core_default,
                             executor: ProcessPoolExecutor=None) -> List[Dict]:


    seed_seq = np.random.SeedSequence()
    child_seeds = seed_seq.spawn(num_simulations)

    if use_parallel_core:
        if executor is None:
            executor = initiate_executor()
            warnings.warn("executor is uninitialized")
        futures = [
            executor.submit(run_single_simulation, initial_K_counts, Ff, seed=seed)
                for seed in child_seeds
        ]  # num_simulations
        results = [future.result() for future in futures]
    else:
        results = [run_single_simulation(initial_K_counts, Ff, seed=seed) for seed in child_seeds]  # num_simulations

    return results


def constant_fitness_1(t):
    return 1.0



def Psi_s(s: float) -> float:
    res = 1 + lambertw(-(1 + s) * np.exp(-(1 + s))).real / (s + 1)
    if isinstance(res, np.ndarray):
        res[np.isnan(res)] = 0.
    else:
        if np.isnan(res):
            res = 0.
    return res

"""
import sympy as sp
Psi, s = sp.symbols('Psi s', real=True)
sp.solve(sp.Eq(1 - Psi, sp.exp(-(1+s) * Psi)), Psi)
[s + sp.LambertW((-s - 1)*sp.exp(-s - 1)) + 1)/(s + 1)]
"""


def stable_ratio(r, s1, s2):
    if not (float(s1).is_integer() and float(s2).is_integer()):
        raise ValueError("s1 and s2 must be integers when r can be negative")
    s1, s2 = int(s1), int(s2)

    if r == 0:
        return 0.
    if r == 1:
        return s1 / (s1 + s2)

    ln_abs_r = np.log(np.abs(r))

    a = 2 * s1 * ln_abs_r
    b = -2 * s2 * ln_abs_r
    m = max(a, b, 0.0)

    num = np.exp(a - m) - np.exp(-m)
    den = np.exp(a - m) - np.exp(b - m)

    if den == 0.0:
        raise ZeroDivisionError("r**s1 == r**-s2; expression is undefined for these s1, s2")

    return num / den


def PFn___(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2

    expected = size1 / N
    results['expected_PFn___'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    first_allele_fixations = [run for run in sim_data if run['first_allele_fixed']]
    p = len(first_allele_fixations) / NUM_RUNS
    results['p'] = p
    results['difference'] = expected - p
    return results


def PFbg_1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    s = (fitness1 - fitness2) / fitness2

    expected = Psi_s(s)
    results['expected_PFbg_1'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    first_allele_fixations = [run for run in sim_data if run['first_allele_fixed']]
    p = len(first_allele_fixations) / NUM_RUNS
    results['p_PFbg_1'] = p
    results['difference'] = expected - p
    return results


def PFbgw1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    s = (fitness1 - fitness2) / fitness2

    expected = 2 * s
    results['expected_PFbgw1'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    first_allele_fixations = [run for run in sim_data if run['first_allele_fixed']]
    p = len(first_allele_fixations) / NUM_RUNS
    results['p'] = p
    results['difference'] = expected - p
    return results


def PFb_w_(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2
    s = (fitness1 - fitness2) / fitness2

    denominator = -np.expm1(-2 * N * s)
    if denominator != 0:
        expected = -np.expm1(-2 * s * size1) / denominator
    else:
        expected = np.nan
    results['expected_PFb_w_'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    first_allele_fixations = [run for run in sim_data if run['first_allele_fixed']]
    p = len(first_allele_fixations) / NUM_RUNS
    results['p'] = p
    results['difference'] = expected - p
    return results


def PF____(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2

    r = fitness1 / fitness2
    expected = stable_ratio(r, size1, size2)
    results['expected_PF____'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    first_allele_fixations = [run for run in sim_data if run['first_allele_fixed']]
    p = len(first_allele_fixations) / NUM_RUNS
    results['p'] = p
    results['difference'] = expected - p
    return results


def TFb_s1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2
    s = (fitness1 - fitness2) / fitness2

    expected = np.nan
    p = size1 / N
    q = size2 / N
    mean_fitness = p * fitness1 + q * fitness2
    Ne = N

    c = 1
    if np.isfinite(Ne) and s != 0 and (c * Ne - 1) > 0 and s > 1 / Ne:
        expected = 2 * np.log(c * Ne - 1) / s
    results['expected_TFb_s1'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    fixed_generations = [run['final_generations'] for run in sim_data if run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


def TFb_w1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    s = (fitness1 - fitness2) / fitness2
    Ne = INITIAL_ALLELE_COUNTS.sum()

    expected = np.nan
    c = 1
    if np.isfinite(Ne) and s != 0 and (c * Ne - 1) > 0 and s > 1 / Ne:
        expected = 2 * np.log(c * Ne - 1) / s
    results['expected_TFb_w1'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    fixed_generations = [run['final_generations'] for run in sim_data if run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


def TFn__1a(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)

    Ne = N
    p = size1 / N
    if p == 0:
        expected = 2 * Ne
    elif p == 1:
        expected = 0.
    else:
        expected = -2 * Ne * (1-p) * np.log1p(-p) / p

    results['expected_TFn__1a'] = expected

    fixed_generations = [run['final_generations'] for run in sim_data if run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


def TFn__1b(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)

    Ne = N
    expected = -2 * N * Ne * (1 - 1 / N) * np.log1p(-1 / N)
    results['expected_TFn__1b'] = expected

    fixed_generations = [run['final_generations'] for run in sim_data if run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


def TFng_1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)

    Ne = N
    expected = 2 * Ne
    results['expected_TFng_1'] = expected
    fixed_generations = [run['final_generations'] for run in sim_data if run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


def TLn__1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)

    Ne = N
    p = size1 / N
    if p == 0:
        expected = 0.
    elif p == 1:
        expected = 2 * Ne
    else:
        expected = -2*Ne*(p/(1-p))*np.log(p)

    results['expected_TLn__1'] = expected

    fixed_generations = [run['final_generations'] for run in sim_data if run['fixed'] and not run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


def TC(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1, size2 = INITIAL_ALLELE_COUNTS[0], INITIAL_ALLELE_COUNTS[1]
    N = size1 + size2
    s = (fitness1 - fitness2) / fitness2

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)

    Ne = N
    c = 1
    if s != 0 and np.isfinite(Ne) and (c * Ne - 1) > 0:
        expected = 2 * np.log(c * Ne - 1) / s
    else:
        expected = np.nan
    results['expected_TC'] = expected

    passage_times = []
    if np.isfinite(Ne) and Ne > 0:
        target_freq_start = 1 / (c * Ne)
        target_freq_end = 1 - 1 / (c * Ne)
        if 0 < target_freq_start < 1 and 0 < target_freq_end < 1 and target_freq_start < target_freq_end:
            for run in sim_data:
                freq_allele_1 = run['allele_frequencies_history'][:, 0]
                start_gen_indices = np.where(freq_allele_1 >= target_freq_start)[0]
                start_gen_idx = start_gen_indices[0] if start_gen_indices.size > 0 else -1
                end_gen_indices = np.where(freq_allele_1 >= target_freq_end)[0]
                end_gen_idx = end_gen_indices[0] if end_gen_indices.size > 0 else -1
                if start_gen_idx != -1 and end_gen_idx != -1 and end_gen_idx > start_gen_idx:
                    passage_times.append(end_gen_idx - start_gen_idx)

    simulated_average_passage_time = np.mean(passage_times) if passage_times else np.nan
    results['simulated_avg_passage_time'] = simulated_average_passage_time

    fixed_generations = [run['final_generations'] for run in sim_data if run['fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


def FFT(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    r1 = FITNESS_FUNCTIONS[0](1)
    r2 = FITNESS_FUNCTIONS[1](1)

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)

    all_differences = []
    for run in sim_data:
        frequencies = run['allele_frequencies_history']
        f1 = frequencies[:, 0]
        mean_fitness = (f1 * r1) + ((1 - f1) * r2)
        fitness_variance = (f1 * (r1 ** 2)) + ((1 - f1) * (r2 ** 2)) - (mean_fitness ** 2)
        delta_mean_fitness = np.diff(mean_fitness)
        variance_at_t = fitness_variance[:-1]
        difference = delta_mean_fitness - variance_at_t
        all_differences.extend(difference)

    difference_in_FFT = float(np.mean(all_differences)) if all_differences else np.nan
    results['difference'] = difference_in_FFT
    return results


ANALYSIS_FUNCTIONS = {
    'PFn___': PFn___, 'PFbg_1': PFbg_1, 'PFbgw1': PFbgw1, 'PFb_w_': PFb_w_, 'PF____': PF____,
    'TFb_w1': TFb_w1, 'TFb_s1': TFb_s1, 'TFn__1a': TFn__1a, 'TFn__1b': TFn__1b,
    'TFng_1': TFng_1, 'TLn__1': TLn__1, 'TC': TC, 'FFT': FFT,
}


ANALYSIS_COLUMN_MAP = {
    'PFn___':  {'observed': 'p',        'predicted': 'expected_PFn___', 'title': 'PF Neutral', 'ylabel': 'Probability of Fixation (allele 1)'},
    'PFbg_1':  {'observed': 'p_PFbg_1', 'predicted': 'expected_PFbg_1', 'title': 'PF Beneficial, Giant Pop (1 allele)', 'ylabel': 'Probability of Fixation (allele 1)'},
    'PFbgw1':  {'observed': 'p',        'predicted': 'expected_PFbgw1', 'title': 'PF Beneficial, Giant Pop, Weak Selection (1 allele)', 'ylabel': 'Probability of Fixation (allele 1)'},
    'PFb_w_':  {'observed': 'p',        'predicted': 'expected_PFb_w_', 'title': 'PF Beneficial, Weak Selection', 'ylabel': 'Probability of Fixation (allele 1)'},
    'PF____':  {'observed': 'p',        'predicted': 'expected_PF____', 'title': 'PF Any Allele (exact)', 'ylabel': 'Probability of Fixation (allele 1)'},
    'TFb_w1':  {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TFb_w1', 'title': 'TF Beneficial, Weak Selection (1 allele)', 'ylabel': 'Generations to Fixation'},
    'TFb_s1':  {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TFb_s1', 'title': 'TF Beneficial, Strong Selection (1 allele)', 'ylabel': 'Generations to Fixation'},
    'TFn__1a': {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TFn__1a', 'title': 'TF Neutral (1 allele, Model A)', 'ylabel': 'Generations to Fixation'},
    'TFn__1b': {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TFn__1b', 'title': 'TF Neutral (1 allele, Model B)', 'ylabel': 'Generations to Fixation'},
    'TFng_1':  {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TFng_1', 'title': 'TF Neutral, Giant Pop (1 allele)', 'ylabel': 'Generations to Fixation'},
    'TLn__1':  {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TLn__1', 'title': 'TL Neutral (1 allele)', 'ylabel': 'Generations to Loss'},
    'TC':      {'observed': 'simulated_avg_passage_time',  'predicted': 'expected_TC', 'title': 'TC Beneficial (1/(cNe) -> 1-1/(cNe) passage time)', 'ylabel': 'Generations (passage time)'},
    'FFT':     {'observed': None, 'predicted': None, 'title': "Fisher's Fundamental Theorem Check", 'ylabel': 'Δ(mean fitness) − Var(fitness)'},
}
