import multiprocessing
import numpy as np
import os
import sys
import warnings

from concurrent.futures import ProcessPoolExecutor
from scipy.special import lambertw
from typing import List, Callable, Iterator, Dict

from scipy.integrate import quad
from scipy.stats import binom


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

S_MODE="log"


from functools import lru_cache
from scipy.stats import binom

EXACT_N_MAX = 2000   # exact chain is O(N^3); 2000 takes a few seconds


@lru_cache(maxsize=None)
def tloss_exact_wf(N, i0=1, s=0.0):
    """Exact mean time to loss, conditional on loss, for YOUR discrete haploid WF model."""
    i = np.arange(N + 1)
    x = i * (1 + s) / (i * (1 + s) + (N - i))          # same as your pvals
    P = binom.pmf(i[None, :], N, x[:, None])
    Q = P[1:N, 1:N]
    I = np.eye(N - 1)
    v = np.linalg.solve(I - Q, P[1:N, 0])              # probability of loss from each state
    Qc = Q * v[None, :] / v[:, None]                   # chain conditioned on loss
    T = np.linalg.solve(I - Qc, np.ones(N - 1))
    return T[i0 - 1]


def tloss_diffusion(N, i0=1):
    """Kimura & Ohta (1969) eq. 16, haploid translation (4Ne -> 2N)."""
    p = i0 / N
    return -2 * N * (p / (1 - p)) * np.log(p)


@lru_cache(maxsize=None)
def _tloss_discreteness_correction(i0=1):
    return tloss_exact_wf(EXACT_N_MAX, i0) - tloss_diffusion(EXACT_N_MAX, i0)


def tloss_neutral_prediction(N, i0=1):
    if N <= EXACT_N_MAX:
        return tloss_exact_wf(N, i0)
    return tloss_diffusion(N, i0) + _tloss_discreteness_correction(i0)

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




def exact_wf_conditional_tfix(N, s, i0=1):
    """Exact mean time to fixation (conditioned on fixation) of the discrete WF chain."""
    i = np.arange(N + 1)
    x = i * (1 + s) / (i * (1 + s) + (N - i))        # same as the sim's pvals
    P = binom.pmf(i[None, :], N, x[:, None])         # P[i, j]
    Q, R = P[1:N, 1:N], P[1:N, N]
    I = np.eye(N - 1)
    u = np.linalg.solve(I - Q, R)                    # fixation probabilities
    Qc = Q * u[None, :] / u[:, None]                 # chain conditioned on fixation
    T = np.linalg.solve(I - Qc, np.ones(N - 1))      # expected steps to absorption
    return T[i0 - 1]

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




def diffusion_s(fitness1, fitness2, mode=S_MODE):
    if mode == "log":
        return float(np.log(fitness1 / fitness2))
    return (fitness1 - fitness2) / fitness2


def tfix_kimura_ohta(N, s, i0=1, include_J2=True):
    """
    Mean time to fixation, conditional on fixation, for a haploid WF population
    (c = 1, Ne = N).

    Kimura & Ohta (1969) eq. 17:  t1(p) = J1 + (1-u(p))/u(p) * J2
    Otto & Whitlock eq. 5 is the J1 term alone, with p = 1/(cNe)
    (set include_J2=False to reproduce eq. 5 exactly).

    Uses the substitution y = logit(x), so dx = x(1-x) dy. This cancels the
    1/(x(1-x)) in both integrands and leaves bounded, overflow-free integrands.
    """
    p = i0 / N
    if s == 0:
        # Kimura & Ohta eq. 14 translated to haploid (4Ne -> 2N)
        return -2 * N * (1 - p) * np.log1p(-p) / p

    # The conditional fixation time is symmetric in s (Otto & Whitlock;
    # Maruyama & Kimura 1974), so deleterious alleles use |s|.
    s = abs(s)
    a = 2 * N * s                 # 2S in Kimura & Ohta, 2cNe*s in Otto & Whitlock
    den = -np.expm1(-a)           # 1 - e^{-2S}

    # J1 integrand: (e^{ax}-1)(e^{a(1-x)}-1) / (e^a - 1), rewritten stably
    def j1(y):
        x, one_minus_x = expit(y), expit(-y)
        return (-np.expm1(-a * x)) * (-np.expm1(-a * one_minus_x)) / den

    # J2 integrand: (e^{ax}-1)(1-e^{-ax}) / (1 - e^{-a})
    def j2(y):
        x = expit(y)
        return np.expm1(a * x) * (-np.expm1(-a * x)) / den

    y_p = np.log(p) - np.log1p(-p)                 # logit(p)
    y_hi = max(y_p + 1.0, np.log(a) + 10.0)        # breakpoint helps quad when a is huge
    J1 = (quad(j1, y_p, y_hi, limit=500)[0] + quad(j1, y_hi, np.inf, limit=500)[0]) / s
    if not include_J2:
        return J1

    J2 = quad(j2, -np.inf, y_p, limit=500)[0] / s
    u = -np.expm1(-a * p) / den                    # fixation probability, eq. 3 / K&O u(p)
    return J1 + (1 - u) / u * J2


def TFb_s1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    N = int(INITIAL_ALLELE_COUNTS.sum())
    s = diffusion_s(fitness1, fitness2)    # ln(1+s) by default; see section 2

    c, Ne = 1, N                           # haploid simulator
    if s > 0 and (c * Ne - 1) > 0:
        expected = 2 * np.log(c * Ne - 1) / s
    else:
        expected = np.nan
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
    size1 = int(INITIAL_ALLELE_COUNTS[0])
    N = int(INITIAL_ALLELE_COUNTS.sum())
    s = diffusion_s(fitness1, fitness2)

    try:
        expected = tfix_kimura_ohta(N, s, i0=size1, include_J2=True)
    except Exception:
        expected = np.nan
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
        expected = -2 * Ne * (1-p) * np.log1p(-p) / p # equation 14

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







TC_K = 10.0   # window starts at x1 = K/(N*s); larger K -> more deterministic, narrower window


def TC(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1 = int(INITIAL_ALLELE_COUNTS[0])
    N = int(INITIAL_ALLELE_COUNTS.sum())
    s = diffusion_s(fitness1, fitness2)            # ln(f1/f2), matches your WF update
    c, Ne = 1, N

    # Literal eq. 6 (window 1/(cNe) -> 1-1/(cNe)), kept for reference only
    results['expected_TC_eq6_literal'] = 2 * np.log(c * Ne - 1) / s if (s > 0 and c * Ne > 2) else np.nan

    # Eq. 6 applied to a drift-safe window
    expected, x1, x2 = np.nan, np.nan, np.nan
    if s > 0:
        x1 = TC_K / (c * Ne * s)
        x2 = 1 - x1
        if x1 < 0.5 and size1 / N < x1:            # window must exist and start above the initial freq
            expected = (np.log(x2 / (1 - x2)) - np.log(x1 / (1 - x1))) / s
    results['expected_TC'] = expected
    results['tc_window'] = (x1, x2)

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS,
                                        use_parallel_core=use_parallel_core, executor=executor)

    passage_times = []
    if np.isfinite(expected):
        for run in sim_data:
            f = run['allele_frequencies_history'][:, 0]
            hits_end = np.flatnonzero(f >= x2)
            if hits_end.size == 0:
                continue                                  # allele never reached x2 (lost)
            end = hits_end[0]
            below = np.flatnonzero(f[:end] < x1)
            if below.size == 0:
                continue
            start = below[-1] + 1                         # last upward crossing of x1 before reaching x2
            passage_times.append(end - start)

    sim_passage = np.mean(passage_times) if passage_times else np.nan
    results['simulated_avg_passage_time'] = sim_passage
    results['n_passages'] = len(passage_times)
    results['difference'] = expected - sim_passage       # compare passage time to passage time
    return results


EXACT_CHAIN_MAX_N = 2500   # dense (N x N) solve; beyond this it gets slow / memory hungry


def TF__w1(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1, size2 = int(INITIAL_ALLELE_COUNTS[0]), int(INITIAL_ALLELE_COUNTS[1])
    N = size1 + size2
    s = (fitness1 - fitness2) / fitness2

    expected = tfix_kimura_ohta(N, s, size1)
    results['expected_TF__w1'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    fixed_generations = [run['final_generations'] for run in sim_data if run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results



from scipy.integrate import quad
from scipy.special import expit

# "log":   s_eff = ln(f1/f2)      -> matches the diffusion to your discrete WF update (recommended)
# "paper": s     = (f1 - f2)/f2   -> literal s from the papers
S_MODE = "log"



def TF____(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False, executor: ProcessPoolExecutor=None) -> List[Dict]:
    results = {}
    fitness1 = FITNESS_FUNCTIONS[0](1)
    fitness2 = FITNESS_FUNCTIONS[1](1)
    size1, size2 = int(INITIAL_ALLELE_COUNTS[0]), int(INITIAL_ALLELE_COUNTS[1])
    N = size1 + size2
    s = (fitness1 - fitness2) / fitness2

    if N <= EXACT_CHAIN_MAX_N:
        expected = exact_wf_conditional_tfix(N, s, size1)
    else:
        expected = np.nan
    results['expected_TF____'] = expected

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core=use_parallel_core, executor=executor)
    fixed_generations = [run['final_generations'] for run in sim_data if run['first_allele_fixed']]
    simulated_average_fixation_time = np.mean(fixed_generations) if fixed_generations else np.nan
    results['simulated_avg_fixation_time'] = simulated_average_fixation_time
    results['difference'] = expected - simulated_average_fixation_time
    return results


from scipy.stats import ttest_1samp, norm


def fft_terms(history: np.ndarray, Ff: List[Callable[[int], float]], N: int):
    """
    Per-generation quantities for Fisher's fundamental theorem, for one run.

    For each transition t -> t+1 (fitness evaluated at time t, exactly as in
    wrightfishersampling), returns:
      delta      : observed change in mean fitness, w̄(p_{t+1}) - w̄(p_t),
                   both measured with the generation-t fitness values
                   (Fisher's 'change due to natural selection')
      predicted  : Var_t(w) / w̄_t          (exact FFT, discrete generations)
      naive      : Var_t(w)                (the form used in the old FFT)
      drift_var  : Var[Δw̄ | p_t] = Var_{p'}(w) / N   (multinomial sampling noise)
    Works for any number of alleles.
    """
    T = history.shape[0] - 1
    if T == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty

    p = history[:-1] / N                       # (T, K) frequencies before selection
    p_next = history[1:] / N                   # (T, K) frequencies after sampling
    W = np.array([[f(t) for f in Ff] for t in range(T)], dtype=float)   # (T, K)

    wbar = np.sum(p * W, axis=1)
    var_w = np.sum(p * W**2, axis=1) - wbar**2
    var_w = np.maximum(var_w, 0.0)             # guard tiny negative round-off

    delta = np.sum(p_next * W, axis=1) - wbar
    predicted = var_w / wbar

    p_sel = p * W / wbar[:, None]              # expected frequencies after selection
    m_sel = np.sum(p_sel * W, axis=1)          # equals wbar + predicted
    drift_var = np.maximum(np.sum(p_sel * W**2, axis=1) - m_sel**2, 0.0) / N

    return delta, predicted, var_w, drift_var


def FFT(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS, use_parallel_core: bool=False,
        executor: ProcessPoolExecutor=None, alpha: float=0.01) -> Dict:
    """
    Tests whether the simulation obeys Fisher's fundamental theorem:
        E[Δw̄ | p_t] = Var_t(w) / w̄_t

    Test 1 (integrated, independent runs): per run, R = Σ_t (Δw̄_t - Var_t/w̄_t).
        R is a stopped martingale, so E[R] = 0 exactly. One-sample t-test across runs.
    Test 2 (per generation): z_t = (Δw̄_t - Var_t/w̄_t) / sqrt(Var[Δw̄ | p_t]).
        E[z] = 0 tests selection; E[z²] = 1 tests that drift is binomial/multinomial.
    """
    results = {}
    N = int(np.sum(INITIAL_ALLELE_COUNTS))

    sim_data = run_multiple_simulations(INITIAL_ALLELE_COUNTS, FITNESS_FUNCTIONS, NUM_RUNS,
                                        use_parallel_core=use_parallel_core, executor=executor)

    run_sums, deltas, preds, naives, zs = [], [], [], [], []
    for run in sim_data:
        d, pr, nv, dv = fft_terms(run['allele_counts_history'], FITNESS_FUNCTIONS, N)
        if d.size == 0:
            continue
        run_sums.append(np.sum(d - pr))
        deltas.append(d); preds.append(pr); naives.append(nv)
        mask = dv > 1e-15                      # zero only if all fitnesses are equal
        zs.append((d[mask] - pr[mask]) / np.sqrt(dv[mask]))

    if not deltas:
        results.update(difference=np.nan, passed=False)
        return results

    delta = np.concatenate(deltas)
    pred = np.concatenate(preds)
    naive = np.concatenate(naives)
    z = np.concatenate(zs) if zs else np.empty(0)
    run_sums = np.asarray(run_sums)

    # Pooled summaries (kept for compatibility with your sweep/plot code)
    results['difference'] = float(np.mean(delta - pred))           # should be ≈ 0
    results['difference_naive'] = float(np.mean(delta - naive))    # old test: Δw̄ − Var
    results['mean_delta_wbar'] = float(np.mean(delta))
    results['mean_predicted'] = float(np.mean(pred))
    results['n_generations'] = int(delta.size)

    # Test 1: integrated residual per run (independent samples)
    results['run_sum_mean'] = float(np.mean(run_sums))
    if run_sums.size > 1 and np.std(run_sums) > 0:
        results['run_sum_pvalue'] = float(ttest_1samp(run_sums, 0.0).pvalue)
    else:
        results['run_sum_pvalue'] = 1.0       # identical runs (e.g. neutral: all residuals 0)

    # Test 2: standardized per-generation residuals
    if z.size > 1:
        results['z_mean'] = float(np.mean(z))
        results['z_mean_pvalue'] = float(2 * norm.sf(abs(np.mean(z)) * np.sqrt(z.size)))
        z2 = z**2
        results['z_var'] = float(np.mean(z2))                         # should be ≈ 1
        z2_se = np.std(z2, ddof=1) / np.sqrt(z.size)
        results['z_var_pvalue'] = float(2 * norm.sf(abs(np.mean(z2) - 1) / z2_se)) if z2_se > 0 else 1.0
    else:
        results.update(z_mean=np.nan, z_mean_pvalue=1.0, z_var=np.nan, z_var_pvalue=1.0)

    results['passed'] = bool(results['run_sum_pvalue'] > alpha and
                             results['z_mean_pvalue'] > alpha and
                             results['z_var_pvalue'] > alpha)
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
    'FFT': {'observed': None, 'predicted': None, 'title': "Fisher's Fundamental Theorem Check",
        'ylabel': 'Standardized residual E[z] (should be 0)'},
}

ANALYSIS_FUNCTIONS.update({'TF__w1': TF__w1, 'TF____': TF____})

ANALYSIS_COLUMN_MAP.update({
    'TF__w1': {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TF__w1',
               'title': 'TF Any Allele, Weak Selection (Kimura–Ohta, 1 allele)', 'ylabel': 'Generations to Fixation'},
    'TF____': {'observed': 'simulated_avg_fixation_time', 'predicted': 'expected_TF____',
               'title': 'TF Any Allele (exact Markov chain)', 'ylabel': 'Generations to Fixation'},
})
