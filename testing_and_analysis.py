# %% [markdown]
# # Wright–Fisher simulation vs. theory: probability and time of fixation
#
# | Graph | x-axis | y-axis | line style / marker = series | colour = formula |
# |---|---|---|---|---|
# | 1 | selection coefficient *s* | P(fixation of allele 1) | population size *N* | PF formulas |
# | 2 | population size *N* | P(fixation of allele 1) | selection coefficient *s* | PF formulas |
# | 3 | selection coefficient *s* | mean generations to fixation (given fixation) | *N* | TF formulas |
# | 4 | population size *N* | mean generations to fixation (given fixation) | *s* | TF formulas |
#
# Each graph is drawn four times, once per starting count: 1 individual, 1 %, 5 % and 10 % of N.
#
# **Starting frequency is held exactly constant.** A percentage start is only used at population sizes
# where it is a whole number of individuals:
# - 1 % → N a multiple of 100
# - 5 % → N a multiple of 20
# - 10 % → N a multiple of 10
#
# Without this restriction, rounding i₀ makes the starting frequency jump around and the curves become jagged.
#
# **Panels**
# - The upper panel shows raw values: thin coloured formula curves and grey simulation dots (± 1 SE).
# - The lower panel shows % deviation from the exact Markov chain on a stretched axis
#   (linear near 0, logarithmic further out).
#
# **Model** (identical to `wrightfishersampling` in `wf_core.py`): haploid WF, $w_1 = 1+s$, $w_2 = 1$,
# $i' \sim \mathrm{Binomial}\!\left(N,\ \frac{i(1+s)}{i(1+s)+N-i}\right)$.

# %%
import sys
import os
import time
import pickle
import warnings
from fractions import Fraction

IN_NOTEBOOK = "ipykernel" in sys.modules

import matplotlib
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import lambertw, expit
from scipy.integrate import quad
from scipy.stats import binom

warnings.filterwarnings("ignore")

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

# %% [markdown]
# ## 1. Configuration

# %%
SEED = 20240601
QUICK_TEST = False
USE_PARALLEL = True
N_JOBS = -1

OUT_DIR = os.path.join(os.getcwd(), "output_overlays")
CACHE_FILE = os.path.join(OUT_DIR, "sim_cache.pkl")
REUSE_CACHE = True
SHOW_INDIVIDUAL_FIGURES = False

START_COUNTS = {
    "1 individual":      ("count", 1),
    "1% of population":  ("frac", 0.01),
    "5% of population":  ("frac", 0.05),
    "10% of population": ("frac", 0.10),
}

# Graphs 2 & 4 (x = N): range and number of simulated / theory points
POP_N_MIN, POP_N_MAX = 10, 3000
POP_SIM_POINTS = 16
THEORY_POINTS_N = 90
POP_SWEEP_S = [0.0, 0.01, 0.1, 1.0]

# Graphs 1 & 3 (x = s): N values must hold every percentage start exactly (multiples of 100)
SEL_SWEEP_S = np.concatenate([[-0.02, -0.01, -0.005, -0.002], [0.0], np.geomspace(1e-3, 2.0, 18)])
SEL_SWEEP_N = [100, 300, 3000]
THEORY_POINTS_S = 160

# Noise control
REPLICATE_RULE = "target_rse"
TARGET_RSE = 0.1
TF_CV = 0.6
LINEAR_RUNS_PER_INDIVIDUAL = 50
MIN_RUNS = 500
MAX_RUNS = 2_000_000
MIN_EXPECTED_FIXATIONS = 5
SIM_CHUNK = 250_000

EXACT_N_MAX = 2000

if QUICK_TEST:
    POP_N_MAX = 2000
    POP_SIM_POINTS, THEORY_POINTS_N = 6, 25
    SEL_SWEEP_S = np.array([-0.01, 0.0, 0.003, 0.03, 0.3, 2.0])
    SEL_SWEEP_N = [100, 300]
    THEORY_POINTS_S = 40
    TARGET_RSE = 0.25
    MIN_RUNS, MAX_RUNS = 100, 50_000
    EXACT_N_MAX = 500

os.makedirs(OUT_DIR, exist_ok=True)


def start_fraction(spec):
    return Fraction(spec[1]).limit_denominator(10_000)


def n_step(spec):
    """Population sizes must be multiples of this so that the start is a whole number of individuals."""
    return 1 if spec[0] == "count" else start_fraction(spec).denominator


def initial_count(N, spec):
    """Exact starting count of allele 1, or None if N cannot hold the requested start exactly."""
    N = int(N)
    if spec[0] == "count":
        i0 = int(spec[1])
    else:
        fr = start_fraction(spec)
        if (N * fr.numerator) % fr.denominator:
            return None
        i0 = N * fr.numerator // fr.denominator
    return i0 if 1 <= i0 < N else None


def snapped_grid(lo, hi, n, step):
    lo = step * int(np.ceil(lo / step))
    hi = step * int(np.floor(hi / step))
    if hi < lo:
        return np.array([], dtype=int)
    return np.unique(np.round(np.geomspace(lo, hi, n) / step).astype(int) * step)


def pop_grid(spec, n_points):
    g = snapped_grid(POP_N_MIN, POP_N_MAX, n_points, n_step(spec))
    return np.array([N for N in g if initial_count(N, spec) is not None], dtype=int)


def dense_s(n):
    s = np.asarray(SEL_SWEEP_S, float)
    parts = [s, [0.0]]
    if s.max() > 0:
        parts.append(np.geomspace(min(1e-4, s[s > 0].min()), s.max(), n))
    if s.min() < 0:
        parts.append(-np.geomspace(min(1e-4, -s[s < 0].max()), -s.min(), max(n // 5, 10)))
    return np.unique(np.concatenate(parts).astype(float))


POP_SIM_N = {lab: pop_grid(spec, POP_SIM_POINTS) for lab, spec in START_COUNTS.items()}
POP_TH_N = {lab: np.union1d(pop_grid(spec, THEORY_POINTS_N), POP_SIM_N[lab]) for lab, spec in START_COUNTS.items()}
THEORY_S = dense_s(THEORY_POINTS_S)

for lab, spec in START_COUNTS.items():
    bad = [N for N in SEL_SWEEP_N if initial_count(N, spec) is None]
    if bad:
        print(f"Note: N = {bad} cannot hold '{lab}' exactly; those series are omitted from that panel.")

# %% [markdown]
# ## 2. Theory
#
# All formulas come from `wf_core.py`.
#
# **Fixation probability**
# - Exact WF Markov chain
# - Kimura diffusion `PFb_w_`
# - `PF____` (diffusion with $s_e = \ln(1+s)$)
# - Neutral $i_0/N$ (`PFn___`)
# - Branching process $1-(1-\Psi)^{i_0}$ (`PFbg_1`)
# - Haldane $2s$ (`PFbgw1`)
#
# **Conditional fixation time**
# - Exact chain
# - Kimura–Ohta eq. 17 with $s_e$ (`TFb_w1`) and with raw $s$ (`TF__w1`)
# - Neutral (`TFn__1a` = `TFn__1b` at $i_0=1$)
# - $2N$ (`TFng_1`)
# - Logistic sweep (`TFb_s1`)

# %%
def fix_prob_diffusion(N, i0, alpha):
    if alpha == 0:
        return i0 / N
    if alpha > 0:
        return float(np.expm1(-alpha * i0) / np.expm1(-alpha * N))
    b = -alpha
    return float(np.exp(-b * (N - i0)) * np.expm1(-b * i0) / np.expm1(-b * N))


def psi_branching(s):
    if s <= 0:
        return 0.0
    w = lambertw(-(1 + s) * np.exp(-(1 + s)), k=0).real
    return float(1 + w / (1 + s))


def tfix_neutral(N, i0):
    p = i0 / N
    return float(-2 * N * (1 - p) * np.log1p(-p) / p)


def tfix_logistic(N, i0, s):
    if s <= 0:
        return np.nan
    se = np.log1p(s)
    p0, p1 = i0 / N, 1 - 1 / N
    if p0 >= p1:
        return np.nan
    return float((np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))) / se)


def tfix_kimura_ohta(N, s, i0=1):
    """Kimura & Ohta (1969) eq. 17, haploid; J2 term rearranged to avoid overflow."""
    p = i0 / N
    if s == 0:
        return tfix_neutral(N, i0)
    s = abs(s)
    a = 2 * N * s
    den = -np.expm1(-a)

    def j1(y):
        x, omx = expit(y), expit(-y)
        return (-np.expm1(-a * x)) * (-np.expm1(-a * omx)) / den

    def j2c(y):
        x = expit(y)
        return np.exp(a * (x - p)) * np.expm1(-a * x) ** 2 / den

    y_p = np.log(p) - np.log1p(-p)
    y_hi = max(y_p + 1.0, np.log(a) + 10.0)
    J1 = (quad(j1, y_p, y_hi, limit=500)[0] + quad(j1, y_hi, np.inf, limit=500)[0]) / s
    w = min(1.0, 30.0 / (a * p * (1 - p)))
    I2 = quad(j2c, y_p - w, y_p, limit=500)[0] + quad(j2c, -np.inf, y_p - w, limit=500)[0]
    u = -np.expm1(-a * p) / den
    C = -np.expm1(-a * (1 - p)) / den
    return float(J1 + C * I2 / (u * s))


def _safe(f, *args):
    try:
        v = f(*args)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def exact_wf_chain(N, s):
    """Exact fixation probability and conditional fixation time for every i0 at once."""
    i = np.arange(N + 1)
    x = i * (1 + s) / (i * (1 + s) + (N - i))
    P = binom.pmf(i[None, :], N, x[:, None])
    Q, R = P[1:N, 1:N], P[1:N, N]
    I = np.eye(N - 1)
    u = np.clip(np.linalg.solve(I - Q, R), 0.0, 1.0)
    with np.errstate(all="ignore"):
        Qc = Q * u[None, :] / u[:, None]
        T = np.linalg.solve(I - Qc, np.ones(N - 1))
    T = np.where(np.isfinite(T) & (u > 1e-300), T, np.nan)
    return u, T


def analytic_predictions(N, s, i0):
    pf = {
        "kimura":    fix_prob_diffusion(N, i0, 2 * s),
        "ratio":     fix_prob_diffusion(N, i0, 2 * np.log1p(s)),
        "neutral":   i0 / N if s == 0 else np.nan,
        "branching": 1 - (1 - psi_branching(s)) ** i0 if s > 0 else np.nan,
        "haldane":   2 * s if (i0 == 1 and 0 < s <= 0.5) else np.nan,
    }
    tf = {
        "ko_log":  _safe(tfix_kimura_ohta, N, np.log1p(s), i0),
        "ko_raw":  _safe(tfix_kimura_ohta, N, s, i0),
        "neutral": tfix_neutral(N, i0) if s == 0 else np.nan,
        "giant":   2.0 * N if (s == 0 and i0 == 1) else np.nan,
        "strong":  tfix_logistic(N, i0, s),
    }
    return {"pf": pf, "tf": tf}


def _analytic_batch(keys):
    return [analytic_predictions(*k) for k in keys]


def attach_exact(k, ana, exact):
    N, s, i0 = k
    ex = exact.get((N, s))
    pf_ex = float(ex[0][i0 - 1]) if ex is not None else np.nan
    tf_ex = float(ex[1][i0 - 1]) if ex is not None else np.nan
    return {"pf": {"exact": pf_ex, **ana["pf"]}, "tf": {"exact": tf_ex, **ana["tf"]}}

# %% [markdown]
# ## 3. Holding noise constant
#
# For each simulated point, the number of replicates is chosen before simulating, from the predicted
# fixation probability $p$, so that every point has the same relative standard error $\varepsilon$:
#
# $$n=\max\!\left(\frac{1-p}{p\,\varepsilon^{2}},\ \frac{\mathrm{CV}_T^{2}}{p\,\varepsilon^{2}}\right)$$
#
# For a neutral allele this gives $n \approx N/(i_0\varepsilon^2)$, i.e. replicates proportional to $N$.
#
# **Caps and skipped points**
# - Points that hit `MAX_RUNS` are drawn as hollow markers (noise target not met).
# - Points expected to give fewer than `MIN_EXPECTED_FIXATIONS` fixations are not simulated.

# %%
def choose_num_runs(N, i0, p_pred):
    if REPLICATE_RULE == "linear_in_N":
        n = LINEAR_RUNS_PER_INDIVIDUAL * N
    else:
        p = float(np.clip(p_pred, 1e-300, 1.0))
        n = max((1 - p) / (p * TARGET_RSE ** 2), TF_CV ** 2 / (p * TARGET_RSE ** 2))
    capped = n > MAX_RUNS
    return int(np.clip(np.ceil(n), MIN_RUNS, MAX_RUNS)), bool(capped)

# %% [markdown]
# ## 4. Simulator — your original generation-by-generation simulation (`wf_core.py`)
#
# **Default engine (`SIM_ENGINE = "wf_core"`)**
# - Every replicate is one call to `wf_core.run_single_simulation`.
# - That call iterates `wrightfishersampling` one generation at a time (multinomial sampling)
#   until one allele is fixed.
# - Per run, exactly as in your `PF…` / `TF…` functions:
#   - fixation of allele 1 = `run["first_allele_fixed"]`
#   - time to fixation = `run["final_generations"]`
#
# Runs are processed in batches across CPU cores. Only the per-run outcome is kept, not the history,
# so millions of runs fit in memory.
#
# **Alternative engine (`SIM_ENGINE = "vectorised"`)**
# - Same Wright–Fisher model, but all replicates of a point are stepped through the generations together.
# - Much faster; results agree statistically.

# %%
WF_CORE_DIR = os.getcwd()     # folder containing wf_core.py
SIM_ENGINE = "wf_core"        # "wf_core" (your original simulator) or "vectorised"
RUNS_PER_TASK = 20_000        # replicates given to one worker at a time

if WF_CORE_DIR not in sys.path:
    sys.path.insert(0, WF_CORE_DIR)
import wf_core                # fails here if wf_core.py is not found
from wf_core import run_single_simulation, ConstantFitness


def point_entropy(k):
    """Seed material tied to the point itself, so results don't depend on which other points exist."""
    N, s, i0 = k
    return [SEED, int(N), int(i0), int(round(s * 1e9)) & 0xFFFFFFFF, int(s < 0)]


def summarize(n_runs, n_fix, t_sum, t_sq):
    p_hat = n_fix / n_runs
    p_se = np.sqrt(p_hat * (1 - p_hat) / n_runs) if n_fix > 0 else np.nan
    if n_fix > 0:
        t_mean = t_sum / n_fix
        t_var = max(t_sq / n_fix - t_mean ** 2, 0.0) * n_fix / max(n_fix - 1, 1)
        t_se = np.sqrt(t_var / n_fix) if n_fix > 1 else np.nan
    else:
        t_mean = t_se = np.nan
    return dict(n_runs=n_runs, n_fix=n_fix, p_hat=p_hat, p_se=p_se, t_mean=t_mean, t_se=t_se)


def _simulate_batch_wf_core(N, s, i0, entropy, start, count, wf_core_dir):
    """
    Runs `count` independent replicates with YOUR wf_core.run_single_simulation, one after another.
    Replicate j uses child seed j of the point's SeedSequence (same as SeedSequence(entropy).spawn(n)[j]).
    """
    import sys
    if wf_core_dir not in sys.path:
        sys.path.insert(0, wf_core_dir)
    import numpy as np
    from wf_core import run_single_simulation, ConstantFitness

    K = np.array([i0, N - i0])
    Ff = [ConstantFitness(1.0 + s), ConstantFitness(1.0)]
    n_fix, t_sum, t_sq = 0, 0.0, 0.0
    for j in range(start, start + count):
        seed = np.random.SeedSequence(entropy, spawn_key=(j,))
        run = run_single_simulation(K, Ff, seed=seed)      # full trajectory, generation by generation
        if run["first_allele_fixed"]:
            t = run["final_generations"]
            n_fix += 1
            t_sum += t
            t_sq += float(t) * t
    return n_fix, t_sum, t_sq


def simulate_point(N, s, i0, n_runs, seed, chunk=250_000):
    """Vectorised engine: same WF model, all replicates stepped through the generations together."""
    rng = np.random.default_rng(seed)
    w = 1.0 + s
    n_fix, t_sum, t_sq = 0, 0.0, 0.0
    remaining = n_runs
    while remaining > 0:
        m = min(remaining, chunk)
        remaining -= m
        cur = np.full(m, i0, dtype=np.int64)
        t = 0
        while cur.size:
            t += 1
            wi = cur * w
            cur = rng.binomial(N, wi / (wi + (N - cur)))
            fixed = cur == N
            done = fixed | (cur == 0)
            if done.any():
                nf = int(np.count_nonzero(fixed))
                if nf:
                    n_fix += nf
                    t_sum += nf * t
                    t_sq += nf * float(t) * t
                cur = cur[~done]
    return summarize(n_runs, n_fix, t_sum, t_sq)


def run_parallel(func, arg_tuples):
    if not arg_tuples:
        return []
    if USE_PARALLEL and HAVE_JOBLIB and len(arg_tuples) > 1:
        return Parallel(n_jobs=N_JOBS, verbose=5)(delayed(func)(*a) for a in arg_tuples)
    return [func(*a) for a in arg_tuples]


def run_simulations(keys, n_runs_of):
    tic = time.time()
    if SIM_ENGINE == "wf_core":
        tasks, owners = [], []
        for k in keys:
            n = n_runs_of[k]
            for start in range(0, n, RUNS_PER_TASK):
                tasks.append((k[0], k[1], k[2], point_entropy(k), start,
                              min(RUNS_PER_TASK, n - start), WF_CORE_DIR))
                owners.append(k)
        print(f"wf_core engine: {len(tasks)} batches of up to {RUNS_PER_TASK:,} runs")
        out = run_parallel(_simulate_batch_wf_core, tasks)
        acc = {k: [0, 0.0, 0.0] for k in keys}
        for k, (nf, ts, tq) in zip(owners, out):
            acc[k][0] += nf
            acc[k][1] += ts
            acc[k][2] += tq
        res = {k: summarize(n_runs_of[k], *acc[k]) for k in keys}
    elif SIM_ENGINE == "vectorised":
        out = run_parallel(simulate_point,
                           [(k[0], k[1], k[2], n_runs_of[k], np.random.SeedSequence(point_entropy(k)), SIM_CHUNK)
                            for k in keys])
        res = dict(zip(keys, out))
    else:
        raise ValueError(f"unknown SIM_ENGINE {SIM_ENGINE!r}")
    for r in res.values():
        r["engine"] = SIM_ENGINE
    print(f"Simulated {len(keys)} points with engine '{SIM_ENGINE}' in {time.time() - tic:.1f} s")
    return res


# %% [markdown]
# ## 5. Run everything (theory on dense grid → replicate plan → simulations)

# %%
t_start = time.time()

POINT_KEYS, THEORY_KEYS = set(), set()
for lab, spec in START_COUNTS.items():
    for s in POP_SWEEP_S:
        for N in POP_SIM_N[lab]:
            POINT_KEYS.add((int(N), float(s), initial_count(N, spec)))
        for N in POP_TH_N[lab]:
            THEORY_KEYS.add((int(N), float(s), initial_count(N, spec)))
    for N in SEL_SWEEP_N:
        i0 = initial_count(N, spec)
        if i0 is None:
            continue
        for s in SEL_SWEEP_S:
            POINT_KEYS.add((int(N), float(s), i0))
        for s in THEORY_S:
            THEORY_KEYS.add((int(N), float(s), i0))
POINT_KEYS = sorted(POINT_KEYS)
ALL_KEYS = sorted(THEORY_KEYS | set(POINT_KEYS))

EXACT, RESULTS, ANALYTIC = {}, {}, {}
if REUSE_CACHE and os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as fh:
        cache = pickle.load(fh)
    EXACT, RESULTS, ANALYTIC = cache.get("exact", {}), cache.get("results", {}), cache.get("analytic", {})
    print(f"Loaded cache: {len(EXACT)} exact chains, {len(ANALYTIC)} analytic points, {len(RESULTS)} simulated points")

# 5a. exact Markov chains (theory only — never used as simulation data)
exact_todo = sorted({(N, s) for (N, s, _) in ALL_KEYS if N <= EXACT_N_MAX} - set(EXACT))
print(f"Exact chains to compute: {len(exact_todo)}")
EXACT.update(zip(exact_todo, run_parallel(exact_wf_chain, exact_todo)))

# 5b. closed-form / integral formulas on the dense grid
todo_a = [k for k in ALL_KEYS if k not in ANALYTIC]
print(f"Analytic points to compute: {len(todo_a)}")
chunks = [todo_a[i:i + 200] for i in range(0, len(todo_a), 200)]
for chunk, res in zip(chunks, run_parallel(_analytic_batch, [(c,) for c in chunks])):
    ANALYTIC.update(zip(chunk, res))

PREDICTIONS = {k: attach_exact(k, ANALYTIC[k], EXACT) for k in ALL_KEYS}

# 5c. replicate plan (theory decides only HOW MANY runs; the results come from the runs)
PLAN = {}
for k in POINT_KEYS:
    N, s, i0 = k
    p_pred = PREDICTIONS[k]["pf"]["exact"]
    if not np.isfinite(p_pred):
        p_pred = PREDICTIONS[k]["pf"]["ratio"]
    n, capped = choose_num_runs(N, i0, p_pred)
    PLAN[k] = dict(p_pred=p_pred, n_runs=n, capped=capped,
                   skip=(n * p_pred < MIN_EXPECTED_FIXATIONS))

# 5d. simulations — cached results from a different engine are re-run
todo = [k for k in POINT_KEYS
        if not PLAN[k]["skip"]
        and (k not in RESULTS
             or RESULTS[k].get("engine") != SIM_ENGINE
             or RESULTS[k]["n_runs"] != PLAN[k]["n_runs"])]
todo.sort(key=lambda k: -PLAN[k]["n_runs"] * (1 + PLAN[k]["p_pred"] * k[0]))   # big jobs first
print(f"Points: {len(POINT_KEYS)} | to simulate: {len(todo)} | "
      f"skipped (unreachable): {sum(p['skip'] for p in PLAN.values())} | "
      f"capped: {sum(p['capped'] and not p['skip'] for p in PLAN.values())} | "
      f"total replicates: {sum(PLAN[k]['n_runs'] for k in todo):,}")

RESULTS.update(run_simulations(todo, {k: PLAN[k]["n_runs"] for k in todo}))
for k in POINT_KEYS:
    if k in RESULTS:
        RESULTS[k]["capped"] = PLAN[k]["capped"]

with open(CACHE_FILE, "wb") as fh:
    pickle.dump({"exact": EXACT, "results": RESULTS, "analytic": ANALYTIC}, fh)
print(f"Done in {time.time() - t_start:.1f} s")

# %% [markdown]
# ## 6. Plotting helpers
#
# - **Colour = formula** (very thin lines).
# - **Line style and marker shape = series.**
# - **Simulation** = grey markers with ± 1 SE and no connecting lines.
#
# **Lower-panel reference:** each series uses a single reference along its whole length, so the
# deviation curves never jump.
# - If the exact chain exists for the series, it is the reference. Above `EXACT_N_MAX` the lower
#   panel is shaded and left empty.
# - If the exact chain exists nowhere in the series (e.g. N = 3000), the diffusion formula is the reference.

# %%
plt.rcParams["lines.scale_dashes"] = False

FORMULA_LW = 0.6
SIM_COLOR = "#333333"
SIM_MS = 4.5
SIM_ALPHA = 0.6
HIDE_FORMULAS = {"pf": set(), "tf": set()}   # e.g. {"pf": {"haldane"}, "tf": {"ko_raw"}}

RESID_LINTHRESH = 1e-4
RESID_YMAX = 10.0
REF_FALLBACK = {"pf": "ratio", "tf": "ko_log"}

SERIES_LINESTYLES = ["-", (0, (6, 3)), (0, (1.5, 2)), (0, (8, 3, 1.5, 3)), (0, (12, 4)), (0, (3, 1.5, 1, 1.5, 1, 1.5))]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P"]
SERIES_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]   # diagnostic plot only

FORMULAS = {
    "pf": {
        "exact":     dict(color="#D55E00", label=f"Exact WF Markov chain (N ≤ {EXACT_N_MAX}) — reference of lower panel"),
        "kimura":    dict(color="#0072B2", label="Kimura diffusion (1−e^(−2s·i₀))/(1−e^(−2Ns))  [PFb_w_]"),
        "ratio":     dict(color="#009E73", label="(1−r^(−2i₀))/(1−r^(−2N)), r = 1+s  [PF____]"),
        "neutral":   dict(color="#CC79A7", label="Neutral i₀/N  [PFn___]  (hollow marker at s = 0)"),
        "branching": dict(color="#E69F00", label="Branching process 1−(1−Ψ(s))^i₀  [PFbg_1; i₀>1 extended]"),
        "haldane":   dict(color="#56B4E9", label="Haldane 2s  [PFbgw1; i₀ = 1, 0 < s ≤ 0.5]"),
    },
    "tf": {
        "exact":   dict(color="#D55E00", label=f"Exact WF Markov chain, given fixation (N ≤ {EXACT_N_MAX}) — reference of lower panel"),
        "ko_log":  dict(color="#0072B2", label="Kimura–Ohta eq. 17, s_e = ln(1+s)  [TFb_w1]"),
        "ko_raw":  dict(color="#009E73", label="Kimura–Ohta eq. 17, s as given  [TF__w1]"),
        "neutral": dict(color="#CC79A7", label="Neutral −2N(1−p)ln(1−p)/p  [TFn__1a = TFn__1b]  (hollow marker at s = 0)"),
        "giant":   dict(color="#E69F00", label="Giant-pop neutral 2N  [TFng_1; i₀ = 1]  (hollow marker)"),
        "strong":  dict(color="#56B4E9", label="Strong selection, logistic sweep  [TFb_s1; general i₀]"),
    },
}
YLABEL = {"pf": "Probability of fixation of allele 1",
          "tf": "Mean generations to fixation\n(given fixation)"}
QTITLE = {"pf": "Probability of fixation", "tf": "Time to fixation"}


def sweep_layout(sweep, start_label):
    spec = START_COUNTS[start_label]
    if sweep == "pop":
        return dict(series=list(POP_SWEEP_S),
                    x_sim=np.asarray(POP_SIM_N[start_label], float),
                    x_th=np.asarray(POP_TH_N[start_label], float),
                    key_of=lambda x, sv: (int(round(x)), float(sv)),
                    xlabel="Population size N",
                    series_title="Selection coefficient\n(line style / marker)",
                    series_label=lambda sv: f"s = {sv:g}")
    return dict(series=[N for N in SEL_SWEEP_N if initial_count(N, spec) is not None],
                x_sim=np.asarray(SEL_SWEEP_S, float), x_th=np.asarray(THEORY_S, float),
                key_of=lambda x, sv: (int(sv), float(x)),
                xlabel="Selection coefficient s = (w₁ − w₂) / w₂",
                series_title="Population size\n(line style / marker)",
                series_label=lambda sv: f"N = {sv}  (i₀ = {initial_count(sv, spec)})"
                                        + ("  [lower-panel ref = diffusion]" if sv > EXACT_N_MAX else ""))


def keys_for(L, xs, sv, spec):
    out = []
    for xv in xs:
        N, s = L["key_of"](xv, sv)
        out.append((N, s, initial_count(N, spec)))
    return out


def series_reference(quantity, keys, mode=None):
    """One reference per series: the exact chain if it exists anywhere in the series, else the diffusion."""
    ex = np.array([PREDICTIONS[k][quantity]["exact"] for k in keys], float)
    ex = np.where(np.isfinite(ex) & (ex > 0), ex, np.nan)
    if mode is None:
        mode = "exact" if np.isfinite(ex).any() else "fallback"
    if mode == "exact":
        return ex, mode
    fb = np.array([PREDICTIONS[k][quantity][REF_FALLBACK[quantity]] for k in keys], float)
    return np.where(np.isfinite(fb) & (fb > 0), fb, np.nan), mode


def _resid_ticks():
    lo = int(np.round(np.log10(RESID_LINTHRESH)))
    hi = int(np.floor(np.log10(RESID_YMAX)))
    pos = [10.0 ** e for e in range(lo, hi + 1)]
    neg = [-d for d in pos if d <= 1.0]
    ticks = sorted(neg + [0.0] + pos)
    labels = ["0" if t == 0 else f"{t * 100:+.4g}%" for t in ticks]
    return ticks, labels


def draw_panel(ax, axr, quantity, sweep, start_label):
    spec = START_COUNTS[start_label]
    L = sweep_layout(sweep, start_label)
    x_th, x_sim = L["x_th"], L["x_sim"]
    used, series_handles, sim_vals = [], [], []

    for ci, sv in enumerate(L["series"]):
        ls = SERIES_LINESTYLES[ci % len(SERIES_LINESTYLES)]
        mk = SERIES_MARKERS[ci % len(SERIES_MARKERS)]
        th_keys = keys_for(L, x_th, sv, spec)
        sim_keys = keys_for(L, x_sim, sv, spec)
        ref_th, mode = series_reference(quantity, th_keys)
        ref_s, _ = series_reference(quantity, sim_keys, mode)

        # ---------------- theory: colour = formula, line style = series ----------------
        for fk, st in FORMULAS[quantity].items():
            if fk in HIDE_FORMULAS.get(quantity, ()):
                continue
            y = np.array([PREDICTIONS[k][quantity][fk] for k in th_keys], float)
            m = np.isfinite(y) & (y > 0)
            if not m.any():
                continue
            if fk not in used:
                used.append(fk)
            rel = y / ref_th - 1.0
            if m.sum() == 1:     # formula defined at a single x
                ax.plot(x_th[m], y[m], ls="none", marker=mk, ms=10, mfc="none",
                        mec=st["color"], mew=1.0, zorder=4)
                if np.isfinite(rel[m]).all():
                    axr.plot(x_th[m], rel[m], ls="none", marker=mk, ms=10, mfc="none",
                             mec=st["color"], mew=1.0, zorder=4)
            else:                # NaN breaks the line instead of bridging gaps
                ax.plot(x_th, np.where(m, y, np.nan), ls=ls, lw=FORMULA_LW, color=st["color"], zorder=3)
                axr.plot(x_th, np.where(m, rel, np.nan), ls=ls, lw=FORMULA_LW, color=st["color"], zorder=3)

        # ---------------- simulation: markers + error bars only ----------------
        obs, err, cap = [], [], []
        for k in sim_keys:
            r = RESULTS.get(k) if not PLAN[k]["skip"] else None
            if r is None:
                obs.append(np.nan); err.append(np.nan); cap.append(False)
                continue
            v, e = (r["p_hat"], r["p_se"]) if quantity == "pf" else (r["t_mean"], r["t_se"])
            obs.append(v); err.append(e); cap.append(r.get("capped", False))
        obs = np.array(obs, float)
        err = np.nan_to_num(np.array(err, float))
        cap = np.array(cap, bool)
        m = np.isfinite(obs) & (obs > 0)
        sim_vals.extend(obs[m])
        rel_obs, rel_err = obs / ref_s - 1.0, err / ref_s
        lo = np.minimum(err, 0.95 * obs)

        for capped_flag in (False, True):
            mm = m & (cap == capped_flag)
            if not mm.any():
                continue
            mfc = "white" if capped_flag else SIM_COLOR
            ax.errorbar(x_sim[mm], obs[mm], yerr=[lo[mm], err[mm]], fmt="none", ecolor=SIM_COLOR,
                        elinewidth=0.6, capsize=1.5, alpha=SIM_ALPHA, zorder=5)
            ax.plot(x_sim[mm], obs[mm], ls="none", marker=mk, ms=SIM_MS, mfc=mfc, mec=SIM_COLOR,
                    mew=0.8, alpha=SIM_ALPHA, zorder=6)
            mr = mm & np.isfinite(rel_obs)
            if mr.any():
                axr.errorbar(x_sim[mr], rel_obs[mr], yerr=rel_err[mr], fmt="none", ecolor=SIM_COLOR,
                             elinewidth=0.6, capsize=1.5, alpha=SIM_ALPHA, zorder=5)
                axr.plot(x_sim[mr], rel_obs[mr], ls="none", marker=mk, ms=SIM_MS, mfc=mfc,
                         mec=SIM_COLOR, mew=0.8, alpha=SIM_ALPHA, zorder=6)

        series_handles.append(Line2D([], [], color="k", ls=ls, lw=1.0, marker=mk, ms=5,
                                     mfc=SIM_COLOR, mec=SIM_COLOR, label=L["series_label"](sv)))

    # ---------------- axes ----------------
    if sweep == "pop":
        ax.set_xscale("log")
        if x_th.size and x_th.max() > EXACT_N_MAX:
            ax.axvline(EXACT_N_MAX, color="grey", ls=":", lw=0.8)
            axr.axvspan(EXACT_N_MAX, x_th.max() * 1.05, color="0.88", zorder=0)
            axr.text(EXACT_N_MAX * 1.05, 0.95, f"no exact chain\n(N > {EXACT_N_MAX})",
                     transform=axr.get_xaxis_transform(), fontsize=7, va="top", color="0.35")
    else:
        ax.set_xscale("symlog", linthresh=1e-3, linscale=0.6)
        for a in (ax, axr):
            a.axvline(0, color="grey", lw=0.5, alpha=0.5)

    ax.set_yscale("log")
    if quantity == "pf" and sim_vals:
        ax.set_ylim(0.4 * min(sim_vals), 1.5)
    ax.set_ylabel(YLABEL[quantity])
    ax.tick_params(labelbottom=False)
    if spec[0] == "frac":
        sub = f"  (i₀ = {spec[1]:g}·N exactly; N in steps of {n_step(spec)})"
    else:
        sub = f"  (i₀ = {spec[1]})"
    ax.set_title(f"Start: {start_label}{sub}", fontsize=10)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.3)

    axr.set_yscale("symlog", linthresh=RESID_LINTHRESH, linscale=1.0)
    axr.set_ylim(-1.05, RESID_YMAX)
    ticks, labels = _resid_ticks()
    axr.set_yticks(ticks)
    axr.set_yticklabels(labels, fontsize=7)
    axr.axhline(0, color="k", lw=0.5)
    axr.set_ylabel("deviation from\nexact chain", fontsize=9)
    axr.set_xlabel(L["xlabel"])
    axr.grid(True, which="major", ls="--", lw=0.4, alpha=0.3)
    return used, series_handles, L["series_title"]


def formula_handles(quantity, used):
    hs = [Line2D([], [], color=st["color"], lw=3, label=st["label"])
          for fk, st in FORMULAS[quantity].items() if fk in used]
    target = f"target RSE {TARGET_RSE:.0%}" if REPLICATE_RULE == "target_rse" else f"{LINEAR_RUNS_PER_INDIVIDUAL}·N runs"
    hs.append(Line2D([], [], color=SIM_COLOR, ls="none", marker="o", alpha=SIM_ALPHA,
                     label=f"Simulation ± 1 SE ({target}); marker shape = series"))
    hs.append(Line2D([], [], color=SIM_COLOR, ls="none", marker="o", mfc="white",
                     label=f"Simulation, run cap {MAX_RUNS:,} reached (noise target not met)"))
    return hs


def make_graph(number, quantity, sweep):
    vs = "selection coefficient" if sweep == "sel" else "population size"
    stub = f"{number}_{quantity}_vs_{'s' if sweep == 'sel' else 'N'}"
    labels = list(START_COUNTS)

    fig = plt.figure(figsize=(17, 17))
    outer = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.95, bottom=0.14, hspace=0.22, wspace=0.2)
    all_used = []
    for idx, lab in enumerate(labels):
        inner = outer[idx // 2, idx % 2].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        used, sh, stitle = draw_panel(ax, axr, quantity, sweep, lab)
        all_used += [u for u in used if u not in all_used]
        ax.legend(handles=sh, title=stitle, fontsize=8, title_fontsize=8, loc="best", handlelength=4.5)
    fig.legend(handles=formula_handles(quantity, all_used), loc="lower center", ncol=2, fontsize=9,
               title="Colour = formula", title_fontsize=9)
    fig.suptitle(f"Graph {number}: {QTITLE[quantity]} vs. {vs} — simulation vs. theory", fontsize=14)
    fig.savefig(os.path.join(OUT_DIR, f"{stub}_overview.png"), dpi=160)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

    for j, lab in enumerate(labels, 1):
        f, (ax, axr) = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
        used, sh, stitle = draw_panel(ax, axr, quantity, sweep, lab)
        leg1 = ax.legend(handles=sh, title=stitle, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                         fontsize=8, handlelength=4.5)
        ax.add_artist(leg1)
        ax.legend(handles=formula_handles(quantity, used), loc="upper left", bbox_to_anchor=(1.01, 0.45),
                  fontsize=8, title="Colour = formula", title_fontsize=8)
        ax.set_title(f"{QTITLE[quantity]} vs. {vs} — start: {lab}", fontsize=11)
        path = os.path.join(OUT_DIR, f"{stub}_{j}_{lab.split()[0].replace('%', 'pct')}.png")
        f.savefig(path, dpi=220, bbox_inches="tight")
        if IN_NOTEBOOK and SHOW_INDIVIDUAL_FIGURES:
            plt.show()
        plt.close(f)

# %% [markdown]
# ## 7. Graph 1 — Probability of fixation vs. selection coefficient

# %%
make_graph(1, "pf", "sel")

# %% [markdown]
# ## 8. Graph 2 — Probability of fixation vs. population size

# %%
make_graph(2, "pf", "pop")

# %% [markdown]
# ## 9. Graph 3 — Time to fixation vs. selection coefficient

# %%
make_graph(3, "tf", "sel")

# %% [markdown]
# ## 10. Graph 4 — Time to fixation vs. population size

# %%
make_graph(4, "tf", "pop")

# %% [markdown]
# ## 11. Diagnostic — was the noise actually held constant?

# %%
markers = ["o", "s", "^", "D"]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, q in zip(axes, ["pf", "tf"]):
    for li, (lab, spec) in enumerate(START_COUNTS.items()):
        for si, s in enumerate(POP_SWEEP_S):
            Ns, rse = [], []
            for N in POP_SIM_N[lab]:
                k = (int(N), float(s), initial_count(N, spec))
                r = RESULTS.get(k)
                if r is None or PLAN[k]["skip"]:
                    continue
                v, e = (r["p_hat"], r["p_se"]) if q == "pf" else (r["t_mean"], r["t_se"])
                if np.isfinite(v) and np.isfinite(e) and v > 0:
                    Ns.append(N); rse.append(e / v)
            ax.plot(Ns, rse, ls="none", marker=markers[li], ms=4, color=SERIES_COLORS[si], alpha=0.8)
    if REPLICATE_RULE == "target_rse":
        ax.axhline(TARGET_RSE, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Population size N"); ax.set_ylabel("Achieved relative SE")
    ax.set_title(f"{QTITLE[q]}: noise per point")
    ax.grid(True, which="both", ls="--", alpha=0.3)
handles = [Line2D([], [], color=SERIES_COLORS[i], lw=4, label=f"s = {s:g}") for i, s in enumerate(POP_SWEEP_S)]
handles += [Line2D([], [], color="k", ls="none", marker=markers[i], label=lab) for i, lab in enumerate(START_COUNTS)]
fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT_DIR, "diagnostic_noise.png"), dpi=130)
if IN_NOTEBOOK:
    plt.show()
plt.close(fig)

skipped = [k for k in POINT_KEYS if PLAN[k]["skip"]]
if skipped:
    print(f"{len(skipped)} points not simulated (expected < {MIN_EXPECTED_FIXATIONS} fixations even at "
          f"{MAX_RUNS:,} runs), e.g.: " + ", ".join(f"(N={N}, s={s:g}, i0={i})" for N, s, i in skipped[:8]))
print(f"Figures written to {OUT_DIR}")