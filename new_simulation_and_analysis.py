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
import textwrap
from fractions import Fraction

IN_NOTEBOOK = "ipykernel" in sys.modules

import matplotlib
import matplotlib.ticker
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import lambertw, expit
from scipy.integrate import quad
from scipy.stats import binom
from scipy.linalg import solve_banded

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
SHOW_INDIVIDUAL_FIGURES = True

START_COUNTS = {
    "1 individual":      ("count", 1),
    "1% of population":  ("frac", 0.01),
    "5% of population":  ("frac", 0.05),
    "10% of population": ("frac", 0.10),
}

# Graphs 2 & 4 (x = N): range and number of simulated / theory points
POP_N_MIN, POP_N_MAX = 10, 7000
POP_SIM_POINTS = 16
THEORY_POINTS_N = 90
POP_SWEEP_S = [-0.1, -0.01, 0.0, 0.01, 0.1]

# Graphs 1 & 3 (x = s): N values must hold every percentage start exactly (multiples of 100)
SEL_SWEEP_S = np.concatenate([[-0.02, -0.01, -0.005, -0.002], [0.0], np.geomspace(1e-3, 2.0, 18)])
SEL_SWEEP_N = [100, 300, 3000]
THEORY_POINTS_S = 160

# Noise control
REPLICATE_RULE = "target_rse"
TARGET_RSE = 0.05
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
LOSS_ONLY_RUNS = max(MIN_RUNS, int(np.ceil(TF_CV ** 2 / TARGET_RSE ** 2)))   # for points where fixation is unreachable


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


def tloss_neutral(N, i0):
    """Kimura & Ohta (1969) eq. 16, haploid (4Ne -> 2N): mean time to loss given loss, neutral allele."""
    p = i0 / N
    return float(-2 * N * (p / (1 - p)) * np.log(p))


def tfix_logistic(N, i0, s):
    if s <= 0:
        return np.nan
    se = np.log1p(s)
    p0, p1 = i0 / N, 1 - 1 / N
    if p0 >= p1:
        return np.nan
    return float((np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))) / se)


def tfix_kimura_ohta(N, s, i0=1, include_J2=True):
    """Kimura & Ohta (1969) eq. 17, haploid; J2 term rearranged to avoid overflow.

    include_J2=False returns the first integral J1 alone, which is Otto & Whitlock eq. (5)
    (exactly so for i0 = 1, where the lower limit is 1/(cNe) = 1/N).
    """
    p = i0 / N
    if s == 0:
        return tfix_neutral(N, i0) if include_J2 else 2.0 * N * (1 - p)
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
    if not include_J2:
        return float(J1)
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
    """Exact fixation probability, conditional fixation time and conditional loss time for every i0 at once."""
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
    # chain conditioned on loss (same construction with the loss probabilities v)
    v = np.clip(np.linalg.solve(I - Q, P[1:N, 0]), 0.0, 1.0)
    with np.errstate(all="ignore"):
        Ql = Q * v[None, :] / v[:, None]
        TL = np.linalg.solve(I - Ql, np.ones(N - 1))
    TL = np.where(np.isfinite(TL) & (v > 1e-300), TL, np.nan)
    return u, T, TL


def analytic_predictions(N, s, i0):
    pf = {
        "kimura":    fix_prob_diffusion(N, i0, 2 * s),
        "ratio":     fix_prob_diffusion(N, i0, 2 * np.log1p(s)),
        "branching": 1 - (1 - psi_branching(s)) ** i0 if s > 0 else np.nan,
        "haldane":   2 * s if (i0 == 1 and 0 < s <= 0.5) else np.nan,
    }
    tf = {
        "ko_log":  _safe(tfix_kimura_ohta, N, np.log1p(s), i0),
        "ko_raw":  _safe(tfix_kimura_ohta, N, s, i0),
        "strong":  tfix_logistic(N, i0, s),
        "ow5":     _safe(tfix_kimura_ohta, N, np.log1p(s), i0, False),
    }
    return {"pf": pf, "tf": tf}


def _analytic_batch(keys):
    return [analytic_predictions(*k) for k in keys]


_MORAN_MEMO = {}


def _log_pi_unnorm(i, b):
    """log of the (unnormalised) Moran fixation probability 1 - r^(-i), r = e^b, computed without overflow."""
    i = np.asarray(i, float)
    if b > 0:
        return np.log(-np.expm1(-b * i))
    if b < 0:
        a = -b
        return a * i + np.log(-np.expm1(-a * i))
    return np.log(i)


def moran_chain(N, s):
    """Moran model (one birth + one death per step, N steps = 1 generation), relative fitness r = 1+s.

    Returns, for every i0 = 1..N-1, the exact fixation probability and the exact mean time to
    fixation conditioned on fixation (in generations), from first-step analysis of the birth–death chain.
    """
    key = (int(N), float(s))
    if key in _MORAN_MEMO:
        return _MORAN_MEMO[key]
    N = int(N)
    b = float(np.log1p(s))
    i = np.arange(1, N)
    lp = _log_pi_unnorm(np.arange(1, N + 1), b)          # log pi_1 .. log pi_N (unnormalised)
    pi = np.exp(lp[:-1] - lp[-1])
    r = 1.0 + s
    tp = r * i / (r * i + N - i) * (N - i) / N          # P(i -> i+1)
    tm = (N - i) / (r * i + N - i) * i / N              # P(i -> i-1)
    up = tp * np.exp(lp[1:] - lp[:-1])                   # conditioned up-step   T+ pi_{i+1}/pi_i
    dn = np.zeros(N - 1)
    dn[1:] = tm[1:] * np.exp(lp[:-2] - lp[1:-1])         # conditioned down-step T- pi_{i-1}/pi_i (0 at i = 1)
    ab = np.zeros((3, N - 1))
    ab[0, 1:] = -up[:-1]
    ab[1, :] = tp + tm
    ab[2, :-1] = -dn[1:]
    tau = solve_banded((1, 1), ab, np.ones(N - 1))       # mean number of birth–death steps
    out = (np.clip(pi, 0, 1), tau / N)
    _MORAN_MEMO[key] = out
    return out


def attach_exact(k, ana, exact):
    N, s, i0 = k
    ex = exact.get((N, s))
    pf_ex = float(ex[0][i0 - 1]) if ex is not None else np.nan
    tf_ex = float(ex[1][i0 - 1]) if ex is not None else np.nan
    tl_ex = float(ex[2][i0 - 1]) if (ex is not None and len(ex) > 2) else np.nan
    # Neutral formulas do not depend on s: drawn as lines at every s (strictly valid only at s = 0).
    # Computed here rather than cached, so old caches still work.
    # Likewise the branching / Haldane / 2N formulas are evaluated for every i0 and s, even outside their range.
    pf_neu = {"neutral": i0 / N,
              "branching": 1 - (1 - psi_branching(s)) ** i0,
              "haldane": 2 * s}
    tf_neu = {"neutral": tfix_neutral(N, i0), "large": 2.0 * N}
    pf = {"exact": pf_ex, **{k: v for k, v in ana["pf"].items() if k != "neutral"}, **pf_neu}
    tf = {"exact": tf_ex, **{k: v for k, v in ana["tf"].items() if k not in tf_neu and k != "giant"}, **tf_neu}  # "giant" = key in old caches
    # Moran model (different model from the simulated Wright–Fisher one), exact, every N, s, i0.
    try:
        mpi, mt = moran_chain(N, s)
        pf["moran"], tf["moran"] = float(mpi[i0 - 1]), float(mt[i0 - 1])
    except Exception:
        pf["moran"] = tf["moran"] = np.nan
    # time to loss (drawn for every s, although eq. 16 is strictly for s = 0)
    tl = {"exact": tl_ex, "ko16": tloss_neutral(N, i0)}
    return {"pf": pf, "tf": tf, "tl": tl}

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
# ## 4. Vectorised simulator

# %%
def wf_step(rng, cur, N, w):
    """One Wright–Fisher generation for haploid counts `cur` of allele 1 (fitness w vs 1).
    Shared by the main simulator and the Fisher's-theorem test, so both test the same update."""
    wi = cur * w
    return rng.binomial(N, wi / (wi + (N - cur)))


def simulate_point(N, s, i0, n_runs, seed, chunk=250_000):
    rng = np.random.default_rng(seed)
    w = 1.0 + s
    n_fix, t_sum, t_sq = 0, 0.0, 0.0
    n_loss, l_sum, l_sq = 0, 0.0, 0.0
    tic = time.perf_counter()
    remaining = n_runs
    while remaining > 0:
        m = min(remaining, chunk)
        remaining -= m
        cur = np.full(m, i0, dtype=np.int64)
        t = 0
        while cur.size:
            t += 1
            cur = wf_step(rng, cur, N, w)
            fixed = cur == N
            done = fixed | (cur == 0)
            if done.any():
                nf = int(np.count_nonzero(fixed))
                if nf:
                    n_fix += nf
                    t_sum += nf * t
                    t_sq += nf * float(t) * t
                nl = int(np.count_nonzero(done)) - nf
                if nl:
                    n_loss += nl
                    l_sum += nl * t
                    l_sq += nl * float(t) * t
                cur = cur[~done]

    p_hat = n_fix / n_runs
    p_se = np.sqrt(p_hat * (1 - p_hat) / n_runs) if n_fix > 0 else np.nan
    if n_fix > 0:
        t_mean = t_sum / n_fix
        t_var = max(t_sq / n_fix - t_mean ** 2, 0.0) * n_fix / max(n_fix - 1, 1)
        t_se = np.sqrt(t_var / n_fix) if n_fix > 1 else np.nan
    else:
        t_mean = t_se = np.nan
    if n_loss > 0:
        tl_mean = l_sum / n_loss
        tl_var = max(l_sq / n_loss - tl_mean ** 2, 0.0) * n_loss / max(n_loss - 1, 1)
        tl_se = np.sqrt(tl_var / n_loss) if n_loss > 1 else np.nan
    else:
        tl_mean = tl_se = np.nan
    return dict(n_runs=n_runs, n_fix=n_fix, p_hat=p_hat, p_se=p_se,
                t_mean=t_mean, t_se=t_se, n_loss=n_loss, tl_mean=tl_mean, tl_se=tl_se,
                seconds=time.perf_counter() - tic)


def run_parallel(func, arg_tuples):
    if not arg_tuples:
        return []
    if USE_PARALLEL and HAVE_JOBLIB and len(arg_tuples) > 1:
        return Parallel(n_jobs=N_JOBS, verbose=5)(delayed(func)(*a) for a in arg_tuples)
    return [func(*a) for a in arg_tuples]

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

# 5a. exact Markov chains (one solve per (N, s) gives every i0)
exact_todo = sorted({(N, s) for (N, s, _) in ALL_KEYS if N <= EXACT_N_MAX
                     and ((N, s) not in EXACT or len(EXACT[(N, s)]) < 3)})   # old caches lack the loss chain
print(f"Exact chains to compute: {len(exact_todo)}")
EXACT.update(zip(exact_todo, run_parallel(exact_wf_chain, exact_todo)))

# 5b. closed-form / integral formulas on the dense grid
todo_a = [k for k in ALL_KEYS if k not in ANALYTIC or "ow5" not in ANALYTIC[k]["tf"]]
print(f"Analytic points to compute: {len(todo_a)}")
chunks = [todo_a[i:i + 200] for i in range(0, len(todo_a), 200)]
for chunk, res in zip(chunks, run_parallel(_analytic_batch, [(c,) for c in chunks])):
    ANALYTIC.update(zip(chunk, res))

PREDICTIONS = {k: attach_exact(k, ANALYTIC[k], EXACT) for k in ALL_KEYS}

# 5c. replicate plan
PLAN = {}
for k in POINT_KEYS:
    N, s, i0 = k
    p_pred = PREDICTIONS[k]["pf"]["exact"]
    if not np.isfinite(p_pred):
        p_pred = PREDICTIONS[k]["pf"]["ratio"]
    n, capped = choose_num_runs(N, i0, p_pred)
    PLAN[k] = dict(p_pred=p_pred, n_runs=n, capped=capped,
                   skip=(n * p_pred < MIN_EXPECTED_FIXATIONS))
    if PLAN[k]["skip"]:
        # fixation essentially never happens here, but losses do: run enough replicates for
        # the time-to-loss graphs only (these runs are not used for fixation results)
        PLAN[k].update(n_runs=LOSS_ONLY_RUNS, capped=False, loss_only=True)

# 5d. simulations (seed derived from the point itself -> independent of which other points exist)
def point_seed(k):
    N, s, i0 = k
    return np.random.SeedSequence([SEED, N, i0, int(round(s * 1e9)) & 0xFFFFFFFF, int(s < 0)])

todo = [k for k in POINT_KEYS
        if k not in RESULTS or RESULTS[k]["n_runs"] != PLAN[k]["n_runs"] or "tl_mean" not in RESULTS[k]]
todo.sort(key=lambda k: -PLAN[k]["n_runs"] * (1 + PLAN[k]["p_pred"] * k[0]))
print(f"Points: {len(POINT_KEYS)} | to simulate: {len(todo)} | "
      f"fixation unreachable (loss-only runs): {sum(p['skip'] for p in PLAN.values())} | "
      f"capped: {sum(p['capped'] and not p['skip'] for p in PLAN.values())} | "
      f"total replicates: {sum(PLAN[k]['n_runs'] for k in todo):,}")

sim_out = run_parallel(simulate_point,
                       [(k[0], k[1], k[2], PLAN[k]["n_runs"], point_seed(k), SIM_CHUNK) for k in todo])
RESULTS.update(zip(todo, sim_out))
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

FORMULA_LW = 1.1
SIM_COLOR = "#7a7a7a"   # mid grey, so it is never confused with the black exact-chain line
SIM_MS = 4.5
SIM_ALPHA = 0.6
HIDE_FORMULAS = {"pf": set(), "tf": set()}   # e.g. {"pf": {"haldane"}, "tf": {"ko_raw"}}

RESID_LINTHRESH = 1e-4      # only used when RESID_AUTO_ZOOM = False
RESID_YMAX = 10.0           # only used when RESID_AUTO_ZOOM = False
REF_FALLBACK = {"pf": "ratio", "tf": "ko_log", "tl": "ko16"}

# ---- Zoom of the deviation panel (this is what makes subtle differences visible) ----
# Each subplot gets its own y-range, fitted to the deviations actually present in it.
RESID_AUTO_ZOOM = True
RESID_FORMULA_CAP = 0.5         # formulas more than ±50 % off are not allowed to widen the window (they run off-scale)
RESID_PAD = 0.12                # 12 % head-room above/below the data
RESID_MIN_HALFWIDTH = 1e-3      # never zoom in tighter than ±0.1 %
RESID_LINTHRESH_QUANTILE = 0.2  # symlog: linear region is sized to the small (20th-percentile) deviations
RESID_LINEAR_IF_RATIO_BELOW = 30  # if largest/smallest deviation < 30, a plain linear axis is used
PANEL_HEIGHTS = (1.0, 1.6)      # (raw values, deviation) — deviation panel is now the larger one

SERIES_LINESTYLES = ["-", (0, (6, 3)), (0, (1.5, 2)), (0, (8, 3, 1.5, 3)), (0, (12, 4)), (0, (3, 1.5, 1, 1.5, 1, 1.5))]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P"]
SERIES_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]   # diagnostic plot only

# Each legend entry: name [wf_core code] / textbook formula (matplotlib mathtext) / population regime it was built for.
_KO_TF = (r"$\bar{t}(p)=\dfrac{1}{s'}\int_{p}^{1}\dfrac{u(x)\,\left(1-e^{-S(1-x)}\right)}{x(1-x)}\,dx"
          r"\;+\;\dfrac{1-u(p)}{s'\,u(p)}\int_{0}^{p}\dfrac{u(x)\,\left(e^{Sx}-1\right)}{x(1-x)}\,dx$" "\n"
          r"$u(x)=\dfrac{1-e^{-Sx}}{1-e^{-S}},\quad S=2Ns',\quad p=\dfrac{i_0}{N}$")


def _entry(name, formula, regime):
    return f"{name}\n{formula}\nDesigned for: {regime}"


FORMULAS = {
    "pf": {
        "exact": dict(color="#000000", label=_entry(
            "Exact Wright–Fisher Markov chain  (reference for the lower panel)",
            r"$\pi_i=\sum_{j=0}^{N}P_{ij}\,\pi_j,\quad \pi_0=0,\ \pi_N=1,\quad"
            r" P_{ij}=\binom{N}{j}x_i^{\,j}(1-x_i)^{N-j},\quad x_i=\dfrac{i(1+s)}{i(1+s)+N-i}$",
            f"any selection s, any initial count i₀; no approximation (computed for N ≤ {EXACT_N_MAX})")),
        "kimura": dict(color="#0047FF", label=_entry(
            "Kimura's diffusion approximation  [PFb_w_]",
            r"$\pi=\dfrac{1-e^{-2s\,i_0}}{1-e^{-2Ns}}$",
            "weak selection (|s| ≪ 1), large population N, any initial count i₀")),
        "ratio": dict(color="#00A33A", label=_entry(
            "Diffusion with multiplicative fitness, $s_e=\\ln(1+s)$  [PF____]",
            r"$\pi=\dfrac{1-(1+s)^{-2i_0}}{1-(1+s)^{-2N}}$",
            "large population N, weak to strong selection (any s > −1), any initial count i₀")),
        "neutral": dict(color="#E8000B", label=_entry(
            "Neutral fixation probability  [PFn___]",
            r"$\pi=\dfrac{i_0}{N}$",
            "no selection (s = 0), any population size N, any initial count i₀")),
        "branching": dict(color="#FF9900", label=_entry(
            "Branching-process survival (Poisson offspring)  [PFbg_1]",
            r"$\pi=1-(1-\Psi)^{i_0},\quad \Psi=1+\dfrac{W_0\!\left(-(1+s)\,e^{-(1+s)}\right)}{1+s}$",
            "beneficial allele (s > 0), very large population N, rare allele (i₀ ≪ N)")),
        "haldane": dict(color="#9B00D9", label=_entry(
            "Haldane's approximation  [PFbgw1]",
            r"$\pi\approx 2s$",
            "weak positive selection (0 < s ≪ 1), large population N, a single new mutant (i₀ = 1)")),
        "moran": dict(color="#8B4513", label=_entry(
            "Moran model, exact fixation probability  (a different model from the simulated Wright–Fisher one)",
            r"$\pi=\dfrac{1-(1+s)^{-i_0}}{1-(1+s)^{-N}}$",
            "Moran model: overlapping generations, one birth and one death per step; any N, any s, any initial count i₀")),
    },
    "tf": {
        "exact": dict(color="#000000", label=_entry(
            "Exact Wright–Fisher Markov chain, conditioned on fixation  (reference for the lower panel)",
            r"$\bar{t}_i=1+\sum_{j=1}^{N-1}P_{ij}\,\dfrac{\pi_j}{\pi_i}\,\bar{t}_j,\quad \bar{t}_N=0$"
            r"$\quad(P_{ij},\ \pi_i$ as in the fixation-probability chain$)$",
            f"any selection s, any initial count i₀; no approximation (computed for N ≤ {EXACT_N_MAX})")),
        "ko_log": dict(color="#0047FF", label=_entry(
            "Kimura–Ohta (1969) eq. 17, with $s'=s_e=\\ln(1+s)$  [TFb_w1]",
            _KO_TF,
            "large population N, weak to strong selection (any s; uses |s|), any initial count i₀")),
        "ko_raw": dict(color="#00A33A", label=_entry(
            "Kimura–Ohta (1969) eq. 17, with $s'=s$ as given  [TF__w1]",
            _KO_TF,
            "large population N, weak selection (|s| ≪ 1; uses |s|), any initial count i₀")),
        "neutral": dict(color="#E8000B", label=_entry(
            "Neutral conditional fixation time  [TFn__1a = TFn__1b]",
            r"$\bar{t}=-\dfrac{2N\,(1-p)\,\ln(1-p)}{p},\quad p=\dfrac{i_0}{N}$",
            "no selection (s = 0), large population N, any initial count i₀")),
        "large": dict(color="#FF9900", label=_entry(
            "Large-population neutral approximation  [TFng_1]",
            r"$\bar{t}\approx 2N$",
            "no selection (s = 0), large population N, a single new mutant (i₀ = 1)")),
        "strong": dict(color="#9B00D9", label=_entry(
            "Strong-selection logistic sweep  [TFb_s1]",
            r"$\bar{t}=\dfrac{1}{\ln(1+s)}\left[\ln\dfrac{p_1}{1-p_1}-\ln\dfrac{p_0}{1-p_0}\right],"
            r"\quad p_0=\dfrac{i_0}{N},\ \ p_1=1-\dfrac{1}{N}$",
            "strong positive selection (Ns ≫ 1, deterministic sweep), any population size N, any initial count i₀")),
        "moran": dict(color="#8B4513", label=_entry(
            "Moran model, exact mean fixation time given fixation  (a different model from the simulated Wright–Fisher one)",
            r"$\tau_i=1+T_i^{+}\dfrac{\pi_{i+1}}{\pi_i}\tau_{i+1}+T_i^{-}\dfrac{\pi_{i-1}}{\pi_i}\tau_{i-1}"
            r"+\left(1-T_i^{+}-T_i^{-}\right)\tau_i,\quad \tau_N=0,\quad \bar{t}=\dfrac{\tau_{i_0}}{N}$" "\n"
            r"$T_i^{+}=\dfrac{(1+s)\,i}{(1+s)\,i+N-i}\cdot\dfrac{N-i}{N},\quad"
            r" T_i^{-}=\dfrac{N-i}{(1+s)\,i+N-i}\cdot\dfrac{i}{N},\quad \pi_i=\dfrac{1-(1+s)^{-i}}{1-(1+s)^{-N}}$",
            "Moran model: overlapping generations, one birth and one death per step (N steps = 1 generation); any N, any s, any i₀")),
        "ow5": dict(color="#00B8D4", label=_entry(
            "Otto & Whitlock eq. (5): single-copy fixation time (first integral of Kimura–Ohta only)",
            r"$\bar{t}\approx\dfrac{1}{s'}\int_{p}^{1}\dfrac{\left(e^{Sx}-1\right)\left(e^{S(1-x)}-1\right)}"
            r"{x(1-x)\left(e^{S}-1\right)}\,dx,\quad S=2Ns',\quad s'=\ln(1+s),\quad p=\dfrac{i_0}{N}$",
            "a single new mutant (i₀ = 1, lower limit 1/(cNₑ)), any selection strength (uses |s|), large population N")),
    },
    "tl": {
        "exact": dict(color="#000000", label=_entry(
            "Exact Wright–Fisher Markov chain, conditioned on loss  (reference for the lower panel)",
            r"$\bar{t}_i=1+\sum_{j=1}^{N-1}P_{ij}\,\dfrac{\lambda_j}{\lambda_i}\,\bar{t}_j,\quad"
            r"\lambda_i=\sum_{j=0}^{N}P_{ij}\lambda_j,\quad \lambda_0=1,\ \lambda_N=0$"
            r"$\quad(\lambda_i$ = probability of loss$)$",
            f"any selection s, any initial count i₀; no approximation (computed for N ≤ {EXACT_N_MAX})")),
        "ko16": dict(color="#E8000B", label=_entry(
            "Neutral conditional time to loss",
            r"$\bar{t}_{\mathrm{loss}}=-2N\,\dfrac{p}{1-p}\,\ln p,\quad p=\dfrac{i_0}{N}$",
            "no selection (s = 0), large population N, any initial count i₀")),
    },
}

# ---- Literature source of every formula (shown as the last line of each legend entry) ----
OW = "Otto & Whitlock (2005), Fixation Probabilities and Times, Encycl. Life Sci."
KO = "Kimura & Ohta (1969), Genetics 61: 763–771"
SOURCES = {
    "pf": {
        "exact":     "not taken from either paper: direct Wright–Fisher transition matrix "
                     f"(the same 'exact matrix calculation' as the Observed column of {OW.split(',')[0]}, Table 1)",
        "kimura":    f"{OW}, eq. (3), with c = 1, Nₑ = N, p = i₀/N",
        "ratio":     f"{OW}, eq. (3), with s replaced by ln(1+s) (this substitution is not in the paper)",
        "neutral":   f"{OW}, p. 1 text (unnumbered): Π = p for a neutral allele; also eq. (3) in the limit s → 0",
        "branching": f"{OW}, eq. (1) (Haldane 1927); closed form via Lambert W and the extension to i₀ copies are not in the paper",
        "haldane":   f"{OW}, eq. (2) (Haldane 1927)",
        "moran":     "not in either uploaded paper: Moran (1958), Proc. Camb. Phil. Soc. 54: 60–71 "
                     "(standard Moran fixation probability)",
    },
    "tf": {
        "exact":   "not taken from either paper: Wright–Fisher transition matrix conditioned on fixation",
        "ko_log":  f"{KO}, eq. (17) (genic selection; built from eq. 12), haploid form with s′ = ln(1+s); "
                   f"for i₀ = 1 cf. {OW.split(',')[0]}, eq. (5)",
        "ko_raw":  f"{KO}, eq. (17) (genic selection; built from eq. 12), haploid form with s′ = s; "
                   f"for i₀ = 1 cf. {OW.split(',')[0]}, eq. (5)",
        "neutral": f"{KO}, eq. (14) (diploid 4Nₑ → haploid 2N)",
        "large":   f"{KO}, eq. (15) (diploid 4Nₑ → haploid 2N); {OW.split(',')[0]}, p. 3 text (unnumbered): ≈ 2cNₑ",
        "strong":  f"{OW}, eq. (6) (Crow & Kimura 1970, eq. 5.3.13); here started from p₀ = i₀/N and with s → ln(1+s), "
                   "which reduces to eq. (6) for i₀ = 1, c = 1",
        "ow5":     f"{OW}, eq. (5) (after Ewens 1979, p. 151); here with s′ = ln(1+s) and lower limit "
                   "i₀/N, which is eq. (5) exactly for i₀ = 1, c = 1",
        "moran":   "not in either uploaded paper: first-step analysis of the Moran birth–death chain, conditioned on "
                   "fixation (see e.g. Antal & Scheuring 2006, Bull. Math. Biol. 68: 1923–1944); solved exactly",
    },
    "tl": {
        "exact": "not taken from either paper: Wright–Fisher transition matrix conditioned on loss",
        "ko16":  f"{KO}, eq. (16), haploid form (diploid 4Nₑ → haploid 2N)",
    },
}

YLABEL = {"pf": "Probability of fixation of allele 1",
          "tf": "Mean generations to fixation\n(given fixation)",
          "tl": "Mean generations to loss\n(given loss)"}
QTITLE = {"pf": "Probability of fixation", "tf": "Time to fixation", "tl": "Time to loss"}


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


def _pct(v, _pos=None):
    if v == 0:
        return "0"
    a = abs(v) * 100
    txt = f"{a:.3g}"
    return ("+" if v > 0 else "−") + txt + "%"


def apply_resid_zoom(axr, formula_dev, sim_lo, sim_hi, sim_mid):
    """Fit the deviation axis to the data in this subplot.

    - The window always contains every simulation point with its ± 1 SE bar.
    - Formula curves widen the window only if they are within ±RESID_FORMULA_CAP;
      formulas further off simply run off the top/bottom.
    - If deviations span a wide range of magnitudes, a symlog axis is used whose linear
      region is sized to the small deviations, so 0.1 % and 10 % are both readable.
    """
    f = np.asarray(formula_dev, float)
    f = f[np.isfinite(f) & (np.abs(f) <= RESID_FORMULA_CAP)]
    lo_s = np.asarray(sim_lo, float); hi_s = np.asarray(sim_hi, float)
    lo_s, hi_s = lo_s[np.isfinite(lo_s)], hi_s[np.isfinite(hi_s)]
    lo = min([0.0] + list(f) + list(lo_s))
    hi = max([0.0] + list(f) + list(hi_s))
    half = max(hi - lo, 2 * RESID_MIN_HALFWIDTH) / 2
    mid = (hi + lo) / 2
    lo, hi = min(lo, mid - half), max(hi, mid + half)
    pad = RESID_PAD * (hi - lo)
    lo, hi = max(lo - pad, -1.0), hi + pad

    mags = np.abs(np.concatenate([f, np.asarray(sim_mid, float)]))
    mags = mags[np.isfinite(mags) & (mags > 1e-7)]
    big = max(abs(lo), abs(hi))
    if mags.size == 0 or big / max(np.quantile(mags, RESID_LINTHRESH_QUANTILE), 1e-12) < RESID_LINEAR_IF_RATIO_BELOW:
        axr.set_yscale("linear")
        axr.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(7, symmetric=False))
    else:
        lt = 10.0 ** np.floor(np.log10(np.quantile(mags, RESID_LINTHRESH_QUANTILE)))
        lt = min(lt, big / 10)
        axr.set_yscale("symlog", linthresh=lt, linscale=1.0)
        axr.yaxis.set_major_locator(matplotlib.ticker.SymmetricalLogLocator(base=10, linthresh=lt))
        axr.yaxis.set_minor_locator(matplotlib.ticker.SymmetricalLogLocator(
            base=10, linthresh=lt, subs=np.arange(2, 10)))
        axr.axhspan(-lt, lt, color="#ececec", zorder=0, lw=0)
    axr.set_ylim(lo, hi)
    axr.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_pct))
    axr.tick_params(axis="y", labelsize=7)


def draw_panel(ax, axr, quantity, sweep, start_label):
    spec = START_COUNTS[start_label]
    L = sweep_layout(sweep, start_label)
    x_th, x_sim = L["x_th"], L["x_sim"]
    used, series_handles, sim_vals = [], [], []
    zoom_f, zoom_lo, zoom_hi, zoom_mid = [], [], [], []
    rep_info = []
    x_lo, x_hi = (x_sim.min(), x_sim.max()) if x_sim.size else (-np.inf, np.inf)

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
            in_x = m & np.isfinite(rel) & (x_th >= x_lo) & (x_th <= x_hi)
            zoom_f.extend(rel[in_x])
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
        runs_list, nfix, n_skip = [], 0, 0
        for k in sim_keys:
            # loss-only runs (fixation unreachable) are used for the time-to-loss graphs only
            usable = quantity == "tl" or not PLAN[k]["skip"]
            r = RESULTS.get(k) if usable else None
            if r is None:
                n_skip += 1
                obs.append(np.nan); err.append(np.nan); cap.append(False)
                continue
            runs_list.append(int(r["n_runs"]))
            nfix += int(r.get("n_loss", 0)) if quantity == "tl" else int(r["n_fix"])
            v, e = {"pf": (r["p_hat"], r["p_se"]), "tf": (r["t_mean"], r["t_se"]),
                    "tl": (r.get("tl_mean", np.nan), r.get("tl_se", np.nan))}[quantity]
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
            zoom_lo.extend(rel_obs[mr] - rel_err[mr])
            zoom_hi.extend(rel_obs[mr] + rel_err[mr])
            zoom_mid.extend(rel_obs[mr])
            if mr.any():
                axr.errorbar(x_sim[mr], rel_obs[mr], yerr=rel_err[mr], fmt="none", ecolor=SIM_COLOR,
                             elinewidth=0.6, capsize=1.5, alpha=SIM_ALPHA, zorder=5)
                axr.plot(x_sim[mr], rel_obs[mr], ls="none", marker=mk, ms=SIM_MS, mfc=mfc,
                         mec=SIM_COLOR, mew=0.8, alpha=SIM_ALPHA, zorder=6)

        rep_info.append(dict(label=L["series_label"](sv), runs=runs_list, nfix=nfix, skipped=n_skip,
                             events="losses" if quantity == "tl" else "fixations",
                             capped=int(sum(RESULTS[k].get("capped", False) for k in sim_keys
                                            if k in RESULTS and not PLAN[k]["skip"]))))
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

    if RESID_AUTO_ZOOM:
        apply_resid_zoom(axr, zoom_f, zoom_lo, zoom_hi, zoom_mid)
    else:
        axr.set_yscale("symlog", linthresh=RESID_LINTHRESH, linscale=1.0)
        axr.set_ylim(-1.05, RESID_YMAX)
        ticks, labels = _resid_ticks()
        axr.set_yticks(ticks)
        axr.set_yticklabels(labels, fontsize=7)
    axr.axhline(0, color="k", lw=0.5)
    axr.set_ylabel("% deviation from\nexact chain (zoomed)", fontsize=9)
    axr.set_xlabel(L["xlabel"])
    axr.grid(True, which="major", ls="--", lw=0.4, alpha=0.3)
    return used, series_handles, L["series_title"], rep_info


def formula_handles(quantity, used):
    hs = [Line2D([], [], color=st["color"], lw=3, label=st["label"] + "\n" + textwrap.fill(
                    "Source: " + SOURCES[quantity][fk], 95, subsequent_indent="    "))
          for fk, st in FORMULAS[quantity].items() if fk in used]
    target = f"target RSE {TARGET_RSE:.0%}" if REPLICATE_RULE == "target_rse" else f"{LINEAR_RUNS_PER_INDIVIDUAL}·N runs"
    hs.append(Line2D([], [], color=SIM_COLOR, ls="none", marker="o", alpha=SIM_ALPHA,
                     label=f"Simulation ± 1 SE ({target}); marker shape = series"))
    hs.append(Line2D([], [], color=SIM_COLOR, ls="none", marker="o", mfc="white",
                     label=f"Simulation, run cap {MAX_RUNS:,} reached (noise target not met)"))
    return hs


def replicate_note(rep_info):
    """Side note: exact number of simulation replicates behind the grey points of one figure."""
    lines = ["Simulation replicates (runs)", ""]
    tot_runs = tot_fix = tot_pts = 0
    for d in rep_info:
        r = d["runs"]
        lines.append(d["label"].split("  [")[0])
        if r:
            lines.append(f"  {sum(r):>12,} runs, {len(r)} points")
            lines.append(f"  per point: {min(r):,} – {max(r):,}")
            lines.append(f"  {d['events']}: {d['nfix']:,}")
        else:
            lines.append("  no points simulated")
        if d["skipped"]:
            lines.append(f"  skipped (unreachable): {d['skipped']}")
        if d["capped"]:
            lines.append(f"  at run cap ({MAX_RUNS:,}): {d['capped']}")
        tot_runs += sum(r); tot_fix += d["nfix"]; tot_pts += len(r)
    ev = rep_info[0]["events"] if rep_info else "fixations"
    lines += ["", f"Total: {tot_runs:,} runs", f"       {tot_pts} points, {tot_fix:,} {ev}"]
    return "\n".join(lines)


def make_graph(number, quantity, sweep):
    """One separate figure per starting count: 4 images per graph, 16 in total."""
    vs = "selection coefficient" if sweep == "sel" else "population size"
    stub = f"graph{number}_{quantity}_vs_{'s' if sweep == 'sel' else 'N'}"
    for j, lab in enumerate(START_COUNTS, 1):
        f, (ax, axr) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                    gridspec_kw=dict(height_ratios=list(PANEL_HEIGHTS), hspace=0.06))
        used, sh, stitle, rep_info = draw_panel(ax, axr, quantity, sweep, lab)
        axr.text(1.01, 1.0, replicate_note(rep_info), transform=axr.transAxes, va="top", ha="left",
                 fontsize=8.5, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#bbbbbb"))
        leg1 = ax.legend(handles=sh, title=stitle, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                         fontsize=9, title_fontsize=9, handlelength=4.5)
        ax.add_artist(leg1)
        f.legend(handles=formula_handles(quantity, used), loc="upper center", bbox_to_anchor=(0.5, 0.0),
                 ncol=2, fontsize=10, title="Colour = formula   ·   each entry: name / equation / population it was designed for",
                 title_fontsize=10.5, labelspacing=1.3, columnspacing=2.5, handlelength=3, borderpad=1.0,
                 alignment="left")
        ax.set_title(f"Graph {number}.{j}: {QTITLE[quantity]} vs. {vs}\n"
                     f"{ax.get_title()}", fontsize=12)
        tag = lab.split()[0].replace("%", "pct")
        path = os.path.join(OUT_DIR, f"{stub}_{j}_start_{tag}.png")
        f.savefig(path, dpi=200, bbox_inches="tight")
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
# ## 10b. Graphs 5 & 6 — Time to loss (given loss) vs. selection coefficient and vs. population size
#
# Same layout as Graphs 3 & 4, but for runs in which allele 1 is **lost**. Compared against
# Kimura & Ohta (1969) eq. 16 (haploid) and the exact Wright–Fisher chain conditioned on loss.

# %%
make_graph(5, "tl", "sel")
make_graph(6, "tl", "pop")

# %% [markdown]
# ## 10c. Graph 7 — Fisher's fundamental theorem
#
# Fisher's fundamental theorem (discrete generations, constant fitness): the expected change in mean
# fitness in one generation equals the variance in fitness divided by the mean fitness,
#
# $$\mathrm{E}[\Delta\bar w \mid p] = \frac{\mathrm{Var}(w)}{\bar w},\qquad \bar w = 1+sp,\quad \mathrm{Var}(w)=s^2p(1-p).$$
#
# **Test.** For a grid of allele frequencies p = i/N, many single generations are run with the *same update
# function* as the main simulator (`wf_step`), and the mean change in mean fitness, Δw̄ = s(p′ − p), is
# compared with the prediction. Replicates per point are chosen so every point has the same relative SE
# (`TARGET_RSE`). The side box also reports observed ÷ expected sampling variance (should be ≈ 1 if the
# random sampling is binomial, as Wright–Fisher assumes).

# %%
FFT_N = [100, 1000] if not QUICK_TEST else [100]
FFT_S = [-0.1, -0.01, 0.01, 0.1, 1.0]          # s = 0 is omitted: Δw̄ = 0 exactly, so there is nothing to plot on a log axis
FFT_P_POINTS = 15 if not QUICK_TEST else 7
FFT_CACHE = os.path.join(OUT_DIR, "fft_cache.pkl")


def fft_p_grid(N, n):
    """Allele counts i (1..N-1) spread evenly on a logit scale."""
    y = np.linspace(np.log(1 / (N - 1)), np.log(N - 1), n)
    return np.unique(np.clip(np.round(N * expit(y)).astype(int), 1, N - 1))


def fft_prediction(p, s):
    wbar = 1 + s * p
    var_w = s ** 2 * p * (1 - p)
    return {"fft": var_w / wbar, "naive": var_w}


def fft_runs(N, s, i):
    p = i / N
    x = i * (1 + s) / (i * (1 + s) + N - i)
    var_dw = s ** 2 * x * (1 - x) / N                      # Var[Δw̄ | p] under binomial sampling
    pred = fft_prediction(p, s)["fft"]
    n = var_dw / (pred ** 2 * TARGET_RSE ** 2)
    return int(np.clip(np.ceil(n), MIN_RUNS, MAX_RUNS)), bool(n > MAX_RUNS), var_dw


def fft_point(N, s, i, n, seed, chunk=1_000_000):
    rng = np.random.default_rng(seed)
    w, p = 1.0 + s, i / N
    tot, tot2, done = 0.0, 0.0, 0
    while done < n:
        m = min(chunk, n - done)
        new = wf_step(rng, np.full(m, i, dtype=np.int64), N, w)
        dw = s * (new / N - p)                             # change in mean fitness, w̄ = 1 + s p
        tot += dw.sum(); tot2 += (dw ** 2).sum(); done += m
    mean = tot / n
    var = max(tot2 / n - mean ** 2, 0.0) * n / (n - 1)
    return dict(n=n, mean=mean, se=np.sqrt(var / n), var=var)


FFT_RES = {}
if REUSE_CACHE and os.path.exists(FFT_CACHE):
    with open(FFT_CACHE, "rb") as fh:
        FFT_RES = pickle.load(fh)
fft_plan = {}
for N in FFT_N:
    for s in FFT_S:
        for i in fft_p_grid(N, FFT_P_POINTS):
            n, capped, var_dw = fft_runs(N, s, int(i))
            fft_plan[(N, float(s), int(i))] = dict(n=n, capped=capped, var_dw=var_dw)
fft_todo = [k for k, v in fft_plan.items() if k not in FFT_RES or FFT_RES[k]["n"] != v["n"]]
print(f"Fisher's theorem: {len(fft_plan)} points, {len(fft_todo)} to simulate, "
      f"{sum(fft_plan[k]['n'] for k in fft_todo):,} single-generation replicates")
fft_out = run_parallel(fft_point, [(k[0], k[1], k[2], fft_plan[k]["n"],
                                    np.random.SeedSequence([SEED, 7, k[0], k[2], int(round(k[1] * 1e9)) & 0xFFFFFFFF,
                                                            int(k[1] < 0)]))
                                   for k in fft_todo])
FFT_RES.update(zip(fft_todo, fft_out))
with open(FFT_CACHE, "wb") as fh:
    pickle.dump(FFT_RES, fh)

FFT_FORMULAS = {
    "fft": dict(color="#000000", label=_entry(
        "Fisher's fundamental theorem, discrete generations  (reference for the lower panel)",
        r"$\mathrm{E}\left[\Delta\bar{w}\mid p\right]=\dfrac{\mathrm{Var}(w)}{\bar{w}},\quad"
        r"\bar{w}=1+s\,p,\quad \mathrm{Var}(w)=s^{2}p(1-p)$",
        "any selection s, any population size N, any allele frequency p; exact in expectation for one generation "
        "of selection followed by random sampling (constant fitness)")),
    "naive": dict(color="#0047FF", label=_entry(
        "Fisher's fundamental theorem, continuous-time form",
        r"$\mathrm{E}\left[\Delta\bar{w}\mid p\right]\approx\mathrm{Var}(w)=s^{2}p(1-p)$",
        "weak selection (|s| ≪ 1, so w̄ ≈ 1), continuous time; any N, any p")),
}
FFT_SOURCES = {
    "fft":   "not in either uploaded paper: Fisher (1930), The Genetical Theory of Natural Selection; "
             "discrete-generation form Δw̄ = Var(w)/w̄ (see e.g. Crow & Kimura 1970)",
    "naive": "not in either uploaded paper: Fisher (1930), The Genetical Theory of Natural Selection "
             "(rate of increase in mean fitness = genetic variance in fitness)",
}


def make_fft_graph(number=7):
    for j, N in enumerate(FFT_N, 1):
        f, (ax, axr) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                    gridspec_kw=dict(height_ratios=list(PANEL_HEIGHTS), hspace=0.06))
        p_th = expit(np.linspace(np.log(1 / (N - 1)), np.log(N - 1), 300))
        zf, zlo, zhi, zmid, series_handles, lines = [], [], [], [], [], ["Single-generation replicates", ""]
        tot_n = tot_pts = 0
        for ci, s in enumerate(FFT_S):
            ls = SERIES_LINESTYLES[ci % len(SERIES_LINESTYLES)]
            mk = SERIES_MARKERS[ci % len(SERIES_MARKERS)]
            th = fft_prediction(p_th, s)
            for fk, st in FFT_FORMULAS.items():
                ax.plot(p_th, th[fk], ls=ls, lw=FORMULA_LW, color=st["color"], zorder=3)
                rel = th[fk] / th["fft"] - 1
                axr.plot(p_th, rel, ls=ls, lw=FORMULA_LW, color=st["color"], zorder=3)
                zf.extend(rel)
            ks = sorted(k for k in fft_plan if k[0] == N and k[1] == float(s))
            p_s = np.array([k[2] / N for k in ks])
            obs = np.array([FFT_RES[k]["mean"] for k in ks]); se = np.array([FFT_RES[k]["se"] for k in ks])
            ref = fft_prediction(p_s, s)["fft"]
            capped = np.array([fft_plan[k]["capped"] for k in ks])
            vr = np.array([FFT_RES[k]["var"] / fft_plan[k]["var_dw"] for k in ks])
            rel, rerr = obs / ref - 1, se / ref
            for cf in (False, True):
                mm = (capped == cf) & np.isfinite(obs)
                if not mm.any():
                    continue
                mfc = "white" if cf else SIM_COLOR
                good = mm & (obs > 0)
                ax.errorbar(p_s[good], obs[good], yerr=[np.minimum(se, 0.95 * obs)[good], se[good]], fmt="none",
                            ecolor=SIM_COLOR, elinewidth=0.6, capsize=1.5, alpha=SIM_ALPHA, zorder=5)
                ax.plot(p_s[good], obs[good], ls="none", marker=mk, ms=SIM_MS, mfc=mfc, mec=SIM_COLOR,
                        mew=0.8, alpha=SIM_ALPHA, zorder=6)
                axr.errorbar(p_s[mm], rel[mm], yerr=rerr[mm], fmt="none", ecolor=SIM_COLOR,
                             elinewidth=0.6, capsize=1.5, alpha=SIM_ALPHA, zorder=5)
                axr.plot(p_s[mm], rel[mm], ls="none", marker=mk, ms=SIM_MS, mfc=mfc, mec=SIM_COLOR,
                         mew=0.8, alpha=SIM_ALPHA, zorder=6)
                zlo.extend(rel[mm] - rerr[mm]); zhi.extend(rel[mm] + rerr[mm]); zmid.extend(rel[mm])
            ns = [FFT_RES[k]["n"] for k in ks]
            tot_n += sum(ns); tot_pts += len(ns)
            lines += [f"s = {s:g}", f"  {sum(ns):>12,} replicates, {len(ns)} points",
                      f"  per point: {min(ns):,} – {max(ns):,}",
                      f"  obs/exp sampling var: {np.mean(vr):.3f}"]
            if capped.any():
                lines.append(f"  at run cap ({MAX_RUNS:,}): {int(capped.sum())}")
            series_handles.append(Line2D([], [], color="k", ls=ls, lw=1.0, marker=mk, ms=5,
                                         mfc=SIM_COLOR, mec=SIM_COLOR, label=f"s = {s:g}"))
        lines += ["", f"Total: {tot_n:,} replicates", f"       {tot_pts} points"]

        ax.set_xscale("logit")
        ax.set_yscale("log")
        ax.set_ylabel("Mean change in mean fitness\nper generation, E[Δw̄]")
        ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.3)
        ax.tick_params(labelbottom=False)
        ax.set_title(f"Graph {number}.{j}: Fisher's fundamental theorem — one generation of selection + sampling\n"
                     f"Population size N = {N}", fontsize=12)
        apply_resid_zoom(axr, zf, zlo, zhi, zmid)
        axr.axhline(0, color="k", lw=0.5)
        axr.set_ylabel("% deviation from\nVar(w)/w̄ (zoomed)", fontsize=9)
        axr.set_xlabel("Frequency of allele 1 before selection, p (logit scale)")
        axr.grid(True, which="major", ls="--", lw=0.4, alpha=0.3)

        leg1 = ax.legend(handles=series_handles, title="Selection coefficient\n(line style / marker)",
                         loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, title_fontsize=9, handlelength=4.5)
        ax.add_artist(leg1)
        axr.text(1.01, 1.0, "\n".join(lines), transform=axr.transAxes, va="top", ha="left", fontsize=8.5,
                 family="monospace", bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#bbbbbb"))
        hs = [Line2D([], [], color=st["color"], lw=3, label=st["label"] + "\n" + textwrap.fill(
                  "Source: " + FFT_SOURCES[fk], 95, subsequent_indent="    "))
              for fk, st in FFT_FORMULAS.items()]
        hs.append(Line2D([], [], color=SIM_COLOR, ls="none", marker="o", alpha=SIM_ALPHA,
                         label=f"Simulation ± 1 SE (target RSE {TARGET_RSE:.0%}); marker shape = series"))
        hs.append(Line2D([], [], color=SIM_COLOR, ls="none", marker="o", mfc="white",
                         label=f"Simulation, run cap {MAX_RUNS:,} reached (noise target not met)"))
        f.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10,
                 title="Colour = formula   ·   each entry: name / equation / population it was designed for",
                 title_fontsize=10.5, labelspacing=1.3, columnspacing=2.5, handlelength=3, borderpad=1.0,
                 alignment="left")
        f.savefig(os.path.join(OUT_DIR, f"graph{number}_fisher_theorem_{j}_N{N}.png"), dpi=200, bbox_inches="tight")
        if IN_NOTEBOOK and SHOW_INDIVIDUAL_FIGURES:
            plt.show()
        plt.close(f)


make_fft_graph(7)

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
    print(f"{len(skipped)} points not simulated for fixation (expected < {MIN_EXPECTED_FIXATIONS} fixations even at "
          f"{MAX_RUNS:,} runs; only short loss-only runs were done), e.g.: " + ", ".join(f"(N={N}, s={s:g}, i0={i})" for N, s, i in skipped[:8]))
print(f"Figures written to {OUT_DIR}")

# %%
