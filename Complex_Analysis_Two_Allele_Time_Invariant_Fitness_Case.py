import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time, os, warnings
from sklearn.metrics import r2_score, mean_absolute_error

from wf_core import (ANALYSIS_FUNCTIONS, ANALYSIS_COLUMN_MAP, ConstantFitness)

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_RUNS = 2000
N_POINTS = 16

# ---------------------------------------------------------------------------
# Sweep configuration per analysis function.
#   pop:  vary population size N (size1 fixed at 1, size2 = N-1), s held fixed
#   sel:  vary selection coefficient s (fitness1 = 1+s, fitness2 = 1), N held fixed
# Ranges were chosen per-function based on the validity domain of each
# formula (e.g. weak-selection formulas need s << 1/Ne; giant-population
# formulas are shown converging as N grows; "any" formulas get wide ranges).
# ---------------------------------------------------------------------------
SWEEPS = {
    'PFn___':  {'pop': {'s': 0.0,  'N': ('log', 10, 3000)},
                'sel': {'N': 300,  's': ('lin', -0.5, 2.0)}},
    'PFbg_1':  {'pop': {'s': 0.3,  'N': ('log', 50, 5000)},
                'sel': {'N': 3000, 's': ('log', 0.01, 2.0)}},
    'PFbgw1':  {'pop': {'s': 0.02, 'N': ('log', 100, 5000)},
                'sel': {'N': 3000, 's': ('log', 0.001, 0.15)}},
    'PFb_w_':  {'pop': {'s': 0.02, 'N': ('log', 10, 2000)},
                'sel': {'N': 300,  's': ('log', 0.001, 0.15)}},
    'PF____':  {'pop': {'s': 0.15, 'N': ('log', 10, 3000)},
                'sel': {'N': 300,  's': ('lin', -0.8, 3.0)}},
    'TFb_w1':  {'pop': {'s': 0.004,'N': ('log', 10, 200)},
                'sel': {'N': 100,  's': ('log', 0.0005, 0.009)}},
    'TFb_s1':  {'pop': {'s': 0.5,  'N': ('log', 10, 3000)},
                'sel': {'N': 300,  's': ('log', 0.05, 3.0)}},
    'TFn__1a': {'pop': {'s': 0.0,  'N': ('log', 10, 2000)},
                'sel': {'N': 200,  's': ('lin', -0.5, 2.0)}},
    'TFn__1b': {'pop': {'s': 0.0,  'N': ('log', 10, 2000)},
                'sel': {'N': 200,  's': ('lin', -0.5, 2.0)}},
    'TFng_1':  {'pop': {'s': 0.0,  'N': ('log', 50, 5000)},
                'sel': {'N': 1000, 's': ('lin', -0.5, 2.0)}},
    'TLn__1':  {'pop': {'s': 0.0,  'N': ('log', 10, 2000)},
                'sel': {'N': 200,  's': ('lin', -0.5, 2.0)}},
    'TC':      {'pop': {'s': 0.2,  'N': ('log', 20, 2000)},
                'sel': {'N': 300,  's': ('log', 0.02, 2.0)}},
    'FFT':     {'pop': {'s': 0.1,  'N': ('log', 10, 2000)},
                'sel': {'N': 300,  's': ('lin', -0.8, 3.0)}},
}

CAVEATS = {
    'TFb_w1': "Note: theory assumes weak selection (s << 1/Ne); points beyond that break down and predicted values become undefined (gap in line).",
    'TFb_s1': "Note: predicted curve is the time to fixation *given the beneficial allele fixes*; the simulated average includes fast early losses too, so a systematic gap is expected.",
    'TFn__1a': "Note: predicted curve is the time to fixation *given fixation occurs*; the simulated average includes fast early losses too, so a systematic gap is expected.",
    'TFn__1b': "Note: predicted curve is the time to fixation *given fixation occurs*; the simulated average includes fast early losses too, so a systematic gap is expected.",
    'TFng_1': "Note: predicted curve is the time to fixation *given fixation occurs*; the simulated average includes fast early losses too, so a systematic gap is expected.",
}


def make_values(spacing, lo, hi, n=N_POINTS):
    if spacing == 'log':
        return np.geomspace(lo, hi, n)
    else:
        return np.linspace(lo, hi, n)


def decide_scale(values):
    finite = np.array([v for v in values if np.isfinite(v)])
    if finite.size == 0:
        return 'linear'
    positive = finite[finite > 0]
    if positive.size == finite.size and positive.size > 0:
        vmax, vmin = positive.max(), positive.min()
        if vmin > 0 and vmax / vmin >= 20:
            return 'log'
    return 'linear'


def run_pop_sweep(func_name, s_fixed, N_values):
    func = ANALYSIS_FUNCTIONS[func_name]
    col = ANALYSIS_COLUMN_MAP[func_name]
    xs, obs, pred = [], [], []
    for N in N_values:
        N = int(round(N))
        if N < 2:
            continue
        K = np.array([1, N - 1])
        Ff = [ConstantFitness(1 + s_fixed), ConstantFitness(1.0)]
        r = func(K, Ff, NUM_RUNS)
        xs.append(N)
        if func_name == 'FFT':
            obs.append(r['difference'])
            pred.append(0.0)
        else:
            obs.append(r.get(col['observed'], np.nan))
            pred.append(r.get(col['predicted'], np.nan))
    return np.array(xs, dtype=float), np.array(obs, dtype=float), np.array(pred, dtype=float)


def run_sel_sweep(func_name, N_fixed, s_values):
    func = ANALYSIS_FUNCTIONS[func_name]
    col = ANALYSIS_COLUMN_MAP[func_name]
    xs, obs, pred = [], [], []
    N = int(N_fixed)
    for s in s_values:
        K = np.array([1, N - 1])
        Ff = [ConstantFitness(1 + s), ConstantFitness(1.0)]
        r = func(K, Ff, NUM_RUNS)
        xs.append(s)
        if func_name == 'FFT':
            obs.append(r['difference'])
            pred.append(0.0)
        else:
            obs.append(r.get(col['observed'], np.nan))
            pred.append(r.get(col['predicted'], np.nan))
    return np.array(xs, dtype=float), np.array(obs, dtype=float), np.array(pred, dtype=float)


def plot_sweep(func_name, sweep_type, xs, obs, pred, xlabel, out_path):
    col = ANALYSIS_COLUMN_MAP[func_name]
    title = col['title']
    ylabel = col['ylabel']

    finite_mask_obs = np.isfinite(obs)
    finite_mask_pred = np.isfinite(pred)
    both_mask = finite_mask_obs & finite_mask_pred

    r2 = mae = np.nan
    if both_mask.sum() >= 2:
        try:
            r2 = r2_score(obs[both_mask], pred[both_mask])
        except Exception:
            r2 = np.nan
        mae = mean_absolute_error(obs[both_mask], pred[both_mask])

    scale_x = decide_scale(xs)
    scale_y = decide_scale(np.concatenate([obs[finite_mask_obs], pred[finite_mask_pred]])) \
        if (finite_mask_obs.any() or finite_mask_pred.any()) else 'linear'

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs[finite_mask_pred], pred[finite_mask_pred], '-', color='#c0392b', linewidth=2,
            label='Theoretical prediction', zorder=2)
    ax.scatter(xs[finite_mask_obs], obs[finite_mask_obs], color='#2980b9', s=32, alpha=0.85,
               edgecolor='white', linewidth=0.5, label=f'Simulated ({NUM_RUNS} runs/point)', zorder=3)

    if scale_x == 'log':
        ax.set_xscale('log')
    if scale_y == 'log':
        ax.set_yscale('log')

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    metrics_str = f"R²={r2:.3f}, MAE={mae:.4g}" if np.isfinite(r2) else (f"MAE={mae:.4g}" if np.isfinite(mae) else "")
    ax.set_title(f"{title}\n{func_name} — {sweep_type}" + (f"  ({metrics_str})" if metrics_str else ""), fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='best')

    caveat = CAVEATS.get(func_name)
    if caveat:
        fig.text(0.5, 0.005, caveat, ha='center', fontsize=7.5, style='italic', color='#555555', wrap=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1] if caveat else None)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    t_start = time.time()
    idx = 0
    func_order = list(ANALYSIS_FUNCTIONS.keys())
    for fname in func_order:
        cfg = SWEEPS[fname]

        # population size sweep
        idx += 1
        spacing, lo, hi = cfg['pop']['N']
        N_values = make_values(spacing, lo, hi)
        s_fixed = cfg['pop']['s']
        t0 = time.time()
        xs, obs, pred = run_pop_sweep(fname, s_fixed, N_values)
        out_path = os.path.join(OUT_DIR, f"{idx:02d}_{fname}_population_size.png")
        plot_sweep(fname, f"varying population size (s fixed at {s_fixed})", xs, obs, pred,
                   "Population Size (N)", out_path)
        print(f"[{idx:02d}] {fname} pop sweep done in {time.time()-t0:.1f}s -> {out_path}")

        # selection coefficient sweep
        idx += 1
        spacing, lo, hi = cfg['sel']['s']
        s_values = make_values(spacing, lo, hi)
        N_fixed = cfg['sel']['N']
        t0 = time.time()
        xs, obs, pred = run_sel_sweep(fname, N_fixed, s_values)
        out_path = os.path.join(OUT_DIR, f"{idx:02d}_{fname}_beneficialness.png")
        plot_sweep(fname, f"varying selection coefficient s (N fixed at {N_fixed})", xs, obs, pred,
                   "Selection Coefficient s = (fitness₁ − fitness₂) / fitness₂", out_path)
        print(f"[{idx:02d}] {fname} sel sweep done in {time.time()-t0:.1f}s -> {out_path}")

    print(f"\nTotal: {idx} graphs in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()