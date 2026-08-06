import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import warnings

from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import r2_score, mean_absolute_error

from wf_core import (
    ANALYSIS_FUNCTIONS,
    ANALYSIS_COLUMN_MAP,
    ConstantFitness,
    initiate_executor,
)

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# multiprocessing
use_parallel: bool = True

NUM_RUNS = 3000
N_POINTS = 40


SWEEPS = {
    'PFn___':  {'pop': {'s': 0.0,  'N': ('log', 10, 7000), 'size1': 8},
                },
    'PFbg_1':  {'pop': {'s': 0.3,  'N': ('log', 3000, 7000), 'num_runs': 20000},
                'sel': {'N': 5000, 's': ('log', 0.01, 2.0), 'num_runs': 20000}},
    'PFbgw1':  {'pop': {'s': 0.02, 'N': ('log', 3000, 7000), 'num_runs': 20000},
                'sel': {'N': 5000, 's': ('log', 0.001, 0.1), 'num_runs': 20000}},
    'PFb_w_':  {'pop': {'s': 0.01, 'N': ('log', 10, 7000),'size1': 8},
                'sel': {'N': 300,  's': ('log', -1.0, 3.0),'size1': 8}},
    'PF____':  {'pop': {'s': 0.15, 'N': ('log', 10, 200),'size1': 8, 'num_runs': 20000},
                'sel': {'N': 300,  's': ('lin', -0.2, 1.0),'size1': 20}},
    'TFb_w1':  {'pop': {'s': 0.01,'N': ('log', 10, 7000), 'num_runs': 20000},
                'sel': {'N': 300,  's': ('log', 0.001, 0.1), 'num_runs': 20000}},
    'TFb_s1':  {'pop': {'s': 2.0,  'N': ('log', 10, 7000), 'num_runs': 20000},
                'sel': {'N': 300,  's': ('log', 1.0, 4.0), 'num_runs': 20000}},
    'TFn__1a': {'pop': {'s': 0.0,  'N': ('log', 10, 7000)},
               },
    'TFn__1b': {'pop': {'s': 0.0,  'N': ('log', 10, 7000)},
                },
    'TFng_1':  {'pop': {'s': 0.0,  'N': ('log', 2000, 4000), 'num_runs': 30000},
                },
    'TLn__1':  {'pop': {'s': 0.0,  'N': ('log', 500, 15000), 'num_runs': 20000},
                },
    'TC':      {'pop': {'s': 0.01,  'N': ('log', 10, 7000),'size1': 8},
                'sel': {'N': 300,  's': ('log', 0.001, 0.1),'size1': 8}},
    'FFT':     {'pop': {'s': 0.1,  'N': ('log', 10, 2000),'size1': 8},
                'sel': {'N': 300,  's': ('lin', -1.0, 3.0),'size1': 8}},
}


CAVEATS = {

    }

Y_LIMITS = {
    ('PFbg_1', 'pop'): (0.0, 0.7),
    ('PFbgw1', 'pop'): (0.0, 0.3),
    ('PF____', 'pop'): (0.0, 1.0),
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


def run_pop_sweep(func_name, s_fixed, N_values, initial_count=1, num_runs=NUM_RUNS, use_parallel_core: bool = False,
                  executor: ProcessPoolExecutor = None):
    func = ANALYSIS_FUNCTIONS[func_name]
    col = ANALYSIS_COLUMN_MAP[func_name]
    xs, obs, pred = [], [], []
    if use_parallel_core and executor is None:
        executor = initiate_executor()
    for N in N_values:
        N = int(round(N))
        if N < 2 or initial_count >= N:
            continue
        K = np.array([initial_count, N - initial_count])
        Ff = [ConstantFitness(1 + s_fixed), ConstantFitness(1.0)]
        r = func(K, Ff, num_runs, use_parallel_core=use_parallel_core, executor=executor)
        xs.append(N)
        if func_name == 'FFT':
            obs.append(r['difference'])
            pred.append(0.0)
        else:
            obs.append(r.get(col['observed'], np.nan))
            pred.append(r.get(col['predicted'], np.nan))
    return np.array(xs, dtype=float), np.array(obs, dtype=float), np.array(pred, dtype=float)


def run_sel_sweep(func_name, N_fixed, s_values, initial_count=1, num_runs=NUM_RUNS, use_parallel_core: bool = False,
                  executor: ProcessPoolExecutor = None):
    func = ANALYSIS_FUNCTIONS[func_name]
    col = ANALYSIS_COLUMN_MAP[func_name]
    xs, obs, pred = [], [], []
    N = int(N_fixed)
    if use_parallel_core and executor is None:
        executor = initiate_executor()
    for s in s_values:
        K = np.array([initial_count, N - initial_count])
        Ff = [ConstantFitness(1 + s), ConstantFitness(1.0)]
        r = func(K, Ff, num_runs, use_parallel_core=use_parallel_core, executor=executor)
        xs.append(s)
        if func_name == 'FFT':
            obs.append(r['difference'])
            pred.append(0.0)
        else:
            obs.append(r.get(col['observed'], np.nan))
            pred.append(r.get(col['predicted'], np.nan))
    return np.array(xs, dtype=float), np.array(obs, dtype=float), np.array(pred, dtype=float)


def plot_sweep(func_name, sweep_type, xs, obs, pred, xlabel, out_path, num_runs=NUM_RUNS, sweep_kind=None):
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
               edgecolor='white', linewidth=0.5, label=f'Simulated ({num_runs} runs/point)', zorder=3)

    if scale_x == 'log':
        ax.set_xscale('log')
    if scale_y == 'log':
        ax.set_yscale('log')

    ylim = Y_LIMITS.get((func_name, sweep_kind))
    if ylim:
        ax.set_ylim(*ylim)

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
    with initiate_executor() as executor:
        # generate results
        results_pop = []
        results_sel = []
        futures_pop = []
        futures_sel = []
        # track which functions have 'sel' sweep (and order)
        # so plotting loop can index results_sel/futures_sel properly
        sel_func_order = []
        out_paths_pop = {}
        out_paths_sel = {}
        t0 = time.time()
        if use_parallel:
            print("Multiprocessing pop and sel sweeps… ", flush=True)
        for fname in func_order:
            cfg = SWEEPS[fname]

            # population size sweep
            idx += 1
            spacing, lo, hi = cfg['pop']['N']
            N_values = make_values(spacing, lo, hi)
            s_fixed = cfg['pop']['s']
            size1_pop = cfg['pop'].get('size1', 1)
            out_paths_pop[fname] = os.path.join(OUT_DIR, f"{idx:02d}_{fname}_population_size.png")
            num_runs_pop = cfg['pop'].get('num_runs', NUM_RUNS)
            if use_parallel:
                futures_pop.append(executor.submit(run_pop_sweep, fname, s_fixed, N_values,
                                                   initial_count=size1_pop, num_runs=num_runs_pop, use_parallel_core=False))
            else:
                t0 = time.time()
                results_pop.append(run_pop_sweep(fname, s_fixed, N_values,
                                                 initial_count=size1_pop, num_runs=num_runs_pop, use_parallel_core=False))
                print(f"[{idx:02d}] {fname} pop sweep done in {time.time() - t0:.1f}s -> {out_paths_pop[fname]}")

            # selection coefficient sweep 
            if 'sel' in cfg:
                idx += 1
                spacing, lo, hi = cfg['sel']['s']
                s_values = make_values(spacing, lo, hi)
                N_fixed = cfg['sel']['N']
                size1_sel = cfg['sel'].get('size1', 1)
                out_paths_sel[fname] = os.path.join(OUT_DIR, f"{idx:02d}_{fname}_beneficialness.png")
                sel_func_order.append(fname)
                num_runs_pop = cfg['sel'].get('num_runs', NUM_RUNS)
                if use_parallel:
                    futures_sel.append(executor.submit(run_sel_sweep, fname, N_fixed, s_values,
                                                       initial_count=size1_sel, num_runs=num_runs_pop, use_parallel_core=False))
                else:
                    t0 = time.time()
                    results_sel.append(run_sel_sweep(fname, N_fixed, s_values,
                                                     initial_count=size1_sel,num_runs=num_runs_pop, use_parallel_core=False))
                    print(f"[{idx:02d}] {fname} sel sweep done in {time.time() - t0:.1f}s -> {out_paths_sel[fname]}")

        if use_parallel:
            results_pop = [future.result() for future in futures_pop]
            results_sel = [future.result() for future in futures_sel]
            print(f"Parallel pop and sel sweeps done in {time.time() - t0:.1f}s")

    idx = 0
    for k, fname in enumerate(func_order):
        cfg = SWEEPS[fname]

        # population size sweep
        idx += 1
        s_fixed = cfg['pop']['s']
        xs, obs, pred = results_pop[k]
        out_path = out_paths_pop[fname]
        plot_sweep(fname, f"varying population size (s fixed at {s_fixed})", xs, obs, pred,
                   "Population Size (N)", out_path, sweep_kind='pop')

        # selection coefficient sweep (optional)
        if 'sel' in cfg:
            idx += 1
            sel_k = sel_func_order.index(fname)
            N_fixed = cfg['sel']['N']
            xs, obs, pred = results_sel[sel_k]
            out_path = out_paths_sel[fname]
            plot_sweep(fname, f"varying selection coefficient s (N fixed at {N_fixed})", xs, obs, pred,
                       "Selection Coefficient s = (fitness₁ − fitness₂) / fitness₂", out_path, sweep_kind='sel')

    print(f"\nTotal: {idx} graphs in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
