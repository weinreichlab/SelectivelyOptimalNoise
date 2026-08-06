import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import warnings

from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import r2_score, mean_absolute_error

import traceback
from concurrent.futures import as_completed

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
QUICK_TEST = False   # True: 3 points/sweep, 50 runs/point -- finishes in minutes, catches crashes


SWEEPS = {
    'PFn___':  {'pop': {'s': 0.0,  'N': ('log', 10, 7000), 'size1': 8},
                },
    'PFbg_1':  {'pop': {'s': 0.3,  'N': ('log', 3000, 7000), 'num_runs': 20000},
                'sel': {'N': 5000, 's': ('log', 0.01, 2.0), 'num_runs': 20000}},
    'PFbgw1':  {'pop': {'s': 0.02, 'N': ('log', 3000, 7000), 'num_runs': 20000},
                'sel': {'N': 5000, 's': ('log', 0.001, 0.1), 'num_runs': 20000}},
    'PFb_w_':  {'pop': {'s': 0.01, 'N': ('log', 10, 7000), 'size1': 8},
            'sel': {'N': 300,  's': ('lin', -0.5, 3.0), 'size1': 8}},
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
    'TLn__1':  {'pop': {'s': 0.0,  'N': ('log', 500, 15000), 'num_runs': 40000},
                },
    'TC':      {'pop': {'s': 0.05,  'N': ('log', 1000, 7000),'size1': 20},
                'sel': {'N': 5000,  's': ('log', 0.005, 0.2),'size1': 20, 'num_runs': 20000}},
    'FFT':     {'pop': {'s': 0.1,  'N': ('log', 10, 2000),'size1': 8},
                'sel': {'N': 300,  's': ('lin', -1.0, 3.0),'size1': 8}},
    'TF__w1': {'pop': {'s': 0.01, 'N': ('log', 10, 2000), 'size1': 5, 'num_runs': 10000},
               'sel': {'N': 300, 's': ('lin', -0.1, 0.1), 'size1': 5, 'num_runs': 10000}},
    'TF____': {'pop': {'s': 0.05, 'N': ('log', 10, 2000), 'size1': 5, 'num_runs': 10000},
               'sel': {'N': 300, 's': ('lin', -0.2, 1.5), 'size1': 8, 'num_runs': 10000}},
}


CAVEATS = {

    }

Y_LIMITS = {
    ('PFbg_1', 'pop'): (0.0, 0.7),
    ('PFbgw1', 'pop'): (0.0, 0.3),
    ('PF____', 'pop'): (0.0, 1.0),
}

X_SCALES = {
    ('TFng_1', 'pop'): 'log',
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
            obs.append(r['z_mean'])
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
            obs.append(r['z_mean'])
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

    scale_x = X_SCALES.get((func_name, sweep_kind), decide_scale(xs))
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


REQUIRED_COLUMN_KEYS = ('observed', 'predicted', 'title', 'ylabel')


def check_config(func_order):
    problems = []
    for fname in func_order:
        if fname not in SWEEPS:
            problems.append(f"{fname}: missing from SWEEPS")
        col = ANALYSIS_COLUMN_MAP.get(fname)
        if col is None:
            problems.append(f"{fname}: missing from ANALYSIS_COLUMN_MAP")
            continue
        for key in REQUIRED_COLUMN_KEYS:
            if key not in col:
                problems.append(f"{fname}: ANALYSIS_COLUMN_MAP entry has no '{key}'")
    if problems:
        raise ValueError("Configuration errors:\n  " + "\n  ".join(problems))


def build_jobs(func_order):
    jobs, idx = [], 0
    for fname in func_order:
        cfg = SWEEPS[fname]
        for kind in ('pop', 'sel'):
            if kind not in cfg:
                continue
            idx += 1
            c = cfg[kind]
            num_runs = c.get('num_runs', NUM_RUNS)
            if kind == 'pop':
                values = make_values(*c['N'])
                args = (fname, c['s'], values)
                fn = run_pop_sweep
                sweep_type = f"varying population size (s fixed at {c['s']})"
                xlabel = "Population Size (N)"
                suffix = "population_size"
            else:
                values = make_values(*c['s'])
                args = (fname, c['N'], values)
                fn = run_sel_sweep
                sweep_type = f"varying selection coefficient s (N fixed at {c['N']})"
                xlabel = "Selection Coefficient s = (fitness₁ − fitness₂) / fitness₂"
                suffix = "beneficialness"

            if QUICK_TEST:
                args = args[:2] + (values[[0, len(values) // 2, -1]],)
                num_runs = min(num_runs, 50)

            jobs.append(dict(
                fname=fname, kind=kind, fn=fn, args=args,
                kwargs=dict(initial_count=c.get('size1', 1), num_runs=num_runs, use_parallel_core=False),
                sweep_type=sweep_type, xlabel=xlabel,
                out_path=os.path.join(OUT_DIR, f"{idx:02d}_{fname}_{suffix}.png"),
            ))
    return jobs


def finish_job(job, get_result, failures, t0):
    label = f"{job['fname']} ({job['kind']})"
    try:
        xs, obs, pred = get_result()
        np.savez(job['out_path'].replace('.png', '.npz'), xs=xs, obs=obs, pred=pred)
        plot_sweep(job['fname'], job['sweep_type'], xs, obs, pred, job['xlabel'], job['out_path'],
                   num_runs=job['kwargs']['num_runs'], sweep_kind=job['kind'])
        print(f"[{time.time() - t0:7.1f}s] done: {label} -> {job['out_path']}", flush=True)
    except Exception:
        print(f"[{time.time() - t0:7.1f}s] FAILED: {label}\n{traceback.format_exc()}", flush=True)
        failures.append(label)


def main():
    t_start = time.time()
    func_order = list(ANALYSIS_FUNCTIONS.keys())
    check_config(func_order)                     # fails in < 1 s instead of after hours
    jobs = build_jobs(func_order)
    failures = []

    if use_parallel:
        print(f"Submitting {len(jobs)} sweeps…", flush=True)
        with initiate_executor() as executor:
            future_to_job = {executor.submit(j['fn'], *j['args'], **j['kwargs']): j for j in jobs}
            for future in as_completed(future_to_job):
                finish_job(future_to_job[future], future.result, failures, t_start)
    else:
        for j in jobs:
            finish_job(j, lambda j=j: j['fn'](*j['args'], **j['kwargs']), failures, t_start)

    print(f"\n{len(jobs) - len(failures)}/{len(jobs)} graphs in {time.time() - t_start:.1f}s")
    if failures:
        print("Failed sweeps:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
