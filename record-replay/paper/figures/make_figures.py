"""Generate the figures of autotuning_study.tex from the experiment archives.

Run inside rr_current.sif:  python3 make_figures.py
Outputs PDF files next to this script. Colors follow the reference
categorical palette in fixed slot order (validated by the palette's authors;
the first three slots pass all pairs, used for scatter plots).
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
X = '/data2/nmn/rr-metrics/experiment-archives'
S = f'{X}/search-study-2026-10-09'

SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#fcfcfb'
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
CLASS_COLOR = {'locality': SLOT[0], 'register': SLOT[1], 'compute': SLOT[2]}
CLASS_MARK = {'locality': 'o', 'register': 's', 'compute': '^'}
CLASS_LABEL = {'locality': 'Locality bound', 'register': 'Register bound', 'compute': 'Compute saturated'}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'lines.linewidth': 1.6, 'legend.frameon': False, 'figure.dpi': 200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
})
ONE, TWO = 3.35, 6.9   # column widths in inches


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)
    print('wrote', name)


def gmean(x):
    return float(np.exp(np.log(np.asarray(x, float)).mean()))


# ---------------------------------------------------------------- 1. clock ramp
def fig_clock_ramp():
    def parse(f):
        rows = []
        for line in open(f):
            if line.startswith('P'):
                rows.append([float(v) * 1e3 for v in line.split()[1:]])
        return rows
    cold = parse(f'{X}/measurement-fix-2026-10-09/probe2-l1721.log')
    warm = parse(f'{X}/measurement-fix-2026-10-09/probe3-l1721-warm300.log')
    fig, ax = plt.subplots(figsize=(ONE, 2.0))
    for i, r in enumerate(cold):
        ax.plot(range(1, len(r) + 1), r, color=SLOT[0], lw=1.1, alpha=0.85,
                label='Cold start (8 processes)' if i == 0 else None)
    for i, r in enumerate(warm):
        ax.plot(range(1, len(r) + 1), r, color=SLOT[1], lw=1.1,
                label='After ~3 s warm-up (8 processes)' if i == 0 else None)
    ax.set_xlabel('Launch index within one replay process')
    ax.set_ylabel('Kernel time (ms)')
    ax.set_ylim(0, 20)
    ax.legend(loc='lower center', ncol=1)
    ax.set_title('LULESH l1721, same configuration')
    save(fig, 'clock_ramp')


# ---------------------------------------------------------------- 2. winner's curse
def fig_winners_curse():
    df = pd.read_csv(f'{X}/optimizer-study-2026-10-09/dataset.csv')
    v = df[(df.bench == 'LULESH') & df.valid]
    rows = []
    for line, g in v.groupby('line'):
        b = {o: g[(g.objective == o) & g.baseline].dur_median.mean() for o in ['speedup', 'energy']}
        for a, o in [('speedup', 'energy'), ('energy', 'speedup')]:
            ga = g[(g.objective == a) & ~g.baseline]
            go = g[(g.objective == o) & ~g.baseline].set_index('key')
            for _, r in ga.nsmallest(5, 'dur_median').iterrows():
                if r.key in go.index and not np.isnan(b[a]) and not np.isnan(b[o]):
                    rows.append((b[a] / r.dur_median, b[o] / go.loc[r.key].dur_median))
    c = np.array(rows)
    fig, ax = plt.subplots(figsize=(ONE, 2.4))
    ax.scatter(c[:, 0], c[:, 1], s=16, color=SLOT[0], edgecolor=SURF, linewidth=0.6, zorder=3)
    lim = [0.75, 2.4]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls='--')
    ax.text(2.25, 2.33, 'replicated = claimed', color=INK2, fontsize=6.5, ha='right', va='bottom')
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ticks = [0.8, 1.0, 1.25, 1.5, 2.0]
    for f in (ax.set_xticks, ax.set_yticks):
        f(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks]); ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Claimed speedup (search that found it)')
    ax.set_ylabel('Speedup in the other search')
    ax.set_title(f'Top-5 configurations, legacy measurement (n={len(c)})')
    save(fig, 'winners_curse')


# ---------------------------------------------------------------- 3. measurement validation
def fig_measurement():
    s = pd.read_csv(f'{X}/measurement-fix-2026-10-09/samples.csv')
    s = s[(s.rc == 0) & ~s.contended]
    cv = s.groupby(['mode', 'line', 'config']).duration.agg(lambda x: x.std() / x.mean())
    med = cv.groupby(['mode', 'line']).median().unstack(0) * 100
    fig, ax = plt.subplots(figsize=(ONE, 2.0))
    xs = np.arange(len(med))
    w = 0.38
    ax.bar(xs - w / 2 - 0.01, med['legacy'], w, color=SLOT[0], label='Legacy (v2)')
    ax.bar(xs + w / 2 + 0.01, med['v3'], w, color=SLOT[1], label='New (v3)')
    for x, val in zip(xs, med['v3']):
        ax.text(x + w / 2, val * 1.15, f'{val:.2f}', ha='center', va='bottom', fontsize=5.5, color=INK2)
    ax.set_yscale('log')
    ax.set_xticks(xs); ax.set_xticklabels([f'l{l}' for l in med.index])
    ax.set_ylabel('Cross-process CV of time (%)')
    ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(['0.1', '1', '10'])
    ax.set_ylim(0.08, 60)
    ax.legend(loc='upper center', ncol=2)
    ax.set_title('Median over five configurations, six rounds, idle GPU')
    save(fig, 'measurement_cv')


# ---------------------------------------------------------------- 4. anytime curves
def fig_anytime():
    cfg = pd.read_csv(f'{S}/configs-v2.csv')
    lut = {(r.line, r.key): (r.speedup if r.valid else 0.0) for r in cfg.itertuples()}
    best = cfg[cfg.valid].groupby('line').speedup.max()
    methods = [('bo-speedup', 'BO (existing)'), ('tpe-speedup', 'TPE'), ('motpe', 'MOTPE'),
               ('random', 'Random'), ('trees-meta', 'Trees + compile metrics')]
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    for i, (m, label) in enumerate(methods):
        curves = []
        for f in glob.glob(f'{S}/runs/l*/{m}-s*/trace_*.csv'):
            line = int(f.split('/runs/l')[1].split('/')[0])
            if best[line] <= 1.01:
                continue
            t = pd.read_csv(f)
            sp = np.array([lut.get((line, k), 0.0) for k in t.key])
            cum = np.maximum.accumulate(np.maximum(sp, 1e-9))
            curves.append(np.log(np.maximum(cum, 1e-9)) / np.log(best[line]))
        c = np.clip(np.mean(curves, axis=0), 0, 1)
        ax.plot(np.arange(1, len(c) + 1), c, color=SLOT[i], marker=MARK[i], markevery=10,
                markersize=4, markeredgecolor=SURF, markeredgewidth=0.6, label=label)
    ax.set_xlabel('Configurations proposed')
    ax.set_ylabel('Share of best known log speedup')
    ax.set_ylim(0, 1); ax.set_xlim(1, 60)
    ax.legend(loc='lower right')
    ax.set_title('LULESH, kernels with a gain, 3 seeds each')
    save(fig, 'anytime')


# ---------------------------------------------------------------- 5. time vs energy
def fig_time_energy():
    cfg = pd.read_csv(f'{S}/configs-v2.csv')
    cfg = cfg[cfg.valid]
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.4))
    for ax, line in zip(axes, [333, 1201]):
        g = cfg[(cfg.line == line) & (cfg.speedup > 0.5) & (cfg.energy_gain > 0.5)]
        pts = g[['speedup', 'energy_gain']].values
        dom = np.array([((pts[:, 0] >= a[0]) & (pts[:, 1] >= a[1]) &
                         ((pts[:, 0] > a[0]) | (pts[:, 1] > a[1]))).any() for a in pts])
        ax.scatter(pts[dom, 0], pts[dom, 1], s=6, color=SLOT[0], alpha=0.35, linewidth=0,
                   label='Measured configuration')
        front = pts[~dom][np.argsort(pts[~dom][:, 0])]
        ax.plot(front[:, 0], front[:, 1], color=SLOT[1], marker='o', markersize=4,
                markeredgecolor=SURF, markeredgewidth=0.6, lw=1.4, label='Pareto front', zorder=3)
        ax.axhline(1, color=INK2, lw=0.6); ax.axvline(1, color=INK2, lw=0.6)
        ax.set_xlabel('Speedup over baseline')
        ax.set_ylabel('Energy gain over baseline')
        ax.set_title(f'LULESH l{line} ({len(g)} configurations)')
    axes[0].legend(loc='upper left')
    save(fig, 'time_energy')


# ---------------------------------------------------------------- 6. DRAM traffic
def metric_rows():
    rows = []
    for f in glob.glob(f'{S}/runs/l*/trees-*-s*/analysis*.json'):
        line = int(f.split('/runs/l')[1].split('/')[0])
        for k, v in json.load(open(f))['Stats'].items():
            m = v.get('metrics')
            if v['valid'] and m:
                rows.append(dict(line=line, key=k, base=k.startswith('NumThreads=None'),
                                 duration=np.median(v['duration']),
                                 dram=m['dram__sectors_read.sum'],
                                 gld=m['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum']))
    return pd.DataFrame(rows).drop_duplicates(['line', 'key'])


def fig_dram():
    d = metric_rows()
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.3))
    for ax, line in zip(axes, [649, 844]):
        g = d[d.line == line]
        b = g[g.base].iloc[0]
        ax.scatter(g.dram / b.dram, g.duration / b.duration, s=7, color=SLOT[0], alpha=0.5,
                   linewidth=0, label='DRAM sectors read')
        ax.scatter(g.gld / b.gld, g.duration / b.duration, s=7, color=SLOT[1], marker='s',
                   alpha=0.5, linewidth=0, label='Global load requests')
        ax.scatter([1], [1], s=40, facecolor='none', edgecolor=INK, linewidth=1.0, zorder=4)
        ax.annotate('baseline', (1, 1), xytext=(6, -10), textcoords='offset points',
                    fontsize=6.5, color=INK2)
        ax.set_xscale('log'); ax.set_yscale('log')
        from matplotlib.ticker import FuncFormatter, NullFormatter
        fmt = FuncFormatter(lambda v, _: f'{v:g}')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(fmt); axis.set_minor_formatter(NullFormatter())
        ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
        lo_x, hi_x = ax.get_xlim(); lo_y, hi_y = ax.get_ylim()
        ax.set_xticks([t for t in ticks if lo_x <= t <= hi_x])
        ax.set_yticks([t for t in [0.4, 0.6, 0.8, 1, 2, 4, 8, 16, 32] if lo_y <= t <= hi_y])
        ax.set_xlabel('Metric relative to baseline')
        ax.set_ylabel('Kernel time relative to baseline')
        ax.set_title(f'LULESH l{line} ({len(g)} configurations)')
    axes[0].legend(loc='upper left')
    save(fig, 'dram_traffic')


# ---------------------------------------------------------------- 7. classification map
def fig_classes():
    w = pd.read_csv(f'{X}/adaptive-2026-10-10/warm_metrics.csv')
    w = w[w['mode'] == 'warm'].groupby(['bench', 'line'])[['sm', 'dram', 'occ', 'spill']].median().reset_index()
    best = {}
    for f, bench in [(f'{S}/configs-v2.csv', 'LULESH'), (f'{S}/configs-minife.csv', 'MINIFE')]:
        c = pd.read_csv(f)
        for l, s in c[c.valid].groupby('line').speedup.max().items():
            best[(bench, l)] = s
    import sys
    sys.path.insert(0, '/data2/nmn/rr-metrics/openmp-rr/record-replay/RR')
    from RRUtil.Search import classify_bottleneck
    names = {'sm': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
             'dram': 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
             'occ': 'sm__warps_active.avg.pct_of_peak_sustained_active',
             'spill': 'l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum'}
    w['klass'] = [classify_bottleneck({names[c]: r[c] for c in names}) for _, r in w.iterrows()]
    w['best'] = [best.get((b, l), np.nan) for b, l in zip(w.bench, w.line)]
    fig, ax = plt.subplots(figsize=(ONE, 2.7))
    ax.axvspan(80, 100, ymin=0, ymax=0.7, color=GRID, alpha=0.6, lw=0)
    ax.text(81, 3, 'compute-saturated\nregion', fontsize=6, color=INK2, va='bottom')
    for k in ['locality', 'register', 'compute']:
        g = w[w.klass == k]
        ax.scatter(g.sm, g.dram, s=26, color=CLASS_COLOR[k], marker=CLASS_MARK[k],
                   edgecolor=SURF, linewidth=0.6, label=CLASS_LABEL[k], zorder=3)
    cluster = (w.sm < 25) & (w.dram > 80)
    for _, r in w[~cluster].iterrows():
        tag = ('L' if r.bench == 'LULESH' else 'M') + str(r.line)
        ax.annotate(f'{tag} ({r.best:.2f}x)' if r.best == r.best else tag, (r.sm, r.dram),
                    xytext=(3, 2), textcoords='offset points', fontsize=5.2, color=INK2)
    items = sorted([(('L' if r.bench == 'LULESH' else 'M') + str(r.line), r.best)
                    for _, r in w[cluster].iterrows()], key=lambda t: -t[1])
    text = 'Upper-left cluster (locality bound):\n' + '\n'.join(
        ', '.join(f'{t} ({b:.2f}x)' for t, b in items[i:i + 3]) for i in range(0, len(items), 3))
    ax.text(18, 60, text, fontsize=5.2, color=INK2, va='center', ha='left')
    ax.annotate('', xy=(14, 84), xytext=(22, 67), arrowprops=dict(arrowstyle='->', color=INK2, lw=0.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel('Baseline SM throughput (% of peak)')
    ax.set_ylabel('Baseline DRAM throughput (% of peak)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handletextpad=0.2)
    save(fig, 'classes')


# ---------------------------------------------------------------- 8. metric features per kernel
def fig_meta_effect():
    t = pd.read_csv(f'{X}/metric-search-2026-10-10/meta_vs_config.csv').set_index('line')
    cls = {333: 'locality', 649: 'locality', 1029: 'locality', 844: 'register', 1080: 'register',
           1201: 'register', 584: 'compute', 1721: 'compute', 1913: 'compute', 2103: 'compute'}
    t['klass'] = t.index.map(cls)
    t = t.sort_values('meta/config (geomean over seeds)')
    fig, ax = plt.subplots(figsize=(ONE, 2.2))
    ys = np.arange(len(t))
    for y, (line, r) in zip(ys, t.iterrows()):
        val = (r['meta/config (geomean over seeds)'] - 1) * 100
        ax.barh(y, val, color=CLASS_COLOR[r.klass], height=0.62)
        ax.text(val + (0.3 if val >= 0 else -0.3), y, f'{val:+.1f}%', va='center',
                ha='left' if val >= 0 else 'right', fontsize=6, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([f'l{l}' for l in t.index])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-12, 11)
    ax.set_xlabel('Shipped speedup, trees-meta vs trees-config (%)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CLASS_COLOR[k], label=CLASS_LABEL[k]) for k in CLASS_COLOR],
              loc='upper left')
    ax.set_title('Effect of compile-time metric features (3 seeds)')
    save(fig, 'meta_effect')


# ---------------------------------------------------------------- 9. cost breakdown
def fig_cost():
    c = pd.read_csv(f'{X}/search-efficiency-2026-10-10/cost_breakdown.csv')
    w = c.groupby('mode').wall.median()
    floor = w['single_launch']
    warm = w['warm1s_10rep'] - floor
    energy = w['warm1s_10rep_energy'] - w['warm1s_10rep']
    metrics = w['metrics_warm']
    parts = [('Compile (opt, llc, ptxas)', 0.4, SLOT[3]),
             ('Timing process 1: start-up and load', floor, SLOT[0]),
             ('Timing process 1: warm-up, 10 launches, energy', warm + energy, SLOT[2]),
             ('Timing process 2: start-up and load', floor, SLOT[0]),
             ('Timing process 2: warm-up, 10 launches, energy', warm + energy, SLOT[2]),
             ('Metrics replay (warm, all counters)', metrics, SLOT[1])]
    fig, ax = plt.subplots(figsize=(TWO, 1.25))
    left = 0
    for name, val, col in parts:
        ax.barh(0, val, left=left, color=col, height=0.5, edgecolor=SURF, linewidth=1.5)
        if val > 0.9:
            ax.text(left + val / 2, 0, f'{val:.1f}s', ha='center', va='center', fontsize=6.5, color='white')
        left += val
    ax.set_xlim(0, left + 0.2)
    ax.set_yticks([]); ax.grid(axis='y', visible=False)
    ax.set_xlabel('Wall time of one evaluation (s), median of three kernels on three GPUs')
    from matplotlib.patches import Patch
    labels = {SLOT[3]: 'Compile (opt, llc, ptxas)',
              SLOT[0]: 'Timing process: start-up and loading the recording',
              SLOT[2]: 'Timing process: 1 s warm-up, 10 timed launches, energy window',
              SLOT[1]: 'Metrics replay (warm-up and all counters)'}
    ax.legend(handles=[Patch(color=c_, label=l) for c_, l in labels.items()], loc='upper center',
              bbox_to_anchor=(0.5, -0.7), ncol=2, columnspacing=1.2)
    ax.spines['left'].set_visible(False)
    save(fig, 'cost_breakdown')


# ---------------------------------------------------------------- 10. initial designs
def fig_initial():
    t = pd.read_csv(f'{X}/search-efficiency-2026-10-10/initial_designs.csv')
    p = t.pivot_table(index='design', columns='k', values='share', aggfunc='mean')
    order = [('random', 'Random'), ('sobol', 'Sobol'), ('heuristic', 'Heuristic'),
             ('portfolio-all', 'Portfolio (all kernels)'), ('portfolio-same-class', 'Portfolio (same class)')]
    fig, ax = plt.subplots(figsize=(ONE, 2.1))
    xs = np.array([0, 1])
    w = 0.16
    for i, (d, label) in enumerate(order):
        vals = [p.loc[d, k] if k in p.columns and not np.isnan(p.loc[d, k]) else np.nan for k in (5, 10)]
        pos = xs + (i - 2) * (w + 0.01)
        ax.bar(pos, vals, w, color=SLOT[i], label=label)
        for x, v in zip(pos, vals):
            if v == v:
                ax.text(x, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=5.3, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(['First 5 configurations', 'First 10 configurations'])
    ax.set_ylabel('Share of best known\nlog speedup')
    ax.set_ylim(0, 0.9)
    ax.legend(loc='upper left', ncol=2, columnspacing=0.8)
    save(fig, 'initial_designs')


# ---------------------------------------------------------------- 11. online avenues
def fig_avenues():
    O = f'{X}/avenues-online-2026-10-11'
    A = f'{X}/adaptive-2026-10-10'
    kernels = [f'{S}/runs/l{l}' for l in [649, 1201, 844, 333, 1080, 1029]] + \
              [f'{A}/runs-minife/l{l}' for l in [521, 290, 205, 210, 157, 254, 169]]
    full = 0.4 + 2 * 4.35 + 5.5
    phases = [('Multi-fidelity vs TPE', 'tpe-mf', 'tpe-speedup'),
              ('Portfolio vs TPE', 'tpe-portfolio', 'tpe-speedup'),
              ('Transfer model vs TPE', 'transfer-model', 'tpe-speedup'),
              ('Prior vs trees-config', 'trees-prior', 'trees-config')]
    budgets = np.arange(1, 61)
    fig, axes = plt.subplots(1, 4, figsize=(TWO, 1.9), sharey=True)
    for ax, (title, new, base) in zip(axes, phases):
        curves = {new: [], base: []}
        for kd in kernels:
            per = {}
            for m in (new, base):
                dirs = glob.glob(f'{kd}/{m}-s*')
                alt = f'{O}/runs-minife-p4/{os.path.basename(kd)}'
                if m in ('trees-prior', 'trees-config') and glob.glob(f'{alt}/{m}-s*'):
                    dirs = glob.glob(f'{alt}/{m}-s*')
                for d in dirs:
                    tr = glob.glob(f'{d}/trace_*.csv')
                    if not tr:
                        continue
                    t = pd.read_csv(tr[0])
                    if 'cost' not in t:
                        t['cost'] = np.cumsum(np.where(t.source == 'repeat', 0, full))
                    t = t[t.source != 'promote']
                    best = np.maximum.accumulate(np.where(t.speedup > 0.05, t.speedup, 0))
                    per.setdefault(m, []).append((t.cost.values, best))
            if not per.get(new) or not per.get(base):
                continue
            bk = max(b.max() for m in per for _, b in per[m])
            if bk <= 1.01:
                continue
            for m in per:
                for cost, best in per[m]:
                    idx = np.searchsorted(cost, budgets * full, side='right') - 1
                    got = np.where(idx >= 0, best[np.clip(idx, 0, None)], 1.0)
                    curves[m].append(np.log(np.maximum(got, 1e-9)) / np.log(bk))
        for i, (m, label) in enumerate([(base, 'Baseline'), (new, 'Avenue')]):
            c = np.clip(np.mean(curves[m], axis=0), 0, 1)
            ax.plot(budgets, c, color=SLOT[i], marker=MARK[i], markevery=10, markersize=3.5,
                    markeredgecolor=SURF, markeredgewidth=0.5, label=label)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel('Cost (full evaluations)', fontsize=7)
        ax.set_xlim(1, 60); ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=6)
    axes[0].set_ylabel('Share of best known\nlog speedup')
    save(fig, 'avenues')


# ---------------------------------------------------------------- 12. learned policies
def fig_rl():
    R = f'{X}/rl-tuning-2026-10-13'
    r = pd.concat([pd.read_csv(f'{R}/rl_results.csv'), pd.read_csv(f'{R}/rl_hybrid_results.csv')])
    opt = {os.path.basename(p)[:-4]: pd.read_csv(p).speedup.max() for p in glob.glob(f'{R}/landscape/*.csv')}
    r = r[r.kernel.map(opt) > 1.05]
    methods = [('random', 'Random'), ('tpe', 'TPE'), ('gp_ei', 'GP-EI'), ('transfer_ei', 'Transfer-EI'),
               ('gp_transfer_ei', 'GP+transfer EI'), ('rl', 'RL'), ('rl_hybrid', 'RL + model inputs')]
    folds = [('loko', 'Leave one kernel out'), ('lulesh->minife', 'LULESH $\\to$ miniFE'),
             ('minife->lulesh', 'miniFE $\\to$ LULESH')]
    fig, axes = plt.subplots(1, 3, figsize=(TWO, 2.0), sharey=True)
    for ax, (fold, title) in zip(axes, folds):
        t = r[r.fold == fold].pivot_table(index='method', columns='n', values='share', aggfunc='mean')
        for i, (m, label) in enumerate(methods):
            if m in t.index:
                ax.plot(t.columns, t.loc[m].values, color=SLOT[i], marker=MARK[i], markersize=3.5,
                        markeredgecolor=SURF, markeredgewidth=0.5, label=label,
                        linewidth=2.2 if m == 'rl_hybrid' else 1.3)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel('Evaluations', fontsize=7)
        ax.set_xlim(1, 30); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Share of optimum\nlog speedup')
    axes[0].legend(loc='lower right', fontsize=5.5, ncol=1)
    save(fig, 'rl')


# ---------------------------------------------------------------- 13. learned transfer
def fig_learned():
    R = f'{X}/rl-tuning-2026-10-13'
    L = f'{X}/learned-transfer-2026-10-13'
    r = pd.concat([pd.read_csv(f'{R}/rl_results.csv'), pd.read_csv(f'{R}/rl_hybrid_results.csv'),
                   pd.read_csv(f'{L}/lt_results.csv'), pd.read_csv(f'{L}/lt_hybrid_results.csv')])
    opt = {os.path.basename(p)[:-4]: pd.read_csv(p).speedup.max() for p in glob.glob(f'{R}/landscape/*.csv')}
    r = r[r.kernel.map(opt) > 1.05]
    methods = [('gp_ei', 'GP-EI'), ('transfer_ei', 'Transfer-EI'), ('rl_hybrid', 'RL + model inputs'),
               ('rgpe', 'RGPE'), ('transfer_rank', 'Ranking model'), ('wave_blr', 'Wave model'),
               ('rank_rgpe', 'Ranking, then RGPE')]
    folds = [('loko', 'Leave one kernel out'), ('lulesh->minife', 'LULESH $\\to$ miniFE'),
             ('minife->lulesh', 'miniFE $\\to$ LULESH')]
    fig, axes = plt.subplots(1, 3, figsize=(TWO, 2.0), sharey=True)
    for ax, (fold, title) in zip(axes, folds):
        t = r[r.fold == fold].pivot_table(index='method', columns='n', values='share', aggfunc='mean')
        for i, (m, label) in enumerate(methods):
            if m in t.index:
                ax.plot(t.columns, t.loc[m].values, color=SLOT[i], marker=MARK[i], markersize=3.5,
                        markeredgecolor=SURF, markeredgewidth=0.5, label=label,
                        linewidth=2.2 if m == 'transfer_rank' else 1.3)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel('Evaluations', fontsize=7)
        ax.set_xlim(1, 30); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel('Share of optimum\nlog speedup')
    axes[0].legend(loc='lower right', fontsize=5.5, ncol=1)
    save(fig, 'learned')


if __name__ == '__main__':
    for f in [fig_clock_ramp, fig_winners_curse, fig_measurement, fig_anytime, fig_time_energy,
              fig_dram, fig_classes, fig_meta_effect, fig_cost, fig_initial, fig_avenues, fig_rl, fig_learned]:
        try:
            f()
        except Exception as e:
            print('FAILED', f.__name__, type(e).__name__, e)
