"""Search strategies over the fractional launch-configuration space.

Every searcher proposes points in the same space that Kernel.BO uses
(getBounds -> convertPointToValues) and is told objective values to
maximize. The first point is always the full launch-bound point, as in
Kernel.BO.
"""
import math


def objective_values(speedup, energy_gain):
    """Log-ratios to maximize; invalid configs already map to 0.01."""
    def safe_log(x):
        # Missing measurements (NaN) are scored like invalid configs.
        return math.log(x) if x == x and x > 1e-6 else math.log(0.01)
    return {'speedup': safe_log(speedup), 'energy': safe_log(energy_gain)}


class BOSearcher:
    """The existing optimizer: bayes_opt GP with UCB (kappa 1.96)."""

    def __init__(self, bounds, seed, n_init, objectives):
        from bayes_opt import BayesianOptimization, UtilityFunction
        assert len(objectives) == 1, 'bo is single-objective'
        self.bounds, self.n_init, self.objective = bounds, n_init, objectives[0]
        self.opt = BayesianOptimization(f=None, pbounds=bounds, verbose=0,
                                        random_state=seed,
                                        allow_duplicate_points=True)
        self.utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
        self.asked = 0

    def ask(self):
        if self.asked == 0:
            point = {k: b[1] for k, b in self.bounds.items()}
        elif self.asked < self.n_init:
            point = self.opt.space.array_to_params(self.opt.space.random_sample())
        else:
            point = self.opt.suggest(self.utility)
        self.asked += 1
        return point

    def tell(self, point, values):
        # Kernel.BO registers raw ratios; keep that so 'bo' is the baseline.
        self.opt.register(params=point, target=math.exp(values[self.objective]))


class OptunaSearcher:
    """Optuna TPE / random / multi-objective TPE via the ask-tell interface."""

    def __init__(self, bounds, seed, n_init, objectives, kind):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if kind == 'random':
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_init,
                                                 multivariate=True)
        self.bounds, self.objectives = bounds, objectives
        self.study = optuna.create_study(directions=['maximize'] * len(objectives),
                                         sampler=sampler)
        self.study.enqueue_trial({k: b[1] for k, b in bounds.items()})
        self.trial = None

    def ask(self):
        self.trial = self.study.ask()
        return {k: self.trial.suggest_float(k, b[0], b[1]) for k, b in self.bounds.items()}

    def tell(self, point, values):
        result = [values[o] for o in self.objectives]
        self.study.tell(self.trial, result[0] if len(result) == 1 else result)


# T4 / sm_75 limits for compile-time features.
SM_COUNT, SM_THREADS, SM_BLOCKS, SM_REGS, REG_UNIT, WARP = 40, 1024, 16, 65536, 256, 32
MAX_REGS_PER_BLOCK = 65536


def theoretical_blocks_per_sm(regs, block):
    warps = int(math.ceil(block / WARP))
    regs_per_warp = int(math.ceil(max(regs, 1) * WARP / REG_UNIT) * REG_UNIT)
    return max(0, min(SM_THREADS // (warps * WARP), (SM_REGS // regs_per_warp) // warps,
                      SM_BLOCKS))


def compile_features(config, table, trip_count):
    """Free (pre-measurement) features of a launch configuration.

    `table` maps (MaxThreads, MinTeams) -> (regs, spill_bytes) from ptxas.
    Returns (features, known_invalid).
    """
    mt, mn = config.get('MaxThreads') or 1024, config.get('MinTeams') or 1
    nt = config.get('NumThreads') or mt
    teams = config.get('NumTeams') or max(1, math.ceil(trip_count / nt))
    regs, spill = table.get((mt, mn), (float('nan'), float('nan')))
    if regs != regs:
        return None, False
    blocks = theoretical_blocks_per_sm(regs, nt)
    occupancy = blocks * math.ceil(nt / WARP) * WARP / SM_THREADS
    waves = teams / max(1, SM_COUNT * blocks)
    tail = waves / math.ceil(waves) if waves > 0 else 0.0
    known_invalid = regs * nt > MAX_REGS_PER_BLOCK
    features = [math.log2(mt), mn, math.log2(nt), math.log2(teams), regs, spill,
                occupancy, math.log2(max(waves, 1e-3)), tail]
    return features, known_invalid


def build_compile_table(kernel, path):
    """ptxas registers/spills for every (MaxThreads, MinTeams) launch bound.

    Compiles in a scratch directory so the kernel's working image is not
    overwritten. Cached as JSON at `path`.
    """
    import json
    import os
    import shutil
    import tempfile
    if os.path.exists(path):
        return {tuple(map(int, k.split(','))): tuple(v)
                for k, v in json.load(open(path)).items()}
    prof = kernel.Profiler
    warps = int(prof.MaxThreadBlockSize / prof.WarpSize)
    pairs = set()
    for w in range(1, warps + 1):
        mt = int(w * prof.WarpSize)
        max_blocks = min(32, int(prof.ThreadsSMP // mt))
        for mn in range(1, max(1, max_blocks) + 1):
            pairs.add((mt, mn))
    cwd = os.getcwd()
    original = os.path.abspath(kernel.original)
    scratch = tempfile.mkdtemp(prefix=f'ct-{kernel.HashName}-', dir=cwd)
    table = {}
    try:
        os.chdir(scratch)
        os.symlink(original, os.path.basename(original))
        for mt, mn in sorted(pairs):
            stats = prof.deviceCompile(os.path.basename(original), kernel.Name,
                                       kernel.HashName, MaxThreads=mt, MinTeams=mn)
            st = (stats or {}).get(kernel.Name)
            if st:
                table[(mt, mn)] = (st[0], st[3] + st[4] if len(st) > 4 else 0)
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch, ignore_errors=True)
    json.dump({f'{a},{b}': v for (a, b), v in table.items()}, open(path, 'w'), indent=1)
    return table


def classify_bottleneck(metrics, sm_saturated=80.0):
    """Bottleneck class of a kernel from its baseline metrics.

    `metrics` should be the per-metric median of several baseline samples.
    Rules from experiment-archives/metric-search-2026-10-10 (10 LULESH kernels):
    compute-saturated kernels had nothing to gain; locality-bound kernels gain
    by cutting DRAM traffic; register-bound kernels gain from launch bounds.
    """
    sm = metrics.get('sm__throughput.avg.pct_of_peak_sustained_elapsed')
    dram = metrics.get('dram__throughput.avg.pct_of_peak_sustained_elapsed')
    occ = metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active')
    spill = metrics.get('l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum', 0) or 0
    if sm is None or dram is None or occ is None:
        return 'unknown'
    # Compute-saturated means the SM pipes are busy *and* memory is not: a
    # memory-heavy kernel profiled while power-capped (lower SM clock) can
    # show high SM utilisation too (l844 read 52-88%; its DRAM was 69-88%,
    # while the compute-saturated kernels stayed <= 66%).
    if sm > sm_saturated and dram < 70:
        return 'compute'
    if occ > 80 and dram > 70:
        return 'locality'
    if spill > 0 or occ < 60:
        return 'register'
    return 'other'


def _z(values):
    import numpy as np
    v = np.asarray(values, float)
    return (v - v.mean()) / (v.std() + 1e-12)


class TreesSearcher:
    """ExtraTrees surrogate + expected improvement over random candidates.

    `featurize(point) -> (features, known_invalid, key)`. With the compile
    featurizer, candidates certain to exceed the register file are never
    proposed, and candidates mapping to an evaluated config are skipped.
    """

    def __init__(self, bounds, seed, n_init, objectives, featurize, n_candidates=1500,
                 target='objective'):
        import numpy as np
        assert len(objectives) == 1, 'trees is single-objective'
        # target='register-mix' steers by 0.5 z(log speedup) + 0.5 z(condition),
        # condition = mean of z(-log1p spill loads) and z(occupancy); all
        # observations are relabelled at every step.
        self.target = target
        self.raw = []
        self.np, self.rng = np, np.random.default_rng(seed)
        self.bounds, self.names = bounds, list(bounds)
        self.n_init, self.objective, self.featurize = n_init, objectives[0], featurize
        self.n_candidates, self.seed = n_candidates, seed
        self.F, self.y, self.points, self.seen = [], [], [], set()
        self.asked = 0

    def _random(self, n):
        lo = self.np.array([self.bounds[k][0] for k in self.names])
        hi = self.np.array([self.bounds[k][1] for k in self.names])
        return lo + self.rng.random((n, len(self.names))) * (hi - lo)

    def _usable(self, arr):
        out = []
        for row in arr:
            point = dict(zip(self.names, row))
            f, invalid, key = self.featurize(point)
            if f is not None and not invalid and key not in self.seen:
                out.append((point, f, key))
        return out

    def ask(self):
        np = self.np
        self.asked += 1
        if self.asked == 1:
            return {k: b[1] for k, b in self.bounds.items()}
        if len(self.y) < self.n_init or len(self.y) < 3:
            cands = self._usable(self._random(200))
            return cands[0][0] if cands else dict(zip(self.names, self._random(1)[0]))
        from sklearn.ensemble import ExtraTreesRegressor
        arr = self._random(self.n_candidates)
        top = np.argsort(self._targets())[-5:]
        local = [np.clip(np.array([self.points[i][k] for k in self.names]) +
                         self.rng.normal(0, 0.05, (60, len(self.names))), 1e-9, 1.0) for i in top]
        cands = self._usable(np.vstack([arr] + local))
        if not cands:
            return dict(zip(self.names, self._random(1)[0]))
        y = self._targets()
        model = ExtraTreesRegressor(200, min_samples_leaf=2, random_state=self.seed).fit(
            np.array(self.F), y)
        Fc = np.array([c[1] for c in cands])
        per_tree = np.stack([t.predict(Fc) for t in model.estimators_])
        mu, sd = per_tree.mean(0), np.maximum(per_tree.std(0), 1e-9)
        best = y.max()
        from scipy.stats import norm
        z = (mu - best) / sd
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        return cands[int(np.argmax(ei))][0]

    def tell(self, point, values):
        f, _, key = self.featurize(point)
        self.seen.add(key)
        v = values[self.objective]
        if f is not None and v > math.log(0.02):   # fit on valid measurements only
            self.F.append(f)
            self.y.append(v)
            self.points.append(point)
            self.raw.append((values.get('spill', 0.0), values.get('occupancy', float('nan'))))

    def _targets(self):
        np = self.np
        y = np.array(self.y)
        if self.target != 'register-mix':
            return y
        spill = np.array([r[0] for r in self.raw], float)
        occ = np.array([r[1] for r in self.raw], float)
        occ = np.where(np.isnan(occ), np.nanmean(occ) if np.isfinite(occ).any() else 0, occ)
        condition = 0.5 * _z(-np.log1p(spill)) + 0.5 * _z(occ)
        return 0.5 * _z(y) + 0.5 * condition


def make_searcher(algorithm, bounds, seed, n_init, objectives, featurize=None,
                  target='objective'):
    if algorithm in ('trees-config', 'trees-meta'):
        assert featurize is not None, f'{algorithm} needs a featurizer'
        return TreesSearcher(bounds, seed, n_init, objectives, featurize, target=target)
    if algorithm == 'bo':
        return BOSearcher(bounds, seed, n_init, objectives)
    if algorithm in ('tpe', 'random', 'motpe'):
        if algorithm == 'motpe':
            assert len(objectives) > 1, 'motpe needs several objectives'
        return OptunaSearcher(bounds, seed, n_init, objectives,
                              'random' if algorithm == 'random' else 'tpe')
    raise ValueError(f'unknown search algorithm {algorithm}')
