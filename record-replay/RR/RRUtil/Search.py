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

    def __init__(self, bounds, seed, n_init, objectives, kind, initial_points=None,
                 constant_liar=False):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if kind == 'random':
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            # constant_liar: proposals asked before earlier ones are told
            # (batch evaluation) avoid the pending points.
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_init,
                                                 multivariate=True,
                                                 constant_liar=constant_liar)
        self.bounds, self.objectives = bounds, objectives
        self.study = optuna.create_study(directions=['maximize'] * len(objectives),
                                         sampler=sampler)
        self.study.enqueue_trial({k: b[1] for k, b in bounds.items()})
        # Warm start: evaluate these points (e.g. a cross-kernel portfolio)
        # before the sampler's own initial samples.
        for point in initial_points or []:
            self.study.enqueue_trial({k: min(max(point[k], bounds[k][0]), bounds[k][1])
                                      for k in bounds})
        self.trial = None

    def ask(self):
        self.trial = self.study.ask()
        return {k: self.trial.suggest_float(k, b[0], b[1]) for k, b in self.bounds.items()}

    def tell(self, point, values):
        result = [values[o] for o in self.objectives]
        self.study.tell(self.trial, result[0] if len(result) == 1 else result)

    def ask_many(self, n):
        """Propose n points before any is told (batch evaluation)."""
        out = []
        for _ in range(n):
            trial = self.study.ask()
            out.append(({k: trial.suggest_float(k, b[0], b[1]) for k, b in self.bounds.items()},
                        trial))
        return out

    def tell_trial(self, trial, values):
        result = [values[o] for o in self.objectives]
        self.study.tell(trial, result[0] if len(result) == 1 else result)


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


def config_to_point(config, trip_count, warp=32, max_block=1024, threads_smp=1024):
    """Inverse of Device.getLaunchBounds: fractional point for a discrete config.

    `config` holds MaxThreads, MinTeams, NumThreads and either NumTeams or
    teams_frac (teams x threads / trip count). A tiny offset keeps the ceil()
    in getLaunchBounds on the intended value.
    """
    eps = 1e-9
    mt = int(config['MaxThreads'])
    max_blocks = min(32, int(threads_smp // mt))
    mn = min(int(config['MinTeams']), max_blocks)
    nt = min(int(config['NumThreads']), mt)
    frac = config.get('teams_frac')
    if frac is None:
        frac = int(config['NumTeams']) * nt / trip_count
    return {'MaxThreads': max(mt / max_block - eps, eps),
            'MinTeams': max(mn / max_blocks - eps, eps),
            'NumThreads': max(nt / mt - eps, eps),
            'NumTeams': min(max(frac - eps, eps), 1.0)}


def build_portfolio(pool, target_kid, klass, trip_count, size=10, per_kernel=3):
    """Initial configurations taken from the best configs of other kernels.

    Leave-one-kernel-out: `pool` (pooled.csv rows) without the target. Same-class
    kernels contribute first, then the others, best source speedup first.
    """
    others = pool[(pool.kid != target_kid) & (pool.speedup > 1.0)]
    per = []
    for kid, g in others.groupby('kid'):
        top = g.nlargest(per_kernel, 'speedup')
        per.append((top.klass.iloc[0] != klass, -top.speedup.iloc[0],
                    [dict(MaxThreads=r.MaxThreads, MinTeams=r.MinTeams,
                          NumThreads=r.NumThreads, teams_frac=r.teams_frac)
                     for r in top.itertuples()]))
    per.sort(key=lambda t: (t[0], t[1]))
    points, seen = [], set()
    for rank in range(per_kernel):          # every kernel's best before any second best
        for _, _, cfgs in per:
            if rank >= len(cfgs):
                continue
            cfg = cfgs[rank]
            key = (cfg['MaxThreads'], cfg['MinTeams'], cfg['NumThreads'],
                   round(cfg['teams_frac'], 3))
            if key in seen:
                continue
            seen.add(key)
            points.append(config_to_point(cfg, trip_count))
            if len(points) >= size:
                return points
    return points


def point_to_config(point, trip_count, warp=32, max_block=1024, threads_smp=1024):
    """Device.getLaunchBounds for the T4 limits (MaxThreads, MinTeams, NumThreads, NumTeams)."""
    mt = int(math.ceil(max(point['MaxThreads'], 1e-12) * (max_block / warp)) * warp)
    max_blocks = min(32, int(threads_smp // mt))
    mn = max(1, int(math.ceil(max(point['MinTeams'], 1e-12) * max_blocks)))
    nt = min(mt, int(math.ceil(max(point['NumThreads'], 1e-12) * mt / warp) * warp))
    teams = int(math.ceil(trip_count / nt * max(point['NumTeams'], 1e-12)))
    return dict(MaxThreads=mt, MinTeams=mn, NumThreads=nt, NumTeams=teams)


def prior_density(P, Z, bandwidth=0.08, floor=0.1):
    """Gaussian KDE over prior points P (prior_coords space), mixed with a uniform floor,
    normalised to a maximum of 1 over the evaluated set Z."""
    import numpy as np
    if P is None or len(P) == 0:
        return np.ones(len(Z))
    d = ((Z[:, None, :] - P[None]) ** 2).sum(-1)
    dens = np.exp(-d / (2 * bandwidth ** 2)).mean(1)
    return floor + (1 - floor) * dens / max(dens.max(), 1e-12)


TRANSFER_FEATURES = ['lnt', 'lfrac', 'regs', 'lspill', 'occ', 'lwaves', 'tail',
                     'is_locality', 'is_register', 'is_compute']


def transfer_frame(df, klass_col='klass'):
    import numpy as np
    col = lambda c: df[c].to_numpy(dtype=float)   # df.tail would be the pandas method
    out = np.c_[np.log2(col('NumThreads')), np.log2(np.clip(col('teams_frac'), 1e-6, None)),
                col('regs'), np.log1p(col('spill')), col('occ'),
                np.log2(np.clip(col('waves'), 1e-3, None)), col('tail')]
    k = df[klass_col].values
    return np.c_[out, k == 'locality', k == 'register', k == 'compute'].astype(float)


class TransferModelSearcher:
    """Rank a deduplicated candidate set with a cost model trained across kernels.

    The ExtraTrees model is fitted on the pooled measurements of all *other*
    kernels (hardware-normalised features: threads, teams fraction, registers,
    spills, theoretical occupancy, waves, tail, bottleneck class) plus this
    kernel's own measurements, which get `own_weight`. Candidates are one
    representative launch bound per distinct binary (registers, spills) x every
    warp multiple of NumThreads x a log grid of teams fractions; candidates with
    the same binary and launch are never proposed twice. No random initial
    design: proposals follow expected improvement from the first step.
    """

    def __init__(self, bounds, seed, objectives, pool, target_kid, klass, table, trip_count,
                 own_weight=20.0, n_frac=28, initial_points=None, prior_points=None, beta=10.0):
        import numpy as np
        import pandas as pd
        from RRUtil.Search import compile_features
        self.np, self.pd, self.seed = np, pd, seed
        self.objective = objectives[0]
        self.trip, self.klass = trip_count, klass
        other = pool[pool.kid != target_kid]
        self.X0 = transfer_frame(other)
        self.y0 = np.log(other.speedup.values)
        self.own_weight = own_weight
        # candidate enumeration
        classes = {}
        for (mt, mn), (regs, spill) in table.items():
            k = (regs, spill)
            if k not in classes or mt > classes[k][0]:
                classes[k] = (mt, mn)
        rows = []
        fracs = np.geomspace(1 / 256, 1.0, n_frac)
        for (regs, spill), (mt, mn) in classes.items():
            for nt in range(32, mt + 1, 32):
                for fr in fracs:
                    teams = max(1, int(math.ceil(trip_count * fr / nt)))
                    cfg = dict(MaxThreads=mt, MinTeams=mn, NumThreads=nt, NumTeams=teams)
                    f, invalid = compile_features(cfg, table, trip_count)
                    if f is None or invalid:
                        continue
                    rows.append(dict(cfg, teams_frac=teams * nt / trip_count, regs=f[4],
                                     spill=f[5], occ=f[6], waves=2 ** f[7], tail=f[8],
                                     klass=klass, canon=(regs, spill, nt, teams)))
        self.cand = pd.DataFrame(rows).drop_duplicates('canon').reset_index(drop=True)
        self.Xc = transfer_frame(self.cand)
        self.own_X, self.own_y, self.seen = [], [], set()
        self.asked = None
        self.table = table
        # Combined mode: evaluate portfolio points first; weight EI by a prior.
        self.initial = list(initial_points or [])
        self.beta = beta
        self.prior_w = None
        if prior_points:
            order = ['MaxThreads', 'MinTeams', 'NumThreads', 'NumTeams']
            P = prior_coords(np.array([[p[k] for k in order] for p in prior_points]))
            U = np.array([[config_to_point(dict(MaxThreads=r.MaxThreads, MinTeams=r.MinTeams,
                                                NumThreads=r.NumThreads, NumTeams=r.NumTeams),
                                           trip_count)[k] for k in order]
                          for r in self.cand.itertuples()])
            self.prior_w = prior_density(P, prior_coords(U))

    def _row_for(self, point):
        from RRUtil.Search import compile_features
        cfg = point_to_config(point, self.trip)
        f, _ = compile_features(cfg, self.table, self.trip)
        if f is None:
            return None
        return self.pd.Series(dict(cfg, teams_frac=cfg['NumTeams'] * cfg['NumThreads'] / self.trip,
                                   regs=f[4], spill=f[5], occ=f[6], waves=2 ** f[7], tail=f[8],
                                   klass=self.klass,
                                   canon=(f[4], f[5], cfg['NumThreads'], cfg['NumTeams'])))

    def ask(self):
        from sklearn.ensemble import ExtraTreesRegressor
        from scipy.stats import norm
        np = self.np
        while self.initial:
            point = self.initial.pop(0)
            row = self._row_for(point)
            if row is not None and row.canon not in self.seen:
                self.asked = row
                return point
        X = np.vstack([self.X0] + ([np.array(self.own_X)] if self.own_X else []))
        y = np.r_[self.y0, np.array(self.own_y)] if self.own_y else self.y0
        w = np.r_[np.ones(len(self.y0)), np.full(len(self.own_y), self.own_weight)]
        m = ExtraTreesRegressor(150, min_samples_leaf=3, random_state=self.seed, n_jobs=2)
        m.fit(X, y, sample_weight=w)
        mask = ~self.cand.canon.isin(self.seen).values
        idx = np.flatnonzero(mask)
        per = np.stack([t.predict(self.Xc[idx]) for t in m.estimators_])
        mu, sd = per.mean(0), np.maximum(per.std(0), 1e-9)
        best = max(self.own_y) if self.own_y else 0.0     # baseline = log(1)
        z = (mu - best) / sd
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        if self.prior_w is not None:
            ei = ei * self.prior_w[idx] ** (self.beta / max(1, len(self.own_y)))
        i = idx[int(np.argmax(ei))]
        r = self.cand.iloc[i]
        self.asked = r
        return config_to_point(dict(MaxThreads=r.MaxThreads, MinTeams=r.MinTeams,
                                    NumThreads=r.NumThreads, NumTeams=r.NumTeams), self.trip)

    def ask_many(self, n):
        """Propose n candidates at once (batch evaluation).

        Remaining portfolio points come first; the rest are the candidates with
        the highest (prior-weighted) expected improvement, distinct from each
        other and from everything already evaluated. `last_max_ei` holds the
        largest expected improvement (log speedup) among unevaluated
        candidates, for convergence-based stopping.
        """
        from sklearn.ensemble import ExtraTreesRegressor
        from scipy.stats import norm
        np = self.np
        out, chosen = [], set()
        while self.initial and len(out) < n:
            point = self.initial.pop(0)
            row = self._row_for(point)
            if row is not None and row.canon not in self.seen and row.canon not in chosen:
                chosen.add(row.canon)
                out.append((point, row))
        X = np.vstack([self.X0] + ([np.array(self.own_X)] if self.own_X else []))
        y = np.r_[self.y0, np.array(self.own_y)] if self.own_y else self.y0
        w = np.r_[np.ones(len(self.y0)), np.full(len(self.own_y), self.own_weight)]
        m = ExtraTreesRegressor(150, min_samples_leaf=3, random_state=self.seed, n_jobs=2)
        m.fit(X, y, sample_weight=w)
        idx = np.flatnonzero(~self.cand.canon.isin(self.seen | chosen).values)
        if len(idx) == 0:
            self.last_max_ei = 0.0
            return out
        per = np.stack([t.predict(self.Xc[idx]) for t in m.estimators_])
        mu, sd = per.mean(0), np.maximum(per.std(0), 1e-9)
        best = max(self.own_y) if self.own_y else 0.0
        z = (mu - best) / sd
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        self.last_max_ei = float(ei.max())
        score = ei * (self.prior_w[idx] ** (self.beta / max(1, len(self.own_y)))
                      if self.prior_w is not None else 1.0)
        for j in np.argsort(-score)[:max(0, n - len(out))]:
            r = self.cand.iloc[idx[j]]
            out.append((config_to_point(dict(MaxThreads=r.MaxThreads, MinTeams=r.MinTeams,
                                             NumThreads=r.NumThreads, NumTeams=r.NumTeams),
                                        self.trip), r))
        return out

    def tell_trial(self, row, values):
        self.asked = row
        self.tell(None, values)

    def tell(self, point, values):
        r = self.asked
        self.seen.add(r.canon)
        v = values[self.objective]
        if v > math.log(0.02):
            self.own_X.append(transfer_frame(self.pd.DataFrame([r]))[0])
            self.own_y.append(v)


def rank_grades(y, levels=32):
    """Percentile grades 0..levels-1 of y (ranking labels, scale-free across kernels)."""
    import numpy as np
    import pandas as pd
    r = pd.Series(y).rank(pct=True).values
    return np.minimum((r * levels).astype(int), levels - 1)


# Scale of the geometric transfer features for Gaussian processes (trees ignore it).
GP_SCALE = [10.0, 8.0, 255.0, 10.0, 1.0, 10.0, 1.0]


class RankTransferSearcher(TransferModelSearcher):
    """Pairwise-ranking cost model across kernels, optionally followed by RGPE.

    Bagged XGBoost rankers (rank:pairwise) are trained on the other kernels'
    measurements in `pool`, one query group per kernel with percentile grades
    as relevance, so every kernel counts equally whatever its speedup range.
    Before each batch, every ranker is fine-tuned with a few trees on this
    kernel's own observations. Candidates (the same deduplicated set as
    TransferModelSearcher) are ranked by mean + std of the z-scored ranker
    outputs. With `rgpe`, after `switch` own observations the acquisition is the
    expected improvement of a ranking-weighted GP ensemble (Feurer et al.
    2018): one GP per other kernel, the fine-tuned ranker, and a GP on this
    kernel's observations, weighted by how often each ranks the observations
    best. `last_max_ei` is the largest expected improvement in log speedup of a
    GP on the kernel's own observations (inf before `min_obs` observations),
    for the convergence stop, since ranker scores are not in speedup units.
    Offline evaluation: experiment-archives/learned-transfer-2026-10-13.
    """

    def __init__(self, bounds, seed, objectives, pool, target_kid, klass, table, trip_count,
                 rgpe=False, switch=5, bags=8, min_obs=32, rounds=150):
        import numpy as np
        import xgboost as xgb
        super().__init__(bounds, seed, objectives, pool, target_kid, klass, table, trip_count)
        self.xgb, self.rgpe, self.switch, self.min_obs = xgb, rgpe, switch, min_obs
        self.rng = np.random.default_rng(seed)
        other = pool[(pool.kid != target_kid) & (pool.speedup > 0)]
        groups = [(transfer_frame(g), np.log(g.speedup.values)) for _, g in other.groupby('kid')]
        self.models = []
        for b in range(bags):
            pick = self.rng.choice(len(groups), len(groups), replace=True)
            X = np.vstack([groups[i][0] for i in pick])
            y = np.concatenate([rank_grades(groups[i][1]) for i in pick])
            q = np.concatenate([np.full(len(groups[i][1]), j) for j, i in enumerate(pick)])
            p = dict(objective='rank:pairwise', eta=0.1, max_depth=6, subsample=0.8,
                     lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=8,
                     nthread=2, seed=int(seed * 100 + b), verbosity=0)
            self.models.append((xgb.train(p, xgb.DMatrix(X, label=y, qid=q),
                                          num_boost_round=rounds), p))
        self.dc = xgb.DMatrix(self.Xc)
        self.base_scores = np.array([m.predict(self.dc) for m, _ in self.models])
        self.Gc = self.Xc[:, :7] / np.array(GP_SCALE)
        if rgpe:
            mus, sds = [], []
            for X, y in groups:
                idx = self.rng.choice(len(y), min(300, len(y)), replace=False)
                gp = self._fit_gp(X[idx, :7] / np.array(GP_SCALE), self._std(y)[idx])
                mu, sd = gp.predict(self.Gc, return_std=True)
                mus.append(mu); sds.append(np.maximum(sd, 1e-6))
            self.BM, self.BS = np.array(mus), np.array(sds)
        self.last_weights = None
        self.own_idx = []

    @staticmethod
    def _std(y):
        s = y.std()
        return (y - y.mean()) / (s if s > 1e-9 else 1.0)

    @staticmethod
    def _fit_gp(X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        return GaussianProcessRegressor(ConstantKernel() * Matern(nu=2.5) + WhiteKernel(1e-3),
                                        normalize_y=False, random_state=0).fit(X, y)

    @staticmethod
    def _ei(mu, sd, best):
        import numpy as np
        from scipy.stats import norm
        sd = np.maximum(sd, 1e-9)
        z = (mu - best) / sd
        return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)

    def _rank_z(self):
        np, xgb = self.np, self.xgb
        if len(self.own_y) >= 2:
            g = rank_grades(np.array(self.own_y), levels=len(self.own_y))
            d = xgb.DMatrix(np.array(self.own_X), label=g, qid=np.zeros(len(g), int))
            sc = np.array([xgb.train(dict(p, eta=0.3, max_depth=3, subsample=1.0), d,
                                     num_boost_round=20, xgb_model=m.copy()).predict(self.dc)
                           for m, p in self.models])
        else:
            sc = self.base_scores
        return (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-9)

    def _rgpe(self, z):
        """RGPE expected improvement (standardised own units) over all candidates."""
        np = self.np
        own = np.array(self.own_y)
        rows = np.array(self.own_idx)
        y = self._std(own)
        G = self.Gc[rows]
        gp = self._fit_gp(G, y)
        K = gp.kernel_(G)
        Ki = np.linalg.inv(K + 1e-10 * np.eye(len(y)))
        mloo, vloo = y - (Ki @ y) / np.diag(Ki), 1.0 / np.diag(Ki)
        mt, st = gp.predict(self.Gc, return_std=True)
        BM = np.vstack([self.BM, z.mean(0)])
        BS = np.vstack([self.BS, np.maximum(z.std(0), 0.05)])
        B, n, S = len(BM), len(y), 256
        iu = np.triu_indices(n, 1)
        dy = np.sign(y[:, None] - y[None, :])[iu]

        def loss(f):
            return (np.sign(f[..., :, None] - f[..., None, :])[(Ellipsis,) + iu] != dy).sum(-1)
        fb = BM[:, rows][:, None, :] + BS[:, rows][:, None, :] * self.rng.standard_normal((B, S, n))
        ft = mloo + np.sqrt(np.maximum(vloo, 1e-12)) * self.rng.standard_normal((S, n))
        L = np.vstack([loss(fb), loss(ft)[None]]) + self.rng.random((B + 1, S)) * 0.1
        w = np.bincount(L.argmin(0), minlength=B + 1) / S
        w[:-1][np.median(L[:-1], axis=1) > np.percentile(L[-1], 95)] = 0.0
        w = w / w.sum() if w.sum() > 0 else np.r_[np.zeros(B), 1.0]
        self.last_weights = dict(target=float(w[-1]), ranker=float(w[-2]))
        mu = w[:-1] @ BM + w[-1] * mt
        sd = np.sqrt((w[:-1] ** 2) @ (BS ** 2) + (w[-1] * st) ** 2)
        return self._ei(mu, sd, y.max())

    def ask(self):
        point, row = self.ask_many(1)[0]
        self.asked = row
        return point

    def ask_many(self, n):
        np = self.np
        idx = np.flatnonzero(~self.cand.canon.isin(self.seen).values)
        if len(idx) == 0:
            self.last_max_ei = 0.0
            return []
        z = self._rank_z()
        if self.rgpe and len(self.own_y) >= max(2, self.switch):
            acq = self._rgpe(z)
        else:
            acq = z.mean(0) + z.std(0)
        acq = acq + self.rng.random(len(acq)) * 1e-9
        # Convergence signal in log speedup units from the kernel's own observations.
        if len(self.own_y) >= self.min_obs:
            gp = self._fit_gp(self.Gc[np.array(self.own_idx)], np.array(self.own_y))
            mu, sd = gp.predict(self.Gc[idx], return_std=True)
            self.last_max_ei = float(self._ei(mu, sd, max(self.own_y)).max())
        else:
            self.last_max_ei = float('inf')
        out = []
        for j in idx[np.argsort(-acq[idx])[:n]]:
            r = self.cand.iloc[j]
            out.append((config_to_point(dict(MaxThreads=r.MaxThreads, MinTeams=r.MinTeams,
                                             NumThreads=r.NumThreads, NumTeams=r.NumTeams),
                                        self.trip), (j, r)))
        return out

    def tell_trial(self, trial, values):
        j, r = trial
        self.seen.add(r.canon)
        v = values[self.objective]
        if v > math.log(0.02):
            self.own_idx.append(int(j))
            self.own_X.append(self.Xc[j])
            self.own_y.append(v)

    def tell(self, point, values):
        self.tell_trial(self.asked, values)


def prior_coords(u):
    """Map fractional points to the space the prior is defined in (log teams)."""
    import numpy as np
    u = np.atleast_2d(u)
    return np.c_[u[:, :3], np.clip(1 + np.log2(np.clip(u[:, 3], 1e-9, 1)) / 10, 0, 1)]


class PriorTreesSearcher(TreesSearcher):
    """TreesSearcher (config features) with a piBO-style prior.

    The prior is a Gaussian KDE over the top configs of other kernels of the
    same class, mixed with a uniform floor. Initial samples are drawn from the
    prior, and expected improvement is multiplied by prior(x)^(beta / n).
    """

    def __init__(self, bounds, seed, n_init, objectives, featurize, prior_points,
                 beta=10.0, bandwidth=0.08, floor=0.1):
        super().__init__(bounds, seed, n_init, objectives, featurize)
        import numpy as np
        order = ['MaxThreads', 'MinTeams', 'NumThreads', 'NumTeams']
        self.order = order
        self.P = prior_coords(np.array([[p[k] for k in order] for p in prior_points])) \
            if prior_points else None
        self.beta, self.bw, self.floor = beta, bandwidth, floor

    def _prior(self, arr):
        np = self.np
        if self.P is None:
            return np.ones(len(arr))
        Z = prior_coords(arr[:, [self.names.index(k) for k in self.order]])
        d = ((Z[:, None, :] - self.P[None]) ** 2).sum(-1)
        dens = np.exp(-d / (2 * self.bw ** 2)).mean(1) / ((2 * np.pi) ** 2 * self.bw ** 4)
        return self.floor + (1 - self.floor) * dens / max(dens.max(), 1e-12)

    def _sample_prior(self, n):
        np = self.np
        if self.P is None:
            return self._random(n)
        # sample prior coords from the KDE, then invert the teams transform
        pick = self.P[self.rng.integers(len(self.P), size=n)]
        z = np.clip(pick + self.rng.normal(0, self.bw, pick.shape), 1e-6, 1.0)
        u = z.copy()
        u[:, 3] = 2 ** ((z[:, 3] - 1) * 10)
        uni = self._random(n)
        use_uniform = self.rng.random(n) < self.floor
        u[use_uniform] = uni[use_uniform][:, [self.names.index(k) for k in self.order]]
        out = np.empty_like(u)
        for j, k in enumerate(self.order):
            out[:, self.names.index(k)] = u[:, j]
        return out

    def ask(self):
        np = self.np
        self.asked += 1
        if self.asked == 1:
            return {k: b[1] for k, b in self.bounds.items()}
        if len(self.y) < self.n_init or len(self.y) < 3:
            cands = self._usable(self._sample_prior(200))
            return cands[0][0] if cands else dict(zip(self.names, self._random(1)[0]))
        from sklearn.ensemble import ExtraTreesRegressor
        from scipy.stats import norm
        arr = np.vstack([self._random(self.n_candidates), self._sample_prior(500)])
        top = np.argsort(self.y)[-5:]
        local = [np.clip(np.array([self.points[i][k] for k in self.names]) +
                         self.rng.normal(0, 0.05, (60, len(self.names))), 1e-9, 1.0) for i in top]
        arr = np.vstack([arr] + local)
        cands = self._usable(arr)
        if not cands:
            return dict(zip(self.names, self._random(1)[0]))
        y = self._targets()
        model = ExtraTreesRegressor(200, min_samples_leaf=2, random_state=self.seed).fit(
            np.array(self.F), y)
        Fc = np.array([c[1] for c in cands])
        per = np.stack([t.predict(Fc) for t in model.estimators_])
        mu, sd = per.mean(0), np.maximum(per.std(0), 1e-9)
        best = y.max()
        z = (mu - best) / sd
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        pts = np.array([[c[0][k] for k in self.names] for c in cands])
        ei = ei * self._prior(pts) ** (self.beta / max(1, len(self.y)))
        return cands[int(np.argmax(ei))][0]


def make_searcher(algorithm, bounds, seed, n_init, objectives, featurize=None,
                  target='objective', options=None):
    options = options or {}
    if algorithm in ('trees-config', 'trees-meta'):
        assert featurize is not None, f'{algorithm} needs a featurizer'
        return TreesSearcher(bounds, seed, n_init, objectives, featurize, target=target)
    if algorithm == 'trees-prior':
        return PriorTreesSearcher(bounds, seed, n_init, objectives, featurize,
                                  options.get('prior_points', []))
    if algorithm == 'transfer-model':
        return TransferModelSearcher(bounds, seed, objectives, options['pool'],
                                     options['target_kid'], options['klass'],
                                     options['table'], options['trip_count'],
                                     initial_points=options.get('initial_points'),
                                     prior_points=options.get('prior_points'))
    if algorithm in ('rank-transfer', 'rank-rgpe'):
        return RankTransferSearcher(bounds, seed, objectives, options['pool'],
                                    options['target_kid'], options['klass'],
                                    options['table'], options['trip_count'],
                                    rgpe=algorithm == 'rank-rgpe')
    if algorithm == 'bo':
        return BOSearcher(bounds, seed, n_init, objectives)
    if algorithm in ('tpe', 'random', 'motpe'):
        if algorithm == 'motpe':
            assert len(objectives) > 1, 'motpe needs several objectives'
        return OptunaSearcher(bounds, seed, n_init, objectives,
                              'random' if algorithm == 'random' else 'tpe',
                              initial_points=options.get('initial_points'),
                              constant_liar=options.get('constant_liar', False))
    raise ValueError(f'unknown search algorithm {algorithm}')
