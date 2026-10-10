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


def make_searcher(algorithm, bounds, seed, n_init, objectives):
    if algorithm == 'bo':
        return BOSearcher(bounds, seed, n_init, objectives)
    if algorithm in ('tpe', 'random', 'motpe'):
        if algorithm == 'motpe':
            assert len(objectives) > 1, 'motpe needs several objectives'
        return OptunaSearcher(bounds, seed, n_init, objectives,
                              'random' if algorithm == 'random' else 'tpe')
    raise ValueError(f'unknown search algorithm {algorithm}')
