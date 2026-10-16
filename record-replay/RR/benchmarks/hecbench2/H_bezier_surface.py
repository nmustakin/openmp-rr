"""HeCBench bezier-surface-omp for record-replay tuning (bulk set, experiment-archives/hecbench-more).

Built from /data2/nmn/rr-metrics/hecbench/src/bezier-surface-omp with Makefile.rr (Makefile.aomp retargeted to nvptx64/sm_75 with the
record-replay offload flags, hecbench/rr-build2). Run with: bs -n 8192
"""
import re

import RRUtil.Benchmark as Bench
from RRUtil.Utilities import execute_command

SRC = '/data2/nmn/rr-metrics/hecbench/src/bezier-surface-omp'


class H_BEZIER_SURFACE(Bench.MakeBenchmark):
    def __init__(self, prof, **kwargs):
        kwargs.pop('executable', None)
        kwargs.pop('resultsdir', None)
        kwargs.pop('root', None)
        super().__init__(prof, 'bs', SRC, **kwargs)
        self.executableDir = SRC

    def build(self, fn=None, useCase=None):
        self.executableDir = SRC
        return execute_command('make -f Makefile.rr', capture_output=True, cwd=SRC, shell=True)

    def clean(self):
        self.executableDir = SRC
        return execute_command('make -f Makefile.rr clean', capture_output=True, cwd=SRC, shell=True)

    def command(self):
        return '/data2/nmn/rr-metrics/hecbench/src/bezier-surface-omp/bs -n 8192'

    def env(self):
        return ''

    def getApplicationTime(self, stdout):
        m = re.findall(r'([0-9.]+(?:[eE][-+]?[0-9]+)?) *(?:s|sec|secs|seconds)\b', stdout or '')
        return m[-1] if m else '0'
