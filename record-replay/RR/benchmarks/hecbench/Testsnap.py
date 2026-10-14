"""HeCBench testSNAP-omp (proxy-app subset) for record-replay tuning.

Built from /data2/nmn/rr-metrics/hecbench/src/testSNAP-omp with Makefile.rr (Makefile.aomp
retargeted to nvptx64/sm_75 with the record-replay offload flags, see
/data2/nmn/rr-metrics/hecbench/rr-build/README.md). Recorded with: main --nsteps 2
"""
import os
import re

import RRUtil.Benchmark as Bench
from RRUtil.Utilities import execute_command

SRC = '/data2/nmn/rr-metrics/hecbench/src/testSNAP-omp'


class TESTSNAP(Bench.MakeBenchmark):
    def __init__(self, prof, **kwargs):
        kwargs.pop('executable', None)
        kwargs.pop('resultsdir', None)
        kwargs.pop('root', None)
        super().__init__(prof, 'main', SRC, **kwargs)
        self.executableDir = SRC

    def build(self, fn=None, useCase=None):
        self.executableDir = SRC
        return execute_command('make -f Makefile.rr', capture_output=True, cwd=SRC, shell=True)

    def clean(self):
        self.executableDir = SRC
        return execute_command('make -f Makefile.rr clean', capture_output=True, cwd=SRC, shell=True)

    def command(self):
        return f'{SRC}/main --nsteps 2'.strip()

    def env(self):
        return ''

    def getApplicationTime(self, stdout):
        m = re.findall(r'([0-9.]+(?:[eE][-+]?[0-9]+)?) *(?:s|sec|secs|seconds)', stdout or '')
        return m[-1] if m else '0'
