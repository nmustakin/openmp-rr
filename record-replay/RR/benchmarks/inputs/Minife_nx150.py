import itertools
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import argparse
import re

"""miniFE at 150^3 (second input of MINIFE, experiment-archives/inputs-2026-10-14)."""
import RRUtil.Benchmark as Bench
import RRUtil.Device as Device
import RRUtil.Utilities as Utilities

MINIFE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'miniFE-omp')


class MINIFE_NX150(Bench.MakeBenchmark):
    def __init__(self, prof, **kwargs):
        kwargs.pop("executable", None)
        kwargs.pop('resultsdir', None)
        kwargs.pop('root', None)
        super().__init__(prof, "miniFE.x", MINIFE_ROOT, **kwargs)

    def build(self, fn=None, useCase=None):
        if fn is None:
            return super().build({ 'LEFLAGS' : Device.GetCompileOptions(self.Device) })

        leOptions = Device.GetCompileOptions(self.Device) + f' -mllvm --bo-omp-autotune={fn}'
        return super().build({'LEFLAGS' : leOptions}, useCase)

    def clean(self):
        return super().clean()

    def command(self):
        return  f"{self.executableDir}/miniFE.x -nx 150 -ny 150 -nz 150"

    def env(self):
        return ''

    def getApplicationTime(self, stdout):
        exec_time_pattern = '__ExecutionTime__:(.*)'
        tmp =  re.findall(exec_time_pattern, stdout)
        return tmp[0]

