from abc import ABC, abstractmethod
import glob
import json
import numpy as np
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import math
import itertools as it
import uuid
import hashlib
import pandas as pd
import re

from bayes_opt import BayesianOptimization, UtilityFunction
import pynvml
from pynvml import nvmlDeviceGetTotalEnergyConsumption as nvmlEnergy

from RRUtil import Device
from RRUtil.Utilities import createDir
from RRUtil.Utilities import execute_command
from RRUtil.DataBase import DB

class Kernel:
    # v3: in-process time-based warm-up (clock ramp) + repeated event-timed
    # launches (median), kernel-
    # level NVML energy over a back-to-back launch window, periodic baseline
    # re-measurement, and median aggregation. Not comparable with v2 samples.
    MeasurementSchema = 'device-kernel-timing-v3'
    # Overridable from the environment for measurement experiments.
    TimingWarmup = int(os.environ.get('RR_TIMING_WARMUP', 3))
    # Clocks ramp from idle over ~0.5 s on T4; time the kernel only after.
    TimingWarmupMs = int(os.environ.get('RR_TIMING_WARMUP_MS', 1000))
    TimingRepeats = int(os.environ.get('RR_TIMING_REPEATS', 10))
    EnergyMinMs = int(os.environ.get('RR_ENERGY_MIN_MS', 300))
    BaselineEvery = int(os.environ.get('RR_BASELINE_EVERY', 20))
    ExclusiveWaitSeconds = int(os.environ.get('RR_EXCLUSIVE_WAIT_SECONDS', 3600))
    ContendedRetries = int(os.environ.get('RR_CONTENDED_RETRIES', 5))
    # Replay processes per configuration (each runs warm-up + repeats).
    TimingTries = int(os.environ.get('RR_TIMING_TRIES', 5))
    # Opt-in early stop: if the baseline already saturates the SM pipes
    # (sm__throughput % of peak above this, e.g. 80), cut the search budget
    # to SaturatedBudget suggestions. On 10 LULESH kernels every kernel above
    # 83% had <= 1.05x to gain and every kernel with a real gain was <= 51%
    # (experiment-archives/metric-search-2026-10-10). 0 disables.
    SaturatedSMPct = float(os.environ.get('RR_SATURATED_SM_PCT', 0))
    SaturatedBudget = int(os.environ.get('RR_SATURATED_BUDGET', 15))
    # Baseline metric samples whose median classifies the kernel (adaptive).
    ClassifySamples = int(os.environ.get('RR_CLASSIFY_SAMPLES', 3))
    # Extra perfworks metrics collected in the metrics replay (validated on
    # T4 / sm_75). Request counts do not vary with launch configuration;
    # these capture traffic, cache behaviour, utilisation and spills.
    ExtraMetrics = os.environ.get('RR_EXTRA_METRICS', ','.join([
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__t_sectors.sum',
        'lts__t_sector_hit_rate.pct',
        'l1tex__t_sector_hit_rate.pct',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'smsp__inst_executed.sum',
        'sm__inst_executed_pipe_fp64.sum',
        'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active',
        'sm__cycles_active.avg',
        'smsp__thread_inst_executed_per_inst_executed.ratio',
        'smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio',
        'gpu__time_duration.sum']))

    class OMPReplayer:
        def __init__(self, MaxMem):
            self.envVars = {
                'OMP_NUM_THREADS' : 1,
                'OMP_TARGET_OFFLOAD' : 'mandatory',
                'LIBOMPTARGET_NEXTGEN_PLUGINS' : 1,
                'LIBOMPTARGET_RR_DEVMEM_SIZE' : MaxMem,
                # Whole-application recording only needs CUPTI activity data.
                # Enable expensive hardware metrics for isolated replays.
                'LIBOMPTARGET_RR_PROFILE_METRICS' : 1
            }
            self._env=' '.join([f'{k}={v}' for k,v in self.envVars.items()])
            self.cmd = 'llvm-omp-kernel-replay'

        def command(self, numThreads=None, numTeams=None, verify=False):
            cmd = self.cmd
            if not isinstance(numThreads, type(None)):
                cmd += ' --num-threads {0}'.format(int(numThreads))
            if not isinstance(numTeams, type(None)):
                cmd += ' --num-teams {0}'.format(int(numTeams))
            if verify:
                cmd += f' --verify'
            return cmd + ' {kernel_name}.json'

        def env(self, profile_metrics=True, profile_activities=True,
                profile_timing=False, energy=True):
            env_vars = dict(self.envVars)
            env_vars['LIBOMPTARGET_RR_PROFILE_METRICS'] = int(profile_metrics)
            env_vars['LIBOMPTARGET_RR_PROFILE_ACTIVITIES'] = int(profile_activities)
            env_vars['LIBOMPTARGET_RR_PROFILE_TIMING'] = int(profile_timing)
            if profile_metrics and Kernel.ExtraMetrics:
                env_vars['LIBOMPTARGET_RR_EXTRA_METRICS'] = Kernel.ExtraMetrics
            if profile_metrics:
                env_vars['LIBOMPTARGET_RR_METRICS_WARMUP_MS'] = Kernel.TimingWarmupMs
            if profile_timing:
                env_vars['LIBOMPTARGET_RR_TIMING_WARMUP'] = Kernel.TimingWarmup
                env_vars['LIBOMPTARGET_RR_TIMING_WARMUP_MS'] = Kernel.TimingWarmupMs
                env_vars['LIBOMPTARGET_RR_TIMING_REPEATS'] = Kernel.TimingRepeats
                env_vars['LIBOMPTARGET_RR_ENERGY_MIN_MS'] = Kernel.EnergyMinMs if energy else 0
            return ' '.join([f'{k}={v}' for k,v in env_vars.items()])

    def __init__(self, Device, name, hashname, dbDir, recorded=True):
        self.Name = name
        self.recorded = recorded
        self.HashName = hashname
        self.original = f'{hashname}.image.orig'
        # This we should fix. Ckpt, should instruct to not copy
        if recorded:
            if not Path(self.original).exists():
                shutil.copyfile(f'{hashname}.image', self.original)
        self.Profiler = Device
        self.Replayer = self.OMPReplayer(4) #request 4Gb memory
        self.dbDir = dbDir
        self.profiled = False
        self.NumRegs = -1
        self.SMem = -1
        self.NumTeams = -1
        self.NumThreads = -1
        self.ExecTime = -1
        self.Metrics = {}

    def getTripCount(self):
        if getattr(self, '_trip_count', None) is None:
            with open(f'{self.HashName}.json', 'r') as fd:
                self._trip_count = json.load(fd)['LoopTripCount']
        return self._trip_count

    def isWorkShare(self):
        if self.getTripCount() > 0:
            return True

        return False

    def getDefaultKey(self):
        # This will serve as our Baseline
        defaultConfig = { 'NumThreads' : None, 'NumTeams' : None }
        #TODO: replace with cupti (might not need to, Profiler seems to be an alias for Device)
        for compilationConfig in self.Profiler.getCompilationConfigs():
            defaultConfig.update( { compilationConfig : None } )
        defaultKey = self.getConfigKey(defaultConfig)
        return defaultKey

    def isOptimized(self, **kwargs):
        configKey = self.getConfigKey(kwargs)
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())
        totalExperiments = db.getTotalSize()
        if totalExperiments == -1:
            return False

        return len(db) == totalExperiments + 1

    def Pending(self, **kwargs):
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())
        totalExperiments = db.getTotalSize()
        if totalExperiments == -1:
            return 2**31

        if (len(db) > totalExperiments ):
            return 0
        return totalExperiments + 1 - len(db)

    def getDefaultTime(self, **kwargs):
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())
        return db.getBaseline()

    def getDefaultEnergy(self, **kwargs):
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())
        return db.getBaselineEnergy()

    def Stats(self, **kwargs):
        configKey = self.getConfigKey(kwargs)
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())
        _, X, Y = db.GetSpeedUps()
        if len(Y) > 0:
            index_max = max(range(len(Y)), key=Y.__getitem__)
            bestConfig = X[index_max]
            bestKey = self.getConfigKey(bestConfig)
            return (configKey, Y[index_max], db.EnergyImprov(bestKey),
                    db.LoadRatio(bestKey), db.StoreRatio(bestKey), bestConfig)
        return configKey, -1, -1, -1, -1, {}

    def ExplorationStats(self, **kwargs):
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())
        return db.GetExplorationStats()
        
    def BuildBest(self, **kwargs):
        configKey = self.getConfigKey(kwargs)
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())

        Configs, X, Y =  db.GetSpeedUps()
        index_max = max(range(len(Y)), key=Y.__getitem__)
        bestConfig = X[index_max]
        best  = ' '.join([f'{k}={v}' for k,v in X[index_max].items()])

        regs = bestConfig['NumRegsPerThread']
        pressure = bestConfig['RegEffort']
        NumThreads = bestConfig['NumThreads']
        NumTeams = bestConfig['NumTeams']
        self.prof.deviceCompile(self.original, self.Name, self.HashName, regs, pressure)
        print('Best :{} {}={} {}={} (s)'.format( best, 'speedup', Y[index_max], 'Duration', db.Duration(Configs[index_max])))
        # TODO: Replayer.command() runs nvprof, need to change that. 
        cmd = self.Replayer.command(NumThreads, NumTeams).format(kernel_name=f'{self.HashName}')
        env = self.Replayer.env(profile_metrics=True, profile_activities=True)
        #TODO: replace with cupti
        #Profiler.command() runs nvprof
        #cmd = self.Profiler.command().format(executable=cmd, output=f'{self.HashName}.csv')
        cmd = self.Profiler.command().format(executable=cmd)
        #env += ' ' + self.Profiler.env()
        print('Cmd:', env + ' ' + cmd)

    @staticmethod
    def extractKernels(kName, kname_to_hash, toDelete, Device):
        imageName = kname_to_hash[kName]
        shutil.copy(f'{imageName}.image', f'{imageName}.image.beforeExtract')

        kernelImage =f'{imageName}.image'

        cmdP =('llvm-extract {input} '
                ' --delete --func {kDelete} -o {outName}')
        for toDeleteKernel in toDelete:
            cmd = cmdP.format(input=kernelImage, kDelete=toDeleteKernel,
                    outName=kernelImage)
            ret, stdout, stderr = execute_command(cmd, capture_output=True,
                    cwd=os.getcwd(), shell=True)
        Device.renameSymbols(kernelImage, kernelImage)
        print('kernelImage', kernelImage)

    @staticmethod
    def getConfigKey(config):
        #filtered_config = {k: v for k, v in config.items() if k != 'deviceID'}
        #return '-'.join([str(k)+'='+str(filtered_config[k]) for k in filtered_config])
        return '-'.join([str(k)+'='+str(config[k]) for k in config])

    def optimize_exhaustive(self, **kwargs):
        @Device.Stage(self.Profiler.getStageDir(), uuid.uuid4().hex, self.HashName)
        def optimize(self, **kwargs):
            config = {}
            for compilationConfig in self.Profiler.getCompilationConfigs():
                print(compilationConfig)
                v = kwargs.get(compilationConfig, None)
                if v:
                    start, end, step = self.Profiler.getDefaultBounds(compilationConfig)
                    config[compilationConfig] = (start, end, step)

            configKey = self.getConfigKey(config)

            defaultKey = self.getDefaultKey()
            db = DB(self.getDBName(**kwargs), defaultKey)
            db.setMeasurementSchema(self.MeasurementSchema)

            # Get data for baseline.
            if not (defaultKey in db):
                #print('sample defaultKey', defaultKey)
                Valid, SStats, DStats, Duration, Energy, GLoads, GStores = self.sample()
                db.Add(defaultKey, [SStats], DStats, [Duration], [Energy], [GLoads], [GStores], Valid)

            experiments = []
            if 'MaxThreads' in config:
                print(config['MaxThreads'][0], config['MaxThreads'][1], config['MaxThreads'][2])
                for tid in range(config['MaxThreads'][0], config['MaxThreads'][1], config['MaxThreads'][2]):
                    if 'MinTeams' in configKey:
                        for tmId in range(config['MinTeams'][0], config['MinTeams'][1], config['MinTeams'][2]):
                            if self.Profiler.isValidConfig(tid, tmId):
                                experiments.append( { 'MaxThreads' : tid,
                                                      'MinTeams' : tmId,
                                                      'NumThreads' : tid } )
                    else:
                        experiments.append( { 'MaxThreads' : tid,
                                              'NumThreads' : tid,
                                              'NumTeams' : None,
                                             } )

            mData =  {
                       'args' : kwargs,
                       'config': config,
                       'totalExperiments' : len(experiments)
                      }

            db.setMetadata(mData)

            bestConfig = None
            bestSpeedup = None
            worstConfig = None
            worstSpeedup = None
            
            bestEnergyConfig = None
            bestImprov = None
            worstEnergyConfig = None
            worstImprov = None
            
            leastLoads = None
            leastLoadConfig = None
            leastStores = None
            leastStoreConfig = None
            mostLoads = None
            mostLoadConfig = None
            mostStores = None
            mostStoreConfig = None

            for e in experiments:
                #print(e)
                key = self.getConfigKey(e)

                Valid, SStats, DStats, Duration, Energy, GLoads, GStores = (0, -1, -1, 1, 1, 1, 1)
                if not (key in db):
                    Valid, SStats, DStats, Duration, Energy, GLoads, GStores = self.sample(Tries=5, **e)
                db.Add(key, [SStats], DStats, [Duration], [Energy], [GLoads], [GStores], Valid)

                speedup = db.Speedup(key)
                energy_improv = db.EnergyImprov(key)
                load_ratio = db.LoadRatio(key)
                store_ratio = db.StoreRatio(key)
                print(f'{key} -> Speedup: {speedup}, Energy_Improv: {energy_improv}, '
                      f'Load_Ratio: {load_ratio}, Store_Ratio: {store_ratio}')
                if bestSpeedup is None or bestSpeedup < speedup:
                    bestSpeedup = speedup
                    bestConfig = e
                if worstSpeedup is None or worstSpeedup > speedup:
                    worstSpeedup = speedup
                    worstConfig = e

                if bestImprov is None or bestImprov < energy_improv:
                    bestImprov = energy_improv
                    bestEnergyConfig = e
                if worstImprov is None or worstImprov > energy_improv:
                    worstImprov = energy_improv
                    worstEnergyConfig = e

                meanGLoads = np.array(GLoads).mean() if isinstance(GLoads, list) else GLoads
                meanGStores = np.array(GStores).mean() if isinstance(GStores, list) else GStores

                if leastLoads is None or leastLoads > meanGLoads:
                    leastLoads = meanGLoads
                    leastLoadConfig = e
                if leastStores is None or leastStores > meanGStores:
                    leastStores = meanGStores
                    leastStoreConfig = e

                if mostLoads is None or mostLoads < meanGLoads:
                    mostLoads = meanGLoads
                    mostLoadConfig = e
                if mostStores is None or mostStores < meanGStores:
                    mostStores = meanGStores
                    mostStoreConfig = e
            
            print(f'Kernel: {self.Name} Best speedup: {bestConfig}; f(x) = {bestSpeedup}')
            print(f'Kernel: {self.Name} Worst speedup: {worstConfig}; f(x) = {worstSpeedup}')
            print(f'Kernel: {self.Name} Least energy: {bestEnergyConfig}; f(x) = {bestImprov}')
            print(f'Kernel: {self.Name} Most energy: {worstEnergyConfig}; f(x) = {worstImprov}')

            print(f'Kernel: {self.Name} Least loads: {leastLoadConfig}; f(x) = {leastLoads}')
            print(f'Kernel: {self.Name} Most loads: {mostLoadConfig}; f(x) = {mostLoads}')
            print(f'Kernel: {self.Name} Least stores: {leastStoreConfig}; f(x) = {leastStores}')
            print(f'Kernel: {self.Name} Most stores: {mostStoreConfig}; f(x) = {mostStores}')

        optimize(self, **kwargs)

    def getBounds(self, **kwargs):
        WarpSize = self.Profiler.WarpSize
        MaxTeamSize = self.Profiler.MaxThreadBlockSize
        MaxWarpsPerTeam = int(MaxTeamSize / WarpSize)
        MaxNumTeams = min(int(self.getTripCount()/WarpSize), (2**31)-1)

        pBounds = {}
        # Over here we optimize launch bounds we will use fractional bounds
        if True: #'MinTeams' in kwargs or 'MaxThreads':
            #We should not include pure 0. We always require at least a single element
            MinTeams = kwargs.get('MinTeams', None)
            if MinTeams is not None:
                pBounds['MinTeams'] = (np.nextafter(0,1), 1.0)

            MaxThreads = kwargs.get('MaxThreads', None)
            if MaxThreads is not None:
                pBounds['MaxThreads'] = (np.nextafter(0, 1), 1.0)

            NumThreads = kwargs.get('NumThreads', None)
            if NumThreads is not None:
                pBounds['NumThreads'] = (np.nextafter(0, 1), 1.0)

            NumTeams = kwargs.get('NumTeams', None)
            if NumTeams is not None:
                pBounds['NumTeams'] = (np.nextafter(0, 1), 1.0)
        else:
            NumThreads = kwargs.get('NumThreads', None)
            if not isinstance(NumThreads, type(None)):
                pBounds['NumThreads'] = [1, MaxWarpsPerTeam + 0.99]

            NumTeams = kwargs.get('NumTeams', None)
            if not isinstance(NumTeams, type(None)):
                pBounds['NumTeams'] = [1, MaxNumTeams + 0.99]

            for compilationConfig in self.Profiler.getCompilationConfigs():
                v = kwargs.get(compilationConfig, None)
                if v:
                    pBounds[compilationConfig] = self.Profiler.getDefaultBounds(compilationConfig)
        return pBounds

    def getDBName(self, **kwargs):
        pBounds = self.getBounds(**kwargs)
        configKey = self.getConfigKey(pBounds)
        hashedKey = hashlib.md5(configKey.encode()).hexdigest()
        return f'{self.dbDir}/analysis{self.HashName}.{hashedKey}.json'

    def convertPointToValues(self, Points):
        # max threads indicates that BO uses a bounding box of 0->1
        # fom all options
        #if 'MaxThreads' in Points.keys():
        return self.Profiler.getLaunchBounds(Points, self.getTripCount())

        config = {}
        if 'NumThreads' in Points.keys():
            NumWarps = int(math.floor(Points['NumThreads']))
            NumThreads = NumWarps * self.Profiler.WarpSize
            config.update( { 'NumThreads' : NumThreads } )

        if 'NumTeams' in Points.keys():
            NumTeams = int(math.floor(Points['NumTeams']))
            config.update( { 'NumTeams' : NumTeams } )

        for compilationConfig in self.Profiler.getCompilationConfigs():
            if compilationConfig not in Points.keys():
                continue

            v = int(math.floor(Points[compilationConfig]))
            config.update( { compilationConfig : v } )
        return config

    def BO(self, **kwargs):
        @Device.Stage(self.Profiler.getStageDir(), uuid.uuid4().hex, self.HashName)
        def optimize(self, **kwargs):
            #default values
            initSamples = kwargs.get('initSamples', 3)
            optSamples = kwargs.get('optSamples', 10)
            deviceID = kwargs.get('device', 0)
            objective = kwargs.get('objective', 'memory')
            #print(initSamples, optSamples)

            pBounds = self.getBounds(**kwargs)
            #print('pBounds', pBounds)

            random_state = kwargs.get('random_state', 1234)
            optimizer = BayesianOptimization(f = None,
                                             pbounds = pBounds,
                                             verbose = 2,
                                             random_state = random_state,
                                             allow_duplicate_points=True
                                             )
            # Here we have the utility function
            utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)

            db = DB(self.getDBName(**kwargs), self.getDefaultKey())
            db.setMeasurementSchema(self.MeasurementSchema)
            stats_file = f'{self.dbDir}/results_{self.HashName}.csv'

            stats = [] 
            
            mData =  {
                       'args' : kwargs,
                       'bounds' : pBounds,
                       'totalExperiments' : initSamples + optSamples
                     }

            db.setMetadata(mData)

            if not (self.getDefaultKey() in db):
                Valid, SStats, DStats, Duration, Energy, GLoads, GStores = self.sample()
                db.Add(self.getDefaultKey(), [SStats], DStats, [Duration], [Energy], [GLoads], [GStores], Valid)

            pending = optSamples + initSamples

            for performed in range (0, pending):
                # Always evaluate the full launch-bound point. Important
                # configurations such as a full thread block occupy only a
                # narrow interval after fractional values are rounded to warp
                # sizes, so purely random initialization frequently misses
                # them and leaves the optimizer with only near-baseline data.
                if performed == 0:
                    NextPoint = {
                        name: bounds[1] for name, bounds in pBounds.items()
                    }
                elif performed < initSamples:
                    NextPoint = optimizer.space.array_to_params(optimizer.space.random_sample())
                else:
                    NextPoint = optimizer.suggest(utility)
                config = self.convertPointToValues(NextPoint)
                config['DeviceID'] = deviceID
                print("================= Next Point ====================")
                print(NextPoint)
                #print(config)
                print(f'Performing {performed} experiment out of {pending}')
                print("=================================================")

                key = self.getConfigKey(config)
                Valid, SStats, DStats, Duration, Energy, GLoads, GStores = (0, -1, -1, 1, 1, -1, -1)
                if not (key in db):
                    Valid, SStats, DStats, Duration, Energy, GLoads, GStores = self.sample(Tries=5, **config)
                db.Add(key, [SStats], DStats, [Duration], [Energy],[GLoads], [GStores], Valid)
                if Valid and getattr(self, 'LastMetrics', None):
                    db.SetMetrics(key, self.LastMetrics)
                # Re-measure the baseline periodically so speedups are taken
                # against a baseline that averages over the run's drift.
                if (self.BaselineEvery > 0 and
                        (performed + 1) % self.BaselineEvery == 0):
                    BValid, _, _, BDuration, BEnergy, _, _ = self.sample()
                    if BValid:
                        db.AddSamples(self.getDefaultKey(), BDuration, BEnergy)
                Speedup = db.Speedup(key)
                EnergyGain = db.EnergyImprov(key)
                LoadRatio = db.LoadRatio(key)
                StoreRatio = db.StoreRatio(key)
                
                if(Speedup >= 1.05 or EnergyGain >= 1): 
                    row_data = {**config, 'Speedup': Speedup,
                                'Energy_Improvment': EnergyGain,
                                'Load_Ratio': LoadRatio,
                                'Store_Ratio': StoreRatio}
                    stats.append(row_data)

                print(f'{key} -> Speedup: {Speedup}, EnergyImprov: {EnergyGain}, '
                      f'Load_Ratio: {LoadRatio}, Store_Ratio: {StoreRatio}')
                try:
                    # Update the optimizer with the evaluation results.
                    # Energy - Delay product - EnergyGain * Speedup
                    #optimizer.register(params = NextPoint, target = EnergyGain * Speedup)
                    targets = {
                        'speedup': Speedup,
                        'energy': EnergyGain,
                        'loads': LoadRatio,
                        'stores': StoreRatio,
                        'memory': LoadRatio * StoreRatio,
                    }
                    optimizer.register(params = NextPoint, target = targets[objective])
                    #optimizer.register(params = NextPoint, target = Speedup)
                except:
                    assert False, 'optimizer error'

            print('Kernel: {} Best result: {}; f_{}(x) = {}.'.format(self.Name, optimizer.max['params'], objective, optimizer.max['target']))
            df = pd.DataFrame(stats)
            df.to_csv(stats_file, index=False)

        optimize(self, **kwargs)

    # Nominal cost (s) of one evaluation, measured on T4 (experiment-archives/
    # search-efficiency-2026-10-10/cost_breakdown.log): compile 0.4 s, a timing
    # process ~4.0 s without and ~4.35 s with the energy window, a warm metrics
    # replay ~5.5 s. Used to give multi-fidelity searches an equal budget.
    COST_SCREEN = 0.4 + 4.0
    COST_FULL = 0.4 + 2 * 4.35 + 5.5
    COST_PROMOTE = 4.35 + 5.5

    def Search(self, algorithm, objectives, budget, initSamples, seed, cacheDB=None,
               device=0, compileTable=None, options=None, **kwargs):
        """Search with a pluggable strategy (RRUtil.Search).

        `budget` counts suggestions, as Kernel.BO does (a suggestion that
        rounds to an already-evaluated config still consumes budget).
        `cacheDB` is a DB path shared by runs on the same kernel and GPU: a
        config measured by an earlier run is reused instead of re-measured.
        Results go to the DB named by getDBName under self.dbDir; the shipped
        configuration is recorded in its metadata ('shipped').

        Strategies beyond the plain searchers:
          adaptive       classify the kernel from baseline metrics, then
                         compute -> tpe with a cut budget, register ->
                         trees-meta steering by speedup + spills/occupancy,
                         other -> tpe
          tpe-mf         multi-fidelity tpe: screen every proposal with one
                         timing process (no metrics), within a nominal cost
                         budget of `budget` full evaluations, then fully measure
                         the `promote` best screens and ship among them
          tpe-portfolio  tpe whose first proposals are the best configs of
                         other kernels (options['pool'], leave-one-out)
          trees-prior    trees-config with a piBO-style prior from the top
                         configs of other same-class kernels
          transfer-model cost model trained on other kernels plus this one,
                         ranking a deduplicated candidate set
        """
        from RRUtil.Search import (make_searcher, objective_values, build_compile_table,
                                   compile_features, classify_bottleneck, build_portfolio,
                                   config_to_point)
        options = dict(options or {})
        pBounds = self.getBounds(**kwargs)
        db = DB(self.getDBName(**kwargs), self.getDefaultKey())
        db.setMeasurementSchema(self.MeasurementSchema)
        cache = None
        if cacheDB is not None:
            cache = DB(cacheDB, self.getDefaultKey())
            cache.setMeasurementSchema(self.MeasurementSchema)

        def measure(config, screen=False):
            return self.sample(Tries=self.TimingTries, Screen=screen, **config)

        if self.getDefaultKey() not in db:
            Valid, SStats, DStats, Duration, Energy, GLoads, GStores = measure({})
            db.Add(self.getDefaultKey(), [SStats], DStats, [Duration], [Energy],
                   [GLoads], [GStores], Valid)
            if Valid:
                db.SetMetrics(self.getDefaultKey(), getattr(self, 'LastMetrics', {}))
        base_metrics = db.Data.get(self.getDefaultKey(), {}).get('metrics', {})

        klass = None
        if algorithm in ('adaptive', 'tpe-portfolio', 'trees-prior', 'transfer-model'):
            # Classify from the median of several baseline metric samples; a
            # single profiled launch can be power-throttled.
            samples = db.Data[self.getDefaultKey()].get('metrics_samples', [base_metrics])
            while len(samples) < self.ClassifySamples:
                BValid, _, _, BDuration, BEnergy, _, _ = measure({})
                if not BValid:
                    break
                db.AddSamples(self.getDefaultKey(), BDuration, BEnergy)
                samples.append(dict(getattr(self, 'LastMetrics', {})))
            db.Data[self.getDefaultKey()]['metrics_samples'] = samples
            db.Update()
            names = set().union(*[m.keys() for m in samples])
            median = {n: float(np.median([m[n] for m in samples if n in m])) for n in names}
            klass = classify_bottleneck(median, self.SaturatedSMPct or 80.0)

        strategy, target, fidelity = algorithm, 'objective', 'full'
        pool = options.pop('pool', None)
        if algorithm == 'adaptive':
            if klass == 'register':
                strategy, target = 'trees-meta', 'register-mix'
            else:
                strategy = 'tpe'
            if klass == 'compute':
                budget = min(budget, self.SaturatedBudget)
            print(f'Adaptive: kernel classified {klass}; strategy {strategy}, target '
                  f'{target}, budget {budget}', flush=True)
        elif algorithm == 'tpe-mf':
            strategy, fidelity = 'tpe', 'mf'
        elif algorithm == 'tpe-portfolio':
            strategy = 'tpe'
            options['initial_points'] = build_portfolio(
                pool, options['target_kid'], klass, self.getTripCount(),
                size=options.get('portfolio_size', 10))
            print(f'Portfolio: {len(options["initial_points"])} points for class {klass}',
                  flush=True)
        elif algorithm == 'trees-prior':
            others = pool[(pool.kid != options['target_kid']) & (pool.klass == klass)]
            if others.empty:
                others = pool[pool.kid != options['target_kid']]
            top = others[others.speedup >= others.groupby('kid').speedup.transform(
                lambda v: v.quantile(0.95))]
            options['prior_points'] = [config_to_point(
                dict(MaxThreads=r.MaxThreads, MinTeams=r.MinTeams, NumThreads=r.NumThreads,
                     teams_frac=r.teams_frac), self.getTripCount()) for r in top.itertuples()]
            print(f'Prior: {len(options["prior_points"])} top configs from '
                  f'{others.kid.nunique()} {klass} kernels', flush=True)
        else:
            sm_pct = base_metrics.get('sm__throughput.avg.pct_of_peak_sustained_elapsed')
            if self.SaturatedSMPct > 0 and sm_pct is not None and sm_pct > self.SaturatedSMPct:
                print(f'Baseline SM throughput {sm_pct:.1f}% > {self.SaturatedSMPct}%: '
                      f'compute-saturated, budget {budget} -> {self.SaturatedBudget}', flush=True)
                budget = min(budget, self.SaturatedBudget)
        db.setMetadata({'args': {**kwargs, 'algorithm': algorithm, 'strategy': strategy,
                                 'target': target, 'class': klass, 'fidelity': fidelity,
                                 'objectives': objectives, 'seed': seed,
                                 'initSamples': initSamples},
                        'bounds': pBounds, 'totalExperiments': budget})

        featurize, table = None, None
        if strategy in ('trees-meta', 'transfer-model'):
            table = build_compile_table(
                self, compileTable or f'{self.dbDir}/compile_table.json')
        if strategy in ('trees-config', 'trees-prior'):
            def featurize(point):
                key = self.getConfigKey({**self.convertPointToValues(point), 'DeviceID': device})
                return [point[k] for k in sorted(point)], False, key
        elif strategy == 'trees-meta':
            trip_count = self.getTripCount()

            def featurize(point):
                config = self.convertPointToValues(point)
                key = self.getConfigKey({**config, 'DeviceID': device})
                f, invalid = compile_features(config, table, trip_count)
                return f, invalid, key
        if strategy == 'transfer-model':
            options.update(pool=pool, klass=klass, table=table, trip_count=self.getTripCount())
        searcher = make_searcher(strategy, pBounds, seed, initSamples, objectives,
                                 featurize, target, options)
        # The register-mix target needs metrics for every observed config, so
        # cached entries measured before metrics were collected are re-measured.
        need_metrics = target != 'objective'

        trace, spent = [], 0.0
        cost_budget = budget * self.COST_FULL
        reserve = (options.get('promote', 5) * self.COST_PROMOTE) if fidelity == 'mf' else 0.0
        step = 0
        while (step < budget) if fidelity == 'full' else (spent + self.COST_SCREEN + reserve
                                                            <= cost_budget):
            point = searcher.ask()
            config = self.convertPointToValues(point)
            config['DeviceID'] = device
            key = self.getConfigKey(config)
            source = 'repeat'
            unit = self.COST_SCREEN if fidelity == 'mf' else self.COST_FULL
            if key not in db:
                cached = cache is not None and key in cache and (
                    not need_metrics or not cache.Data[key]['valid']
                    or 'metrics_warmup_ms' in cache.Data[key].get('metrics', {}))
                if cached:
                    db.CopyEntry(key, cache)
                    source = 'cache'
                else:
                    Valid, SStats, DStats, Duration, Energy, GLoads, GStores = measure(
                        config, screen=fidelity == 'mf')
                    db.Add(key, [SStats], DStats, [Duration], [Energy], [GLoads],
                           [GStores], Valid)
                    if Valid and fidelity == 'full':
                        db.SetMetrics(key, getattr(self, 'LastMetrics', {}))
                    if fidelity == 'mf':
                        db.Data[key]['fidelity'] = 'screen'
                        db.Update()
                    elif cache is not None:
                        cache.CopyEntry(key, db)
                    source = 'measured'
                spent += unit          # cache hits are charged their nominal cost
            else:
                db.Add(key, None, None, None, None, None, None, None)
            if (self.BaselineEvery > 0 and (step + 1) % self.BaselineEvery == 0
                    and fidelity == 'full'):
                BValid, _, _, BDuration, BEnergy, _, _ = measure({})
                if BValid:
                    db.AddSamples(self.getDefaultKey(), BDuration, BEnergy)
            values = objective_values(db.Speedup(key), db.EnergyImprov(key))
            metrics = db.Data[key].get('metrics', {})
            values['spill'] = metrics.get('l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum', 0.0)
            values['occupancy'] = metrics.get(
                'sm__warps_active.avg.pct_of_peak_sustained_active', float('nan'))
            searcher.tell(point, values)
            trace.append({'step': step, 'key': key, 'source': source, 'class': klass,
                          'cost': spent, 'speedup': db.Speedup(key),
                          'energy_gain': db.EnergyImprov(key)})
            print(f'[{algorithm}/{"+".join(objectives)}/s{seed}] {step + 1} cost={spent:.0f}s '
                  f'{source:8s} {key} speedup={trace[-1]["speedup"]:.4f}', flush=True)
            step += 1

        shipped = None
        if fidelity == 'mf':
            # Promote the best screens to a full measurement and ship among them.
            rows = [(db.Speedup(k), k) for k in {t['key'] for t in trace}
                    if db.Data[k]['valid']]
            for _, k in sorted(rows, reverse=True)[:options.get('promote', 5)]:
                config = {name: int(v) for name, v in (p.split('=') for p in k.split('-'))
                          if name != 'DeviceID' and v != 'None'}
                Valid, _, _, Duration, Energy, _, _ = measure(config)
                if Valid:
                    db.AddSamples(k, Duration, Energy)
                    db.SetMetrics(k, getattr(self, 'LastMetrics', {}))
                    db.Data[k]['fidelity'] = 'promoted'
                    db.Update()
                spent += self.COST_PROMOTE
                trace.append({'step': step, 'key': k, 'source': 'promote', 'class': klass,
                              'cost': spent, 'speedup': db.Speedup(k),
                              'energy_gain': db.EnergyImprov(k)})
                step += 1
            promoted = [k for k in db.Data if db.Data[k].get('fidelity') == 'promoted']
            if promoted:
                shipped = max(promoted, key=db.Speedup)
        if shipped is not None:
            db.MetaData['shipped'] = shipped
            db.Update()
        pd.DataFrame(trace).to_csv(
            f'{self.dbDir}/trace_{self.HashName}.csv', index=False)
        return trace

    # returns isValid, SStats, DStats, execTime, execEnergy, gLoads, gStores
    def sample(self, **kwargs):
        Tries = kwargs.pop('Tries', 5)
        # Screen: one timing process, no energy window, no metrics replay
        # (the cheap fidelity of multi-fidelity search).
        screen = kwargs.pop('Screen', False)
        if screen:
            Tries = 1
        deviceID = kwargs.get('deviceID', 0)
        NumThreads = kwargs.get('NumThreads', None)
        NumTeams = kwargs.get('NumTeams', None)
       
        device_handle = self._nvml_handle(deviceID)
        StaticStats = self.Profiler.deviceCompile(self.original, self.Name, self.HashName, **kwargs)
        SStats = None
        if StaticStats:
            SStats = StaticStats[self.Name]
            if NumThreads is not None:
                if ( SStats[0] * NumThreads > self.Profiler.MaxRegBlock ):
                    return False, None, None, None, None, None, None
            if ( SStats[1] > self.Profiler.MaxSharedMemBlock ) :
                return False, None, None, None, None, None, None

        DStats = list()
        res, durations, energies, GLoads, GStores = self.execute(
            NumThreads, NumTeams, device_handle=device_handle, maxIters=Tries,
            metrics=not screen, energy=not screen)
        if len(res) == 0:
            return False, None, None, None, None, None, None

        DStats += res
        #stats = np.array(DStats)[:,-3]
        end = time.time()
        #end energy here
        end_energy = nvmlEnergy(device_handle)

        return True, SStats, DStats, durations, energies, GLoads, GStores

    @staticmethod
    def _nvml_handle(cuda_ordinal):
        """NVML handle for a CUDA device ordinal.

        NVML ignores CUDA_VISIBLE_DEVICES, so map the ordinal through it
        (numeric entries, CUDA_DEVICE_ORDER=PCI_BUS_ID assumed).
        """
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible:
            entries = [v.strip() for v in visible.split(',') if v.strip()]
            if all(v.isdigit() for v in entries) and cuda_ordinal < len(entries):
                return pynvml.nvmlDeviceGetHandleByIndex(int(entries[cuda_ordinal]))
        return pynvml.nvmlDeviceGetHandleByIndex(cuda_ordinal)

    @staticmethod
    def _foreign_gpu_processes(device_handle):
        if device_handle is None:
            return []
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
        except pynvml.NVMLError:
            return []
        return [p.pid for p in procs if p.pid != os.getpid()]

    def _wait_for_exclusive_gpu(self, device_handle):
        waited = 0
        while self._foreign_gpu_processes(device_handle):
            if waited % 60 == 0:
                print(f'GPU busy with processes '
                      f'{self._foreign_gpu_processes(device_handle)}; waiting')
            if waited >= self.ExclusiveWaitSeconds:
                print('Gave up waiting for an exclusive GPU')
                return False
            time.sleep(5)
            waited += 5
        return True

    def execute(self, NumThreads=None, NumTeams=None, maxIters = 9,
                keepMaxTeam=True, dropSingleTeams=False, device_handle=None,
                metrics=True, energy=True):
        #TODO: replace reference to nvprof in command()
        cmd = self.Replayer.command(NumThreads, NumTeams).format(kernel_name=f'{self.HashName}')
        metrics_env = self.Replayer.env(profile_metrics=True,
                                        profile_activities=True)
        timing_env = self.Replayer.env(profile_metrics=False,
                                       profile_activities=False,
                                       profile_timing=True, energy=energy)

        #cmd = self.Profiler.command().format(executable=cmd, output=f'{self.HashName}.csv')
        #env += ' ' + self.Profiler.env()
        results = []
        durations = []
        energies = []
        global_loads = []
        global_stores = []
        activities_file = Path('kernel_activities.csv')
        #print("MaxIters ", maxIters)
        contended_retries = 0
        clocks = []
        self.LastMetrics = {}
        while len(durations) < maxIters:
            activities_file.unlink(missing_ok=True)
            # Another process on this GPU slows the kernel and pollutes the
            # device-wide NVML energy counter, so measure only when the GPU
            # is exclusively ours and discard samples that overlapped one.
            self._wait_for_exclusive_gpu(device_handle)
            start = time.perf_counter()
            ret, stdout, stderr = execute_command(timing_env + ' timeout 180s ' + cmd,
                    capture_output=True, cwd=os.getcwd(),
                    shell=True, ContinueOnFailure=True)
            end = time.perf_counter()
            if self._foreign_gpu_processes(device_handle):
                contended_retries += 1
                print('GPU shared during timing replay; discarding sample')
                if contended_retries <= self.ContendedRetries:
                    continue
                print('GPU stayed shared; keeping contended sample')
            if ret != 0:
                activities_file.unlink(missing_ok=True)
                return [], [], [], [], []
            timing_match = re.search(
                r'\|\|ReplayTiming\|\| duration_seconds: ([-+0-9.eE]+)',
                stdout)
            if timing_match is None:
                print('Replay did not report a device kernel duration')
                return [], [], [], [], []
            durations.append(float(timing_match.group(1)))
            clock_match = re.search(r'\|\|ReplayClock\|\| sm_clock_mhz: (\d+)', stdout)
            if clock_match:
                clocks.append(float(clock_match.group(1)))
            # Kernel-level energy from the plugin's back-to-back launch window.
            # Process-level NVML energy (replay start-up, data loading) is
            # dominated by overhead unrelated to the launch configuration.
            energy_match = re.search(
                r'\|\|ReplayEnergy\|\| joules_per_launch: ([-+0-9.eE]+)', stdout)
            if energy_match is not None:
                energies.append(float(energy_match.group(1)))
            else:
                if energy and Kernel.EnergyMinMs > 0:
                    print('Replay did not report kernel energy; recording NaN')
                energies.append(float('nan'))

        if not metrics:
            return [[]], durations, energies, [None], [None]

        # Metrics are deterministic enough for optimization and expensive to
        # collect, so profile them once after all timing/energy samples.
        activities_file.unlink(missing_ok=True)
        ret, stdout, stderr = execute_command(metrics_env + ' timeout 180s ' + cmd,
                capture_output=True, cwd=os.getcwd(),
                shell=True, ContinueOnFailure=True)
        if ret != 0:
            activities_file.unlink(missing_ok=True)
            return [], [], [], [], []

        try:
            kernelDescr = self.Profiler.parse(str(activities_file),
                                              keepMaxTeam,
                                              dropSingleTeams)[self.Name]
        finally:
            activities_file.unlink(missing_ok=True)
        # All metrics printed by the plugin, plus the SM clock seen during
        # the timing replays. Stored per config via DB.SetMetrics.
        self.LastMetrics = {m.group(1): float(m.group(2)) for m in re.finditer(
            r'\|\|NVMetric\|\| (\S+): (\S+)', stdout)}
        if clocks:
            self.LastMetrics['sm_clock_mhz'] = float(np.median(clocks))
        # Utilisation metrics are only comparable when profiled at a steady
        # clock; tag them so consumers can ignore metrics from cold replays.
        metrics_clock = re.search(r'\|\|MetricsClock\|\| sm_clock_mhz: (\d+)', stdout)
        if metrics_clock:
            self.LastMetrics['metrics_warmup_ms'] = float(Kernel.TimingWarmupMs)
            self.LastMetrics['metrics_sm_clock_mhz'] = float(metrics_clock.group(1))
        global_loads.append(kernelDescr.pop('GLoads', None))
        global_stores.append(kernelDescr.pop('GStores', None))
        results.append(list(kernelDescr.values()))

        # The legacy CUDA plugin reports nvmetrics on stdout instead of the
        # CUPTI activity CSV used by the nextgen plugin. Use stdout as a
        # fallback so both plugins populate the same database fields.
        if global_loads[-1] is None or global_stores[-1] is None:
            load_match = re.search(r'\|\|NVMetrics\|\| global_load_requests: ([-+0-9.eE]+)', stdout)
            store_match = re.search(r'\|\|NVMetrics\|\| global_store_requests: ([-+0-9.eE]+)', stdout)
            global_loads[-1] = float(load_match.group(1)) if load_match else 0.0
            global_stores[-1] = float(store_match.group(1)) if store_match else 0.0
        

        return results, durations, energies, global_loads, global_stores

    def setOrigDescr(self, Regs, SMem, NumTeams, NumThreads, ExecTime,
                     **metrics):
        self.NumRegs = Regs
        self.SMem = SMem
        self.NumTeams = NumTeams
        self.NumThreads = NumThreads
        self.ExecTime = ExecTime
        self.Metrics = {
            name: value for name, value in metrics.items()
            if value is not None
        }
        self.profiled = True

    def IsProfiled(self):
        return self.profiled

    def __getstate__(self):
        state = self.__dict__.copy()
        # Assuming Profiler can be converted to and from a dict for serialization purposes.
        # Modify or remove this line if Profiler doesn't need special handling.
        #TODO: replace with CUPTI
        state['Profiler'] = self.Profiler.to_dict() if hasattr(self.Profiler, 'to_dict') else None
        # Assume Replayer can be fully reconstructed from other attributes, so we don't serialize it.
        del state['Replayer']
        return state

    def __setstate__(self, state):
        # Restore Profiler from dict if necessary. This assumes `Device.getDevice` or similar functionality.
        if 'Profiler' in state and state['Profiler'] is not None:
            state['Profiler'] = Device.getDevice(state['Profiler']['Name'])
        # Recreate Replayer based on the restored state.
        self.__dict__.update(state)
        self.Replayer = self.OMPReplayer(4)  # Recreate Replayer assuming it can be fully reconstructed.

    def to_dict(self):
        ret = dict()
        ret['Name'] = self.Name
        ret['HashName'] = self.HashName
        ret['NumRegs'] = self.NumRegs
        ret['SMem'] = self.SMem
        ret['NumTeams'] = self.NumTeams
        ret['NumThreads'] = self.NumThreads
        ret['ExecTime'] = self.ExecTime
        ret['Metrics'] = self.Metrics
        ret['DB'] = self.dbDir
        ret['Profiler'] = self.Profiler.to_dict()
        ret['recorded'] = self.recorded
        return ret

    @classmethod
    def from_dict(cls, d):
        prof = Device.getDevice(d['Profiler']['Name'])
        kernel = cls(prof, d['Name'], d['HashName'], d['DB'], d['recorded'])
        kernel.setOrigDescr(d['NumRegs'], d['SMem'], d['NumTeams'],
                            d['NumThreads'], d['ExecTime'],
                            **d.get('Metrics', {}))
        return kernel

class BenchmarkExecution:
    def __init__(self, name, buildTime, oTime, iTime, buildEnergy, oEnergy, binary, kernels):
        self.name = name
        self.bTime = buildTime
        self.oTime = oTime
        self.iTime = iTime
        self.buildEnergy = buildEnergy
        self.oEnergy = oEnergy
        self.binary = binary
        self.kernels = kernels

    def to_dict(self):
        ret = dict()
        ret['name'] = self.name
        ret['BuildTime'] = self.bTime
        ret['OuterTime'] = self.oTime
        ret['InnerTime'] = self.iTime
        ret['BuildEnergy'] = self.buildEnergy
        ret['OuterEnergy'] = self.oEnergy
        ret['ExecutableDir'] = self.binary
        ret['Kernels'] = dict()
        for k, v in self.kernels.items():
            ret['Kernels'][k] = v.to_dict()
        return ret

    # Implementing __getstate__ for serialization
    def __getstate__(self):
        state = self.__dict__.copy()
        # Convert kernel objects to a serializable form (dictionaries)
        state['kernels'] = {k: v.to_dict() for k, v in self.kernels.items()}
        return state

    # Implementing __setstate__ for deserialization
    def __setstate__(self, state):
        # Assuming there's a way to reconstruct Kernel objects from dictionaries
        # This requires a from_dict method or similar in your Kernel class
        state['kernels'] = {k: Kernel.from_dict(v) for k, v in state['kernels'].items()}
        self.__dict__.update(state)

    def getKernels(self):
        return self.kernels

    def printKernels(self):
        print('Kernel Name:RR Kernel Handler')
        for k, v in self.kernels.items():
            print('{0}:{1}'.format(v.Name, k))

    def getKernelName(self, k):
        return self.kernels[k].Name

    def getExecutionTimes(self):
        times = dict()
        times['EndToEnd'] = self.iTime
        for k, v in self.kernels.items():
            times[v.Name.split('_')[-1]] = float(v.ExecTime) * 1E-9
        return times


    @classmethod
    def from_dict(cls, name, **kwargs):
        for t in ['BuildTime', 'OuterTime', 'InnerTime', 'ExecutableDir', 'Kernels']:
            if t not in kwargs:
                raise RuntimeError(f'Cannot Find {t} in Execution dictionary')

        kernels = dict()
        for k, v in kwargs['Kernels'].items():
            kernels[k] = Kernel.from_dict(v)

        return cls(name, kwargs['BuildTime'],
            kwargs['OuterTime'],
            kwargs['InnerTime'],
            kwargs['BuildEnergy'],
            kwargs['OuterEnergy'],
            kwargs['ExecutableDir'], kernels)

class BaseBenchmark(ABC):
    class OMPRecorder:
        def __init__(self, MaxMem):
            self.envVars = {
                'OMP_NUM_THREADS' : 2,
                'LIBOMPTARGET_RR_DEVMEM_SIZE':  MaxMem,
                'LIBOMPTARGET_RR_SAVE_OUTPUT':  1,
                'OMP_TARGET_OFFLOAD':           'mandatory',
                'LIBOMPTARGET_NEXTGEN_PLUGINS': 1,
                'LIBOMPTARGET_RECORD':          1
            }
            self._env=' '.join([f'{k}={v}' for k,v in self.envVars.items()])
            self.cmd = '{executable}'

        def command(self):
            return self.cmd

        def env(self, profile_metrics=True, profile_activities=True,
                profile_timing=False):
            env_vars = dict(self.envVars)
            env_vars['LIBOMPTARGET_RR_PROFILE_METRICS'] = int(profile_metrics)
            env_vars['LIBOMPTARGET_RR_PROFILE_ACTIVITIES'] = int(profile_activities)
            env_vars['LIBOMPTARGET_RR_PROFILE_TIMING'] = int(profile_timing)
            return ' '.join([f'{k}={v}' for k,v in env_vars.items()])

        def threads(self):
            return '{0}={1}'.format('OMP_NUM_THREAD', self.envVars['OMP_NUM_THREADS'])

    def __init__(self, Device, executable, root, **kwargs):
        self.executable = executable
        self.root = root
        self.versions = dict()
        self.Device = Device
        self.recorder = self.OMPRecorder(4)

        self.resultsDir = kwargs.get('resultsDir')

        if 'versions' in kwargs:
            for k, v in kwargs['versions'].items():
                self.versions[k] = BenchmarkExecution.from_dict(**v)

    def to_dict(self):
        ret = dict()
        ret['executable'] = self.executable
        ret['resultsdir'] = self.resultsDir
        ret['root'] = self.root
        ret['Profiler'] = self.Device.to_dict()
        ret['versions'] = dict()
        for k, v in self.versions.items():
            ret['versions'][k] = v.to_dict()
        return ret

    # For the BaseBenchmark class, if we need to directly support pickling:
    def __getstate__(self):
        state = self.__dict__.copy()
        # Convert the Device to a dict if necessary. Adjust this based on actual implementation.
        state['Device'] = self.Device.to_dict() if hasattr(self.Device, 'to_dict') else None
        # The recorder and versions are assumed to be serializable or have been converted to a serializable form.
        return state

    def __setstate__(self, state):
        # Restore the Device from its dict representation if necessary.
        if 'Device' in state and state['Device'] is not None:
            state['Device'] = Device.getDevice(state['Device']['Name'])  # Assuming a method to recreate Device
        self.__dict__.update(state)
        # Assuming the recorder and versions can be directly restored from their serialized state.

    def printKernels(self):
        if 'record' not in self.versions:
            print('Benchmark is not recorded')
            return

        self.versions['record'].printKernels()

    def getKernelName(self, k):
        if 'record' not in self.versions:
            print('Benchmark is not recorded')
            return
        self.versions['record'].getKernelName()

    def getKernels(self, version):
        if version not in self.versions:
            print('Version does not exist')
            return {}

        return self.versions[version].getKernels()


    def getExecutionTimes(self, drop=None):
        exec_times = {}
        for k, v in self.versions.items():
            if k in drop:
                continue
            exec_times[k] = v.getExecutionTimes()
        return exec_times

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def command(self):
        pass

    @abstractmethod
    def env(self):
        pass

    @abstractmethod
    def getApplicationTime(self, stdout):
      pass

    def compile(self, clean=True):
        if clean:
            self.clean()
        self.build()

    def execute(self, Profile=True, Record=True):
        env='OMP_TARGET_OFFLOAD=mandatory LIBOMPTARGET_NEXTGEN_PLUGINS=1' + self.env()
        command = self.command()
        if Record:
            #TODO: replace command()
            command = self.recorder.command().format(executable=command)
            env += ' '  + self.recorder.env()
        else:
            env += ' ' + self.recorder.threads()

        if Profile:
            csvFile=self.executable.replace('/', '_')
            print("csvFile Name: ", csvFile)
            #command = self.Device.command().format(executable=command, output=f'{csvFile}.csv')
            command = self.Device.command().format(executable=command)
            env += ' ' + self.Device.env()

        #print(env + ' ' + command)
        ret, stdout, stderr = execute_command(env + ' ' + command, capture_output=True, cwd=os.getcwd(), shell=True, check=False)
        print("NEW ret: " , ret, "\n")
        print("NEW stdout: ", stdout, "\n")
        print("NEW stderr: ", stderr, "\nEND\n")
        application_time = self.getApplicationTime(stdout)

        if Profile:
            csvFile=self.executable.replace('/', '_')
            print("Reading csvfile: ", csvFile)
            kernelDescr = self.Device.parse('kernel_activities.csv',
                                            keepMaxTeam=True, dropSingleTeams=True)

        if not Record:
            kernels = dict()
            for k, v in kernelDescr.items():
                kernel = Kernel(self.Device, k, k, self.resultsDir, recorded=False)
                print('Not Recordkernel', kernel)
                if isinstance(self.Device, Device.NVIDIA):
                    kernel.setOrigDescr( v['Registers Per Thread'],
                                         v['Static SMem'],
                                         v['Grid X'],
                                         v['Block X'],
                                         v['Duration'],
                                         GLoads=v.get('GLoads'),
                                         GStores=v.get('GStores'),
                                         Occupancy=v.get('Occupancy'),
                                         BranchEfficiency=v.get(
                                             'Branch Efficiency'))
                kernels[f'{k}.json'] = kernel
            return application_time, kernels

        # We may record more kernels than the ones we will optimize, so we find
        # out all recorded kernels and from the image of each kernel delete
        # all the others to make compilation times faster.
        kname_to_hash = {}
        recordedKernels = []
        for kernelInfo in Path('./').glob('__omp*.json'):
            with open(kernelInfo, 'r') as f:
                info = json.load(f)
                kname = info['Name']
                recordedKernels.append(kname)
                kname_to_hash[kname] = str(kernelInfo).replace('.json', '')

        for k in kernelDescr.keys():
            toDeleteKernelList = [r for r in recordedKernels if r != k]
            if toDeleteKernelList:
                Kernel.extractKernels(k, kname_to_hash, toDeleteKernelList, self.Device)

        kernels = dict()
        for k, v in kernelDescr.items():
            if Path(f'{kname_to_hash[k]}.json').exists():
                kernel = Kernel(self.Device, k, kname_to_hash[k], self.resultsDir)
                print('Recordkernel', kernel)
                #FIXME: We should also support AMD in this point
                if isinstance(self.Device, Device.NVIDIA):
                    kernel.setOrigDescr( v['Registers Per Thread'],
                                         v['Static SMem'],
                                         v['Grid X'],
                                         v['Block X'],
                                         v['Duration'],
                                         GLoads=v.get('GLoads'),
                                         GStores=v.get('GStores'),
                                         Occupancy=v.get('Occupancy'),
                                         BranchEfficiency=v.get(
                                             'Branch Efficiency'))
                kernels[f'{kname_to_hash[k]}.json'] = kernel
            else:
                raise RuntimeError(f'Kernel {k} was profiled but not recorded')

        return application_time, kernels

    def name(self):
        return type(self).__name__

    def register_variant(self, vName, bTime, rTime, aTime, bEnergy, rEnergy, kernels): 
        self.versions[vName] = BenchmarkExecution(vName, bTime, rTime, aTime, bEnergy, rEnergy, self.executableDir, kernels)

    def ckpt(self, fn):
        ckpt_fn = f'{fn}/{self.__class__.__name__}.ckpt.json'
        with open(ckpt_fn, 'w') as fd:
            json.dump(self.to_dict(), fd, indent = 6)

    #This function is called before "execute" and performs any necessary data preparation steps
    def preprocess(self, currentDirectory):
        pass

    @classmethod
    def load_ckpt(cls, path, device):
        ckpt_fn = f'{path}/{cls.__name__}.ckpt.json'
        if Path(ckpt_fn).exists():
            with open(ckpt_fn, 'r') as fd:
                data = json.load(fd)
            prof = Device.getDevice(device)
            bench = cls(prof, **data)
            return bench
        return None

class MakeBenchmark(BaseBenchmark):
    BUILDDIR='build'
    def __init__(self, Device, executable, root, **kwargs):
        super().__init__(Device, executable, root, **kwargs)
        self.executableDir = None
        self.buildDir = None

    def build(self, flags=None, optimal=None):
        if optimal is None:
          self.executableDir = self.buildDir = createDir(self.BUILDDIR)
        else:
          self.executableDir = self.buildDir = createDir(self.BUILDDIR + '_best_' + optimal)
        makeCmd = 'BUILDDIR={0} make '.format(self.buildDir)
        if flags is not None:
            makeCmd+=' '.join(['{0}="{1}"'.format(k,v) for k,v in flags.items() ])
        return execute_command(makeCmd, capture_output=True, cwd=self.root, shell=True)

    def clean(self, flags=None):
        self.executableDir = self.buildDir = createDir(self.BUILDDIR)
        makeCmd = 'BUILDDIR={0} make clean '.format(self.buildDir)
        if flags is not None:
            makeCmd+=' '.join(['{0}="{1}"'.format(k,v) for k,v in flags.items() ])
        return execute_command(makeCmd, capture_output=True, cwd=self.root, shell=True)

    @abstractmethod
    def env(self):
        return ''

    @abstractmethod
    def command(self):
        return ''

#There should be a better way to do this.
validBenchmarks=[BaseBenchmark.__name__, MakeBenchmark.__name__]
