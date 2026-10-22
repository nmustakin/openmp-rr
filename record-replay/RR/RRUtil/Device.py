import re
import pandas as pd
import subprocess
from pathlib import Path
import shutil
import os
import math

from RRUtil.Utilities import execute_command
from RRUtil.Utilities import createDir

def Stage(StageRootDir, StageTempDir, StagePattern):
    def StageDecorator(decoratedFunc):
        def wrapper(*args,**kwargs):
            if StageRootDir is None:
                result = decoratedFunc(*args, **kwargs)
                return result

            currentDir = Path().absolute()
            tmpDir = createDir(StageRootDir / Path(StageTempDir) )
            for fn in currentDir.glob(StagePattern + "*"):
                shutil.copyfile(fn, Path(tmpDir) / Path(fn.name))
                print('Copy: Src: {0} Dst: {1}'.format(fn, Path(tmpDir) / Path(fn.name)))
            print('Push Directory')
            os.chdir(tmpDir)
            result = decoratedFunc(*args, **kwargs)
            print('Pop directory')
            os.chdir(currentDir)
            shutil.rmtree(tmpDir)
            return result
        return wrapper
    return StageDecorator

class DeviceSingleton(type):
    def __init__(self, *args, **kwargs):
        super(DeviceSingleton, self).__init__(*args, **kwargs)
        self.__instances__ = dict()

    def __call__(self, *args, **kwargs):
        dev = 0
        if 'device' in kwargs:
            dev = kwargs['device']

        if dev not in self.__instances__:
            self.__instances__[dev] = super(DeviceSingleton, self).__call__(*args, **kwargs)
        return self.__instances__[dev]

class NVIDIA(metaclass=DeviceSingleton):
    __loaded__ = False
    def __init__(self, drv, StageDir=None, device=0):
        self.drv = drv
        self.StageDir = StageDir
        self.NumDevices = self.drv.Device.count()
        self.Arch = 'sm_75'
        self.Target = 'nvptx64'
        if device < self.NumDevices:
            gpuDevice = self.drv.Device(device)
            self.Capability = float( '%d.%d' % gpuDevice.compute_capability() )
            self.TotalMemory = gpuDevice.total_memory()
            self.WarpSize = float(gpuDevice.WARP_SIZE)
            self.ThreadsSMP = float(gpuDevice.MAX_THREADS_PER_MULTIPROCESSOR)
            self.WarpsSMP = float(self.ThreadsSMP / self.WarpSize)
            self.SharedMemoySMP = float(gpuDevice.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
            self.MaxSharedMemBlock = float(gpuDevice.MAX_SHARED_MEMORY_PER_BLOCK)
            self.MaxRegBlock = float(gpuDevice.MAX_REGISTERS_PER_BLOCK)
            self.RegFileSizeSMP = float(gpuDevice.MAX_REGISTERS_PER_MULTIPROCESSOR)
            self.MaxRegBlock = float(gpuDevice.MAX_REGISTERS_PER_BLOCK)
            self.MaxThreadBlockSize = float(gpuDevice.MAX_THREADS_PER_BLOCK)
            #self._profile = 'nvprof --print-gpu-trace --normalized-time-unit ns --csv --log-file {output} {executable}'
            self._profile = '{executable}'
            self._env = ''
            #self.Name = gpuDevice.name()
            self.Name = 'nvidia'
            self.optConfigs = ['MaxThreads', 'MinTeams', 'NumThreads', 'NumTeams']
            self.optMap = {
                'MaxThreads' : 'maxntidx',
                'MinTeams' : 'minctasm',
                'NumThreads' : 'threads',
                'NumTeams' : 'teams'
            }
#            self.optConfigs = [
#                                   ['NumWarps', 'NumTeams', 'NumRegsPerThread', 'RegEffort'],
#                                   ['NumWarps', 'NumTeams'],
#                               ]
            self.compilationConfigs = [ 'MaxThreads', 'MinTeams']

    def to_dict(self):
        return { 'Name' : self.Name,
                 'Capability' : self.Capability }

    def print(self):
        print("WarpSize: ", self.WarpSize)
        print("Warps per SMP: " , self.WarpsSMP)
        print("Threads Per SMP:" , self.ThreadsSMP)
        print("Shared Mem Per SMP", self.SharedMemoySMP)
        print("Max Shared Mem", self.MaxSharedMemBlock)
        print("Reg File SMP:", self.RegFileSizeSMP)
        print("Max Registers per Block:", self.MaxRegBlock)
        print("Max Thread Block Size", self.MaxThreadBlockSize)

    def getMaxMem(self):
        return 15

    def convert(self, k, v):
      if v is None:
          return { k : -1 }
      return { k : v }

    def transformConfig(self, options):
      config = {}
      for k, v in self.optMap.items():
        config.update( self.convert( v, options.get(k, None) ) )
      return config


    def getLaunchBounds(self, options, TripCount):
        MaxThreads = self.MaxThreadBlockSize
        WarpSize = self.WarpSize
        MaxNumberOfWarps = MaxThreads / WarpSize
        MaxThreadsPerMP =  self.ThreadsSMP

        MaxThreadFrac = options.get('MaxThreads', 1.0)
        MaxThreadsPerBlock = int(math.ceil(MaxThreadFrac * MaxNumberOfWarps) * WarpSize)

        MaxBlocks = int(math.floor(MaxThreadsPerMP / MaxThreadsPerBlock))

        if MaxBlocks > 32:
            MaxBlocks = 32

        MinFracTeam = options.get('MinTeams', None)
        if MinFracTeam is None:
            MinBlocksPerMultiprocessor = 1
        else:
            MinBlocksPerMultiprocessor = int(math.ceil(MinFracTeam* (MaxBlocks)))
            # Corner case where MinFracTeam > 1.0. We should not take ceil, but floor
            if MinBlocksPerMultiprocessor * MaxThreadsPerBlock > MaxThreadsPerMP:
                MinBlocksPerMultiprocessor = int(math.floor(MaxThreadsPerMP/MaxThreadsPerBlock))

        # If we are not setting the number of threads explicitly, then use the one we compile with
        ThreadFrac = options.get('NumThreads', 1.0)
        NumThreads = int((math.ceil((ThreadFrac * MaxThreadsPerBlock)/WarpSize) * WarpSize))
        if NumThreads > MaxThreadsPerBlock:
            NumThreads = MaxThreadsPerBlock

        fracTeams = options.get('NumTeams', None)
        NumTeams = None
        if fracTeams is not None:
            NumTeams = int(math.ceil( (TripCount / NumThreads) * fracTeams ))

        return {
                 'MaxThreads'   : MaxThreadsPerBlock if ('MaxThreads' in options) else None,
                 'MinTeams'     : MinBlocksPerMultiprocessor if ('MinTeams' in options) else None,
                 'NumThreads'   : NumThreads if ('NumThreads' in options ) else None,
                 'NumTeams'     : NumTeams if ('NumTeams' in options ) else None
               }

    def parse(self, fn, keepMaxTeam=True, dropSingleTeams=False):
        
        header_rows = 0
        with open(fn, 'r') as file:
           for line in file: 
               if line.startswith('=='):
                   header_rows += 1
               else:
                   break

        df = pd.read_csv(fn, sep=',', skiprows=header_rows)
        #print("df columns: \n", df.columns)
        #metric = df.Duration[0]
        roi_columns = ['Name', 'Device', 'Grid X', 'Block X', 'Registers Per Thread', 'Static SMem', 'Dynamic SMem', 'Duration']
        for metric_column in ['GLoads', 'GStores', 'Occupancy', 'Branch Efficiency']:
            if metric_column in df.columns:
                roi_columns.append(metric_column)
        Types = { 'Grid X' : 'int32',
                  'Block X' : 'int32',
                  'Registers Per Thread' :'int32',
                  'Static SMem' : 'float32',
                  'Dynamic SMem' : 'float32',
                  'Duration' : 'float32',
                  'GLoads' : 'float32',
                  'GStores' : 'float32',
                  'Occupancy' : 'float32',
                  'Branch Efficiency' : 'float32' }
        Types = {k: v for k, v in Types.items() if k in df.columns}
        df = df[df.Name.notna()]
        df = df[df.Name.str.contains('__omp_offloading')]
        df = df[roi_columns]
        df = df.astype(Types)
        df = df.groupby(['Name', 'Device', 'Grid X', 'Block X', 'Registers Per Thread', 'Static SMem', 'Dynamic SMem']).mean().reset_index()
        df = df.set_index("Name")
        pd.set_option('display.max_rows', None)
        pd.options.display.max_colwidth = 1000
        # Find indexes with maximum Grid value
        if keepMaxTeam:
            idx = df.groupby(['Name'])['Grid X'].transform(max) == df['Grid X']
            df = df[idx]
        # Drop cases of single team kernels. Those are likely to be constructors
        # or kernels that we cannot optimize
        if dropSingleTeams:
            idx = df.groupby(['Name'])['Grid X'].transform(max) != 1
            df = df[idx]
        return df.to_dict(orient='index')

    def renameSymbols(self, inputName, outName):
        opt_cmd = ['opt']
        tuning_pass = os.getenv('RR_TUNING_PASS')
        assert tuning_pass, 'Expected env var for the tuning pass .so'
        opt_cmd.append(f'--load={tuning_pass}')
        opt_cmd.append('--enable-new-pm=0')
        opt_cmd.append('--global-symbol-rename-pass')
        opt_cmd.append(f'{inputName}')
        opt_cmd.append(f'-o {outName}')
        opt_cmd = ' '.join(opt_cmd)
        print(opt_cmd)
        execute_command(opt_cmd, capture_output=True, shell=True)

    def deviceCompile(self, original, Name, HashName, **kwargs):
        numRegs = kwargs.get('NumRegsPerThread', None)
        regAggr = kwargs.get('RegEffort', None)
        MaxThreads = kwargs.get('MaxThreads', None)
        MinTeams = kwargs.get('MinTeams', None)

        opt_cmd= ['opt', f'{original}', f'-o {HashName}.opt.out']

        if (MaxThreads is not None) or (MinTeams is not None) :
            tuning_pass = os.getenv('RR_TUNING_PASS')
            assert tuning_pass, 'Expected env var for the tuning pass .so'
            opt_cmd.append(f'--load={tuning_pass}')
            opt_cmd.append('--enable-new-pm=0')
            opt_cmd.append('--legacy-nvptx-tune-pass')
            opt_cmd.append(f'--nvptx-function-name={Name}')

        if MaxThreads is not None:
            opt_cmd.append(f'--max-threads={MaxThreads}')

        if MinTeams is not None:
            opt_cmd.append(f'--min-blocks={MinTeams}')
        opt_cmd = ' '.join(opt_cmd)
        #print(opt_cmd)
        execute_command(opt_cmd, capture_output=True, shell=True)

        llc_cmd= ' '.join(['llc','-mcpu=sm_75','-mattr=+ptx76', '-O3', f'-o {HashName}.llc.out', f'{HashName}.opt.out'])
        #print(llc_cmd)
        llc_ps = execute_command(llc_cmd, capture_output=True, shell=True)
        ptxas_cmd = ['ptxas', '-v', '--gpu-name', 'sm_75', '-O3', '--warn-on-spills',
             f'{HashName}.llc.out', '-o', f'{HashName}.image']
        if not isinstance(numRegs, type(None)):
            ptxas_cmd.append(f'-maxrregcount {numRegs}')
        if not isinstance(regAggr, type(None)):
            ptxas_cmd.append(f'-regUsageLevel {regAggr}')
        ptxas_cmd = ' '.join(ptxas_cmd)
        #print(ptxas_cmd)
        code, stdout, stderr = execute_command(ptxas_cmd, capture_output=True, shell=True)
        #print(stdout)
        #print(stderr)
        return self.parsePTXAS(stdout, stderr)

    def parsePTXAS(self, stdout, stderr):
        ptxasOut = stderr.split("\n")
        startFlag = False
        kernelName = ''
        kernels = []
        spills = []
        registerRE = "Used (\d+) registers"
        shmemRE = "(\d+) bytes smem"
        cmemRE = "(\d+) bytes cmem\[(\d+)\]"
        spillRe = "Registers are spilled to local memory in function '(\w+)', (\d+) bytes spill stores, (\d+) bytes spill loads"

        for l in ptxasOut:
            if "ptxas warning : Registers are spilled to local memory in function " in l:
                tmp = re.findall(spillRe, l)
                for c in tmp:
                    spills.append([c[0], int(c[1]), int(c[2])])
            elif "ptxas info    : Compiling entry function" in l:
                if not startFlag:
                    startFlag = True
                    kernelName = l.split(' ')[9].replace("'", "")
                    print('Found:', kernelName)
            if ("ptxas info    : Used" in l) and startFlag:
                regs = re.findall(registerRE, l)
                shmem = re.findall(shmemRE, l)
                cmem = re.findall(cmemRE,l)
                totalRegs = 0
                totalShMem = 0
                totalCMem = 0
                for r in regs:
                    totalRegs += int(r)
                for s in shmem:
                    totalShMem += int(s)
                for c in cmem:
                    #Currently I am ignoring banks
                    totalCMem += int(c[0])
                kernels.append([kernelName, totalRegs, totalShMem, totalCMem])
                startFlag = False
            elif ("ptxas info    : Used" in l):
                print("Should never happen")
        ret = dict()
        for k in kernels:
            ret[k[0]] = [k[1], k[2], k[3], 0, 0]

        for s in spills:
            if s[0] in ret:
                ret[s[0]][3] = s[1]
                ret[s[0]][4] = s[2]
        return ret

    def recordPrefix(self):
        return ''

    def metricsCommand(self, cmd):
        return f'timeout 180s {cmd}'

    def parseMetrics(self, stdout, activitiesFile):
        """All metrics printed by the plugin."""
        return {m.group(1): float(m.group(2)) for m in re.finditer(
            r'\|\|NVMetric\|\| (\S+): (\S+)', stdout)}

    def getRecordMem(self):
        return 4

    def getReplayMem(self):
        # The replay region is mapped at the recorded address (mMap).
        return 4

    def dropSingleTeams(self):
        """Whether kernel discovery drops single-team launches (constructors)."""
        return True

    def blockFits(self, regs, threads):
        """Whether a block of `threads` with `regs` registers per thread fits."""
        return regs * threads <= self.MaxRegBlock

    def command(self):
        return self._profile

    def env(self):
        return self._env

    def getNumDevices(self):
        return self.NumDevices

    def getOptConfigs(self):
        return self.optConfigs

    def getCompilationConfigs(self):
        return self.compilationConfigs

    def isValidConfig(self, NumThreads, NumTeams):
        if ( NumThreads == 32 ):
            return NumTeams <= 32
        return NumThreads * NumTeams <= self.ThreadsSMP

    def getDefaultBounds(self, config):
        assert config in self.compilationConfigs, f'Config {config} is not valid'
        if config == 'NumRegsPerThread':
            return (32, 255.99, 1)
        elif config == 'RegEffort':
            return (1, 10.99, 1)
        elif config == 'MaxThreads':
            return (32, 1024+1, int(self.WarpSize))
        elif config == 'MinTeams':
            return (1, int(self.ThreadsSMP/ self.WarpSize)+1, 1)
        else:
            assert False, f'getDefaultBounds not impl. for {config}'

    def getTarget(self):
        return self.Target

    def getArch(self):
        return self.Arch

    def getStageDir(self):
        return self.StageDir

    @classmethod
    def Load(cls, StageDir, device):
        try:
            import pycuda
            import pycuda.driver as drv
            if not cls.__loaded__:
                drv.init()
                cls.__loaded__ = True
            return cls(drv, StageDir=StageDir, device=device)
        except ImportError:
            raise Exception('NVIDIA requires pycuda installed')

def _rocm_path():
    return os.environ.get('ROCM_PATH', '/opt/rocm')


def _query_amd_agents():
    """GPU agents reported by rocminfo, in ROCR order, as dicts with the
    fields the tuner needs. Empty if rocminfo is unavailable."""
    try:
        out = subprocess.run([f'{_rocm_path()}/bin/rocminfo'], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    agents = []
    for block in out.split('*******')[1:]:
        if 'Device Type:             GPU' not in block:
            continue
        def field(name, cast=int):
            m = re.search(rf'^\s*{name}:\s*(\S+)', block, re.M)
            return cast(m.group(1).split('(')[0]) if m else None
        agents.append({'Arch': field('Name', str), 'CUs': field('Compute Unit'),
                       'WaveSize': field('Wavefront Size'),
                       'MaxWavesPerCU': field('Max Waves Per CU'),
                       'MaxBlock': field('Workgroup Max Size')})
    return agents


class AMDGPU(metaclass=DeviceSingleton):
    __loaded__ = False
    # Per-pass rocprof counter sets (each line is one replay process). They
    # fit the gfx906 counter limits; SQ_ACCUM_PREV must follow SQ_LEVEL_WAVES.
    MetricPasses = [
        'SQ_LEVEL_WAVES SQ_ACCUM_PREV SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM_RD '
        'SQ_INSTS_VMEM_WR SQ_ACTIVE_INST_VALU SQ_INSTS_SALU GRBM_GUI_ACTIVE GRBM_COUNT',
        'FETCH_SIZE',
        'WRITE_SIZE',
        'TCC_HIT_sum TCC_MISS_sum MemUnitBusy',
    ]

    # MI50 has no multi-step clock ramp: a 0.42 ms kernel settles within ~20
    # launches, and warm-ups of 100 ms to 3 s give the same median.
    TimingWarmupMs = 100

    def __init__(self, StageDir=None, device=0):
        self.Target='amdgcn-amd-amdhsa'
        self.StageDir = StageDir
        agents = _query_amd_agents()
        self.NumDevices = len(agents) or 8
        # Defaults are the MI50 (gfx906) when rocminfo is unavailable.
        agent = agents[device] if device < len(agents) else {}
        self.Arch = agent.get('Arch') or 'gfx906'
        self.CUs = agent.get('CUs') or 60
        self.MaxWavesPerCU = agent.get('MaxWavesPerCU') or 40
        # HBM2 peak of the MI50/MI60, for DRAM throughput as a percentage.
        self.PeakDramBytesPerSec = float(os.environ.get('RR_AMD_PEAK_DRAM_GBS', 1024)) * 1e9

        if device < self.NumDevices:
            print(f'Setting up device {device} ({self.Arch}) from {self.NumDevices}')
            self.TotalMemory = 16 * 1024**3
            self.WarpSize = agent.get('WaveSize') or 64
            self.ThreadsSMP = self.MaxWavesPerCU * self.WarpSize
            self.WarpsSMP = self.MaxWavesPerCU
            self.SharedMemoySMP = 65536
            self.MaxSharedMemBlock = 65536
            self.MaxRegBlock = 0
            self.RegFileSizeSMP = 0
            self.MaxRegBlock = 0
            self.MaxThreadBlockSize = 1024
            # XXX: rocprof with --stats APPENDS to the <kernel name>.json name,
            # making it unusable for the replay. We use timestamps and calculate
            # the duration the same as --stats does.
            # Recording runs write the per-dispatch rocprof CSV to the file
            # name the CUDA plugin uses; parse() reads its columns.
            self._profile = (f'{_rocm_path()}/bin/rocprof --timestamp on '
                             '-o kernel_activities.csv {executable}')
            self._env = ''
            self.Name = 'amd'
            #self.optConfigs = [ ['NumWarps', 'NumTeams', 'FlatWorkGroupSize', 'WavesPerEU' ], [ 'NumWarps', 'NumTeams' ] ]
            #self.optConfigs = [ ['NumWarps', 'NumTeams', 'FlatWorkGroupSize', 'WavesPerEU' ]]
            #self.optConfigs = [ ['NumWarps', 'NumTeams', 'WavesPerEU' ]]
            #self.optConfigs = [ ['WavesPerEU'] ]
            self.optConfigs = ['MaxThreads', 'MinTeams', 'NumThreads', 'NumTeams']
            #self.compilationConfigs = [ 'FlatWorkGroupSize', 'WavesPerEU' ]
            self.compilationConfigs = [ 'MaxThreads', 'MinTeams']

            self.optMap = {
                'MaxThreads' : 'maxntidx',
                'MinTeams' : 'minctasm',
                'NumThreads' : 'threads',
                'NumTeams' : 'teams'
            }

    def getMaxMem(self):
        # Asking for more than the device has makes the replay region step
        # down through failed allocations.
        return int(self.TotalMemory // 1024**3) - 1

    def isValidConfig(self, NumThreads, NumTeams):
        return NumThreads * NumTeams <= self.ThreadsSMP

    def getWavesPerEU(self, MaxThreads, MinTeams):
        """amdgpu-waves-per-eu for at least MinTeams work-groups of MaxThreads
        per CU (the analogue of minBlocksPerMultiprocessor). A CU has 4 SIMDs
        (EUs); the value must be in [1, waves per SIMD]."""
        threads = MaxThreads if MaxThreads is not None else self.MaxThreadBlockSize
        waves_per_cu = MinTeams * math.ceil(threads / self.WarpSize)
        return int(min(max(1, math.ceil(waves_per_cu / 4)), self.MaxWavesPerCU // 4))

    def transformConfig(self, options):
      config = {}
      if 'MaxThreads' in options:
          config['maxntidx'] = options['MaxThreads']
      if 'MinTeams' in options:
          config['minctasm'] = self.getWavesPerEU(options['MaxThreads'], options['MinTeams'])

      if 'NumTeams' in options:
          config['teams'] = options['NumTeams']

      if 'NumThreads' in options:
          config['threads'] = options['NumThreads']
      return config

    def getLaunchBounds(self, options, TripCount):
        MaxThreads = self.MaxThreadBlockSize
        WarpSize = self.WarpSize
        MaxNumberOfWarps = MaxThreads / WarpSize
        MaxThreadsPerMP =  self.ThreadsSMP

        MaxThreadFrac = options.get('MaxThreads', 1.0)
        MaxThreadsPerBlock = int(math.ceil(MaxThreadFrac * MaxNumberOfWarps) * WarpSize)

        # Same MinTeams range as the search's point mapping (Search.LIMITS).
        MaxBlocks = min(32, int(math.floor(MaxThreadsPerMP / MaxThreadsPerBlock)))

        MinFracTeam = options.get('MinTeams', None)
        if MinFracTeam is None:
            MinBlocksPerMultiprocessor = 1
        else:
            MinBlocksPerMultiprocessor = int(math.ceil(MinFracTeam* (MaxBlocks)))
            # Corner case where MinFracTeam > 1.0. We should not take ceil, but floor
            if MinBlocksPerMultiprocessor * MaxThreadsPerBlock > MaxThreadsPerMP:
                MinBlocksPerMultiprocessor = int(math.floor(MaxThreadsPerMP/MaxThreadsPerBlock))

        # If we are not setting the number of threads explicitly, then use the one we compile with
        ThreadFrac = options.get('NumThreads', 1.0)
        NumThreads = int((math.ceil((ThreadFrac * MaxThreadsPerBlock)/WarpSize) * WarpSize))
        if NumThreads > MaxThreadsPerBlock:
            NumThreads = MaxThreadsPerBlock

        fracTeams = options.get('NumTeams', None)
        NumTeams = None
        if fracTeams is not None:
            NumTeams = int(math.ceil( (TripCount / NumThreads) * fracTeams ))

        return {
                 'MaxThreads'   : MaxThreadsPerBlock if ('MaxThreads' in options) else None,
                 'MinTeams'     : MinBlocksPerMultiprocessor if ('MinTeams' in options) else None,
                 'NumThreads'   : NumThreads if ('NumThreads' in options ) else None,
                 'NumTeams'     : NumTeams if ('NumTeams' in options ) else None
               }

    def to_dict(self):
        return { 'Name' : self.Name }

    def print(self):
        print("WarpSize: ", self.WarpSize)
        print("Warps per SMP: " , self.WarpsSMP)
        print("Threads Per SMP:" , self.ThreadsSMP)
        print("Shared Mem Per SMP", self.SharedMemoySMP)
        print("Max Shared Mem", self.MaxSharedMemBlock)
        print("Reg File SMP:", self.RegFileSizeSMP)
        print("Max Registers per Block:", self.MaxRegBlock)
        print("Max Thread Block Size", self.MaxThreadBlockSize)

    def parse(self, fn, keepMaxTeam=True, dropSingleTeams=False):
        df = pd.read_csv(fn, sep=',')
        df.rename(columns={'KernelName' : 'Name'}, inplace=True)
        df['Duration'] = df['EndNs'] - df['BeginNs']
        #metric = df.Duration[0]
        roi_columns = ['Name','gpu-id','grd','wgr','lds','scr', 'arch_vgpr','accum_vgpr','sgpr','wave_size','Duration']
        df.Duration = pd.to_numeric(df.Duration)
        df = df[df.Name.str.contains('__omp_offloading')]
        df = df[roi_columns]
        grp_columns = ['Name','gpu-id','grd','wgr','lds','scr', 'arch_vgpr','accum_vgpr','sgpr','wave_size']
        df = df.groupby(grp_columns).mean().reset_index()
        df['Name'] = df['Name'].str.replace('.kd', '', regex=False)
        # Same keys, in the same order, as NVIDIA.parse (the tuner stores the
        # values positionally); AMD-only fields follow. rocprof's grd counts
        # work-items, CUPTI's Grid X counts blocks.
        df = pd.DataFrame({
            'Name': df['Name'], 'Device': df['gpu-id'],
            'Grid X': (df['grd'] // df['wgr']).astype('int32'),
            'Block X': df['wgr'].astype('int32'),
            'Registers Per Thread': (df['arch_vgpr'] + df['accum_vgpr']).astype('int32'),
            'Static SMem': df['lds'].astype('float32'),
            'Dynamic SMem': 0.0,
            'Duration': df['Duration'].astype('float32'),
            'SGPRs': df['sgpr'].astype('int32'),
            'Scratch': df['scr'].astype('float32'),
        }).set_index('Name')
        pd.options.display.max_colwidth = 1000
        if keepMaxTeam:
            idx = df.groupby(['Name'])['Grid X'].transform(max) == df['Grid X']
            df = df[idx]

        if dropSingleTeams:
            idx = df.groupby(['Name'])['Grid X'].transform(max) != 1
            df = df[idx]

        return df.to_dict(orient='index')

    def _rocprofMetricsDir(self):
        """Directory holding rr_metrics.xml next to symlinks to the ROCm
        metrics files, because rocprofiler only resolves relative includes."""
        lib = Path(_rocm_path()) / 'lib' / 'rocprofiler'
        d = Path(os.environ.get('TMPDIR', '/tmp')) / f'rr-rocprof-{os.getuid()}'
        d.mkdir(parents=True, exist_ok=True)
        for name in ['metrics.xml', 'gfx_metrics.xml']:
            link = d / name
            if not link.exists():
                link.symlink_to(lib / name)
        shutil.copyfile(Path(__file__).parent / 'amd' / 'rr_metrics.xml',
                        d / 'rr_metrics.xml')
        (d / 'passes.txt').write_text(
            ''.join(f'pmc: {p}\n' for p in self.MetricPasses))
        return d

    def metricsCommand(self, cmd):
        """Metrics replay: rocprof runs one replay process per counter pass,
        so every pass starts from the recorded inputs."""
        d = self._rocprofMetricsDir()
        timeout = 180 * len(self.MetricPasses)
        return (f'timeout {timeout}s {_rocm_path()}/bin/rocprof -m {d}/rr_metrics.xml '
                f'-i {d}/passes.txt --timestamp on -o kernel_activities.csv {cmd}')

    def parseMetrics(self, stdout, activitiesFile):
        """Counters of the profiled dispatch and metrics derived from them."""
        df = pd.read_csv(activitiesFile)
        df = df[df.KernelName.str.contains('__omp_offloading', na=False)]
        if df.empty:
            return {}
        r = df.iloc[-1]
        counters = [c for p in self.MetricPasses for c in p.split()]
        m = {c: float(r[c]) for c in counters if c in r and pd.notna(r[c])}
        seconds = (r['EndNs'] - r['BeginNs']) * 1e-9
        m['gpu_time_seconds'] = seconds
        gui = m.get('GRBM_GUI_ACTIVE')
        if gui:
            waves = 4 * m.get('SQ_ACCUM_PREV', 0) / gui / self.CUs
            m['waves_per_cu'] = waves
            m['achieved_occupancy_pct'] = 100 * waves / self.MaxWavesPerCU
            m['valu_busy_pct'] = (100 * m.get('SQ_ACTIVE_INST_VALU', 0) * 4
                                  / (4 * self.CUs) / gui)
        if 'FETCH_SIZE' in m:
            m['dram_read_bytes'] = m['FETCH_SIZE'] * 1024
        if 'WRITE_SIZE' in m:
            m['dram_write_bytes'] = m['WRITE_SIZE'] * 1024
        if seconds > 0 and 'FETCH_SIZE' in m and 'WRITE_SIZE' in m:
            m['dram_throughput_pct'] = (100 * (m['dram_read_bytes'] + m['dram_write_bytes'])
                                        / seconds / self.PeakDramBytesPerSec)
        hits, misses = m.get('TCC_HIT_sum'), m.get('TCC_MISS_sum')
        if hits is not None and misses is not None and hits + misses > 0:
            m['l2_hit_pct'] = 100 * hits / (hits + misses)
        # Per-work-item scratch (spills and stack), from the code object.
        m['scratch_bytes'] = float(r['scr'])
        return m

    #This is not needed for AMD AFAIK. Keep it though
    def renameSymbols(self, inputName, outName):
        opt_cmd = ['opt']
        tuning_pass = os.getenv('RR_TUNING_PASS')
        assert tuning_pass, 'Expected env var for the tuning pass .so'
        opt_cmd.append(f'--load={tuning_pass}')
        opt_cmd.append('--enable-new-pm=0')
        opt_cmd.append('--global-symbol-rename-pass')
        opt_cmd.append(f'{inputName}')
        opt_cmd.append(f'-o {outName}')
        opt_cmd = ' '.join(opt_cmd)
        print(opt_cmd)
        execute_command(opt_cmd, capture_output=True, shell=True)

    def deviceCompile(self, original, Name, HashName, **kwargs):
        """Apply the launch bounds and return the backend's resource usage.

        The image stays LLVM bitcode, which the AMDGPU plugin JIT-compiles.
        Resource usage comes from compiling the same bitcode with llc, in the
        NVIDIA layout {kernel: [VGPRs, LDS bytes, SGPRs, spill bytes per lane,
        0]} (NVIDIA: [regs, smem, cmem, spill stores, spill loads]).
        """
        MaxThreads = kwargs.get('MaxThreads', None)
        MinTeams = kwargs.get('MinTeams', None)
        compile_cmd = ['opt', f'{original}', f'-o {HashName}.image']
        if (MinTeams is not None) or (MaxThreads is not None):
            tuning_pass = os.getenv('RR_TUNING_PASS')
            assert tuning_pass, 'Expected env var for the tuning pass .so'
            compile_cmd += [f'--load={tuning_pass}', '--enable-new-pm=0',
                            '--legacy-amdgpuattribute-pass']
        # Same attribute values as OpenMPOpt's --bo-omp-autotune for AMDGCN.
        if MaxThreads is not None:
            compile_cmd.append(f'--amdgpu-flat-work-group-size=1,{int(MaxThreads)}')
        if MinTeams is not None:
            compile_cmd.append(
                f'--amdgpu-waves-per-eu={self.getWavesPerEU(MaxThreads, MinTeams)}')
        execute_command(' '.join(compile_cmd), capture_output=True, shell=True)

        # Different MinTeams often map to the same waves-per-EU, i.e. the same
        # binary, so reuse the (slow) llc statistics.
        wpe = self.getWavesPerEU(MaxThreads, MinTeams) if MinTeams is not None else None
        src = os.path.realpath(original)
        key = (src, os.path.getmtime(src), MaxThreads, wpe)
        cache = self.__dict__.setdefault('_resourceCache', {})
        if key not in cache:
            llc_cmd = (f'llc -mtriple={self.Target} -mcpu={self.Arch} -O3 '
                       f'-pass-remarks-analysis=kernel-resource-usage '
                       f'{HashName}.image -o /dev/null')
            _, _, stderr = execute_command(llc_cmd, capture_output=True, shell=True)
            cache[key] = self.parseResourceUsage(stderr)
        return {k: list(v) for k, v in cache[key].items()}

    @staticmethod
    def parseResourceUsage(remarks):
        """Parse llc -pass-remarks-analysis=kernel-resource-usage output."""
        ret, name = {}, None
        fields = {'VGPRs': 0, 'LDS Size [bytes/block]': 1, 'SGPRs': 2}
        for line in remarks.splitlines():
            m = re.search(r'remark: .*?(Function Name|[A-Za-z][A-Za-z ]*(?: \[[^]]*\])?): (\S+)', line)
            if not m:
                continue
            key, value = m.group(1).strip(), m.group(2)
            if key == 'Function Name':
                name = value
                ret[name] = [0, 0, 0, 0, 0]
            elif name is None:
                continue
            elif key in fields:
                ret[name][fields[key]] = int(value)
            elif key == 'VGPRs Spill':
                # 4 bytes per spilled VGPR as counted by the backend: a relative
                # spill measure like ptxas' spill bytes, not scratch traffic.
                ret[name][3] = 4 * int(value)
        return ret

    def recordPrefix(self):
        # The replay region has no fixed-address mapping on AMDGPU, so it lands
        # at an ASLR-dependent address. Record without ASLR; the replay tool
        # disables it for itself.
        return 'setarch $(uname -m) -R '

    def getRecordMem(self):
        # The heuristic region loses up to 2 GB to alignment and keeps 1 GB of
        # headroom (LIBOMPTARGET_RR_DEVMEM_HEADROOM_MB), so allocate more.
        return 12

    def getReplayMem(self):
        # Without a fixed-address mapping the replay block must be large
        # enough to reach down to the recorded address.
        return self.getMaxMem()

    def dropSingleTeams(self):
        # Short loops (e.g. minisweep, 200-240 iterations) launch a single team
        # of the default 256 threads; BaseBenchmark.execute keeps those with a
        # recorded trip count instead.
        return False

    def blockFits(self, vgprs, threads):
        """Whether one work-group of `threads` with `vgprs` VGPRs fits a CU:
        VGPRs are allocated in blocks of 4 out of 256 per lane and SIMD, and the
        work-group's waves are spread over the 4 SIMDs."""
        waves = math.ceil(threads / self.WarpSize)
        per_simd = min(self.MaxWavesPerCU // 4, 256 // (math.ceil(max(vgprs, 1) / 4) * 4))
        return waves <= 4 * per_simd

    def command(self):
        return self._profile

    def env(self):
        return self._env

    def getNumDevices(self):
        return self.NumDevices

    def getOptConfigs(self):
        return self.optConfigs

    def getCompilationConfigs(self):
        return self.compilationConfigs

    def getDefaultBounds(self, config):
        assert config in self.compilationConfigs, f'Config {config} is not valid'
        if config == 'FlatWorkGroupSize':
            return (1, 1024.99)
        elif config == 'WavesPerEU':
            # TODO: what are the bounds?
            return (1, 100.99)
        elif config == 'MaxThreads':
            return (self.WarpSize, 1024+1, (self.WarpSize))
        elif config == 'MinTeams':
            return (1, int(self.ThreadsSMP/ self.WarpSize)+1, 1)
        else:
            assert False, f'getDefaultBounds not impl. for {config}'

    def getTarget(self):
        return self.Target

    def getArch(self):
        return self.Arch

    def getStageDir(self):
        return self.StageDir

def getDevice(device, StageDir=None, device_no=0):
    if device == 'nvidia':
        NV = NVIDIA.Load(StageDir, device_no)
        if isinstance(NV, NVIDIA):
            if NV.getNumDevices() >= 1:
                return NV
    elif device == 'amd':
        return AMDGPU(StageDir, device_no)
    else:
        raise Exception('Device must be nvidia or amd')

def GetCompileOptions(Device):
    return  ('-fopenmp  -fopenmp-offload-mandatory -foffload-lto'
             ' -fopenmp-targets={0}'
             ' -Xopenmp-target={0} -march={1}'
             ' -Wl,--embed-bitcode').format(Device.getTarget(), Device.getArch())

