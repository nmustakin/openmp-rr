"""Vendor-neutral access to the GPU state the tuner needs: which other
processes use a GPU (exclusive-GPU guard), the board energy counter, and the
environment variable that restricts a process to one GPU.

get_monitor() returns an NVML monitor on NVIDIA and a KFD/ROCm SMI monitor on
AMD. Handles are opaque; None means the device could not be resolved, and every
query then reports nothing (no foreign processes, no energy).
"""
import ctypes
import os
from pathlib import Path


class NVMLMonitor:
    visible_env = 'CUDA_VISIBLE_DEVICES'

    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()

    def init_child(self):
        """Re-initialize after fork."""
        self.nvml.nvmlInit()

    def handle(self, ordinal):
        """NVML handle for a CUDA device ordinal.

        NVML ignores CUDA_VISIBLE_DEVICES, so map the ordinal through it
        (numeric entries, CUDA_DEVICE_ORDER=PCI_BUS_ID assumed).
        """
        visible = os.environ.get(self.visible_env)
        if visible:
            entries = [v.strip() for v in visible.split(',') if v.strip()]
            if all(v.isdigit() for v in entries) and ordinal < len(entries):
                return self.nvml.nvmlDeviceGetHandleByIndex(int(entries[ordinal]))
        return self.nvml.nvmlDeviceGetHandleByIndex(ordinal)

    def foreign_pids(self, handle):
        if handle is None:
            return []
        try:
            procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except self.nvml.NVMLError:
            return []
        return [p.pid for p in procs if p.pid != os.getpid()]

    def energy_mj(self, handle):
        """Board energy counter in mJ, or None."""
        if handle is None:
            return None
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        except self.nvml.NVMLError:
            return None


class AMDMonitor:
    """KFD sysfs for processes on a GPU, ROCm SMI for energy.

    ROCm numbers GPUs in KFD topology-node order; ROCR_VISIBLE_DEVICES selects
    and reorders them. A process uses a GPU when it has a queue on it.
    """
    visible_env = 'ROCR_VISIBLE_DEVICES'
    KFD = Path('/sys/class/kfd/kfd')

    def __init__(self):
        self.gpus = []      # (gpu_id, PCI location id) in ROCm order
        nodes = sorted((self.KFD / 'topology' / 'nodes').iterdir(),
                       key=lambda p: int(p.name))
        for node in nodes:
            gpu_id = int((node / 'gpu_id').read_text())
            if gpu_id == 0:          # CPU node
                continue
            props = dict(line.split() for line in
                         (node / 'properties').read_text().splitlines()
                         if len(line.split()) == 2)
            self.gpus.append((gpu_id, int(props.get('location_id', 0))))
        self.smi = None
        rocm = os.environ.get('ROCM_PATH', '/opt/rocm')
        try:
            self.smi = ctypes.CDLL(f'{rocm}/lib/librocm_smi64.so')
            if self.smi.rsmi_init(ctypes.c_uint64(0)) != 0:
                self.smi = None
        except OSError:
            self.smi = None

    def init_child(self):
        pass

    def handle(self, ordinal):
        """(gpu_id, PCI location id) of a ROCm device ordinal, or None."""
        visible = os.environ.get(self.visible_env)
        index = ordinal
        if visible:
            entries = [v.strip() for v in visible.split(',') if v.strip()]
            if all(v.isdigit() for v in entries) and ordinal < len(entries):
                index = int(entries[ordinal])
        return self.gpus[index] if index < len(self.gpus) else None

    def foreign_pids(self, handle):
        if handle is None:
            return []
        gpu_id = str(handle[0])
        pids = []
        for proc in (self.KFD / 'proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                if any((q / 'gpuid').read_text().strip() == gpu_id
                       for q in (proc / 'queues').iterdir()):
                    pids.append(int(proc.name))
            except OSError:          # the process exited while we looked
                continue
        return pids

    def energy_mj(self, handle):
        """Board energy counter in mJ, or None where it is unusable (MI50
        reports a constant 0xFFFFFFFF; the plugin integrates power instead)."""
        if handle is None or self.smi is None:
            return None
        index = self._smi_index(handle[1])
        if index is None:
            return None
        counter, resolution, stamp = ctypes.c_uint64(), ctypes.c_float(), ctypes.c_uint64()
        if self.smi.rsmi_dev_energy_count_get(index, ctypes.byref(counter),
                                              ctypes.byref(resolution),
                                              ctypes.byref(stamp)) != 0:
            return None
        if counter.value in (0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
            return None
        return counter.value * resolution.value / 1e3

    def _smi_index(self, location_id):
        count = ctypes.c_uint32()
        if self.smi.rsmi_num_monitor_devices(ctypes.byref(count)) != 0:
            return None
        for i in range(count.value):
            bdf = ctypes.c_uint64()
            if (self.smi.rsmi_dev_pci_id_get(i, ctypes.byref(bdf)) == 0
                    and bdf.value & 0xffff == location_id):
                return i
        return None


_monitor = None


def get_monitor(vendor=None):
    """The monitor for this machine: `vendor` is 'nvidia' or 'amd'; by default
    AMD when the KFD driver is present."""
    global _monitor
    if _monitor is None:
        if vendor is None:
            vendor = 'amd' if AMDMonitor.KFD.exists() else 'nvidia'
        _monitor = AMDMonitor() if vendor == 'amd' else NVMLMonitor()
    return _monitor
