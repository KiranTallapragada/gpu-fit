"""GPU catalog. Memory is what `nvidia-smi --query-gpu=memory.total` reports (MiB),
which is what vLLM budgets against, not the marketing number."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_mib: int
    fp8: bool  # native FP8 (Ada / Hopper / Blackwell)
    bandwidth_gbs: int = 0  # memory bandwidth, GB/s (vendor spec); decode speed scales with it
    note: str = ""

    @property
    def memory_gib(self) -> float:
        return self.memory_mib / 1024


GPUS = {
    g.name.lower(): g
    for g in [
        Gpu("L4", 23034, True, 300),
        Gpu("A10G", 23028, False, 600),
        Gpu("RTX4090", 24564, True, 1008),
        Gpu("A100-40", 40960, False, 1555),
        Gpu("L40S", 46068, True, 864),
        Gpu("RTX6000-Ada", 49140, True, 960),
        Gpu("A100-80", 81920, False, 2039),
        Gpu("H100", 81559, True, 3350, "80 GB SXM (PCIe: ~2,000 GB/s)"),
        Gpu("H200", 143771, True, 4800),
        Gpu("B200", 183359, True, 8000),
    ]
}


def get_gpu(name: str) -> Gpu:
    key = name.lower().replace("_", "-").replace(" ", "-")
    if key in GPUS:
        return GPUS[key]
    for k, g in GPUS.items():
        if k.replace("-", "") == key.replace("-", ""):
            return g
    known = ", ".join(g.name for g in GPUS.values())
    raise SystemExit(f"Unknown GPU '{name}'. Known: {known}. Or pass --gpu-mem-gib.")
