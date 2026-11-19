"""GPU catalog. Memory is what `nvidia-smi --query-gpu=memory.total` reports (MiB),
which is what vLLM budgets against, not the marketing number."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_mib: int
    fp8: bool  # native FP8 (Ada / Hopper / Blackwell)
    note: str = ""

    @property
    def memory_gib(self) -> float:
        return self.memory_mib / 1024


GPUS = {
    g.name.lower(): g
    for g in [
        Gpu("L4", 23034, True),
        Gpu("A10G", 23028, False),
        Gpu("RTX4090", 24564, True),
        Gpu("A100-40", 40960, False),
        Gpu("L40S", 46068, True),
        Gpu("RTX6000-Ada", 49140, True),
        Gpu("A100-80", 81920, False),
        Gpu("H100", 81559, True, "80 GB SXM/PCIe"),
        Gpu("H200", 143771, True),
        Gpu("B200", 183359, True),
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
