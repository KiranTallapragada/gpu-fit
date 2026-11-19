"""Check the browser port (gpu-fit-core.js) gives the same answers as the Python package.

Usage: PYTHONPATH=src python3 bench/parity.py <models.json> <gpu-fit-core.js>
Needs node on PATH. Exits non-zero on any mismatch.
"""

import json
import subprocess
import sys

from gpu_fit.calc import estimate
from gpu_fit.model import ModelSpec

CONTEXTS = [8192, 32768, 131072]
TPS = [1, 2, 8]
KVS = [2, 1]
SEQS = [32, 256]


def main() -> int:
    data = json.load(open(sys.argv[1]))
    cases, expected = [], []
    for m in data["models"]:
        spec = ModelSpec(**m)
        for g in data["gpus"]:
            mem = g["memory_mib"] / 1024
            for ctx in CONTEXTS:
                for tp in TPS:
                    for kvb in KVS:
                        for seqs in SEQS:
                            f = estimate(spec, mem, ctx, tp, kvb, 0.9, None, seqs)
                            cases.append([m["name"], mem, ctx, tp, kvb, seqs])
                            expected.append([round(f.kv_pool_gib, 6), f.kv_tokens, f.max_context_one_request,
                                             f.hybrid_slots, f.fits, round(min(f.concurrent, 1e9), 6)])
    js = f"""
const G = require({json.dumps(sys.argv[2])});
const data = require({json.dumps(sys.argv[1])});
const byName = Object.fromEntries(data.models.map(m => [m.name, m]));
const cases = {json.dumps(cases)};
const out = cases.map(([n, mem, ctx, tp, kvb, seqs]) => {{
  const f = G.estimate(byName[n], mem, ctx, {{tp, kvBytes: kvb, util: 0.9, maxNumSeqs: seqs}});
  return [Math.round(f.kvPoolGib * 1e6) / 1e6, f.kvTokens, f.maxContextOneRequest, f.hybridSlots, f.fits,
          Math.round(Math.min(f.concurrent, 1e9) * 1e6) / 1e6];
}});
console.log(JSON.stringify(out));
"""
    got = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    bad = [(c, e, g) for c, e, g in zip(cases, expected, got)
           if e[1:5] != g[1:5] or abs(e[0] - g[0]) > 1e-5 or abs(e[5] - g[5]) > 1e-4 * max(1, e[5])]
    print(f"{len(cases)} cases, {len(bad)} mismatches")
    for b in bad[:10]:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
