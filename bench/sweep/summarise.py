"""Turn a `vllm bench sweep serve` output folder into one table: where does it stop scaling?

Repeated runs at the same concurrency are combined by their median. Warm-up results (files or
folders with "warmup" in the name) are skipped. Failed requests are reported when the results
include them.

Usage: python3 summarise.py <output-dir> [--markdown]
"""
import json
import pathlib
import statistics
import sys

COLS = [("request_throughput", "req/s", "{:.2f}"), ("output_throughput", "out tok/s", "{:.0f}"),
        ("median_ttft_ms", "TTFT p50 ms", "{:.0f}"), ("median_e2el_ms", "latency p50 s", "{:.1f}"),
        ("p99_e2el_ms", "latency p99 s", "{:.1f}")]


def load(root: pathlib.Path) -> dict:
    """Results grouped by concurrency."""
    groups: dict = {}
    for f in root.rglob("*.json"):
        if "warmup" in str(f.relative_to(root)).lower():
            continue
        try:
            d = json.loads(f.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and "request_throughput" in d and d.get("max_concurrency"):
            groups.setdefault(int(d["max_concurrency"]), []).append(d)
    return groups


def median_of(runs: list, key: str):
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def main() -> None:
    root, md = pathlib.Path(sys.argv[1]), "--markdown" in sys.argv
    groups = load(root)
    if not groups:
        sys.exit(f"no benchmark results under {root}")
    rows = []
    for conc in sorted(groups):
        runs = groups[conc]
        row = {"conc": conc, "runs": len(runs)}
        for key, _, _ in COLS:
            row[key] = median_of(runs, key)
        row["failed"] = sum(int(r.get("failed") or 0) for r in runs)
        rows.append(row)

    base = rows[0]
    per_req = (base["output_throughput"] or 0) / base["conc"]
    head = ["at once", "runs"] + [c[1] for c in COLS] + ["failed", "scaling efficiency"]
    table = []
    for r in rows:
        vals = [str(r["conc"]), str(r["runs"])]
        for key, _, fmt in COLS:
            v = r[key]
            vals.append("-" if v is None else fmt.format(v / 1000 if key.endswith("e2el_ms") else v))
        vals.append(str(r["failed"]))
        if per_req > 0 and r["output_throughput"] is not None:
            vals.append(f"{r['output_throughput'] / (per_req * r['conc']) * 100:.0f}%")
        else:
            vals.append("-")
        table.append(vals)

    if md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for t in table:
            print("| " + " | ".join(t) + " |")
    else:
        for t in [head] + table:
            print("\t".join(t))
    if per_req <= 0:
        print("\nnote: the lowest-concurrency result has no output throughput, so scaling"
              " efficiency can't be computed; check that run for errors")
    if any(r["failed"] for r in rows):
        print("note: some requests failed; treat those rows with care")
    valid = [r for r in rows if r["output_throughput"]]
    if valid:
        best = max(valid, key=lambda r: r["output_throughput"])
        print(f"\npeak output throughput at {best['conc']} at once ({best['output_throughput']:.0f} tok/s)")


if __name__ == "__main__":
    main()
