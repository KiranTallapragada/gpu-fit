"""Turn a `vllm bench sweep serve` output folder into one table: where does it stop scaling?

Usage: python3 summarise.py <output-dir> [--markdown]
"""
import json
import pathlib
import sys

COLS = [("max_concurrency", "at once", "{:.0f}"), ("request_throughput", "req/s", "{:.2f}"),
        ("output_throughput", "out tok/s", "{:.0f}"), ("median_ttft_ms", "TTFT p50 ms", "{:.0f}"),
        ("median_e2el_ms", "latency p50 s", "{:.1f}"), ("p99_e2el_ms", "latency p99 s", "{:.1f}")]


def main() -> None:
    root, md = pathlib.Path(sys.argv[1]), "--markdown" in sys.argv
    rows = []
    for f in root.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and "request_throughput" in d and d.get("max_concurrency"):
            rows.append(d)
    rows.sort(key=lambda d: d["max_concurrency"])
    if not rows:
        sys.exit(f"no benchmark results under {root}")
    base = rows[0]["output_throughput"] / rows[0]["max_concurrency"]
    head = [c[1] for c in COLS] + ["scaling efficiency"]
    table = []
    for d in rows:
        vals = []
        for key, _, fmt in COLS:
            v = d.get(key)
            vals.append("-" if v is None else fmt.format(v / 1000 if key.endswith("e2el_ms") else v))
        vals.append(f"{d['output_throughput'] / (base * d['max_concurrency']) * 100:.0f}%")
        table.append(vals)
    if md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for t in table:
            print("| " + " | ".join(t) + " |")
    else:
        for t in [head] + table:
            print("\t".join(t))
    best = max(rows, key=lambda d: d["output_throughput"])
    print(f"\npeak output throughput at {best['max_concurrency']} at once"
          f" ({best['output_throughput']:.0f} tok/s)")


if __name__ == "__main__":
    main()
