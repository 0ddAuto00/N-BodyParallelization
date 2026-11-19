#!/usr/bin/env python3
"""Plot runtime and speedup from results.csv (threads, sequential, rowwise, tiled).

Usage:
    python plot_results.py [path/to/results.csv]
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "../docs/prelim_report/results.csv")

threads, seq, row, tiled = [], [], [], []
with open(csv_path, newline="") as f:
    for r in csv.DictReader(f):
        threads.append(int(r["threads"]))
        seq.append(float(r["sequential"]))
        row.append(float(r["rowwise"]))
        tiled.append(float(r["tiled"]))

# Speedup relative to the sequential baseline (first sequential measurement)
base = seq[0]
row_speedup = [base / t for t in row]
tiled_speedup = [base / t for t in tiled]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Runtime
ax1.plot(threads, seq, "k--", label="Sequential")
ax1.plot(threads, row, "o-", label="Rowwise")
ax1.plot(threads, tiled, "s-", label="Tiled")
ax1.set_xlabel("Threads")
ax1.set_ylabel("Time (s)")
ax1.set_title("Runtime vs. threads")
ax1.set_xticks(threads)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Speedup
ax2.plot(threads, threads, "k--", label="Ideal")
ax2.plot(threads, row_speedup, "o-", label="Rowwise")
ax2.plot(threads, tiled_speedup, "s-", label="Tiled")
ax2.set_xlabel("Threads")
ax2.set_ylabel("Speedup (vs. sequential)")
ax2.set_title("Speedup vs. threads")
ax2.set_xticks(threads)
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.tight_layout()
out = csv_path.with_name("results.png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
