#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical analysis for GPT-Image evaluation results.
We classify tasks into "Visual" and "Transfer" categories by the outer JSON keys.

Input
-----
JSON file structure like:
{
  "visual": [
    { "class": "bar", "fig_id": "001", "metrics": {"ssim": 0.71, "psnr": 8.23, ...}, ... },
    ...
  ],
  "transfer": [
    { "class": "line", "fig_id": "003", "metrics": {"ssim": 0.80, "psnr": 9.10, ...}, ... },
    ...
  ]
}

Output
------
Prints summary statistics for each task type:
- Number of entries
- Average of each metric
"""

import json
from pathlib import Path
import statistics

# === Configuration ===
INPUT_JSON = Path("../GPT-Image_eval_results.json")  # fixed path

def safe_mean(values):
    """Return mean if list not empty, else None"""
    return statistics.mean(values) if values else None

def summarize_task(name, entries):
    """Summarize one task type (visual/transfer)."""
    print(f"\n==== {name.upper()} TASKS ====")
    print(f"Total entries: {len(entries)}")

    # collect metrics
    metrics_accum = {}
    for e in entries:
        metrics = e.get("metrics", {})
        for k, v in metrics.items():
            if v is None:
                continue
            metrics_accum.setdefault(k, []).append(v)

    # compute mean per metric
    for metric, vals in metrics_accum.items():
        avg = safe_mean(vals)
        print(f"{metric:>10s}: {avg:.4f} (n={len(vals)})")

def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    # iterate keys
    for key, entries in data.items():
        if not isinstance(entries, list):
            continue
        if "visual" in key.lower():
            summarize_task("Visual", entries)
        elif "transfer" in key.lower():
            summarize_task("Transfer", entries)
        else:
            print(f"[warn] Unknown task key: {key}")

if __name__ == "__main__":
    main()