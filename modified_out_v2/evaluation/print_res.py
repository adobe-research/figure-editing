#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
overall_res.py

Goal:
- Read results from JSONL (one object per line) or JSON (dict/list).
- Do NOT group by instruction; only report overall metrics for:
    SINGLE
    MULTI (COMBO + SEQ combined)
- If a record has no 'phase', infer default from filename:
    filename contains 'combo' -> COMBO else SINGLE.
- Accept 'instruction' or 'instr' but ignore for aggregation.
- Skip None values (do not coerce to 0).
- Metrics aggregated: ssim, lpips, clip, psnr, ocr, llm_overall/llm_score, llm_instr, llm_content, llm_quality.
"""

import sys
import os
import json
from typing import Any, Dict, Iterable

MAX_LINES = 1000  # only for JSONL mode

METRIC_KEYS = [
    "ssim", "lpips", "clip", "psnr", "ocr",
    "llm_score", "llm_instr", "llm_content", "llm_quality"
]

def infer_default_phase_from_fname(argv) -> str:
    if len(argv) > 1 and argv[1] != "-":
        fname = os.path.basename(argv[1]).lower()
        return "COMBO" if "combo" in fname else "SINGLE"
    return "SINGLE"

def coerce_float_or_skip(val: Any) -> Iterable[float]:
    if val is None or val == "":
        return []
    try:
        return [float(val)]
    except (TypeError, ValueError):
        return []

def add_metrics(acc: Dict[str, Dict[str, float]], metrics: Dict[str, Any]):
    # map llm_overall/llm_score to llm_score
    overall = metrics.get("llm_overall")
    if overall is None:
        overall = metrics.get("llm_score")
    m = {
        "ssim": metrics.get("ssim"),
        "lpips": metrics.get("lpips"),
        "clip": metrics.get("clip"),
        "psnr": metrics.get("psnr"),
        "ocr": metrics.get("ocr"),
        "llm_score": overall,
        "llm_instr": metrics.get("llm_instr"),
        "llm_content": metrics.get("llm_content"),
        "llm_quality": metrics.get("llm_quality"),
    }
    for k, v in m.items():
        for x in coerce_float_or_skip(v):
            acc[k]["sum"] += x
            acc[k]["cnt"] += 1

def new_acc():
    return {k: {"sum": 0.0, "cnt": 0} for k in METRIC_KEYS}

def mean(vsum: float, cnt: int) -> float:
    return (vsum / cnt) if cnt > 0 else float("nan")

def print_block(title: str, acc: Dict[str, Dict[str, float]]):
    print(f"\n{'='*20} {title} {'='*20}")
    total_cnt = acc["ssim"]["cnt"]
    if total_cnt == 0:
        print("(no results)")
        return
    print(f"samples     : {total_cnt}")
    print(f"SSIM        : {mean(acc['ssim']['sum'], acc['ssim']['cnt']):.4f}")
    print(f"LPIPS       : {mean(acc['lpips']['sum'], acc['lpips']['cnt']):.4f}")
    print(f"CLIP        : {mean(acc['clip']['sum'], acc['clip']['cnt']):.4f}")
    print(f"PSNR        : {mean(acc['psnr']['sum'], acc['psnr']['cnt']):.2f}")
    print(f"OCR         : {mean(acc['ocr']['sum'], acc['ocr']['cnt']):.4f}")
    print(f"LLM Overall : {mean(acc['llm_score']['sum'], acc['llm_score']['cnt']):.2f}")
    print(f"  - IF      : {mean(acc['llm_instr']['sum'], acc['llm_instr']['cnt']):.2f}")
    print(f"  - Content : {mean(acc['llm_content']['sum'], acc['llm_content']['cnt']):.2f}")
    print(f"  - Quality : {mean(acc['llm_quality']['sum'], acc['llm_quality']['cnt']):.2f}")

def iter_jsonl(fh, default_phase: str):
    for i, raw in enumerate(fh):
        if i >= MAX_LINES:
            break
        s = raw.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not rec.get("phase"):
            rec["phase"] = default_phase
        yield rec

def iter_from_loaded_obj(obj: Any, default_phase: str):
    def with_phase(items, phase_name: str):
        for rec in items:
            r = dict(rec)
            r["phase"] = r.get("phase", phase_name) or phase_name
            yield r

    if isinstance(obj, dict):
        low = {k.lower(): k for k in obj.keys()}
        had = False
        if "single" in low:
            had = True
            yield from with_phase(obj[low["single"]], "SINGLE")
        if "combo" in low:
            had = True
            yield from with_phase(obj[low["combo"]], "COMBO")
        if "seq" in low:
            had = True
            yield from with_phase(obj[low["seq"]], "SEQ")
        if not had:
            # treat any list values as records with default phase
            for v in obj.values():
                if isinstance(v, list):
                    for rec in v:
                        r = dict(rec)
                        r["phase"] = r.get("phase", default_phase) or default_phase
                        yield r
    elif isinstance(obj, list):
        for rec in obj:
            r = dict(rec)
            r["phase"] = r.get("phase", default_phase) or default_phase
            yield r

def main():
    default_phase = infer_default_phase_from_fname(sys.argv)

    acc_single = new_acc()
    acc_multi = new_acc()  # COMBO + SEQ

    # open source
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as fh:
            buf = fh.read()
        # try JSON first
        try:
            obj = json.loads(buf)
            rec_iter = iter_from_loaded_obj(obj, default_phase)
        except json.JSONDecodeError:
            # JSONL
            rec_iter = iter_jsonl(buf.splitlines(True), default_phase)  # type: ignore
    else:
        # stdin
        buf = sys.stdin.read()
        try:
            obj = json.loads(buf)
            rec_iter = iter_from_loaded_obj(obj, default_phase)
        except json.JSONDecodeError:
            rec_iter = iter_jsonl(buf.splitlines(True), default_phase)  # type: ignore

    for rec in rec_iter:
        phase = str(rec.get("phase", "")).upper()
        metrics = rec.get("metrics", {})
        if phase == "SINGLE":
            add_metrics(acc_single, metrics)
        elif phase in ("COMBO", "SEQ"):
            add_metrics(acc_multi, metrics)
        else:
            # unknown -> use filename default
            if default_phase == "COMBO":
                add_metrics(acc_multi, metrics)
            else:
                add_metrics(acc_single, metrics)

    print_block("SINGLE (overall)", acc_single)
    print_block("MULTI = COMBO + SEQ (overall)", acc_multi)

if __name__ == "__main__":
    main()