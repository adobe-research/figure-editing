#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Evaluation Script for GPT-Image Results.
Tasks: SINGLE / MULTI / CONV / VISUAL / TRANSFER

Usage
-----
python eval_unified.py \
  --out_root /abs/path/to/modified_out \
  --model_dir_name GPT-Image \
  --disable_llm \
  --disable_clip \
  --disable_ocr

Environment Variables (for LLM Judge)
-------------------------------------
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION   (default: 2024-12-01-preview)
AZURE_OPENAI_DEPLOYMENT    (default: gpt-4o-mini-20240718)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

# ===== Local metrics =====
from ssim import compute_ssim
from IPIPS import compute_lpips
from PSNR import compute_psnr
from OCRSim import compute_ocr_similarity

# ===== Optional CLIP =====
try:
    from CLIPSim import compute_clip_similarity
except Exception:
    compute_clip_similarity = None

# ===== Optional LLM Judge =====
try:
    from llmjudge import LLMJudge
except Exception:
    LLMJudge = None


# --------------------------
# Helpers: IO & JSON state
# --------------------------
def load_json_safe(p: Path, default: Any) -> Any:
    """Load JSON from file; return default if unreadable or missing."""
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def save_json_atomic(p: Path, obj: Any) -> None:
    """Atomically save JSON so the file is never left half-written."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def png_of(json_path: Path) -> Path:
    """Replace .json with .png to locate the GT image file."""
    return json_path.with_suffix(".png")


def read_text_safe(p: Path) -> str:
    """Read text file; return empty string on failure."""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def is_file(p: Path) -> bool:
    return p.exists() and p.is_file()


def figure_dir(out_root: Path, chart_type: str, fig_id: str) -> Path:
    """<out_root>/<class>/<fig_id>"""
    return out_root / chart_type / fig_id


def rebase_to_root(out_root: Path, p: Path, strip_heads: Tuple[str, ...] = ("modified_out",)) -> Path:
    """
    Normalize a path from instructions.json to live under out_root.
    Includes logic to strip specific path heads (like 'modified_out') if present.
    """
    if p is None:
        return out_root
    p = Path(p)
    if p.is_absolute():
        return p

    parts = [seg for seg in p.parts if seg != "."]
    if not parts:
        return out_root

    cand_a = out_root / Path(*parts)  # as-is
    cand_b = None
    if parts and (parts[0] == out_root.name or parts[0] in strip_heads):
        cand_b = out_root / Path(*parts[1:])

    # Prefer the one that exists
    for c in (cand_a, cand_b):
        if c is not None and c.exists():
            return c

    # If neither exists, prefer the stripped version when a head was present
    return cand_b or cand_a


def find_test_png(fig_dir: Path, model_dir: str, gt_png_name: str) -> Optional[Path]:
    """
    Flexible search for Visual/Transfer tasks only.
    1) <fig_dir>/<model_dir>/<gt_png_name>
    2) <fig_dir>/<model_dir>/<sub>/<gt_png_name>
    3) bounded rglob
    """
    base = fig_dir / model_dir
    cand = base / gt_png_name
    if is_file(cand):
        return cand

    subs = ["Visual", "Transfer", "Style", "styles", "vis"]
    for s in subs:
        p = base / s / gt_png_name
        if is_file(p):
            return p

    # bounded glob (depth <= 4)
    try:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".png" and p.name == gt_png_name:
                if len(p.relative_to(base).parts) <= 4:
                    return p
    except Exception:
        pass
    return None


# --------------------------
# Dedup keys (resume/skip)
# --------------------------
def make_key(task: str, chart_type: str, fig_id: str, model_dir: str, test_filename: str) -> str:
    return f"{task}|{chart_type}|{fig_id}|{model_dir}|{test_filename}"


def load_completed_keys(result_path: Path) -> Set[str]:
    """Build a set of unique keys from the existing results JSON to support resuming."""
    completed: Set[str] = set()
    default_struct = {k: [] for k in ["single", "multi", "conv", "visual", "transfer"]}
    data = load_json_safe(result_path, default_struct)
    
    for task in default_struct.keys():
        for rec in data.get(task, []):
            chart = rec.get("chart_type", "")
            fid   = str(rec.get("figure_id", ""))
            model = rec.get("model", "")
            test  = Path(rec.get("test_path", "")).name
            if chart and fid and model and test:
                completed.add(make_key(task, chart, fid, model, test))
    return completed


# --------------------------
# Metrics computation
# --------------------------
def compute_all_metrics(
    gt_path: Path,
    test_path: Path,
    instr_text: str,
    llm: Optional["LLMJudge"],
    disable_clip: bool,
    disable_ocr: bool,
) -> Dict[str, Any]:
    """Compute all metrics that are enabled and available."""
    ssim_score  = compute_ssim(str(gt_path), str(test_path))
    lpips_score = compute_lpips(str(gt_path), str(test_path))

    if disable_clip or compute_clip_similarity is None:
        clip_score = None
    else:
        clip_score = compute_clip_similarity(str(gt_path), str(test_path))

    psnr_score  = compute_psnr(str(gt_path), str(test_path))

    if disable_ocr:
        ocr_score = None
    else:
        try:
            ocr_score = compute_ocr_similarity(str(gt_path), str(test_path))
        except Exception as e:
            print(f"[warn] OCR metric disabled at runtime: {e}")
            ocr_score = None

    llm_overall = llm_if = llm_cp = llm_iq = None
    if llm is not None:
        try:
            llm_if, llm_cp, llm_iq, llm_overall, _ = llm.evaluate(
                str(gt_path), str(test_path), instr_text
            )
        except Exception as e:
            print(f"[warn] LLM judge failed: {e}")

    return {
        "ssim": float(ssim_score),
        "lpips": float(lpips_score),
        "clip": float(clip_score) if isinstance(clip_score, (int, float)) else None,
        "psnr": float(psnr_score),
        "ocr":  float(ocr_score)  if isinstance(ocr_score,  (int, float)) else None,
        "llm_overall": float(llm_overall) if isinstance(llm_overall, (int, float)) else None,
        "llm_instr":   float(llm_if)      if isinstance(llm_if,       (int, float)) else None,
        "llm_content": float(llm_cp)      if isinstance(llm_cp,       (int, float)) else None,
        "llm_quality": float(llm_iq)      if isinstance(llm_iq,       (int, float)) else None,
    }


def load_entries_from_index(index_path: Path) -> List[Dict[str, Any]]:
    obj = load_json_safe(index_path, {})
    if not obj:
        return []
    entries = obj.get("entries", [])
    return entries if isinstance(entries, list) else []


# --------------------------
# Task 1: SINGLE (Strict Search, instructions.json only)
# --------------------------
def eval_single(
    out_root: Path,
    model_dir: str,
    result_data: Dict[str, Any],
    completed_keys: Set[str],
    llm: Optional["LLMJudge"],
    disable_clip: bool,
    disable_ocr: bool,
    result_path: Path,
) -> None:
    main_idx = out_root / "instructions.json"
    if not main_idx.exists():
        print("[info] SINGLE: instructions.json not found, skipping.")
        return

    entries = load_entries_from_index(main_idx)
    for e in entries:
        # STRICT: Only process explicit 'single' mode here
        mode = str(e.get("mode", "")).lower()
        if mode != "single":
            continue

        chart = e.get("class", "")
        fid   = str(e.get("fig_id", ""))
        if not chart or not fid:
            continue

        instr_file = rebase_to_root(out_root, Path(e.get("instruction_file", "")))
        instr_text = read_text_safe(instr_file)

        edited_files = e.get("edited_files", [])
        if not edited_files:
            continue
        gt_json = rebase_to_root(out_root, Path(edited_files[0]))
        gt_png  = png_of(gt_json)
        if not is_file(gt_png):
            continue

        # STRICT SEARCH: <out>/<chart>/<fid>/<model>/<filename>
        fig_dir   = figure_dir(out_root, chart, fid)
        test_png  = fig_dir / model_dir / gt_png.name
        if not is_file(test_png):
            # Unlike Visual/Transfer, we do not search recursively here
            continue

        key = make_key("single", chart, fid, model_dir, test_png.name)
        if key in completed_keys:
            continue

        try:
            metrics = compute_all_metrics(gt_png, test_png, instr_text, llm, disable_clip, disable_ocr)
        except Exception as ex:
            print(f"[error][single] {chart}/{fid}/{test_png.name}: {ex}")
            continue

        record = {
            "chart_type": chart,
            "figure_id": fid,
            "model": model_dir,
            "instruction": instr_text,
            "gt_path": str(gt_png),
            "test_path": str(test_png),
            "metrics": metrics
        }
        result_data["single"].append(record)
        completed_keys.add(key)
        save_json_atomic(result_path, result_data)
        print(f"[ok][single] {chart}/{fid}/{test_png.name} -> saved")


# --------------------------
# Task 2: MULTI (Strict Search)
# --------------------------
def eval_multi(
    out_root: Path,
    model_dir: str,
    result_data: Dict[str, Any],
    completed_keys: Set[str],
    llm: Optional["LLMJudge"],
    disable_clip: bool,
    disable_ocr: bool,
    result_path: Path,
) -> None:
    idx_path = out_root / "instructions.json"
    entries = load_entries_from_index(idx_path)
    if not entries:
        return

    for e in entries:
        if str(e.get("mode", "")).lower() != "multi":
            continue

        chart = e.get("class", "")
        fid   = str(e.get("fig_id", ""))
        if not chart or not fid:
            continue

        instr_text = read_text_safe(rebase_to_root(out_root, Path(e.get("instruction_file", ""))))
        edited_files = e.get("edited_files", [])
        if not edited_files:
            continue
        gt_json = rebase_to_root(out_root, Path(edited_files[0]))
        gt_png  = png_of(gt_json)
        if not is_file(gt_png):
            continue

        # STRICT SEARCH
        fig_dir  = figure_dir(out_root, chart, fid)
        test_png = fig_dir / model_dir / gt_png.name
        if not is_file(test_png):
            continue

        key = make_key("multi", chart, fid, model_dir, test_png.name)
        if key in completed_keys:
            continue

        try:
            metrics = compute_all_metrics(gt_png, test_png, instr_text, llm, disable_clip, disable_ocr)
        except Exception as ex:
            print(f"[error][multi] {chart}/{fid}/{test_png.name}: {ex}")
            continue

        record = {
            "chart_type": chart,
            "figure_id": fid,
            "model": model_dir,
            "instruction": instr_text,
            "gt_path": str(gt_png),
            "test_path": str(test_png),
            "metrics": metrics
        }
        result_data["multi"].append(record)
        completed_keys.add(key)
        save_json_atomic(result_path, result_data)
        print(f"[ok][multi] {chart}/{fid}/{test_png.name} -> saved")


# --------------------------
# Task 3: CONV (Strict Search in Conv/)
# --------------------------
def eval_conv(
    out_root: Path,
    model_dir: str,
    result_data: Dict[str, Any],
    completed_keys: Set[str],
    llm: Optional["LLMJudge"],
    disable_clip: bool,
    disable_ocr: bool,
    result_path: Path,
) -> None:
    idx_path = out_root / "instructions.json"
    entries  = load_entries_from_index(idx_path)
    multi_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    
    # Pre-load multi GTs
    for e in entries:
        if str(e.get("mode", "")).lower() != "multi":
            continue
        chart = e.get("class", "")
        fid   = str(e.get("fig_id", ""))
        if not chart or not fid:
            continue
        edited_files = e.get("edited_files", [])
        if not edited_files:
            continue
        gt_json = rebase_to_root(out_root, Path(edited_files[0]))
        gt_png  = png_of(gt_json)
        if not is_file(gt_png):
            continue
        instr_text = read_text_safe(rebase_to_root(out_root, Path(e.get("instruction_file", ""))))
        multi_map.setdefault((chart, fid), []).append({
            "gt_png": gt_png,
            "instr": instr_text
        })

    # Iterate over file system
    for chart_dir in out_root.iterdir():
        if not chart_dir.is_dir():
            continue
        chart = chart_dir.name

        for fig_dir in chart_dir.iterdir():
            if not fig_dir.is_dir():
                continue
            fid = fig_dir.name
            
            # STRICT SEARCH: Conv/ folder
            conv_dir = fig_dir / model_dir / "Conv"
            if not conv_dir.exists() or not conv_dir.is_dir():
                continue

            mlist = multi_map.get((chart, fid), [])
            if len(mlist) != 1:
                # Ambiguity or no GT found
                continue
            gt_png = mlist[0]["gt_png"]
            instr  = mlist[0]["instr"]

            for p in conv_dir.iterdir():
                if not (p.is_file() and p.suffix.lower() == ".png" and p.name.startswith("seq_final_")):
                    continue

                key = make_key("conv", chart, fid, model_dir, p.name)
                if key in completed_keys:
                    continue

                try:
                    metrics = compute_all_metrics(gt_png, p, instr, llm, disable_clip, disable_ocr)
                except Exception as ex:
                    print(f"[error][conv] {chart}/{fid}/{p.name}: {ex}")
                    continue

                record = {
                    "chart_type": chart,
                    "figure_id": fid,
                    "model": model_dir,
                    "instruction": instr,
                    "gt_path": str(gt_png),
                    "test_path": str(p),
                    "metrics": metrics
                }
                result_data["conv"].append(record)
                completed_keys.add(key)
                save_json_atomic(result_path, result_data)
                print(f"[ok][conv] {chart}/{fid}/{p.name} -> saved")


# --------------------------
# Task 4: VISUAL (Flexible Search, instructions_visual...json)
# --------------------------
def eval_visual(
    out_root: Path,
    model_dir: str,
    result_data: Dict[str, Any],
    completed_keys: Set[str],
    llm: Optional["LLMJudge"],
    disable_clip: bool,
    disable_ocr: bool,
    result_path: Path,
) -> None:
    vis_idx = out_root / "instructions_visual_circle_from_tags.json"
    if not vis_idx.exists():
        print("[info][VISUAL] index not found, skipping.")
        return

    entries = load_entries_from_index(vis_idx)
    for e in entries:
        mode = str(e.get("mode", "")).lower()
        if mode != "single_visual_circle_from_tags":
            continue

        chart = e.get("class", "")
        fid   = str(e.get("fig_id", ""))
        if not chart or not fid:
            continue

        instr_text = read_text_safe(rebase_to_root(out_root, Path(e.get("instruction_file", ""))))
        edited_files = e.get("edited_files", [])
        if not edited_files:
            continue

        gt_json = rebase_to_root(out_root, Path(edited_files[0]))
        gt_png  = png_of(gt_json)
        if not is_file(gt_png):
            continue

        fig_dir  = figure_dir(out_root, chart, fid)
        
        # FLEXIBLE SEARCH
        test_png = find_test_png(fig_dir, model_dir, gt_png.name)
        if not (test_png and is_file(test_png)):
            continue

        key = make_key("visual", chart, fid, model_dir, Path(test_png).name)
        if key in completed_keys:
            continue

        try:
            metrics = compute_all_metrics(gt_png, test_png, instr_text, llm, disable_clip, disable_ocr)
        except Exception as ex:
            print(f"[error][VISUAL] {chart}/{fid}/{Path(test_png).name}: {ex}")
            continue

        record = {
            "chart_type": chart,
            "figure_id": fid,
            "model": model_dir,
            "instruction": instr_text,
            "gt_path": str(gt_png),
            "test_path": str(test_png),
            "metrics": metrics,
            "visual_input": str(rebase_to_root(out_root, Path(e.get("input_image", ""))))
        }
        result_data["visual"].append(record)
        completed_keys.add(key)
        save_json_atomic(result_path, result_data)
        print(f"[ok][VISUAL] {chart}/{fid}/{Path(test_png).name} -> saved")


# --------------------------
# Task 5: TRANSFER (Flexible Search, instructions_style...json)
# --------------------------
def eval_transfer(
    out_root: Path,
    model_dir: str,
    result_data: Dict[str, Any],
    completed_keys: Set[str],
    llm: Optional["LLMJudge"],
    disable_clip: bool,
    disable_ocr: bool,
    result_path: Path,
) -> None:
    tr_idx = out_root / "instructions_style_transfer_single.json"
    if not tr_idx.exists():
        print("[info][TRANSFER] index not found, skipping.")
        return

    entries = load_entries_from_index(tr_idx)
    for e in entries:
        mode = str(e.get("mode", "")).lower()
        if mode != "style_transfer_single":
            continue

        chart = e.get("class", "")
        fid   = str(e.get("fig_id", ""))
        if not chart or not fid:
            continue

        instr_text = read_text_safe(rebase_to_root(out_root, Path(e.get("instruction_file", ""))))
        edited_files = e.get("edited_files", [])
        if not edited_files:
            continue

        gt_json = rebase_to_root(out_root, Path(edited_files[0]))
        gt_png  = png_of(gt_json)
        if not is_file(gt_png):
            continue

        fig_dir  = figure_dir(out_root, chart, fid)
        
        # FLEXIBLE SEARCH
        test_png = find_test_png(fig_dir, model_dir, gt_png.name)
        if not (test_png and is_file(test_png)):
            continue

        key = make_key("transfer", chart, fid, model_dir, Path(test_png).name)
        if key in completed_keys:
            continue

        try:
            metrics = compute_all_metrics(gt_png, test_png, instr_text, llm, disable_clip, disable_ocr)
        except Exception as ex:
            print(f"[error][TRANSFER] {chart}/{fid}/{Path(test_png).name}: {ex}")
            continue

        style_src = e.get("style_source", {})
        style_img = style_src.get("image")
        record = {
            "chart_type": chart,
            "figure_id": fid,
            "model": model_dir,
            "instruction": instr_text,
            "gt_path": str(gt_png),
            "test_path": str(test_png),
            "metrics": metrics,
            "style_source": {
                "class": style_src.get("class", ""),
                "fig_id": style_src.get("fig_id", ""),
                "image": str(rebase_to_root(out_root, Path(style_img))) if style_img else ""
            }
        }
        result_data["transfer"].append(record)
        completed_keys.add(key)
        save_json_atomic(result_path, result_data)
        print(f"[ok][TRANSFER] {chart}/{fid}/{Path(test_png).name} -> saved")


# --------------------------
# LLM Judge Init (Env Vars)
# --------------------------
def make_llm_judge(disable_llm: bool) -> Optional["LLMJudge"]:
    if disable_llm or LLMJudge is None:
        return None
    key   = os.getenv("AZURE_OPENAI_API_KEY", "")
    ep    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    ver   = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    dep   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-20240718")

    if not (key and ep):
        print("[info] LLM judge disabled: missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT.")
        return None

    try:
        llm = LLMJudge(api_key=key, azure_endpoint=ep, api_version=ver, model_deployment_name=dep)
        print(f"LLM Judge initialized with model deployment '{dep}' at endpoint '{ep}'.")
        return llm
    except Exception as e:
        print(f"[warn] LLM judge init failed: {e}")
        return None


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate GPT-Image results for SINGLE/MULTI/CONV/VISUAL/TRANSFER.")
    ap.add_argument("--out_root", type=str, required=True, help="Root directory of outputs (e.g. modified_out)")
    ap.add_argument("--model_dir_name", type=str, default="GPT-Image", help="Folder under each figure with model outputs")
    ap.add_argument("--result_name", type=str, default=None, help="Output JSON filename (default: <model>_eval_results.json)")
    
    # Toggle tasks
    ap.add_argument("--disable_single", action="store_true", help="Disable SINGLE task")
    ap.add_argument("--disable_multi",  action="store_true", help="Disable MULTI task")
    ap.add_argument("--disable_conv",   action="store_true", help="Disable CONV task")
    ap.add_argument("--disable_visual", action="store_true", help="Disable VISUAL task")
    ap.add_argument("--disable_transfer",action="store_true", help="Disable TRANSFER task")
    
    # Toggle metrics
    ap.add_argument("--disable_llm",    action="store_true", help="Disable LLM judge")
    ap.add_argument("--disable_clip",   action="store_true", help="Disable CLIP metric")
    ap.add_argument("--disable_ocr",    action="store_true", help="Disable OCR metric")
    
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.exists():
        raise FileNotFoundError(f"Out root not found: {out_root}")

    # Determine result JSON path
    res_fname = args.result_name if args.result_name else f"{args.model_dir_name}_eval_results.json"
    result_path = out_root / res_fname

    # Load existing to resume
    keys_list = ["single", "multi", "conv", "visual", "transfer"]
    result_data = load_json_safe(result_path, {k: [] for k in keys_list})
    # Ensure all keys exist
    for k in keys_list:
        if k not in result_data or not isinstance(result_data[k], list):
            result_data[k] = []

    completed = load_completed_keys(result_path)

    # Init LLM Judge
    llm = make_llm_judge(disable_llm=args.disable_llm)

    print(f"=== Starting Evaluation: {args.model_dir_name} ===")
    print(f"Output: {result_path}")

    # Run enabled tasks
    if not args.disable_single:
        eval_single(out_root, args.model_dir_name, result_data, completed, llm,
                    args.disable_clip, args.disable_ocr, result_path)

    if not args.disable_multi:
        eval_multi(out_root, args.model_dir_name, result_data, completed, llm,
                   args.disable_clip, args.disable_ocr, result_path)

    if not args.disable_conv:
        eval_conv(out_root, args.model_dir_name, result_data, completed, llm,
                  args.disable_clip, args.disable_ocr, result_path)
    
    if not args.disable_visual:
        eval_visual(out_root, args.model_dir_name, result_data, completed, llm,
                    args.disable_clip, args.disable_ocr, result_path)

    if not args.disable_transfer:
        eval_transfer(out_root, args.model_dir_name, result_data, completed, llm,
                      args.disable_clip, args.disable_ocr, result_path)

    # Final save & summary
    save_json_atomic(result_path, result_data)
    print("\n=== DONE ===")
    print(f"Results saved to: {result_path}")
    for k in keys_list:
        print(f"  {k.upper():<8}: {len(result_data[k])}")


if __name__ == "__main__":
    main()