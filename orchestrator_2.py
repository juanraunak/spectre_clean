#!/usr/bin/env python3
"""
SHADE + MIRAGE + CIPHER Orchestrator (interactive, keeps your inputs)
=====================================================================
Flow:
  1) Run SHADE for a company (asks for company + n).
  2) Optionally run MIRAGE on the SHADE report (asks yes/no + competitors).
  3) Optionally run CIPHER (asks yes/no). Feeds:
        - main company report (from SHADE)
        - all competitor reports (from MIRAGE, or a hint dir you provide)

New: After MIRAGE finishes, copy the SHADE main report and all MIRAGE
competitor reports into ./employee_data/ as timestamped JSON files.
"""

import os, sys, json, time, shutil, logging, glob
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - orchestrator - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("orchestrator")

# -----------------------------
# Imports
# -----------------------------
try:
    from shade import run as shade_run
    logger.info("SHADE module imported successfully")
except Exception as e:
    logger.error(f"Failed to import SHADE: {e}")
    sys.exit(1)

try:
    from mirage import run as mirage_run
    MIRAGE_AVAILABLE = True
    logger.info("MIRAGE runner imported successfully")
except Exception as e:
    MIRAGE_AVAILABLE = False
    logger.warning(f"MIRAGE not available: {e}")

try:
    import cipher  # must expose run_sync(context)
    CIPHER_AVAILABLE = hasattr(cipher, "run_sync")
    if CIPHER_AVAILABLE:
        logger.info("CIPHER module (run_sync) imported successfully")
    else:
        logger.warning("CIPHER module imported, but run_sync not found")
except Exception as e:
    CIPHER_AVAILABLE = False
    logger.warning(f"CIPHER not available: {e}")

# -----------------------------
# Helpers
# -----------------------------
def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "").strip()).strip("_") or "company"

def _write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _pick(value, *keys, default=None):
    for k in keys:
        if isinstance(value, dict) and value.get(k):
            return value[k]
    return default

def _copy_if_exists(src: str | Path, dst: Path) -> Optional[Path]:
    try:
        p = Path(src)
        if p.exists() and p.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(p), str(dst))
            return dst
    except Exception:
        pass
    return None

def _harvest_competitor_json_paths(mirage_result: Dict[str, Any]) -> List[str]:
    """Tries multiple common shapes to find competitor report JSONs in MIRAGE output."""
    paths: List[str] = []
    if not isinstance(mirage_result, dict):
        return paths

    agents = mirage_result.get("agents") or {}
    mir = agents.get("mirage") or {}
    res = mir.get("result") or {}

    candidates: List[str] = []

    arr = res.get("company_json_paths") or []
    if isinstance(arr, list):
        candidates.extend(arr)

    arr = res.get("competitor_reports") or []
    if isinstance(arr, list):
        for item in arr:
            p = _pick(item, "json_path", "report_path", "output_path")
            if p:
                candidates.append(p)

    outputs = res.get("outputs") or {}
    reports_dir = _pick(outputs, "reports_dir", "dir", "folder")
    if reports_dir and Path(reports_dir).exists():
        for p in Path(reports_dir).glob("*.json"):
            candidates.append(str(p.resolve()))

    companies = res.get("companies") or []
    if isinstance(companies, list):
        for c in companies:
            p = _pick(c, "json_path", "report_path", "output_path")
            if p:
                candidates.append(p)

    seen = set()
    for p in candidates:
        try:
            rp = str(Path(p).resolve())
        except Exception:
            continue
        if rp not in seen:
            seen.add(rp)
            paths.append(rp)

    return paths

def _glob_recent_jsons(root: Path, since_seconds: int = 3600) -> List[str]:
    cutoff = time.time() - since_seconds
    out: List[str] = []
    for p in root.rglob("*.json"):
        try:
            if p.stat().st_mtime >= cutoff:
                out.append(str(p.resolve()))
        except Exception:
            pass
    return out

def _prepare_cipher_directory(
    base_output: Path,
    company_name: str,
    shade_report_path: str,
    mirage_result: Optional[Dict[str, Any]],
    mirage_reports_dir_hint: Optional[str] = None,
) -> Path:
    """Create a folder with main + competitor JSONs for CIPHER."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = base_output / "cipher_input" / f"{_slug(company_name)}_{ts}"
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preparing CIPHER input at {data_dir}")

    # Main company report
    main_dst = data_dir / f"{_slug(company_name)}_report.json"
    if not _copy_if_exists(shade_report_path, main_dst):
        # last resort: try to re-open and rewrite
        try:
            with open(shade_report_path, "r", encoding="utf-8") as f:
                _write_json(json.load(f), main_dst)
        except Exception:
            logger.warning("Could not copy/write main SHADE report for CIPHER")

    # Competitors from MIRAGE
    comp_paths: List[str] = _harvest_competitor_json_paths(mirage_result) if mirage_result else []

    # Optional hint directory
    if not comp_paths and mirage_reports_dir_hint:
        hint = Path(mirage_reports_dir_hint)
        if hint.exists():
            comp_paths.extend([str(p.resolve()) for p in hint.glob("*.json")])

    # Fallback: recent JSONs in CWD
    if not comp_paths:
        logger.info("No competitor paths found in MIRAGE; globbing recent *.json near CWD")
        comp_paths = _glob_recent_jsons(Path.cwd(), since_seconds=3600)

    copied = 0
    for p in comp_paths:
        name = Path(p).stem
        if _copy_if_exists(p, data_dir / f"{name}.json"):
            copied += 1
    logger.info(f"Copied {copied} competitor JSONs for CIPHER")
    return data_dir

def _print_mirage_summary(result: Dict[str, Any]):
    try:
        meta = result.get("agents", {}).get("mirage", {}).get("result", {}).get("ghost_mirage_metadata", {})
        target = meta.get("target_company", "Unknown")
        total = meta.get("total_competitors_detected", "n/a")
        print("\n=== MIRAGE SUMMARY ===")
        print(f"Target Company: {target}")
        print(f"Competitors Detected: {total}")
        print("======================\n")
    except Exception:
        pass

# === NEW: export SHADE + MIRAGE reports into ./employee_data ==================
def _export_employee_data(
    company_name: str,
    shade_report_path: str,
    mirage_result: Optional[Dict[str, Any]],
    dest_dir: Path = Path("employee_data")
) -> int:
    """
    Copies the SHADE main report and all MIRAGE competitor report JSONs
    into ./employee_data as uniquely named files. Returns number of files copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    copied = 0

    # 1) Copy SHADE main (Spectre company) report
    shade_dst = dest_dir / f"{_slug(company_name)}__main__{ts}.json"
    if not _copy_if_exists(shade_report_path, shade_dst):
        # last resort: open+rewrite if copy failed
        try:
            with open(shade_report_path, "r", encoding="utf-8") as f:
                _write_json(json.load(f), shade_dst)
            copied += 1
        except Exception as e:
            logger.warning(f"Could not export SHADE report to employee_data: {e}")
    else:
        copied += 1

    # 2) Copy MIRAGE competitor reports (if present)
    comp_paths: List[str] = _harvest_competitor_json_paths(mirage_result) if mirage_result else []
    for idx, p in enumerate(comp_paths, start=1):
        stem = Path(p).stem
        dst = dest_dir / f"{_slug(company_name)}__competitor_{idx:02d}__{stem}__{ts}.json"
        if _copy_if_exists(p, dst):
            copied += 1

    logger.info(f"[employee_data] Exported {copied} report(s) to {dest_dir.resolve()}")
    return copied
# ==============================================================================

# -----------------------------
# Main (interactive)
# -----------------------------
def main():
    base_output = Path("output")
    base_output.mkdir(parents=True, exist_ok=True)

    while True:
        company = input("\nEnter company name (or 'quit'): ").strip()
        if company.lower() in ("quit", "exit", "q"):
            break

        n = input("Number of employees for SHADE [50]: ").strip()
        num_employees = int(n) if n.isdigit() else 50

        # ==== 1) SHADE ====
        logger.info(f"Running SHADE for {company} (n={num_employees})")
        shade_ctx = {"company_name": company, "spectre_n": num_employees}
        shade_res = shade_run(shade_ctx)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shade_fname = f"{_slug(company)}_{ts}_shade_result.json"
        shade_out = base_output / shade_fname
        _write_json(shade_res, shade_out)
        logger.info(f"Saved SHADE report to {shade_out}")

        report_path = shade_res.get("intelligence_report_path") or str(shade_out)

        # ==== 2) MIRAGE (optional) ====
        mirage_res: Optional[Dict[str, Any]] = None
        if MIRAGE_AVAILABLE:
            choice = input("Run MIRAGE on this report? (y/n): ").strip().lower()
            if choice in ("y", "yes"):
                c = input("How many competitors? [10]: ").strip()
                competitors = int(c) if c.isdigit() else 10
                ctx = {"inputs": {"intelligence_report_path": report_path, "num_competitors": competitors}}
                logger.info("Running MIRAGE...")
                mirage_res = mirage_run(ctx)
                if isinstance(mirage_res, dict):
                    _print_mirage_summary(mirage_res)
                else:
                    logger.warning("MIRAGE returned a non-dict result")
        else:
            logger.warning("MIRAGE not available; skipping competitor analysis")

        # === NEW: Export SHADE + MIRAGE reports to ./employee_data ============
        try:
            _export_employee_data(company_name=company, shade_report_path=report_path, mirage_result=mirage_res)
        except Exception as e:
            logger.error(f"Failed exporting reports to employee_data: {e}")
        # =====================================================================

        # ==== 3) CIPHER (optional) ====
        if CIPHER_AVAILABLE:
            run_cipher = input("Run CIPHER (skills extraction) now? (y/n): ").strip().lower()
            if run_cipher in ("y", "yes"):
                hint = input("If MIRAGE reports are in a folder, enter path (or leave blank): ").strip() or None
                out_override = input("CIPHER output dir [blank = ./output/company_skills]: ").strip() or None
                test_ping = input("Test Azure/OpenAI connection first? (y/n): ").strip().lower() in ("y", "yes")

                data_dir = _prepare_cipher_directory(
                    base_output=base_output,
                    company_name=company,
                    shade_report_path=report_path,
                    mirage_result=mirage_res,
                    mirage_reports_dir_hint=hint,
                )

                cipher_inputs = {
                    "data_directory": str(data_dir),
                    "output_dir": out_override or str(base_output / "company_skills"),
                    "test_connection": bool(test_ping),
                }
                logger.info(f"Running CIPHER with inputs: {cipher_inputs}")
                try:
                    ctx = {"inputs": cipher_inputs}
                    cipher.run_sync(ctx)
                    logger.info("CIPHER finished.")
                except Exception as e:
                    logger.error(f"CIPHER errored: {e}")
        else:
            logger.warning("CIPHER not available; skipping skills extraction")

if __name__ == "__main__":
    main()
