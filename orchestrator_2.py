#!/usr/bin/env python3
"""
SHADE + MIRAGE + CIPHER + FRACTAL Orchestrator (interactive)
============================================================

Flow:
  1) Run SHADE for a company (asks for company + n).
  2) Optionally run MIRAGE on the SHADE report (asks yes/no + competitors).
  3) Export:
        - SHADE main report and all MIRAGE competitor reports
        - into ./employee_data/ as timestamped JSONs
  4) Optionally run CIPHER (Agent 3):
        - Reads from ./employee_data/
        - Writes to ./company_skills/ (root, not nested)
  5) Automatically run FRACTAL (Agent 4) AFTER Agent 3:
        - Reads from:
            - ./employee_data/
            - ./company_skills/
            - ./spectre_matches.json (Mirage matches file, project root)
        - Writes summary files to project root

Resilience:
  - MIRAGE errors never stop the flow; we continue to export + CIPHER.
  - FRACTAL only runs immediately after Agent 3 (if Agent 3 was run).

Requirements:
  - shade.run(ctx_dict) -> dict (may include "intelligence_report_path")
  - mirage.run(ctx_dict) -> dict (safe: should not raise; but we wrap anyway)
  - cipher.run_sync(ctx_dict) -> None (expects {"inputs": {...}})
  - fractal.run_sync(ctx_dict) -> None (expects {"inputs": {...}})
"""

import os, sys, json, shutil, logging
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
    logger.info("MIRAGE runner (mirage.run) imported successfully")
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

try:
    import fractal  # Agent 4 — must expose run_sync(context)
    FRACTAL_AVAILABLE = hasattr(fractal, "run_sync")
    if FRACTAL_AVAILABLE:
        logger.info("FRACTAL (Agent 4) module (run_sync) imported successfully")
    else:
        logger.warning("FRACTAL module imported, but run_sync not found")
except Exception as e:
    FRACTAL_AVAILABLE = False
    logger.warning(f"FRACTAL (Agent 4) not available: {e}")

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
    """Try multiple shapes to find competitor report JSONs in MIRAGE output."""
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

def _export_employee_data(
    company_name: str,
    shade_report_path: str,
    mirage_result: Optional[Dict[str, Any]],
    dest_dir: Path = Path("employee_data")
) -> int:
    """
    Copies the SHADE main report and all MIRAGE competitor report JSONs
    into ./employee_data as uniquely named, timestamped files.
    Returns number of files copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    copied = 0

    # SHADE main (Spectre company) report
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

    # MIRAGE competitor reports (if present)
    comp_paths: List[str] = _harvest_competitor_json_paths(mirage_result) if mirage_result else []
    for idx, p in enumerate(comp_paths, start=1):
        stem = Path(p).stem
        dst = dest_dir / f"{_slug(company_name)}__competitor_{idx:02d}__{stem}__{ts}.json"
        if _copy_if_exists(p, dst):
            copied += 1

    logger.info(f"[employee_data] Exported {copied} report(s) to {dest_dir.resolve()}")
    return copied

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

# -----------------------------
# Main (interactive)
# -----------------------------
def main():
    # Root-level directories
    employee_data_dir = Path("employee_data")
    company_skills_dir = Path("company_skills")

    # Ensure outputs root exists
    company_skills_dir.mkdir(parents=True, exist_ok=True)

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

        # The SHADE report path: prefer explicit path from SHADE, else dump full result
        report_path = shade_res.get("intelligence_report_path")
        if not report_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shade_dump = Path("output") / f"{_slug(company)}_{ts}_shade_result.json"
            shade_dump.parent.mkdir(parents=True, exist_ok=True)
            _write_json(shade_res, shade_dump)
            logger.info(f"Saved SHADE full result to {shade_dump}")
            report_path = str(shade_dump)

        # ==== 2) MIRAGE (optional, non-blocking) ====
        mirage_res: Optional[Dict[str, Any]] = None
        if MIRAGE_AVAILABLE:
            choice = input("Run MIRAGE on this report? (y/n): ").strip().lower()
            if choice in ("y", "yes"):
                c = input("How many competitors? [10]: ").strip()
                competitors = int(c) if c.isdigit() else 10
                ctx = {"inputs": {"intelligence_report_path": report_path, "num_competitors": competitors}}
                logger.info("Running MIRAGE...")
                try:
                    tmp_res = mirage_run(ctx)
                    if not isinstance(tmp_res, dict):
                        logger.warning("MIRAGE returned non-dict; coercing to {}")
                        mirage_res = {}
                    else:
                        mirage_res = tmp_res
                    if isinstance(mirage_res, dict):
                        _print_mirage_summary(mirage_res)
                except Exception as e:
                    logger.error(f"MIRAGE errored but continuing: {e}", exc_info=True)
                    mirage_res = {}
        else:
            logger.warning("MIRAGE not available; skipping competitor analysis")

        # ==== 3) EXPORT to ./employee_data (always do this) ====
        try:
            employee_data_dir.mkdir(parents=True, exist_ok=True)
            _ = _export_employee_data(
                company_name=company,
                shade_report_path=report_path,
                mirage_result=mirage_res,
                dest_dir=employee_data_dir
            )
            logger.info("All reports written to employee_data")
        except Exception as e:
            logger.error(f"Failed exporting reports to employee_data: {e}")

        # ==== 4) CIPHER (optional) — reads ./employee_data, writes ./company_skills ====
        ran_cipher = False
        if CIPHER_AVAILABLE:
            run_cipher = input("Run CIPHER (skills extraction) now? (y/n): ").strip().lower()
            if run_cipher in ("y", "yes"):
                cipher_inputs = {
                    "data_directory": str(employee_data_dir.resolve()),
                    "output_dir": str(company_skills_dir.resolve()),
                    "test_connection": False,  # set True if you want a ping first
                }
                logger.info(f"Running CIPHER with inputs: {cipher_inputs}")
                try:
                    ctx = {"inputs": cipher_inputs}
                    cipher.run_sync(ctx)
                    ran_cipher = True
                    logger.info("CIPHER finished.")
                except Exception as e:
                    logger.error(f"CIPHER errored: {e}", exc_info=True)
        else:
            logger.warning("CIPHER not available; skipping skills extraction")

        # ==== 5) FRACTAL (Agent 4) — auto-run ONLY if Agent 3 ran ====
        if ran_cipher and FRACTAL_AVAILABLE:
            try:
                spectre_matches_path = "spectre_matches.json"
                if not Path(spectre_matches_path).exists():
                    logger.warning(f"Missing {spectre_matches_path}; Agent 4 will skip.")
                else:
                    agent4_ctx = {
                        "inputs": {
                            # strict IO: only these two dirs + this one file
                            "skills_dir": str(company_skills_dir.resolve()),
                            "raw_dir": str(employee_data_dir.resolve()),
                            "spectre_path": spectre_matches_path,

                            # keep name aligned with user input
                            "spectre_company": company,

                            # set to True if you want GPT blurbs in outputs
                            "use_llm": False,

                            # filenames for outputs (root)
                            "outputs": {
                                "step1_detailed": "step1_missing_skills.json",
                                "step1_basic": "step1_missing_skills_basic.json",
                                "step2_detailed": "final_skill_gaps_detailed_gpt.json",
                                "final_summary": "final_skill_gaps.json",
                            },
                        }
                    }
                    logger.info("Running Agent 4 (FRACTAL) for skill-gap analysis…")
                    fractal.run_sync(agent4_ctx)
                    logger.info("Agent 4 (FRACTAL) finished. Files written to project root.")
            except Exception as e:
                logger.error(f"FRACTAL (Agent 4) errored: {e}", exc_info=True)
        elif ran_cipher and not FRACTAL_AVAILABLE:
            logger.warning("FRACTAL (Agent 4) not available; skipping skill-gap analysis")

if __name__ == "__main__":
    main()
