#!/usr/bin/env python3
"""
SHADE + MIRAGE + CIPHER + FRACTAL Orchestrator (streamlined)
============================================================

Flow:
  1) Collect ALL user inputs upfront (no interruptions)
  2) Run complete pipeline automatically based on those inputs
  3) Export and archive everything at the end

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
import re, asyncio, runpy, shutil  # add if missing
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

def collect_user_inputs() -> Optional[Dict[str, Any]]:
    """
    Collect ALL user inputs upfront before running any pipeline steps.
    Returns None if user wants to quit.
    """
    print("\n" + "="*60)
    print("WELCOME TO THE STREAMLINED ORCHESTRATOR")
    print("All questions first, then complete pipeline execution")
    print("="*60)
    
    # Company name
    company = input("\n[1/7] Enter company name (or 'quit'): ").strip()
    if company.lower() in ("quit", "exit", "q"):
        return None
    
    # Number of employees for SHADE
    n = input("[2/7] Number of employees for SHADE [50]: ").strip()
    num_employees = int(n) if n.isdigit() else 50

    # Seed employees (names)
    seed_names_raw = input("[3/7] Seed employee NAMES (comma-separated, optional): ").strip()
    seed_employee_names = [s.strip() for s in seed_names_raw.split(",") if s.strip()] if seed_names_raw else []

    # Seed employees (LinkedIn URLs)
    seed_urls_raw = input("[4/7] Seed employee LinkedIn URLS (comma-separated, optional): ").strip()
    seed_employee_urls = [u.strip() for u in seed_urls_raw.split(",") if u.strip()] if seed_urls_raw else []

    # Note about topping-up
    print(f"      → Will include {len(seed_employee_names) + len(seed_employee_urls)} seed(s) "
          f"and auto-discover the remaining to reach {num_employees}.")

    # MIRAGE settings
    run_mirage = input("[5/7] Run MIRAGE competitor analysis? (y/n): ").strip().lower() in ("y", "yes")
    competitors = 10
    if run_mirage and MIRAGE_AVAILABLE:
        c = input("      How many competitors? [10]: ").strip()
        competitors = int(c) if c.isdigit() else 10
    elif run_mirage and not MIRAGE_AVAILABLE:
        print("      Note: MIRAGE not available, will skip this step")
        run_mirage = False
    
    # CIPHER settings
    run_cipher = input("[6/7] Run CIPHER (skills extraction)? (y/n): ").strip().lower() in ("y", "yes")
    if run_cipher and not CIPHER_AVAILABLE:
        print("      Note: CIPHER not available, will skip this step")
        run_cipher = False
    
    # Auto-run confirmation for dependent agents
    auto_agents_info = ""
    if run_cipher:
        agents_to_run = []
        if FRACTAL_AVAILABLE:
            agents_to_run.append("FRACTAL (skill gap analysis)")
        try:
            import spectre_spider
            agents_to_run.append("SPECTRE SPIDER (course recommendations)")
        except:
            pass
        try:
            from report import SkillBoostPlanGenerator
            agents_to_run.append("REPORT GENERATOR (final reports)")
        except:
            pass
        if agents_to_run:
            auto_agents_info = f"      Will also auto-run: {', '.join(agents_to_run)}"
    
    print("[7/7] Pipeline will auto-run remaining agents after CIPHER")
    if auto_agents_info:
        print(auto_agents_info)
    
    confirm = input("      Proceed with this configuration? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        return None
    
    return {
        "company": company,
        "num_employees": num_employees,
        "seed_employee_names": seed_employee_names,
        "seed_employee_urls": seed_employee_urls,
        "run_mirage": run_mirage,
        "competitors": competitors,
        "run_cipher": run_cipher,
    }

def run_pipeline(config: Dict[str, Any]):
    """
    Run the complete pipeline without interruption based on collected config.
    """
    company = config["company"]
    num_employees = config["num_employees"]
    run_mirage = config["run_mirage"]
    competitors = config["competitors"]
    run_cipher = config["run_cipher"]
    
    # Root-level directories
    employee_data_dir = Path("employee_data")
    company_skills_dir = Path("company_skills")
    company_skills_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting pipeline for '{company}' with {num_employees} employees...")
    print("="*60)
    
    # ==== 1) SHADE ====
    print(f"\n[Step 1/8] Running SHADE for {company}...")
    shade_ctx = {
        "company_name": company,
        "spectre_n": num_employees,
        "seed_employee_names": config.get("seed_employee_names", []),
        "seed_employee_urls": config.get("seed_employee_urls", []),
    }
    shade_res = shade_run(shade_ctx)

    # Get SHADE report path
    report_path = shade_res.get("intelligence_report_path")
    if not report_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shade_dump = Path("output") / f"{_slug(company)}_{ts}_shade_result.json"
        shade_dump.parent.mkdir(parents=True, exist_ok=True)
        _write_json(shade_res, shade_dump)
        logger.info(f"Saved SHADE full result to {shade_dump}")
        report_path = str(shade_dump)
    
    print(f"✅ SHADE completed. Report: {report_path}")
    
    # ==== 2) MIRAGE ====
    mirage_res: Optional[Dict[str, Any]] = None
    if run_mirage:
        print(f"\n[Step 2/8] Running MIRAGE with {competitors} competitors...")
        ctx = {"inputs": {"intelligence_report_path": report_path, "num_competitors": competitors}}
        try:
            tmp_res = mirage_run(ctx)
            if not isinstance(tmp_res, dict):
                logger.warning("MIRAGE returned non-dict; coercing to {}")
                mirage_res = {}
            else:
                mirage_res = tmp_res
            if isinstance(mirage_res, dict):
                _print_mirage_summary(mirage_res)
            print("✅ MIRAGE completed.")
        except Exception as e:
            logger.error(f"MIRAGE errored but continuing: {e}", exc_info=True)
            mirage_res = {}
            print("⚠️  MIRAGE failed, continuing pipeline...")
    else:
        print("\n[Step 2/8] Skipping MIRAGE (not requested)")
    
    # ==== 3) EXPORT to ./employee_data ====
    print(f"\n[Step 3/8] Exporting data to employee_data...")
    try:
        employee_data_dir.mkdir(parents=True, exist_ok=True)
        exported_files = _export_employee_data(
            company_name=company,
            shade_report_path=report_path,
            mirage_result=mirage_res,
            dest_dir=employee_data_dir
        )
        print(f"✅ Exported {exported_files} file(s) to employee_data")
    except Exception as e:
        logger.error(f"Failed exporting reports to employee_data: {e}")
        print("⚠️  Export failed, continuing pipeline...")
    
    # ==== 4) CIPHER ====
    ran_cipher = False
    if run_cipher:
        print(f"\n[Step 4/8] Running CIPHER (skills extraction)...")
        cipher_inputs = {
            "data_directory": str(employee_data_dir.resolve()),
            "output_dir": str(company_skills_dir.resolve()),
            "test_connection": False,
        }
        logger.info(f"CIPHER inputs: {cipher_inputs}")
        try:
            ctx = {"inputs": cipher_inputs}
            cipher.run_sync(ctx)
            ran_cipher = True
            print("✅ CIPHER completed.")
        except Exception as e:
            logger.error(f"CIPHER errored: {e}", exc_info=True)
            print("⚠️  CIPHER failed, stopping dependent agents...")
    else:
        print("\n[Step 4/8] Skipping CIPHER (not requested)")
    
    # ==== 5) FRACTAL (Agent 4) — auto-run ONLY if CIPHER ran ====
    if ran_cipher and FRACTAL_AVAILABLE:
        print(f"\n[Step 5/8] Running FRACTAL (Agent 4) - skill gap analysis...")
        try:
            spectre_matches_path = "spectre_matches.json"
            if not Path(spectre_matches_path).exists():
                logger.warning(f"Missing {spectre_matches_path}; Agent 4 will skip.")
                print("⚠️  Missing spectre_matches.json, skipping FRACTAL...")
            else:
                agent4_ctx = {
                    "inputs": {
                        "skills_dir": str(company_skills_dir.resolve()),
                        "raw_dir": str(employee_data_dir.resolve()),
                        "spectre_path": spectre_matches_path,
                        "spectre_company": company,
                        "use_llm": True,
                        "outputs": {
                            "step1_detailed": "step1_missing_skills.json",
                            "step1_basic": "step1_missing_skills_basic.json",
                            "step2_detailed": "final_skill_gaps_detailed_gpt.json",
                            "final_summary": "final_skill_gaps.json",
                        },
                    }
                }
                fractal.run_sync(agent4_ctx)
                print("✅ FRACTAL completed.")
        except Exception as e:
            logger.error(f"FRACTAL (Agent 4) errored: {e}", exc_info=True)
            print("⚠️  FRACTAL failed, stopping dependent agents...")
            ran_cipher = False  # Stop the chain
    elif ran_cipher and not FRACTAL_AVAILABLE:
        print(f"\n[Step 5/8] Skipping FRACTAL (not available)")
    else:
        print(f"\n[Step 5/8] Skipping FRACTAL (CIPHER didn't run)")
    
 

       # ==== 6) AGENT 5 (SPECTRE SPIDER — just run) ====
    print(f"\n[Step 6/8] Running Agent 5 (SPECTRE SPIDER)…")
    ran_spider = False
    try:
        target_path = Path(__file__).with_name("spectre_spider.py")
        if not target_path.exists():
            raise FileNotFoundError(f"{target_path} not found")

        # Run the spider exactly as if invoked directly
        runpy.run_path(str(target_path), run_name="__main__")

        print("✅ AGENT 5 completed.")
        ran_spider = True
    except Exception as e:
        logger.error(f"Agent 5 (Spectre Spider) errored: {e}", exc_info=True)
        print("⚠️  AGENT 5 failed, continuing...")
        ran_spider = False


        # ==== 7) AGENT 6 (REPORT GENERATOR) — always run ====
    print(f"\n[Step 7/8] Running Agent 6 (REPORT GENERATOR) - final reports.")
    try:
        from report import SkillBoostPlanGenerator

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # report.py expects this file by default; no prompts here
        input_json = "spectre_courses.json"

        generator = SkillBoostPlanGenerator()  # ctor takes no args
        generator.process_all_employees(input_json, str(reports_dir))
        print(f"✅ REPORT GENERATOR completed. Reports in {reports_dir.resolve()}")
    except Exception as e:
        logger.error(f"Agent 6 (REPORT GENERATOR) errored: {e}", exc_info=True)
        print("⚠️  REPORT GENERATOR failed, continuing.")


        # ==== 8) FINAL ARCHIVE ====
    print(f"\n[Step 8/8] Creating final archive for '{company}'...")
    try:
        archive_root = Path(_slug(company))
        archive_root.mkdir(parents=True, exist_ok=True)

        # List of directories to archive (add more as needed)
        dirs_to_archive = [
            "employee_data",
            "company_skills",
            "reports",
            "matched_data",   # NEW
            "runs",           # NEW
        ]

        archived_dirs = []
        for d in dirs_to_archive:
            src = Path(d)
            if src.exists():
                dst = archive_root / d
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
                archived_dirs.append(d)

        # Key output files to archive
        files_to_archive = [
            "step1_missing_skills.json",
            "step1_missing_skills_basic.json",
            "final_skill_gaps_detailed_gpt.json",
            "final_skill_gaps.json",
            f"{_slug(company)}_spectre_courses.json",
        ]

        archived_files = []
        for f in files_to_archive:
            p = Path(f)
            if p.exists():
                dst = archive_root / p.name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(p), str(dst))
                archived_files.append(f)

        print(f"✅ Archive created: {archive_root.resolve()}")
        if archived_dirs:
            print(f"   Directories moved: {', '.join(archived_dirs)}")
        if archived_files:
            print(f"   Files moved: {', '.join(archived_files)}")

    except Exception as e:
        logger.error(f"Archiving failed for {company}: {e}", exc_info=True)
        print("⚠️  Archiving failed")

    print(f"\n🎉 Pipeline completed for '{company}'!")
    print("="*60)


# -----------------------------
# Main
# -----------------------------
def main():
    while True:
        # Collect ALL inputs first
        config = collect_user_inputs()
        if config is None:
            print("Goodbye!")
            break
        
        # Run complete pipeline without interruption
        run_pipeline(config)
        
        # Ask if they want to run another company
        another = input("\nRun pipeline for another company? (y/n): ").strip().lower()
        if another not in ("y", "yes"):
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()