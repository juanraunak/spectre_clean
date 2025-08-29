#!/usr/bin/env python3
"""
Spectre Orchestrator — hard override of spectre_spider's finder
- Input arg: Spectre company name (e.g., "Manipal Fintech")
- Always reads:  final_skill_gaps_detailed_gpt.json
- Always writes: spectre_courses_cleaned.json
- Strategy:
    * Define a fuzzy find_company_skills_file in the globals
    * Pre-set SPECTRE_COMPANY / INPUT_JSON / OUTPUT_JSON
    * (Optional) pre-resolve a skills file and expose SKILLS_FILE
    * Run spectre_spider.py via runpy with our globals injected
"""

import os
import sys
import re
import shutil
import asyncio
import runpy
from pathlib import Path

FIXED_INPUT  = "final_skill_gaps_detailed_gpt.json"
FIXED_OUTPUT = "spectre_courses_cleaned.json"
SKILLS_DIR   = "company_skills"

# ---------- Fuzzy helpers ----------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _tokens(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

def _score_match(company: str, fname_stem: str) -> int:
    """
    Robust fuzzy score:
      +50 if normalized company contains file stem or vice versa
      +10 per shared token
      + len of intersection of character sets
    """
    c_norm = _norm(company)
    f_norm = _norm(fname_stem)
    score = 0
    if c_norm in f_norm or f_norm in c_norm:
        score += 50
    score += 10 * len(set(_tokens(company)) & set(_tokens(fname_stem)))
    score += len(set(c_norm) & set(f_norm))
    return score

def _best_company_token(company: str, fname_stem: str) -> str:
    toks = _tokens(company)
    if not toks:
        return company.strip()
    return max(toks, key=lambda t: (t in fname_stem.lower(), len(t)))

def _fuzzy_pick_file(company: str, skills_dir: str = SKILLS_DIR) -> Path:
    p = Path(skills_dir)
    if not p.exists():
        raise FileNotFoundError(f"Skills directory not found: {skills_dir}")
    candidates = [f for f in p.iterdir() if f.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files found in {skills_dir}")
    scored = sorted(((f, _score_match(company, f.stem)) for f in candidates),
                    key=lambda x: x[1], reverse=True)
    best, score = scored[0]
    if score == 0:
        raise FileNotFoundError(f"No sufficiently similar file in {skills_dir} for '{company}'")
    # If wrong extension (e.g., .sjon), clone to .json so downstream loaders are happy
    if best.suffix.lower() != ".json":
        json_twin = best.with_suffix(".json")
        shutil.copy2(best, json_twin)
        best = json_twin
    return best

# ---------- This function will override the module's finder ----------
def find_company_skills_file(spectre_company: str, skills_dir: str = SKILLS_DIR) -> str:
    """
    DROP-IN replacement for spectre_spider.find_company_skills_file
    Uses fuzzy matching and fixes wrong extensions automatically.
    """
    path = _fuzzy_pick_file(spectre_company, skills_dir)
    print(f"✅ [override] Using skills file: {path}")
    return str(path)

# ---------- Runner ----------
def _run_entry(ns: dict):
    # Prefer async builder if present
    if "build_comprehensive_courses_with_prior_knowledge" in ns:
        fn = ns["build_comprehensive_courses_with_prior_knowledge"]
        if asyncio.iscoroutinefunction(fn):
            return asyncio.run(fn())
        return fn()
    # Fallbacks
    for name in ("run", "main"):
        if name in ns and callable(ns[name]):
            return ns[name]()
    raise RuntimeError("No runnable entrypoint found in spectre_spider namespace")

def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: python orchestrator_override.py "<Spectre Company Name>"')
        sys.exit(1)

    company = sys.argv[1].strip()

    # Pre-resolve a skills file (helps if the module also reads SKILLS_FILE directly)
    try:
        skills_file = str(_fuzzy_pick_file(company))
    except Exception:
        skills_file = ""

    # Prepare injected globals so spectre_spider imports cleanly
    init_globals = {
        # our override is installed under the exact name spectre_spider expects
        "find_company_skills_file": find_company_skills_file,
        # predefine constants/env-backed vars the module reads at import time
        "SPECTRE_COMPANY": _best_company_token(company, Path(skills_file).stem if skills_file else ""),
        "INPUT_JSON": FIXED_INPUT,
        "OUTPUT_JSON": FIXED_OUTPUT,
    }
    if skills_file:
        init_globals["SKILLS_FILE"] = skills_file  # if the module reads this directly

    # Execute the target file with our globals preloaded
    target_path = str(Path(__file__).with_name("spectre_spider.py"))
    ns = runpy.run_path(target_path, init_globals=init_globals)

    # Run its entrypoint
    _run_entry(ns)

    print("\n🎯 Done.")
    print(f"→ Input JSON:  {FIXED_INPUT}")
    print(f"→ Output JSON: {FIXED_OUTPUT} (for report.py)")

if __name__ == "__main__":
    main()
