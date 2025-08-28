#!/usr/bin/env python3
"""
Builds spectre_matches.json for Fractal by:
1) Reading all per-company match files in matched_data/
2) Converting them into Fractal schema
3) Aligning every manipal_name to your actual IBM employee name
   (loaded from employee_data/ibm.json + company_skills/ibm_skills.json)
"""

import json, sys, glob
from pathlib import Path
from collections import defaultdict

# ---------- CONFIG ----------
INPUT_DIR = Path("matched_data")                    # folder with per-company match JSONs
OUTPUT_PATH = Path("spectre_matches.json")         # Fractal input
RAW_DIR = Path("employee_data")                    # your raw company data
SKILL_DIR = Path("company_skills")                 # your skills data
SPECTRE_COMPANY = "ibm"                            # your Spectre company key
# ----------------------------

def _norm_company_key(s: str | None) -> str:
    if not s: return "unknown"
    return s.strip().lower().replace(" ", "_").replace("-", "_")

def _f32(x):
    try:
        return float(x)
    except Exception:
        return None

def _ensure_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _safe_read_list(path: Path):
    try:
        d = _read_json(path)
        return d if isinstance(d, list) else []
    except Exception:
        return []

def _normalize_name(s: str | None) -> str:
    return (s or "").strip().lower()

def _load_ibm_employee_name() -> str:
    """Merge RAW + SKILLS for IBM by name and return the single IBM employee name.
       If multiple, pick the first (and print a warning). If none, exit with error."""
    raw = _safe_read_list(RAW_DIR / f"{SPECTRE_COMPANY}.json")
    skills = _safe_read_list(SKILL_DIR / f"{SPECTRE_COMPANY}_skills.json")

    by_name = {}
    for e in raw:
        n = _normalize_name(e.get("name"))
        if n: by_name[n] = dict(e)
    for e in skills:
        n = _normalize_name(e.get("name"))
        if not n: continue
        cur = by_name.setdefault(n, {"name": e.get("name")})
        cur.setdefault("skills", [])
        cur["skills"] = sorted(set((cur.get("skills") or []) + (e.get("skills") or [])))

    names = [v.get("name") for v in by_name.values() if v.get("name")]
    if not names:
        print(f"[ERR] No IBM employees found in {RAW_DIR}/{SPECTRE_COMPANY}.json or {SKILL_DIR}/{SPECTRE_COMPANY}_skills.json")
        sys.exit(1)

    if len(names) > 1:
        print(f"[WARN] Found {len(names)} IBM employees; aligning to the first: {names[0]}. "
              "You can change SPECTRE_COMPANY data or script logic if needed.")
    return names[0]

def main():
    if not INPUT_DIR.exists():
        print(f"[ERR] Input folder not found: {INPUT_DIR.resolve()}")
        sys.exit(1)

    files = sorted(glob.glob(str(INPUT_DIR / "*.json")))
    print(f"[INFO] Found {len(files)} files in {INPUT_DIR}")
    if not files:
        print(f"[WARN] No JSON files found; writing empty output.")
    
    # Build Fractal schema bucket first (before alignment)
    # out: { company_key: [ { manipal_name: str, matches: [{company,name,similarity}, ...] }, ... ] }
    bucket: dict[tuple[str, str], dict] = {}  # key=(company_key, target_employee_lower)

    for fp in files:
        p = Path(fp)
        try:
            data = _read_json(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        # CASE A: "details" schema like your inmagine_matched_details.json
        # { "company": "Inmagine", "matched": [ { target_employee, competitor_employee, competitor_company, similarity_score, ... }, ... ] }
        if isinstance(data, dict) and "matched" in data:
            file_company = _norm_company_key(data.get("company"))
            for row in _ensure_list(data.get("matched")):
                target = (row.get("target_employee") or "").strip()
                if not target:
                    continue
                ckey = _norm_company_key(row.get("competitor_company") or data.get("company") or file_company)
                key = (ckey, _normalize_name(target))
                dst = bucket.setdefault(key, {"manipal_name": target, "matches": []})

                comp_name = (row.get("competitor_employee") or "").strip()
                if comp_name:
                    sim = _f32(row.get("similarity_score"))
                    if not any((m.get("name")==comp_name and _norm_company_key(m.get("company"))==ckey) for m in dst["matches"]):
                        dst["matches"].append({"company": ckey, "name": comp_name, "similarity": sim})
            continue

        # CASE B: Already in Fractal schema → merge directly
        # { "inmagine": [ {manipal_name: "...", matches: [...]}, ... ], "xcompany": [...] }
        if isinstance(data, dict):
            merged_any = False
            for ck, rows in data.items():
                if not isinstance(rows, list): 
                    continue
                ckey = _norm_company_key(ck)
                for r in rows:
                    tname = (r.get("manipal_name") or "").strip()
                    if not tname:
                        continue
                    key = (ckey, _normalize_name(tname))
                    dst = bucket.setdefault(key, {"manipal_name": tname, "matches": []})
                    for m in _ensure_list(r.get("matches")):
                        m_name = (m.get("name") or "").strip()
                        if not m_name:
                            continue
                        m_ckey = _norm_company_key(m.get("company") or ck)
                        m_sim = _f32(m.get("similarity"))
                        if not any((x.get("name")==m_name and _norm_company_key(x.get("company"))==m_ckey) for x in dst["matches"]):
                            dst["matches"].append({"company": m_ckey, "name": m_name, "similarity": m_sim})
                    merged_any = True
            if merged_any:
                continue

        print(f"[WARN] {p.name}: Unrecognized schema, skipping.")

    # Convert bucket → out map grouped by competitor company
    out: dict[str, list[dict]] = defaultdict(list)
    for (company_key, _target_lower), payload in bucket.items():
        out[company_key].append(payload)

    # === ALIGNMENT STEP: force all manipal_name to the actual IBM employee ===
    ibm_name = _load_ibm_employee_name()
    align_count = 0
    for ck, rows in out.items():
        for r in rows:
            if r.get("manipal_name") != ibm_name:
                r["manipal_name"] = ibm_name
                align_count += 1

    # Sort for readability
    out_sorted = {}
    for ck, rows in out.items():
        rows_sorted = sorted(rows, key=lambda r: r["manipal_name"].lower())
        for r in rows_sorted:
            r["matches"] = sorted(
                r["matches"],
                key=lambda m: (-(m["similarity"] if m["similarity"] is not None else -1), m["name"].lower())
            )
        out_sorted[ck] = rows_sorted

    OUTPUT_PATH.write_text(json.dumps(out_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    total_targets = sum(len(v) for v in out_sorted.values())
    total_matches = sum(len(r["matches"]) for v in out_sorted.values() for r in v)
    print(f"[OK] Wrote {OUTPUT_PATH}  companies={len(out_sorted)}  targets={total_targets}  matches={total_matches}")
    print(f"[INFO] Aligned {align_count} entries to IBM employee: {ibm_name}")
    for ck, rows in out_sorted.items():
        print(f"  - {ck}: {len(rows)} targets")

if __name__ == "__main__":
    main()
