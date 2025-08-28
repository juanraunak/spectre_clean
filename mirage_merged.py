#!/usr/bin/env python3
# peer_finder.py
# GPT-only Competitor & Peer Finder with rich logging
# - Uses Google CSE only to fetch web snippets
# - All parsing/classification/ranking is GPT-only (no regex fallbacks)

import os, re, json, time, logging, random, urllib.parse
from typing import List, Dict, Any, Optional
import requests

# ============================== Config ===============================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "2be1544b3dc14327b60a870fe8b94f35")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://notedai.openai.azure.com")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBsa_JCmZy5cJANA3-ksT3sPvwYqhuUQ4s")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "55d9d391fe2394876")

CSE_PER_QUERY = 6
CSE_DELAY_SEC = 0.9
CSE_MAX_RETRIES = 5
GPT_MAX_RETRIES = 5
REQUEST_TIMEOUT = 30

CACHE_DIR = ".pf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================== Logging ==============================
logging.basicConfig(
    level=logging.INFO,  # change to logging.DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("peer_finder")

# ============================== Utilities ============================
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def ensure_az_endpoint() -> str:
    base = (AZURE_OPENAI_ENDPOINT or "").rstrip("/")
    if not base.startswith("http"):
        raise ValueError(f"Invalid AZURE_OPENAI_ENDPOINT '{AZURE_OPENAI_ENDPOINT}'. Must start with https://")
    return f"{base}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_ID}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

def sleep_jitter(base: float) -> None:
    time.sleep(base + random.random() * 0.35)

def extract_name_from_url(linkedin_url: str) -> str:
    """We may extract the name seed from the LinkedIn slug; GPT will verify actual fields."""
    try:
        path = urllib.parse.urlparse(linkedin_url).path
        slug = path.split("/in/")[1].strip("/") if "/in/" in path else path.strip("/")
        slug = re.sub(r"-\d{3,}$", "", slug)
        name = re.sub(r"[-_]+", " ", slug)
        name = normalize_space(re.sub(r"\b(mba|phd|cfa|iim|iit|engg)\b", "", name, flags=re.I))
        name = " ".join(w.capitalize() for w in name.split())
        log.info(f"[Profile] Name seed from URL: '{name}'")
        return name
    except Exception:
        log.warning("[Profile] Could not parse name from URL")
        return ""

# ============================== Google CSE ===========================
def cse_search(query: str, num: int = CSE_PER_QUERY) -> List[Dict[str, Any]]:
    q = normalize_space(query)
    num = max(1, min(10, int(num)))
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q, "num": num, "safe": "off"}

    log.info(f"[CSE] Search: '{q}' (num={num})")
    for attempt in range(1, CSE_MAX_RETRIES + 1):
        try:
            log.debug(f"[CSE] Attempt {attempt} for '{q}'")
            sleep_jitter(CSE_DELAY_SEC)
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                items = r.json().get("items", []) or []
                log.info(f"[CSE] {len(items)} items for '{q}'")
                return items
            if r.status_code in (429, 500, 503):
                backoff = min(8, 0.8 * attempt**2)
                log.warning(f"[CSE] {r.status_code} on '{q}'. Backoff {backoff:.1f}s")
                time.sleep(backoff)
            else:
                log.warning(f"[CSE] {r.status_code}: {r.text[:200]}")
                break
        except requests.RequestException as e:
            backoff = min(8, 0.8 * attempt**2)
            log.warning(f"[CSE] Exception on '{q}': {e}. Retry in {backoff:.1f}s")
            time.sleep(backoff)
    return []

def collect_cse_items(queries: List[str], per_query: int = CSE_PER_QUERY, max_items: int = 40) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()
    log.info(f"[CSE] Running {len(queries)} queries (max_items={max_items})")
    for q in queries:
        log.info(f"[CSE] -> Query: {q}")
        for it in cse_search(q, num=per_query):
            link = it.get("link") or ""
            if link in seen:
                continue
            seen.add(link)
            item = {
                "title": it.get("title", "")[:250],
                "snippet": (it.get("snippet") or it.get("htmlSnippet") or "")[:500],
                "link": link
            }
            log.debug(f"[CSE] + Item: {item['title'][:120]} | {item['link']}")
            items.append(item)
            if len(items) >= max_items:
                log.info(f"[CSE] Reached max_items={max_items}")
                return items
    log.info(f"[CSE] Collected total items: {len(items)}")
    return items

# ============================== Azure GPT ============================
def gpt_json(system_msg: str, user_msg: str, temperature: float = 0.0) -> Dict[str, Any]:
    url = ensure_az_endpoint()
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",
             "content": (
                 "Return ONLY strict JSON (double-quoted keys/strings). "
                 "Do not add backticks or commentary.\n\n" + user_msg
             )},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    log.info("[GPT] Requesting completion")
    log.debug("[GPT] System: %s", system_msg[:400])
    log.debug("[GPT] User: %s", user_msg[:400])

    for attempt in range(1, GPT_MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                content = (r.json()["choices"][0]["message"]["content"] or "").strip()
                log.debug("[GPT] Raw: %s", content[:800])
                try:
                    parsed = json.loads(content)
                    log.info("[GPT] Parsed JSON OK")
                    return parsed
                except json.JSONDecodeError:
                    log.warning("[GPT] JSON parse failed; retrying with stricter prompt")
                    payload["messages"][-1]["content"] = "Only valid JSON. No prose.\n\n" + user_msg
                    time.sleep(0.7)
            elif r.status_code in (429, 500, 503):
                backoff = min(10, 0.9 * attempt**2)
                log.warning(f"[GPT] {r.status_code}. Backoff {backoff:.1f}s")
                time.sleep(backoff)
            else:
                log.error(f"[GPT] Error {r.status_code}: {r.text[:300]}")
                time.sleep(0.8 * attempt)
        except requests.RequestException as e:
            backoff = min(10, 0.9 * attempt**2)
            log.warning(f"[GPT] Exception: {e}. Retry in {backoff:.1f}s")
            time.sleep(backoff)
    log.error("[GPT] Failed after retries")
    return {}

# ============================== GPT Tasks ============================
DEPT_CHOICES = [
    "Engineering/Tech","Data/AI/ML","Product","Design/UX","Finance","Sales/BD",
    "Marketing/Growth","HR/People","Operations/Supply","Customer Success/Support",
    "Legal/Compliance","Other"
]

def gpt_profile_from_cse(name: str, company_hint: Optional[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    system = (
        "Extract a person's current professional profile ONLY from the provided search results. "
        "Use evidence from titles/snippets/links (e.g., LinkedIn). Do not invent facts. "
        "If unknown, leave empty or null.\n\nReturn JSON exactly:\n"
        "{\n"
        "  \"name\": \"\",\n"
        "  \"company\": \"\",\n"
        "  \"title\": \"\",\n"
        "  \"experience_years\": null,\n"
        "  \"skills\": [],\n"
        "  \"summary\": \"\"\n"
        "}"
    )
    user = json.dumps({
        "target_name": name,
        "company_hint": company_hint,
        "search_items": items[:40]
    })
    log.info("[Profile] Sending CSE items to GPT for profile extraction")
    resp = gpt_json(system, user, temperature=0.0) or {}
    log.info("[Profile] GPT says: name=%s company=%s title=%s",
             resp.get("name"), resp.get("company"), resp.get("title"))
    return resp

def gpt_department(profile: Dict[str, Any]) -> str:
    system = (
        "Classify the profile into exactly one of these departments: "
        + ", ".join(DEPT_CHOICES) +
        ". Return JSON: {\"department\":\"<one>\"}. If unclear, use \"Other\"."
    )
    user = json.dumps({"profile": profile, "choices": DEPT_CHOICES})
    log.info("[Dept] Classifying department via GPT")
    resp = gpt_json(system, user, temperature=0.0)
    dept = (resp or {}).get("department")
    log.info("[Dept] Department = %s", dept)
    return dept if isinstance(dept, str) else "Other"

def gpt_competitors_from_cse(target_company: str, department: str, items: List[Dict[str, Any]], max_out: int) -> List[str]:
    system = (
        "From the provided search results, identify companies that are realistic competitors to the target company, "
        f"focusing on the '{department}' function. Use ONLY the given results; do not invent names. "
        "Reject person names and non-company entities. Return JSON: {\"ranked_competitors\": [\"...\"]}."
    )
    user = json.dumps({
        "target_company": target_company,
        "department": department,
        "search_items": items[:60],
        "max": max_out
    })
    log.info("[Competitors] Asking GPT to rank competitors (max=%d)", max_out)
    resp = gpt_json(system, user, temperature=0.1) or {}
    ranked = resp.get("ranked_competitors") or []
    log.info("[Competitors] GPT ranked %d competitors", len(ranked))
    log.debug("[Competitors] List: %s", ranked)
    return ranked[:max_out]

def gpt_employees_from_cse(company: str, department: str, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    system = (
        "From these search results, extract employees who appear to work at the given company in the given department. "
        "Use ONLY provided results; DO NOT infer beyond them. Prefer linkedin.com/in links. "
        "Return JSON: {\"employees\": [{\"name\":\"\",\"title\":\"\",\"linkedin_url\":\"\"}, ...]} "
        f"with at most {limit} employees."
    )
    user = json.dumps({"company": company, "department": department, "search_items": items[:60], "limit": limit})
    log.info("[Employees] GPT extracting employees for %s (limit=%d)", company, limit)
    resp = gpt_json(system, user, temperature=0.0) or {}
    emps = resp.get("employees") or []
    log.info("[Employees] GPT found %d employees @ %s", len(emps), company)
    log.debug("[Employees] Raw: %s", emps)
    cleaned = []
    for e in emps[:limit]:
        cleaned.append({
            "name": e.get("name",""),
            "title": e.get("title",""),
            "linkedin_url": e.get("linkedin_url",""),
            "company": company
        })
    return cleaned

def gpt_similarity(target_profile: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "Compare the two professional profiles based solely on provided fields. "
        "Score strictly: Role compatibility (0-30), Experience match (0-25), Skill alignment (0-25), Department fit (0-20). "
        "Return JSON exactly:\n"
        "{\n"
        "  \"inferred_title\": \"\",\n"
        "  \"inferred_department\": \"\",\n"
        "  \"role_score\": 0,\n"
        "  \"experience_score\": 0,\n"
        "  \"skill_score\": 0,\n"
        "  \"department_score\": 0,\n"
        "  \"total_score\": 0,\n"
        "  \"rationale\": \"\"\n"
        "}"
    )
    user = json.dumps({"target_profile": target_profile, "candidate_profile": candidate})
    log.info("[Similarity] GPT scoring %s (%s)", candidate.get("name"), candidate.get("company"))
    resp = gpt_json(system, user, temperature=0.0) or {}
    log.debug("[Similarity] GPT resp: %s", resp)
    return resp

# ============================== Pipeline =============================
def extract_linkedin_profile(linkedin_url: str, company_hint: Optional[str]) -> Dict[str, Any]:
    name = extract_name_from_url(linkedin_url)
    queries = [
        f'"{name}" site:linkedin.com/in',
        f'"{name}" "{company_hint}" site:linkedin.com/in' if company_hint else f'"{name}" current company',
        f'"{name}" job title',
        f'{name} "{company_hint}" profile' if company_hint else f'{name} profile'
    ]
    log.info("[Profile] Running CSE to collect profile evidence (%d queries)", len(queries))
    items = collect_cse_items([q for q in queries if q], per_query=CSE_PER_QUERY, max_items=30)
    profile = gpt_profile_from_cse(name, company_hint, items)
    profile["name"] = profile.get("name") or name
    profile["linkedin_url"] = linkedin_url
    return profile

def find_competitors_with_departments(company: str, department: str, max_out: int = 10) -> List[str]:
    log.info("[Competitors] Searching for competitors of '%s' (dept=%s)", company, department)
    queries = [
        f"{company} competitors {department}",
        f"companies like {company} {department}",
        f"{department} competitors of {company}",
        f"{company} alternatives {department}",
        f"{company} competitors",
        f"top companies similar to {company} {department}"
    ]
    items = collect_cse_items(queries, per_query=CSE_PER_QUERY, max_items=50)
    return gpt_competitors_from_cse(company, department, items, max_out)

def find_department_employees(competitors: List[str], department: str, per_company_limit: int = 8) -> List[Dict[str, Any]]:
    log.info("[Employees] Fetching employees for %d competitors (dept=%s)", len(competitors), department)
    all_emps: List[Dict[str, Any]] = []
    for comp in competitors:
        log.info("[Employees] -> Company: %s", comp)
        queries = [
            f'site:linkedin.com/in "{comp}" {department}',
            f'"{comp}" {department} team site:linkedin.com/in',
            f'"{comp}" "{department}" Director OR Manager OR Lead site:linkedin.com/in'
        ]
        items = collect_cse_items(queries, per_query=CSE_PER_QUERY, max_items=50)
        emps = gpt_employees_from_cse(comp, department, items, per_company_limit)
        log.info("[Employees] + %d employees @ %s", len(emps), comp)
        all_emps.extend(emps)
    log.info("[Employees] Total collected employees: %d", len(all_emps))
    return all_emps

def analyze_peer_similarity(target_profile: Dict[str, Any], candidate_employees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    log.info("[Similarity] Scoring %d candidates", len(candidate_employees))
    scored: List[Dict[str, Any]] = []
    for cand in candidate_employees:
        resp = gpt_similarity(target_profile, cand)
        scored.append({
            "name": cand.get("name",""),
            "company": cand.get("company",""),
            "job_title": resp.get("inferred_title", cand.get("title","")),
            "department": resp.get("inferred_department",""),
            "similarity_score": int(resp.get("total_score", 0) or 0),
            "role_compatibility": int(resp.get("role_score", 0) or 0),
            "experience_match": int(resp.get("experience_score", 0) or 0),
            "skill_alignment": int(resp.get("skill_score", 0) or 0),
            "department_fit": int(resp.get("department_score", 0) or 0),
            "linkedin_url": cand.get("linkedin_url",""),
            "rationale": resp.get("rationale","")
        })
    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    log.info("[Similarity] Ranking complete")
    return scored

def enhanced_peer_finder(linkedin_url: str, company_hint: Optional[str],
                         max_competitors: int = 10, per_company_limit: int = 8) -> List[Dict[str, Any]]:
    log.info("=== PIPELINE START ===")
    log.info("Target URL: %s | Company hint: %s", linkedin_url, company_hint or "(none)")

    log.info("Step 1/5: Extract profile")
    target_profile = extract_linkedin_profile(linkedin_url, company_hint)

    log.info("Step 2/5: Classify department")
    department = gpt_department(target_profile)
    target_profile["department"] = department

    log.info("Step 3/5: Find competitors")
    competitors = find_competitors_with_departments(target_profile.get("company","") or (company_hint or ""),
                                                    department, max_out=max_competitors)
    log.info("[Competitors] Final list (%d): %s", len(competitors), competitors)

    log.info("Step 4/5: Discover competitor employees")
    employees = find_department_employees(competitors, department, per_company_limit=per_company_limit)

    log.info("Step 5/5: Similarity scoring")
    scored = analyze_peer_similarity(target_profile, employees)

    out_file = "enhanced_competitor_peers.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    log.info("Saved %d results to %s", len(scored), out_file)
    log.info("=== PIPELINE END ===")
    return scored

# ============================== CLI =================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT-only Competitor & Peer Finder")
    parser.add_argument("linkedin_url", help="Target LinkedIn profile URL (https://www.linkedin.com/in/...)")
    parser.add_argument("--company", help="Known company name (hint to improve CSE)", default=None)
    parser.add_argument("--max-competitors", type=int, default=10)
    parser.add_argument("--per-company-limit", type=int, default=8)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    # Fail fast if Azure endpoint misconfigured
    try:
        _ = ensure_az_endpoint()
    except Exception as e:
        log.error(f"Azure endpoint config error: {e}")
        raise SystemExit(1)

    results = enhanced_peer_finder(args.linkedin_url, args.company,
                                   max_competitors=args.max_competitors,
                                   per_company_limit=args.per_company_limit)

    print("\nTop 5 Similar Peers:")
    for i, peer in enumerate(results[:5], 1):
        print(f"{i}. {peer['name']} - {peer['company']}")
        print(f"   Role: {peer.get('job_title','')}")
        print(f"   Score: {peer['similarity_score']}/100")
        print(f"   LinkedIn: {peer['linkedin_url']}\n")

if __name__ == "__main__":
    main()
