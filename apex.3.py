#!/usr/bin/env python3
"""
send_course_trpc_bruteforce_v2.py
Robust sender for /api/trpc/addSyncFlowJob

- Handles BOTH single JSON and multi-course JSON files
- One-by-one sending with delay, start/limit windowing
- Optional forced shape (S1..S7) or auto-try sequence
- Retries with exponential backoff on transient errors
- Preflight check for Azure "Web App is stopped" 403 page
- Clear diagnostics for 4xx, 5xx, SSL, and proxy issues
"""

import json, sys, time, argparse, requests, urllib.parse, re
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

# ---- Default endpoint (can be overridden via flags) ----
DEFAULT_BASE  = "https://mynotedbe-guh7ekdxajcddvd2.southindia-01.azurewebsites.net"
DEFAULT_ROUTE = "/api/trpc/addSyncFlowJob"

# ---- Defaults ----
TIMEOUT = 180
USER_AGENT = "syncflow-client/2.0"

def make_urls(base: str, route: str) -> Tuple[str, str]:
    base = base.rstrip("/")
    if not route.startswith("/"):
        route = "/" + route
    url = f"{base}{route}"
    batch_url = f"{url}?batch=1"
    return url, batch_url

def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def ok(r: requests.Response) -> bool:
    return 200 <= r.status_code < 300

def show(label: str, r: requests.Response):
    body = (r.text or "")
    snippet = body[:2000]
    print(f"\n=== {label} ===")
    print("Status:", r.status_code)
    print("Body  :", snippet)

def extract_job_id(r: requests.Response):
    try:
        data = r.json()
    except Exception:
        print("\n[OK] Request succeeded (non-JSON / unexpected shape).")
        return

    # Try several tRPC / custom result shapes
    if isinstance(data, dict):
        result = data.get("result", {}).get("data", data)
        if isinstance(result, dict):
            jid = result.get("JobId") or result.get("jobId") or result.get("id")
            if jid:
                print(f"\n[OK] Sync job added. JobId: {jid}")
                return
        # tRPC batch envelope array commonly arrives directly, but some servers pack dict->list
        if isinstance(result, list) and result:
            try:
                jid = (
                    result[0].get("result", {})
                              .get("data", {})
                              .get("JobId")
                )
                if jid:
                    print(f"\n[OK] Sync job added. JobId: {jid}")
                    return
            except Exception:
                pass

    print("\n[OK] Request succeeded (no JobId field found).")

AZURE_STOPPED_RE = re.compile(r"Error 403\s*-\s*This web app is stopped", re.I)

def azure_stopped(text: str) -> bool:
    return bool(AZURE_STOPPED_RE.search(text or ""))

def attempt(label: str, *, url: str, json_body: Any = None, data_body: Optional[str] = None,
            headers: Optional[Dict[str,str]] = None, timeout: int = TIMEOUT,
            verify: bool = True) -> requests.Response:
    h = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.post(
        url,
        json=json_body if data_body is None else None,
        data=data_body if data_body is not None else None,
        headers=h,
        timeout=timeout,
        allow_redirects=False,
        verify=verify,
    )
    show(label, r)
    return r

def send_one_course(obj: Dict[str, Any], url: str, batch_url: str, *, force_shape: Optional[str],
                    timeout: int, verify: bool) -> bool:
    """
    Try shapes S1..S7 unless force_shape is given ('S1'...'S7').
    Returns True if any attempt returns 2xx.
    """
    shapes = [
        ("S1 NON-BATCH {\"id\":0,\"json\": obj}",
            dict(url=url, json_body={"id": 0, "json": obj},
                 headers={"Content-Type": "application/json; charset=utf-8"})),
        ("S2 NON-BATCH {\"input\": obj}",
            dict(url=url, json_body={"input": obj},
                 headers={"Content-Type": "application/json; charset=utf-8"})),
        ("S3 NON-BATCH raw body (obj)",
            dict(url=url, json_body=obj,
                 headers={"Content-Type": "application/json; charset=utf-8"})),
        ("S4 BATCH [{\"id\":0,\"json\": obj}]",
            dict(url=batch_url, json_body=[{"id": 0, "json": obj}],
                 headers={"Content-Type": "application/json; charset=utf-8"})),
        ("S5 BATCH index-map {\"0\":{\"json\": obj}}",
            dict(url=batch_url, json_body={"0": {"json": obj}},
                 headers={"Content-Type": "application/json; charset=utf-8"})),
        ("S6 NON-BATCH urlencoded in body (input=<json>)",
            dict(url=url, data_body=f"input={urllib.parse.quote(json.dumps(obj, ensure_ascii=False))}",
                 headers={"Content-Type": "application/x-www-form-urlencoded"})),
        ("S7 NON-BATCH raw stringified JSON",
            dict(url=url, data_body=json.dumps(obj, ensure_ascii=False),
                 headers={"Content-Type": "application/json; charset=utf-8"})),
    ]

    # If user forced a shape, only try that one
    if force_shape:
        idx = int(force_shape.upper().lstrip("S")) - 1
        if not (0 <= idx < len(shapes)):
            raise ValueError(f"Invalid --shape '{force_shape}'. Use S1..S7 or omit for auto.")
        label, kw = shapes[idx]
        r = attempt(label, timeout=timeout, verify=verify, **kw)
        if ok(r):
            extract_job_id(r)
            return True
        else:
            if r.status_code == 403 and azure_stopped(r.text):
                print("\n[FATAL] Azure says the Web App is STOPPED. Start the app in Azure Portal and try again.")
            return False

    # Auto mode: try in order
    for label, kw in shapes:
        r = attempt(label, timeout=timeout, verify=verify, **kw)
        if ok(r):
            extract_job_id(r)
            return True
        if r.status_code == 403 and azure_stopped(r.text):
            print("\n[FATAL] Azure says the Web App is STOPPED. Start the app in Azure Portal and try again.")
            return False

    print("\n[FAIL] All shapes returned non-2xx.")
    print("Hints:")
    print("  • Ensure Fastify body parsing is enabled and tRPC Fastify adapter is used.")
    print("  • If behind Azure Front Door/Functions, confirm request body isn't stripped by middleware.")
    print("  • Log req.body inside resolver to see what actually arrives.")
    return False

def normalize_courses(data: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list[ { courseName, course:[...]} ]  (already course-shaped)
      - list[ { ..., \"course\": { courseName, course:[...] } } ] (wrapped)
      - dict with 'items': [...]
      - single dict { courseName, course:[...] }
    Returns list of course objects.
    """
    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    if isinstance(data, dict):
        # single object
        if ("course" in data and isinstance(data["course"], list)) or ("courseName" in data):
            return [data]
        else:
            raise ValueError("Single dict found but missing 'courseName'/'course' fields.")

    if not isinstance(data, list):
        raise ValueError("Input must be a list (or dict with 'items' or a single course object).")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and "course" in item and isinstance(item["course"], dict):
            out.append(item["course"])
        elif isinstance(item, dict) and "courseName" in item:
            out.append(item)
        else:
            # allow direct course shape: {"courseName":..., "course":[...]}
            if isinstance(item, dict) and "course" in item and isinstance(item["course"], list):
                out.append(item)
            else:
                print(f"[WARN] Skipping index {i}: unrecognized shape (no 'course' or 'courseName').")
    if not out:
        raise ValueError("No course objects found after normalization.")
    return out

def preflight(base_url: str, *, timeout: int, verify: bool) -> None:
    """
    Quick GET to base URL to detect 'Web App is stopped' Azure splash early.
    """
    try:
        r = requests.get(base_url.rstrip("/"), timeout=timeout, allow_redirects=False, verify=verify,
                         headers={"User-Agent": USER_AGENT})
        if r.status_code == 403 and azure_stopped(r.text):
            print("[PRECHECK] Azure 403 page indicates the app is STOPPED. Start it in Azure Portal.")
    except Exception as e:
        # Preflight failures shouldn't block; just log
        print(f"[PRECHECK] Non-fatal preflight error: {e}")

def run_send(course: Dict[str, Any], url: str, batch_url: str, args) -> bool:
    # retries with exponential backoff (on network-type errors)
    attempt_num = 0
    backoff = args.retry_wait
    while True:
        attempt_num += 1
        try:
            ok_ = send_one_course(course, url, batch_url,
                                  force_shape=args.shape, timeout=args.timeout, verify=not args.insecure)
            return ok_
        except (requests.ReadTimeout, requests.ConnectTimeout) as e:
            print(f"[RETRYABLE] Timeout: {e}")
        except requests.SSLError as e:
            print(f"[RETRYABLE] SSL error: {e}")
            if args.insecure:
                print("          (Insecure mode enabled; cert validation already disabled.)")
        except requests.ConnectionError as e:
            print(f"[RETRYABLE] Connection error: {e}")
        except Exception as e:
            print(f"[NON-RETRYABLE] {e}")
            return False

        if attempt_num >= args.retries + 1:  # initial try + retries
            print("[GIVEUP] Exhausted retries.")
            return False

        print(f"[BACKOFF] Sleeping {backoff:.1f}s before retry {attempt_num}/{args.retries}...")
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)  # cap growth

def main():
    ap = argparse.ArgumentParser(description="Send courses via tRPC brute-force shapes, single or many.")
    ap.add_argument("--file", "-f", default="1234.json",
                    help="Path to JSON file (single course object or list). Default: %(default)s")
    ap.add_argument("--base", "-b", default=DEFAULT_BASE, help="Base URL. Default: %(default)s")
    ap.add_argument("--route", "-r", default=DEFAULT_ROUTE, help="Route path. Default: %(default)s")
    ap.add_argument("--delay", "-d", type=float, default=5.0, help="Seconds between sends. Default: %(default)s")
    ap.add_argument("--limit", "-n", type=int, default=0, help="Max number to send (0=all).")
    ap.add_argument("--start", "-s", type=int, default=0, help="Start index (skip first N).")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be sent, do not POST.")
    ap.add_argument("--shape", choices=[f"S{i}" for i in range(1,8)], default=None,
                    help="Force a specific request shape (S1..S7). Omit for auto.")
    ap.add_argument("--retries", type=int, default=2, help="Number of retry cycles on network errors. Default: %(default)s")
    ap.add_argument("--retry-wait", type=float, default=3.0, help="Initial backoff seconds. Default: %(default)s")
    ap.add_argument("--timeout", type=int, default=TIMEOUT, help="HTTP timeout (s). Default: %(default)s")
    ap.add_argument("--insecure", action="store_true", help="Disable TLS verification (USE ONLY FOR DEBUG).")
    ap.add_argument("--no-preflight", action="store_true", help="Skip Azure 'stopped' preflight check.")
    # Legacy positional: if provided, overrides --file and always treats as single object
    ap.add_argument("single_obj_path", nargs="?", default=None,
                    help="(Optional) Path to a SINGLE course JSON to send once (overrides --file).")
    args = ap.parse_args()

    url, batch_url = make_urls(args.base, args.route)

    # Optional Azure preflight
    if not args.no_preflight:
        preflight(args.base, timeout=10, verify=not args.insecure)

    # Single-object override (legacy behavior)
    if args.single_obj_path:
        obj = load_json(args.single_obj_path)
        if args.dry_run:
            name = obj.get("courseName") if isinstance(obj, dict) else "(single JSON)"
            print(f"[DRY-RUN] Would send single object: {name}")
            return
        ok_ = run_send(obj, url, batch_url, args)
        print(f"    -> success={ok_}")
        return

    # Multi-or-single from --file (auto-detect)
    try:
        data = load_json(args.file)
        courses = normalize_courses(data)
    except Exception as e:
        print(f"[ERROR] Failed to load/normalize: {e}")
        sys.exit(1)

    total = len(courses)
    start = max(0, args.start)
    end = total if not args.limit or args.limit < 0 else min(total, start + args.limit)

    if start >= total:
        print(f"[INFO] start index {start} >= total {total}; nothing to send.")
        return

    print(f"[INFO] Loaded {total} course object(s) from {args.file}")
    print(f"[INFO] Will send indices [{start}:{end}) with delay={args.delay}s  shape={args.shape or 'auto'}  retries={args.retries}")
    print(f"[INFO] Endpoint: {url}\n")

    sent = 0
    for i in range(start, end):
        course = courses[i]
        cname = course.get("courseName") or f"Course#{i}"
        chaps = len(course.get("course", [])) if isinstance(course, dict) else "?"
        print(f"\n--- [{i+1-start}/{end-start}] Sending: {cname} | Preview: {{'courseName': '{cname}', 'chapters': {chaps}}}")

        if args.dry_run:
            print("    [DRY-RUN] Skipped POST.")
        else:
            ok_ = run_send(course, url, batch_url, args)
            print(f"    -> success={ok_}")

        sent += 1
        if i < end - 1:
            time.sleep(max(0.0, args.delay))

    print(f"\n[DONE] Attempted {sent} course(s).")

if __name__ == "__main__":
    main()
