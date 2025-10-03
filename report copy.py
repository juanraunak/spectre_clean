#!/usr/bin/env python3
"""
Standalone tester for Bright Data scraping (dataset mode only) — hard-coded config.
"""

import json, time, logging, requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("brightdata_test")

# --- Hard-coded config ---
BRIGHT_DATA_API_KEY = "c00d7b6c54b3df6a2aae1f2b015ca32142040c12d431bb3cd9baad1a15aa13f0"
BRIGHT_DATA_DATASET_ID = "gd_l1viktl72bvl7bjuj0"
BASE_URL = "https://api.brightdata.com"
TIMEOUT = 600

TEST_URLS = [
    "https://in.linkedin.com/in/sarit-shah-bbb58a376",
    "https://in.linkedin.com/in/krish-rajpal-b60310209",
]

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {BRIGHT_DATA_API_KEY}",
    "Content-Type": "application/json",
})

def trigger_dataset(urls):
    endpoint = f"{BASE_URL}/datasets/v3/trigger"
    payload = [{"url": u} for u in urls]
    params = {"dataset_id": BRIGHT_DATA_DATASET_ID, "include_errors": "true"}
    log.info("Triggering DATASET → %s?dataset_id=%s", endpoint, BRIGHT_DATA_DATASET_ID)
    r = session.post(endpoint, params=params, data=json.dumps(payload), timeout=60)
    log.info("Response %d: %s", r.status_code, r.text[:500])
    if not r.ok:
        raise RuntimeError(f"Trigger failed: {r.status_code} {r.text}")
    data = r.json()
    return data.get("id") or data.get("request_id") or data.get("snapshot_id")

def wait_until_ready(snapshot_id):
    url = f"{BASE_URL}/datasets/v3/progress/{snapshot_id}"
    start = time.time()
    while True:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        status = (r.json().get("status") or "").lower()
        log.info("Progress status=%s", status)
        if status in {"ready", "done", "succeeded", "completed"}:
            return
        if time.time() - start > TIMEOUT:
            raise TimeoutError(f"Timed out waiting for snapshot {snapshot_id}")
        time.sleep(5)

def download_ndjson(snapshot_id):
    url = f"{BASE_URL}/datasets/v3/snapshot/{snapshot_id}"
    r = session.get(url, params={"format": "ndjson"}, timeout=120)
    r.raise_for_status()
    rows = []
    for line in r.text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                log.warning("Skipping bad line: %s", line[:200])
    log.info("Downloaded %d rows", len(rows))
    print(json.dumps(rows, indent=2))
    return rows

if __name__ == "__main__":
    log.info("Running in DATASET mode only")
    snap = trigger_dataset(TEST_URLS)
    log.info("snapshot_id=%s", snap)
    wait_until_ready(snap)
    download_ndjson(snap)
