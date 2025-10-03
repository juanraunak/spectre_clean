#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHADE — Company + Employee Discovery (Improved, Single-File Edition)
===================================================================

What this script does:
  • Seed-aware discovery of employees (names and/or LinkedIn URLs)
  • Hard cap: if you want N employees, you get <= N (seeds prioritized)
  • Google CSE → LinkedIn discovery (robust, multi-strategy, paginated)
  • Optional Bright Data scrape for LinkedIn profile details
  • Lightweight company research (CSE + aiohttp + BeautifulSoup)
  • Comprehensive JSON report writer with analytics + executive summary
  • Async-safe runner + sync adapters + simple CLI

Environment variables (recommended):
  GOOGLE_API_KEY
  GOOGLE_CSE_ID
  BRIGHT_DATA_API_KEY          (optional, only if you use Bright Data)
  BRIGHT_DATA_DATASET_ID       (optional, dataset id)
  AZURE_OPENAI_API_KEY         (optional, for AI analysis)
  AZURE_OPENAI_ENDPOINT        (optional)
  AZURE_OPENAI_DEPLOYMENT_ID   (optional, default "gpt-4o")
  AZURE_OPENAI_API_VERSION     (optional, default "2024-06-01")

Usage (CLI):
  $ python shade_improved.py
  Target company: Manipal Fintech
  How many employees to gather? [default 50]: 4
  Seed employee NAMES (comma-separated, optional): Aishwary Sinha, Ashish M., Vaishali Bhat, Ajit Kumar
  Seed employee LinkedIn URLS (comma-separated, optional): https://www.linkedin.com/in/ashish-m-4a6a695/, ...

Outputs:
  • run folder: ./runs/<company-slug>-<ts>/
  • report: runs/<...>/data/<company>_complete_intelligence_report.json
  • legacy copy: ./data/<company>_complete_intelligence_report.json
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import logging
import asyncio
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote, quote

import aiohttp
from bs4 import BeautifulSoup

# =============================================================================
# Configuration
# =============================================================================

HARDCODE = True  # set True to use the hardcoded placeholders below

# --- Placeholders (only used when HARDCODE=True) ---
GOOGLE_API_KEY_HARDCODE = "AIzaSyAohBAGNUxv_QpPXoMjvAXRipIqdhb1DY4"
GOOGLE_CSE_ID_HARDCODE  = "9539617f2a9e14131"

BRIGHT_DATA_API_KEY_HARDCODE  = "c00d7b6c54b3df6a2aae1f2b015ca32142040c12d431bb3cd9baad1a15aa13f0"
BRIGHT_DATA_DATASET_ID_HARDCODE = "gd_l1viktl72bvl7bjuj0"

AZURE_OPENAI_API_KEY_HARDCODE = "2be1544b3dc14327b60a870fe8b94f35"
AZURE_OPENAI_ENDPOINT_HARDCODE = "https://notedai.openai.azure.com"
AZURE_OPENAI_DEPLOYMENT_ID_HARDCODE = "gpt-4o"
AZURE_OPENAI_API_VERSION_HARDCODE = "2024-06-01"

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SHADE - %(levelname)s - %(message)s"
)
logger = logging.getLogger("shade")

# =============================================================================
# Helpers
# =============================================================================

def _slug(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "company"

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _guess_name_from_linkedin_url(url: str) -> str:
    """
    Attempt to pretty-print a name from a LinkedIn /in/ slug.
    """
    try:
        p = urlparse(url)
        parts = [x for x in p.path.split("/") if x]
        # Typical: /in/<slug>/...
        slug = ""
        if len(parts) >= 2 and parts[0] == "in":
            slug = parts[1]
        elif len(parts) >= 1:
            slug = parts[0]
        slug = unquote(slug).split("?")[0].strip("-_/")
        slug = re.sub(r"-\d+$", "", slug)
        name = slug.replace("-", " ").replace("_", " ").strip()
        if not name:
            return "(seed from URL)"
        return " ".join(w.capitalize() for w in name.split())
    except Exception:
        return "(seed from URL)"

def _normalize_name(n: str) -> str:
    return re.sub(r"[^\w\s]", "", (n or "").lower()).strip()

def _name_token_set(n: str) -> set:
    return set(_normalize_name(n).split())

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# =============================================================================
# Data models
# =============================================================================

@dataclass
class RawEmployee:
    name: str
    linkedin_url: str
    snippet: str
    company: str
    is_seed: bool = False

@dataclass
class CompanyData:
    name: str
    website: str = ""
    description: str = ""
    industry: str = ""
    headquarters: str = ""
    founded_year: str = ""
    employee_estimate: str = ""
    revenue_estimate: str = ""
    funding_info: str = ""
    tech_stack: List[str] | None = None
    social_links: Dict[str, str] | None = None
    recent_news: List[Dict[str, str]] | None = None
    financial_data: Dict[str, Any] | None = None
    business_model: str = ""
    key_products: str = ""
    market_position: str = ""
    competitive_analysis: str = ""
    growth_metrics: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if self.tech_stack is None: self.tech_stack = []
        if self.social_links is None: self.social_links = {}
        if self.recent_news is None: self.recent_news = []
        if self.financial_data is None: self.financial_data = {}
        if not self.timestamp: self.timestamp = _now_iso()

# =============================================================================
# Google CSE Employee Finder
# =============================================================================

class GoogleCSEEmployeeFinder:
    """
    Google Custom Search integration for discovering LinkedIn /in/ profiles.
    - Bulk discovery (company scoped)
    - Improved seed-by-name finder with multiple query strategies + paging
    """

    def __init__(self):
        if HARDCODE:
            self.api_key = GOOGLE_API_KEY_HARDCODE
            self.cse_id  = GOOGLE_CSE_ID_HARDCODE
        else:
            self.api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY_HARDCODE if HARDCODE else "")
            self.cse_id  = os.getenv("GOOGLE_CSE_ID", GOOGLE_CSE_ID_HARDCODE if HARDCODE else "")
        if not self.api_key or not self.cse_id:
            raise ValueError("Google CSE keys missing. Set GOOGLE_API_KEY and GOOGLE_CSE_ID.")

        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        })

    # ---------- public ----------

    def find_employees(self, company_name: str, max_results: int = 50) -> List[RawEmployee]:
        """
        Company-scoped discover: site:linkedin.com/in "<company>"
        Stops once we collected max_results.
        """
        if max_results <= 0:
            return []

        query = f'site:linkedin.com/in "{company_name}"'
        logger.info(f"🔎 Discovery query: {query} (limit={max_results})")

        out: List[RawEmployee] = []
        per_page = 10
        pages = math.ceil(max_results / per_page)
        start_index = 1

        for page in range(pages):
            remaining = max_results - len(out)
            if remaining <= 0:
                break

            items = self._search_page(query, start_index=start_index)
            if not items:
                if page == 0:
                    logger.warning("No discovery results.")
                break

            for it in items:
                if len(out) >= max_results:
                    break
                emp = self._extract_employee_from_result(it, company_name, skip_c_suite=True)
                if emp:
                    out.append(emp)

            start_index += per_page
            time.sleep(random.uniform(0.5, 1.0))

        logger.info(f"🎯 Discovery found {len(out)} employees")
        return out

    def find_employee_by_name_improved(self, company_name: str, full_name: str) -> Optional[RawEmployee]:
        """
        Multi-strategy, paginated seed-by-name finder.
        Returns first confident match (LinkedIn /in/).
        """
        full_name = (full_name or "").strip()
        if not full_name:
            return None

        clean = _normalize_name(full_name)
        parts = clean.split()
        variants = [
            f'site:linkedin.com/in "{full_name}" "{company_name}"',
            f'site:linkedin.com/in "{full_name}" {company_name}',
            f'"{full_name}" "{company_name}" site:linkedin.com',
            f'"{full_name}" {company_name} linkedin',
        ]

        # Handle sanitized variant
        if clean and clean != _normalize_name(full_name):
            variants += [
                f'site:linkedin.com/in "{clean}" "{company_name}"',
                f'site:linkedin.com/in "{clean}" {company_name}',
            ]

        if len(parts) >= 2:
            fn, ln = parts[0], parts[-1]
            variants += [
                f'site:linkedin.com/in "{fn} {ln}" "{company_name}"',
                f'site:linkedin.com/in {quote(fn)} {quote(ln)} "{company_name}"',
            ]

        # Try up to 3 pages for each variant
        for vi, q in enumerate(_dedupe_preserve_order(variants), 1):
            for start in (1, 11, 21):
                logger.info(f"🔎 Seed search v{vi}, start={start}: {q}")
                items = self._search_page(q, start_index=start)
                for it in items or []:
                    emp = self._extract_employee_from_result(it, company_name, skip_c_suite=False)
                    if not emp:
                        continue
                    if self._name_matches(emp.name, full_name):
                        emp.is_seed = True
                        logger.info(f"✅ SEED MATCH: {emp.name} ({emp.linkedin_url})")
                        return emp
                time.sleep(0.5)
        logger.warning(f"❌ Seed not found: {full_name}")
        return None

    # ---------- private ----------

    def _name_matches(self, found_name: str, target_name: str) -> bool:
        found = _name_token_set(found_name)
        target = _name_token_set(target_name)
        if not target:
            return False
        if found == target:
            return True
        if len(target) == 1:
            return len(found & target) >= 1
        return len(found & target) >= min(2, len(target))


    def _search_page(self, query: str, start_index: int) -> List[Dict[str, Any]]:
        """
        Fetch one CSE page for `query`, starting at `start_index`.
        Returns a list of result items with keys: title, link, snippet.
        Retries on transient errors and auto-recovers if the `fields` mask is rejected.
        """
        base_params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "start": start_index,
            "num": 10,
        }

        # Preferred minimal mask (faster, smaller payload)
        masked_params = dict(base_params)
        masked_params["fields"] = "items(title,link,snippet),searchInformation(totalResults)"

        def _do_request(params: dict) -> Optional[dict]:
            r = self.session.get(self.base_url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()

        # Try with mask; if a 400 mentions invalid field selection, retry once without mask
        for attempt in range(3):
            try:
                # attempt 0: with mask, attempt 1/2: with/without depending on earlier failure
                params = masked_params if attempt == 0 else base_params
                data = _do_request(params)

                # Sometimes API returns 200 with an "error" body; handle gracefully
                if isinstance(data, dict) and "error" in data:
                    # If it’s an invalid field selection and we still had a mask, retry without it
                    err_msg = str(data["error"])
                    if "Invalid field selection" in err_msg and "fields" in params:
                        logger.warning("CSE: fields mask rejected; retrying without mask.")
                        data = _do_request(base_params)
                        if "error" in data:
                            logger.warning(f"CSE error after unmasked retry: {data['error']}")
                            return []
                    else:
                        logger.warning(f"CSE error payload: {data['error']}")
                        return []

                items = (data or {}).get("items", [])
                if not items:
                    tot = ((data or {}).get("searchInformation") or {}).get("totalResults", "0")
                    logger.debug(f"No items for '{query}' (start={start_index}, totalResults={tot})")
                    return []
                return items

            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                body = getattr(e.response, "text", "") or ""
                # If mask was present and Google says invalid field selection, drop the mask and retry
                if status == 400 and "Invalid field selection" in body and attempt == 0:
                    logger.warning("CSE 400 Invalid field selection; retrying without mask.")
                    continue
                # Non-retriable 4xx except 429
                if status and status != 429 and 400 <= status < 500:
                    logger.warning(f"CSE HTTP {status} (non-retriable) for '{query}': {body[:200]}")
                    return []
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                logger.warning(f"CSE request failed (attempt {attempt+1}/3) for '{query}': {e}")
                time.sleep(0.5 * (attempt + 1))

        return []



    def _extract_employee_from_result(
        self,
        result: Dict[str, Any],
        company_name: str,
        skip_c_suite: bool = True
    ) -> Optional[RawEmployee]:
        try:
            link = result.get("link", "")
            if "linkedin.com/in/" not in (link or ""):
                return None

            title = result.get("title", "") or ""
            name = self._extract_name_from_title(title)
            if not name:
                return None

            # C-suite filter only for bulk discovery (not for seed lookups)
            if skip_c_suite:
                t = title.upper()
                if any(x in t for x in (" CEO ", "CFO", "CTO", "COO", "CMO", "CIO", "CHRO", "CXO", "CPO ", " CDO ")):
                    return None

            snippet = self._clean_snippet(result.get("snippet", "") or "")
            return RawEmployee(name=name, linkedin_url=link, snippet=snippet, company=company_name)
        except Exception as e:
            logger.debug(f"extract_employee error: {e}")
            return None

    def _extract_name_from_title(self, title: str) -> str:
        if not title:
            return ""
        # Common format: "John Doe - Role at Company | LinkedIn"
        parts = re.split(r"[|\-–—]", title)
        if parts:
            candidate = parts[0].strip()
            if candidate and "linkedin" not in candidate.lower():
                return candidate
        return title.replace("LinkedIn", "").strip()

    def _clean_snippet(self, s: str) -> str:
        if not s:
            return ""
        junk = [
            "LinkedIn is the world's largest professional network",
            "View the profiles of professionals named",
            "View the profiles of people named",
            "There are ",
            " professionals named",
            "on LinkedIn."
        ]
        for j in junk:
            s = s.replace(j, "")
        return re.sub(r"\s+", " ", s).strip()

# =============================================================================
# Bright Data Scraper
# =============================================================================

class BrightDataScraper:
    """
    Simple Bright Data dataset trigger + poll + fetch.
    You can skip this if you don't use Bright Data (the rest still works).
    """

    def __init__(self):
        if HARDCODE:
            self.api_key = BRIGHT_DATA_API_KEY_HARDCODE
            self.dataset_id = BRIGHT_DATA_DATASET_ID_HARDCODE
        else:
            self.api_key = os.getenv("BRIGHT_DATA_API_KEY", BRIGHT_DATA_API_KEY_HARDCODE if HARDCODE else "")
            self.dataset_id = os.getenv("BRIGHT_DATA_DATASET_ID", BRIGHT_DATA_DATASET_ID_HARDCODE if HARDCODE else "")
        self.trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={self.dataset_id}&include_errors=true"
        self.status_url = "https://api.brightdata.com/datasets/v3/progress/"
        self.result_url = "https://api.brightdata.com/datasets/v3/snapshot/"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def is_configured(self) -> bool:
        return bool(self.api_key and self.dataset_id)

    def scrape_profiles_one_shot(self, urls: List[str], timeout_sec: int = 1800) -> List[Dict[str, Any]]:
        if not self.is_configured():
            logger.info("Bright Data not configured; skipping profile scrape.")
            return []
        urls = [u for u in urls if u]
        urls = _dedupe_preserve_order(urls)[:100]
        if not urls:
            return []

        snap = self._trigger(urls)
        if not snap:
            logger.error("Failed to trigger Bright Data scrape.")
            return []
        ok = self._wait_ready(snap, timeout=timeout_sec, interval=10)
        if not ok:
            logger.error("Bright Data scrape did not complete in time.")
            return []
        return self._fetch(snap)

    def _trigger(self, urls: List[str]) -> Optional[str]:
        try:
            payload = [{"url": u} for u in urls]
            r = requests.post(self.trigger_url, headers=self.headers, json=payload, timeout=30)
            logger.info(f"BrightData trigger status: {r.status_code}")
            if r.ok:
                js = r.json()
                return js.get("snapshot_id") or js.get("snapshot") or js.get("id")
            logger.error(f"Trigger error: {r.text}")
        except Exception as e:
            logger.error(f"Trigger exception: {e}")
        return None

    def _wait_ready(self, snap: str, timeout: int, interval: int) -> bool:
        logger.info(f"Waiting for Bright Data snapshot {snap} ...")
        elapsed = 0
        while elapsed <= timeout:
            try:
                r = requests.get(self.status_url + snap, headers=self.headers, timeout=15)
                if r.ok:
                    js = r.json()
                    st = (js.get("status") or js.get("state") or "").lower()
                    logger.info(f"  {elapsed}s: status={st}")
                    if st == "ready":
                        return True
                    if st == "error":
                        logger.error(f"Bright Data error: {js}")
                        return False
            except Exception as e:
                logger.warning(f"Status check error: {e}")
            time.sleep(interval)
            elapsed += interval
        return False

    def _fetch(self, snap: str) -> List[Dict[str, Any]]:
        try:
            r = requests.get(self.result_url + snap, headers=self.headers, timeout=120)
            if not r.ok:
                logger.error(f"Snapshot fetch failed: {r.status_code} {r.text}")
                return []
            data = []
            for line in r.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
            logger.info(f"Fetched {len(data)} profiles from Bright Data snapshot")
            return data
        except Exception as e:
            logger.error(f"Fetch exception: {e}")
            return []

# =============================================================================
# Company Research
# =============================================================================

class CompanyReportGenerator:
    """
    Quick company intel via:
      - Google CSE to gather a handful of URLs
      - aiohttp fetch + BeautifulSoup cleanup
      - simple pattern extraction (revenue, headcount, HQ, founded)
      - optional AI (Azure OpenAI) summary if configured
    """

    def __init__(self):
        if HARDCODE:
            self.azure_api_key = AZURE_OPENAI_API_KEY_HARDCODE
            self.azure_endpoint = AZURE_OPENAI_ENDPOINT_HARDCODE
            self.azure_deployment_id = AZURE_OPENAI_DEPLOYMENT_ID_HARDCODE
            self.azure_api_version = AZURE_OPENAI_API_VERSION_HARDCODE
            self.google_api_key = GOOGLE_API_KEY_HARDCODE
            self.google_cx = GOOGLE_CSE_ID_HARDCODE
        else:
            self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            self.azure_deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")
            self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
            self.google_cx = os.getenv("GOOGLE_CSE_ID", "")

        self.max_links_per_query = 5
        self.request_timeout = 20
        self.delay_between_requests = 0.5
        self.max_concurrent_requests = 6
        self.ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )

    def _search_queries(self, company_name: str) -> List[str]:
        return [
            f'"{company_name}" official website',
            f'"{company_name}" press release',
            f'"{company_name}" about',
            f'"{company_name}" headquarters location',
            f'"{company_name}" revenue ARR',
            f'"{company_name}" funding crunchbase',
            f'"{company_name}" LinkedIn',
        ]

    async def google_search(self, q: str) -> List[str]:
        if not (self.google_api_key and self.google_cx):
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"q": q, "key": self.google_api_key, "cx": self.google_cx, "num": self.max_links_per_query}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, params=params, timeout=self.request_timeout) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    items = data.get("items", [])[: self.max_links_per_query]
                    return [it.get("link", "") for it in items if it.get("link")]
        except Exception:
            return []

    async def _fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            headers = {"User-Agent": self.ua}
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, headers=headers, timeout=self.request_timeout) as resp:
                    if resp.status != 200:
                        return None
                    try:
                        html = await resp.text()
                    except Exception:
                        return None
            soup = BeautifulSoup(html, "html.parser")
            # strip non-content
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            meta = soup.find("meta", attrs={"name": "description"})
            desc = (meta.get("content", "").strip() if meta else "")
            # content
            content = ""
            for css in ["main", "article", ".content", ".post-content", "#content", ".main-content", "body"]:
                node = soup.select_one(css)
                if node:
                    content = node.get_text(" ", strip=True)
                    break
            if not content:
                content = soup.get_text(" ", strip=True)
            content = re.sub(r"\s+", " ", content)
            content = re.sub(r"[^\w\s.,;:!?()\-–—/]", " ", content)
            return {
                "url": url,
                "title": title,
                "description": desc,
                "content": content[:15000],
                "social_links": self._extract_social(soup),
                "company_info": self._extract_info(content)
            }
        except Exception:
            return None

    def _extract_social(self, soup: BeautifulSoup) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            if "linkedin.com" in href and "linkedin" not in out:
                out["linkedin"] = a["href"]
            elif ("twitter.com" in href or "x.com" in href) and "twitter" not in out:
                out["twitter"] = a["href"]
            elif "facebook.com" in href and "facebook" not in out:
                out["facebook"] = a["href"]
            elif "instagram.com" in href and "instagram" not in out:
                out["instagram"] = a["href"]
            elif "youtube.com" in href and "youtube" not in out:
                out["youtube"] = a["href"]
            elif "crunchbase.com" in href and "crunchbase" not in out:
                out["crunchbase"] = a["href"]
            elif "github.com" in href and "github" not in out:
                out["github"] = a["href"]
        return out

    def _extract_info(self, content: str) -> Dict[str, str]:
        info: Dict[str, str] = {}
        # revenue patterns
        rev = re.search(r"\$\s?(\d+(?:\.\d+)?)\s*(billion|million|b|m)\b.*?(revenue|arr|sales)", content, re.I)
        if rev:
            info["revenue_estimate"] = rev.group(0)
        # headcount
        emp = re.search(r"(\d{2,3}(?:,\d{3})?)\s*(employees|people|team)", content, re.I)
        if emp:
            info["employee_estimate"] = emp.group(0)
        # founded year
        fy = re.search(r"(founded|established|launched)\s+(in\s+)?(19|20)\d{2}", content, re.I)
        if fy:
            info["founded_year"] = re.search(r"(19|20)\d{2}", fy.group(0)).group(0)
        # headquarters
        hq = re.search(r"(headquarters|based)\s+(in|at)\s+([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)?)", content, re.I)
        if hq:
            info["headquarters"] = hq.group(3)
        return info

    async def generate_company_report(self, company_name: str) -> CompanyData:
        logger.info(f"🏢 Company research for: {company_name}")
        urls: List[str] = []
        # gather URLs via several queries
        tasks = [asyncio.create_task(self.google_search(q)) for q in self._search_queries(company_name)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                urls.extend(r)
        urls = _dedupe_preserve_order(urls)[:20]

        if not urls:
            return CompanyData(name=company_name)

        # fetch in parallel limited
        sem = asyncio.Semaphore(self.max_concurrent_requests)
        pages: List[Dict[str, Any]] = []

        async def worker(u: str):
            async with sem:
                data = await self._fetch_page(u)
                await asyncio.sleep(self.delay_between_requests)
                if data:
                    pages.append(data)

        await asyncio.gather(*[asyncio.create_task(worker(u)) for u in urls])
        if not pages:
            return CompanyData(name=company_name)

        cd = CompanyData(
            name=company_name,
            website=pages[0]["url"] if pages else "",
        )
        # merge info
        for p in pages:
            # social
            for k, v in (p.get("social_links") or {}).items():
                cd.social_links.setdefault(k, v)
            # info bits (first come basis)
            info = p.get("company_info") or {}
            if info.get("revenue_estimate") and not cd.revenue_estimate: cd.revenue_estimate = info["revenue_estimate"]
            if info.get("employee_estimate") and not cd.employee_estimate: cd.employee_estimate = info["employee_estimate"]
            if info.get("founded_year") and not cd.founded_year: cd.founded_year = info["founded_year"]
            if info.get("headquarters") and not cd.headquarters: cd.headquarters = info["headquarters"]

        # default description
        cd.description = f"Automated web scan from {len(pages)} sources. Social: {', '.join(cd.social_links.keys()) or 'none'}."

        # optional AI summary
        if self.azure_api_key and self.azure_endpoint:
            try:
                chunk = "\n\n".join(
                    [f"Title: {p['title']}\nDesc: {p['description']}\nContent: {p['content'][:2000]}" for p in pages[:3]]
                )[:8000]
                prompt = f"Provide a concise factual profile of {company_name}: industry, products, business model, position, notable metrics.\n\nContext:\n{chunk}"
                ai_sum = await self._azure_summary(prompt)
                if ai_sum:
                    cd.description = ai_sum
                    cd.business_model = ai_sum
            except Exception as e:
                logger.warning(f"Azure summary failed: {e}")

        return cd

    async def _azure_summary(self, user_content: str) -> str:
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment_id}/chat/completions?api-version={self.azure_api_version}"
        headers = {"Content-Type": "application/json", "api-key": self.azure_api_key}
        payload = {
            "messages": [
                {"role": "system", "content": "You are a concise, factual business analyst."},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_tokens": 1200
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, headers=headers, json=payload, timeout=30) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.json()
                    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except Exception:
            return ""

# =============================================================================
# Comprehensive Data Writer
# =============================================================================
class ComprehensiveDataManager:
    """
    Save an all-in-one JSON for downstream agents (MIRAGE, CIPHER, etc.)
    FIXED: Properly matches and integrates Bright Data scraped profiles
    """

    def __init__(self, base_dir: str = "data"):
        self.data_dir = base_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _normalize_linkedin_url(self, url: str) -> str:
        """
        Normalize LinkedIn URL for matching by:
        - Converting to lowercase
        - Removing trailing slashes
        - Removing query parameters
        - Standardizing domain (in.linkedin.com -> linkedin.com)
        """
        if not url:
            return ""
        try:
            url = url.lower().strip()
            # Remove query params and fragments
            url = url.split('?')[0].split('#')[0]
            # Remove trailing slash
            url = url.rstrip('/')
            # Standardize domain
            url = url.replace('in.linkedin.com', 'linkedin.com')
            url = url.replace('www.linkedin.com', 'linkedin.com')
            return url
        except Exception:
            return url.lower()

    def _extract_slug_from_url(self, url: str) -> str:
        """Extract the profile slug from LinkedIn URL (e.g., 'pujasingh27' from any variant)"""
        try:
            url = url.lower().strip()
            # Match pattern: .../in/SLUG or .../in/SLUG/
            match = re.search(r'/in/([^/?#]+)', url)
            if match:
                return match.group(1).strip('-_/')
            return ""
        except Exception:
            return ""

    def _fuzzy_match_profile(self, employee: RawEmployee, profiles: List[Dict[str, Any]], 
                            norm_url: str) -> Optional[Dict[str, Any]]:
        """
        Fuzzy match profile by:
        1. URL slug matching
        2. Name similarity
        """
        employee_slug = self._extract_slug_from_url(employee.linkedin_url)
        employee_name_normalized = _normalize_name(employee.name)
        employee_name_tokens = _name_token_set(employee.name)
        
        logger.info(f"Fuzzy matching: slug='{employee_slug}', name='{employee_name_normalized}'")
        
        best_match = None
        best_score = 0
        
        for profile in profiles:
            score = 0
            
            # Check all possible URL fields
            for url_field in ['url', 'profile_url', 'linkedin_url', 'link']:
                profile_url = profile.get(url_field, '')
                if profile_url:
                    profile_slug = self._extract_slug_from_url(profile_url)
                    if profile_slug and employee_slug and profile_slug == employee_slug:
                        logger.info(f"✓ Slug match: {profile_slug}")
                        score += 100  # Strong match
                        break
            
            # Check name fields
            for name_field in ['name', 'full_name', 'title']:
                profile_name = profile.get(name_field, '')
                if profile_name:
                    profile_name_tokens = _name_token_set(str(profile_name))
                    if profile_name_tokens and employee_name_tokens:
                        common_tokens = profile_name_tokens & employee_name_tokens
                        if common_tokens:
                            token_score = len(common_tokens) / max(len(profile_name_tokens), len(employee_name_tokens))
                            score += token_score * 50
                            logger.info(f"✓ Name overlap: {common_tokens} (score +{token_score * 50:.1f})")
            
            if score > best_score:
                best_score = score
                best_match = profile
        
        # Require minimum score for match
        if best_score >= 50:  # Either slug match or strong name overlap
            logger.info(f"✅ Fuzzy match found with score {best_score:.1f}")
            return best_match
        
        logger.info(f"❌ No fuzzy match (best score: {best_score:.1f})")
        return None

    def save_comprehensive_data(
        self,
        company_name: str,
        raw_employees: List[RawEmployee],
        detailed_profiles: Optional[List[Dict[str, Any]]] = None,
        company_data: Optional[CompanyData] = None,
    ) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", company_name.lower())
        fn = f"{safe}_complete_intelligence_report.json"
        fp = os.path.join(self.data_dir, fn)

        # Build normalized URL mapping for Bright Data profiles
        detail_map: Dict[str, Dict[str, Any]] = {}
        profile_list: List[Dict[str, Any]] = []  # Keep all profiles for fuzzy matching
        
        if detailed_profiles:
            logger.info(f"Processing {len(detailed_profiles)} Bright Data profiles...")
            logger.info(f"Raw profile data: {json.dumps(detailed_profiles, indent=2)[:500]}...")  # Debug: show structure
            
            for p in detailed_profiles:
                if not p:
                    continue
                
                profile_list.append(p)
                
                # Try multiple possible URL fields
                url = (p.get("url") or 
                       p.get("profile_url") or 
                       p.get("linkedin_url") or 
                       p.get("link") or "")
                
                if url and "linkedin.com" in url:
                    norm_url = self._normalize_linkedin_url(url)
                    detail_map[norm_url] = p
                    logger.info(f"Mapped profile URL: {norm_url}")

        employees_block: List[Dict[str, Any]] = []
        scraped_count = 0
        matched_count = 0

        for e in raw_employees:
            # Normalize the employee's LinkedIn URL
            norm_employee_url = self._normalize_linkedin_url(e.linkedin_url)
            logger.info(f"Looking for match: {e.name} -> {norm_employee_url}")
            
            # Try exact URL match first
            detail = detail_map.get(norm_employee_url)
            
            # If no exact match, try fuzzy matching by name and URL slug
            if not detail and profile_list:
                detail = self._fuzzy_match_profile(e, profile_list, norm_employee_url)
            
            if detail:
                scraped = True
                scraped_count += 1
                matched_count += 1
                logger.info(f"✅ Matched profile for: {e.name}")
                summary = self._summarize_profile(detail, fallback_name=e.name)
            else:
                scraped = False
                logger.warning(f"❌ No profile match for: {e.name} ({norm_employee_url})")
                logger.warning(f"Available normalized URLs: {list(detail_map.keys())}")
                summary = {
                    "full_name": e.name,
                    "current_position": "Not available",
                    "location": "Not available",
                    "experience_years": 0,
                    "skills_count": 0,
                    "education_count": 0,
                    "connections": "Not available",
                }

            employees_block.append({
                "basic_info": {
                    "name": e.name,
                    "linkedin_url": e.linkedin_url,
                    "company": e.company,
                    "search_snippet": e.snippet,
                    "is_seed": e.is_seed,
                },
                "detailed_profile": detail if detail else None,
                "data_status": {
                    "found_in_search": True,
                    "detailed_scraped": scraped,
                    "scraping_error": None if scraped else (
                        "Profile scraped but not matched" if detailed_profiles and not detail 
                        else "Scrape not attempted" if not detailed_profiles 
                        else "Not scraped or not found"
                    )
                },
                "summary": summary
            })

        total = len(raw_employees)
        success_rate = round((scraped_count / total * 100), 2) if total else 0.0

        logger.info(f"📊 Profile matching: {matched_count}/{total} matched, success rate: {success_rate}%")

        data: Dict[str, Any] = {
            "Spectre_company": {
                "company_name": company_name,
                "report_type": "Complete Intelligence Report",
                "generation_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "data_sources": _dedupe_preserve_order(
                    ["Google Custom Search Engine"]
                    + (["Bright Data LinkedIn Scraper"] if detailed_profiles else [])
                    + (["Company Web Research"] if company_data else [])
                ),
                "report_sections": ["company_intelligence", "employee_intelligence", "analytics"]
            },
            "report_metadata": {"company_name": company_name, "generated_at": _now_iso()},
            "company_intelligence": self._company_block(company_name, company_data),
            "employee_intelligence": {
                "summary": {
                    "total_employees_found": total,
                    "detailed_profiles_scraped": scraped_count,
                    "scraping_success_rate": success_rate,
                    "scraping_completed": bool(detailed_profiles),
                },
                "employees": employees_block
            },
            "analytics": self._analytics(employees_block, company_data),
            "executive_summary": self._exec_summary(company_name, employees_block, company_data, success_rate)
        }

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"📁 Report saved: {fp}")
        return fp

    def _summarize_profile(self, detail: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
        """Extract key info from Bright Data profile"""
        name = (detail.get("name") or 
                detail.get("full_name") or 
                detail.get("title") or 
                fallback_name or "").strip()
        
        location = str(detail.get("location") or 
                      detail.get("geo") or "").strip()
        
        connections = str(detail.get("connections") or 
                         detail.get("connection_count") or "").strip()
        
        # Experience analysis
        experiences = (detail.get("experience") or 
                      detail.get("experiences") or 
                      detail.get("positions") or [])
        
        current_title = ""
        total_months = 0
        
        try:
            if isinstance(experiences, list) and experiences:
                first = experiences[0] or {}
                current_title = str(first.get("title") or 
                                  first.get("position") or "").strip()
                
                for exp in experiences:
                    if not exp:
                        continue
                    dur = str(exp.get("duration") or 
                             exp.get("time_period") or "")
                    yrs = re.findall(r"(\d+)\s*yr", dur)
                    mos = re.findall(r"(\d+)\s*mo", dur)
                    months = (int(yrs[0]) * 12 if yrs else 0) + (int(mos[0]) if mos else 0)
                    total_months += months
        except Exception as e:
            logger.debug(f"Experience parsing error: {e}")
        
        exp_years = round(total_months / 12, 1) if total_months > 0 else 0.0
        
        skills = (detail.get("skills") or 
                 detail.get("skill_list") or [])
        
        education = (detail.get("education") or 
                    detail.get("schools") or [])
        
        return {
            "full_name": name,
            "current_position": current_title or "Not available",
            "location": location or "Not available",
            "experience_years": exp_years,
            "skills_count": len(skills) if isinstance(skills, list) else 0,
            "education_count": len(education) if isinstance(education, list) else 0,
            "connections": connections or "Not available",
        }

    def _company_block(self, company_name: str, c: Optional[CompanyData]) -> Dict[str, Any]:
        if not c:
            return {"basic_info": {"name": company_name, "note": "Company research not available"}}
        return {
            "basic_info": {
                "name": c.name, "website": c.website, "industry": c.industry,
                "headquarters": c.headquarters, "founded_year": c.founded_year,
                "employee_estimate": c.employee_estimate, "revenue_estimate": c.revenue_estimate
            },
            "business_analysis": {
                "description": c.description, "business_model": c.business_model,
                "key_products": c.key_products, "market_position": c.market_position
            },
            "digital_presence": {"social_links": c.social_links, "tech_stack": c.tech_stack},
            "financial_data": c.financial_data, "recent_news": c.recent_news
        }

    def _analytics(self, employees_block: List[Dict[str, Any]], company_data: Optional[CompanyData]) -> Dict[str, Any]:
        from collections import Counter
        positions, skills, locs = [], [], []
        exp_bins = {"0-2_years": 0, "3-5_years": 0, "6-10_years": 0, "11+_years": 0}
        scraped_count = 0
        
        for e in employees_block:
            st = e.get("data_status", {})
            if st.get("detailed_scraped"):
                scraped_count += 1
                s = e.get("summary") or {}
                p = (s.get("current_position") or "").strip()
                l = (s.get("location") or "").strip()
                if p and p != "Not available": 
                    positions.append(p)
                if l and l != "Not available": 
                    locs.append(l)
                
                ey = s.get("experience_years") or 0
                try:
                    y = float(ey)
                    if y <= 2: exp_bins["0-2_years"] += 1
                    elif y <= 5: exp_bins["3-5_years"] += 1
                    elif y <= 10: exp_bins["6-10_years"] += 1
                    else: exp_bins["11+_years"] += 1
                except Exception:
                    pass
                
                detail = e.get("detailed_profile") or {}
                for sk in (detail.get("skills") or []):
                    if isinstance(sk, dict):
                        nm = sk.get("name")
                        if nm: skills.append(nm)
                    elif isinstance(sk, str):
                        skills.append(sk)

        emp_tot = len(employees_block)
        pos_top = [{"position": p, "count": c} for p, c in Counter(positions).most_common(10)]
        skl_top = [{"skill": s, "count": c} for s, c in Counter(skills).most_common(20)]
        loc_top = [{"location": l, "count": c} for l, c in Counter(locs).most_common(10)]

        analytics = {
            "employee_analytics": {
                "totals": {
                    "employees_found": emp_tot,
                    "profiles_scraped": scraped_count,
                    "scraping_success_rate": round((scraped_count / emp_tot * 100), 2) if emp_tot else 0.0
                },
                "top_positions": pos_top,
                "top_skills": skl_top,
                "top_locations": loc_top,
                "experience_distribution": exp_bins
            },
            "company_analytics": {},
            "data_quality": {"sources_analyzed": 0, "confidence_score": 0, "completeness_score": 0}
        }

        # Company analytics
        completeness = 0.0
        comp_weight = 40.0
        emp_weight = 60.0

        if company_data:
            analytics["company_analytics"] = {
                "web_presence": {
                    "social_platforms": len(company_data.social_links), 
                    "platforms": list(company_data.social_links.keys())
                },
                "data_richness": {
                    "has_revenue_data": bool(company_data.revenue_estimate),
                    "has_employee_estimate": bool(company_data.employee_estimate),
                    "has_founding_info": bool(company_data.founded_year),
                    "has_location_info": bool(company_data.headquarters),
                    "has_business_description": bool(company_data.description),
                }
            }
            fields = [
                company_data.description, company_data.industry, company_data.headquarters, 
                company_data.founded_year, company_data.employee_estimate, company_data.revenue_estimate, 
                company_data.social_links, company_data.business_model
            ]
            filled = sum(1 for f in fields if f)
            completeness += (filled / len(fields) * comp_weight)

        if emp_tot:
            completeness += ((scraped_count / emp_tot) * emp_weight)

        analytics["data_quality"]["completeness_score"] = round(completeness, 2)
        analytics["data_quality"]["confidence_score"] = analytics["employee_analytics"]["totals"]["scraping_success_rate"]
        return analytics

    def _exec_summary(self, company_name: str, employees_block: List[Dict[str, Any]], 
                     company_data: Optional[CompanyData], success_rate: float) -> str:
        top_pos = ""
        try:
            top_pos = (self._analytics(employees_block, company_data)["employee_analytics"]["top_positions"] or [{}])[0].get("position", "")
        except Exception:
            pass
        
        basics = []
        if company_data:
            if company_data.industry: basics.append(f"industry: {company_data.industry}")
            if company_data.headquarters: basics.append(f"HQ: {company_data.headquarters}")
            if company_data.founded_year: basics.append(f"founded: {company_data.founded_year}")
        
        bits = [
            f"Company Overview — {company_name}",
            ("; ".join(basics) if basics else "No basic company data extracted."),
            f"Employees: {len(employees_block)} found; detailed profiles success: {success_rate:.2f}%.",
            (f"Most common position: {top_pos}." if top_pos else "")
        ]
        return " ".join([b for b in bits if b]).strip()

# =============================================================================
# Orchestrator-style adapters and main pipeline
# =============================================================================

async def _gather_intel_improved(
    company_name: str,
    max_employees: int,
    seed_names: Optional[List[str]] = None,
    seed_urls: Optional[List[str]] = None,
) -> Tuple[List[RawEmployee], Optional[CompanyData]]:
    """
    Behavior:
      1) If LinkedIn seed URLs are provided, use them as-is (up to cap).
      2) Only if we still have headroom, resolve seed NAMES.
      3) Only if we still have headroom after (1)+(2), do bulk discovery.
      4) Always run company research in parallel; skip unnecessary searches.
    """
    logger.info(f"🚀 Gathering intelligence for {company_name} (cap={max_employees})")
    finder = GoogleCSEEmployeeFinder()
    researcher = CompanyReportGenerator()

    # Normalize inputs
    seed_names = [s.strip() for s in (seed_names or []) if s and s.strip()]
    seed_urls  = [u.strip() for u in (seed_urls or []) if u and u.strip()]

    def _norm_li_url(u: str) -> str:
        try:
            # remove query/fragment, trim trailing slash, normalize locale hosts
            u = u.split("?", 1)[0].split("#", 1)[0].rstrip("/")
            # keep both in.linkedin.com and linkedin.com as-is (they identify the same profile); no forced rewrite
            return u
        except Exception:
            return u

    # 1) Seed URLs → use directly, no searching
    url_seeds: List[RawEmployee] = []
    for u in seed_urls:
        if "linkedin.com/in/" in u:
            url = _norm_li_url(u)
            url_seeds.append(RawEmployee(
                name=_guess_name_from_linkedin_url(url),
                linkedin_url=url,
                snippet="Provided by user seed (URL).",
                company=company_name,
                is_seed=True
            ))

    # De-dupe seed URLs (first occurrence wins)
    seen_urls = set()
    deduped_url_seeds: List[RawEmployee] = []
    for e in url_seeds:
        if e.linkedin_url and e.linkedin_url not in seen_urls:
            seen_urls.add(e.linkedin_url)
            deduped_url_seeds.append(e)
    url_seeds = deduped_url_seeds

    # If seed URLs already satisfy the cap, skip all search work
    if len(url_seeds) >= max_employees:
        logger.info(f"🌱 Using {max_employees}/{len(url_seeds)} provided LinkedIn URLs; skipping seed-name lookup & discovery.")
        company_data = await researcher.generate_company_report(company_name)
        # Cap to requested size
        return url_seeds[:max_employees], company_data

    # Otherwise, we may still need more
    remaining_needed = max(0, int(max_employees) - len(url_seeds))
    logger.info(f"🌱 Seed URLs: {len(url_seeds)}; still need: {remaining_needed}")

    # 2) Resolve seed names ONLY if we still have headroom
    name_hits: List[RawEmployee] = []
    if remaining_needed > 0 and seed_names:
        async def find_by_name(nm: str):
            return await asyncio.to_thread(finder.find_employee_by_name_improved, company_name, nm)

        hits = await asyncio.gather(*[asyncio.create_task(find_by_name(n)) for n in seed_names], return_exceptions=True)
        for h in hits:
            if isinstance(h, RawEmployee) and h.linkedin_url and ("linkedin.com/in/" in h.linkedin_url):
                if h.linkedin_url not in seen_urls:  # prefer URL seeds, avoid dupes
                    h.is_seed = True
                    name_hits.append(h)
                    seen_urls.add(h.linkedin_url)

        # Cap name hits to remaining slots
        if len(name_hits) > remaining_needed:
            name_hits = name_hits[:remaining_needed]

    # Update remaining after name hits
    remaining_needed = max(0, int(max_employees) - (len(url_seeds) + len(name_hits)))

    # 3) Do discovery ONLY if we still need more
    discovered: List[RawEmployee] = []
    company_task = asyncio.create_task(researcher.generate_company_report(company_name))

    if remaining_needed > 0:
        disc = await asyncio.to_thread(finder.find_employees, company_name, remaining_needed)
        for e in disc or []:
            if e.linkedin_url and e.linkedin_url not in seen_urls:
                discovered.append(e)
                seen_urls.add(e.linkedin_url)
            if len(discovered) >= remaining_needed:
                break

    # 4) Merge (priority: URL seeds > name seeds > discovered), cap to max
    all_emps = (url_seeds + name_hits + discovered)[:max_employees]
    company_data = await company_task

    logger.info(f"✅ Prepared {len(all_emps)} employees (target={max_employees})")
    return all_emps, company_data

async def _run_async(context: Dict[str, Any]) -> Dict[str, Any]:
    company_name = (
        context.get("company_name")
        or context.get("spectre_company")
        or context.get("target_company")
    )
    if not company_name:
        raise ValueError("SHADE.run: 'company_name' is required")

    max_employees = _safe_int(
        context.get("spectre_n") or context.get("limit") or context.get("max_employees") or 50,
        default=50
    )

    run_dir = context.get("run_dir") or os.path.join(
        os.getcwd(), "runs", f"{_slug(company_name)}-{int(time.time())}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # 1) Discover employees (seed-aware) + company research
    seed_names = context.get("seed_employee_names") or []
    seed_urls  = context.get("seed_employee_urls") or []

    employees, company_data = await _gather_intel_improved(
        company_name=company_name,
        max_employees=max_employees,
        seed_names=seed_names,
        seed_urls=seed_urls
    )

    # 2) Optional Bright Data scrape
    scraper = BrightDataScraper()
    detailed_profiles: List[Dict[str, Any]] = []
    if context.get("use_bright_data", True):
        detailed_profiles = await asyncio.to_thread(
            scraper.scrape_profiles_one_shot,
            [e.linkedin_url for e in employees]
        )

    # 3) Save comprehensive report (under run_dir/data)
    data_dir = os.path.join(run_dir, "data")
    writer = ComprehensiveDataManager(base_dir=data_dir)
    report_path = writer.save_comprehensive_data(company_name, employees, detailed_profiles, company_data)

    # Legacy copy in ./data
    try:
        os.makedirs("data", exist_ok=True)
        legacy = os.path.join("data", os.path.basename(report_path))
        if os.path.abspath(legacy) != os.path.abspath(report_path):
            with open(report_path, "r", encoding="utf-8") as src, open(legacy, "w", encoding="utf-8") as dst:
                dst.write(src.read())
    except Exception as e:
        logger.warning(f"Legacy copy failed: {e}")

    # Return context-like result
    context.update({
        "intelligence_report_path": report_path,
        "spectre_employees": [{
            "name": e.name,
            "linkedin_url": e.linkedin_url,
            "company": e.company,
            "snippet": e.snippet,
            "is_seed": e.is_seed
        } for e in employees],
        "company_data_available": bool(company_data),
        "shade_status": "ok",
    })
    return context

# ---- public adapters ----

def run(context: Dict[str, Any]) -> Dict[str, Any]:
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run_async(context))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"SHADE.run error: {e}", exc_info=True)
        out = dict(context or {})
        out["shade_status"] = f"error: {e}"
        return out

async def discover_company_employees_async(
    company: str,
    limit: int = 50,
    seed_names: Optional[List[str]] = None,
    seed_urls: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    employees, company_data = await _gather_intel_improved(company, limit, seed_names, seed_urls)
    emps = [{
        "name": e.name, "linkedin_url": e.linkedin_url, "company": e.company, "snippet": e.snippet, "is_seed": e.is_seed
    } for e in employees]
    comp = {"report_metadata": {"company_name": company, "generated_at": _now_iso()},
            "company_data": company_data.__dict__ if isinstance(company_data, CompanyData) else (company_data or {})}
    return emps, comp

def discover_company_employees(
    company: str,
    limit: int = 50,
    seed_names: Optional[List[str]] = None,
    seed_urls: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(discover_company_employees_async(company, limit, seed_names, seed_urls))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"discover_company_employees error: {e}", exc_info=True)
        return [], {"report_metadata": {"company_name": company}, "company_data": {}}

# =============================================================================
# CLI
# =============================================================================

def _prompt(prompt: str, default: Optional[str] = None) -> str:
    s = input(prompt).strip()
    if not s and default is not None:
        return default
    return s

def _split_csv(s: str) -> List[str]:
    if not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    print("\n" + "="*68)
    print(" SHADE — Company + Employee Discovery (Improved)")
    print("="*68)

    company = _prompt("Target company: ")
    if not company:
        print("Company is required.")
        return

    max_emp_s = _prompt("How many employees to gather? [default 50]: ", default="50")
    try:
        max_emps = int(max_emp_s)
    except Exception:
        max_emps = 50

    seed_names_raw = _prompt("Seed employee NAMES (comma-separated, optional): ", default="")
    seed_urls_raw  = _prompt("Seed employee LinkedIn URLS (comma-separated, optional): ", default="")

    seed_names = _split_csv(seed_names_raw)
    seed_urls  = _split_csv(seed_urls_raw)

    ctx: Dict[str, Any] = {
        "company_name": company,
        "spectre_n": max_emps,
        "seed_employee_names": seed_names,
        "seed_employee_urls": seed_urls,
        # toggle Bright Data usage if you want to skip scraping
        "use_bright_data": True,
    }

    print("\nRunning…\n")
    out = run(ctx)

    print("\n✅ Done. Key outputs:")
    print(f" - intelligence_report_path: {out.get('intelligence_report_path')}")
    print(f" - employees found: {len(out.get('spectre_employees') or [])}")
    print(f" - company_data_available: {out.get('company_data_available')}")
    if out.get("spectre_employees"):
        print(" - first few employees:")
        for e in out["spectre_employees"][:10]:
            flag = "🌱" if e.get("is_seed") else "🔍"
            print(f"   • {flag} {e['name']} — {e['linkedin_url']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
