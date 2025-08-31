"""
Agent 1 — SHADE (Company + Employee Discovery)
------------------------------------------------
Cleaned, structured, and adapter-added version of your script.
Core logic preserved:
- Google CSE → LinkedIn employee discovery
- Optional Bright Data scrape (single-shot) of LinkedIn profiles
- Lightweight company web research with Google CSE + aiohttp + BeautifulSoup
- Comprehensive JSON report writer

Additions (non-breaking):
- `run(context) -> context` entrypoint for the 5-stage pipeline
- Async-safe runner that won’t collide with orchestrator loops
- BrightDataScraper now also exposes `scrape_profiles_in_batches(...)`
- Report now includes `report_metadata.company_name` for downstream compatibility
- Optional `base_dir` in ComprehensiveDataManager to write under a run folder
- Defensive fixes (removed stray prints, better error handling)

Expected context inputs (any subset):
- company_name: str (required)
- spectre_n: int (how many Spectre employees to discover; default 50)
- run_dir: str (folder where this agent can write; optional)

Context outputs added:
- intelligence_report_path: str (path to complete report JSON)
- spectre_employees: List[dict] (basic info for discovered employees)
- company_data_available: bool
- shade_status: str ("ok" or "error: ...")
"""

from __future__ import annotations

import json
import os
import time
import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random
import re
from urllib.parse import quote
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime

# ==== HARDCODED KEYS (edit here) ====
HARDCODE = True

GOOGLE_API_KEY = "AIzaSyBsa_JCmZy5cJANA3-ksT3sPvwYqhuUQ4s"
GOOGLE_CSE_ID  = "55d9d391fe2394876"

# Optional (only if you want to hardcode too)
BRIGHT_DATA_API_KEY = "8bda8a8ccf119c9ee2bf9d16591fb28cf591c7d3d7e382aec56ff567e7743da4"
BRIGHT_DATA_DATASET_ID = "gd_l1viktl72bvl7bjuj0"
# ====================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------------------------
# Data models
# -------------------------
@dataclass
class RawEmployee:
    """Data class for raw employee information from Google Search"""
    name: str
    linkedin_url: str
    snippet: str
    company: str


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
        if self.tech_stack is None:
            self.tech_stack = []
        if self.social_links is None:
            self.social_links = {}
        if self.recent_news is None:
            self.recent_news = []
        if self.financial_data is None:
            self.financial_data = {}


# -------------------------
# Google CSE Employee Finder
# -------------------------
class GoogleCSEEmployeeFinder:
    """Google Custom Search Engine integration for finding LinkedIn employee profiles"""

    def __init__(self):
        if HARDCODE:
            self.api_key = GOOGLE_API_KEY
            self.cse_id  = GOOGLE_CSE_ID
        else:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.cse_id  = os.getenv("GOOGLE_CSE_ID")

        self.base_url = "https://www.googleapis.com/customsearch/v1"

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        if not self.cse_id:
            raise ValueError("GOOGLE_CSE_ID is required")

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        })

    def find_employees(self, company_name: str, max_results: int = 50) -> List[RawEmployee]:

        """
        Find employees using Google Custom Search Engine
        """
        logger.info(f"🔍 Starting Google CSE search for employees of: {company_name}")
        logger.info(f"📊 Target: {max_results} employee profiles")

        search_query = f'site:linkedin.com/in "{company_name}"'
        logger.info(f"🔎 Search query: {search_query}")

        employees: List[RawEmployee] = []
        start_index = 1
        results_per_page = 10
        max_pages = min(10, (max_results + results_per_page - 1) // results_per_page)

        for page in range(max_pages):
            try:
                logger.info(
                    f"📄 Fetching page {page + 1}/{max_pages} "
                    f"(results {start_index}-{min(start_index + 9, max_results)})"
                )
                page_results = self._search_page(search_query, start_index)
                if not page_results:
                    logger.warning(f"No results returned for page {page + 1}")
                    break

                for result in page_results:
                    if len(employees) >= max_results:
                        break
                    employee = self._extract_employee_from_result(result, company_name)
                    if employee:
                        employees.append(employee)
                        logger.info(f"✅ Found: {employee.name} - {employee.snippet}")

                if len(employees) >= max_results:
                    break

                delay = random.uniform(2, 5)
                logger.info(f"⏱️ Rate limiting: waiting {delay:.1f} seconds...")
                time.sleep(delay)
                start_index += results_per_page

            except Exception as e:
                logger.error(f"Error fetching page {page + 1}: {e}")
                continue

        logger.info(f"🎯 Successfully found {len(employees)} employees for {company_name}")
        return employees

    def _search_page(self, query: str, start_index: int) -> List[Dict[str, Any]]:
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'start': start_index,
            'num': 10,
            'fields': 'items(title,link,snippet)'
        }
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'items' not in data:
                logger.warning("No 'items' field in API response")
                return []
            return data['items']
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []

    def _extract_employee_from_result(self, result: Dict[str, Any], company_name: str) -> Optional[RawEmployee]:
        try:
            linkedin_url = result.get('link', '')
            if not linkedin_url or 'linkedin.com/in/' not in linkedin_url:
                return None
            title = result.get('title', '')
            name = self._extract_name_from_title(title)
            if not name:
                return None
            c_suite_titles = ['CEO', 'CFO', 'CTO', 'COO', 'CMO', 'CIO', 'CHRO', 'CXO', 'CPO', 'CDO']
            if any(title.upper().startswith(cs_title) for cs_title in c_suite_titles):
                return None
            snippet = self._clean_snippet(result.get('snippet', ''))
            return RawEmployee(name=name, linkedin_url=linkedin_url, snippet=snippet, company=company_name)
        except Exception as e:
            logger.warning(f"Error extracting employee from result: {e}")
            return None

    def _extract_name_from_title(self, title: str) -> str:
        if not title:
            return ""
        separators = ['|', '-', '–', '—']
        for separator in separators:
            if separator in title:
                parts = title.split(separator)
                name_part = parts[0].strip()
                if len(name_part) >= 2 and not any(w in name_part.lower() for w in ['linkedin', 'profile', 'www.']):
                    return name_part
        clean_title = title.replace('LinkedIn', '').replace('- LinkedIn', '').strip()
        return clean_title if len(clean_title) >= 2 else ""

    def _clean_snippet(self, snippet: str) -> str:
        if not snippet:
            return ""
        unwanted = [
            'View the profiles of professionals named',
            'View the profiles of people named',
            'There are ',
            ' professionals named',
            'on LinkedIn.',
            "LinkedIn is the world's largest professional network",
        ]
        for phrase in unwanted:
            snippet = snippet.replace(phrase, '')
        return ' '.join(snippet.split()).strip()


# -------------------------
# Bright Data Scraper
# -------------------------
class BrightDataScraper:
    """Bright Data integration for scraping LinkedIn profiles"""

    def __init__(self):
        self.api_key = os.getenv("BRIGHT_DATA_API_KEY", BRIGHT_DATA_API_KEY)
        self.dataset_id = os.getenv("BRIGHT_DATA_DATASET_ID", BRIGHT_DATA_DATASET_ID)
        self.trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={self.dataset_id}&include_errors=true"
        self.status_url = "https://api.brightdata.com/datasets/v3/progress/"
        self.result_url = "https://api.brightdata.com/datasets/v3/snapshot/"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def scrape_profiles_one_shot(self, urls: List[str], timeout_sec: int = 100000) -> List[Dict[str, Any]]:
        urls = [u for u in urls if u]
        urls = list(dict.fromkeys(urls))[:100]
        if not urls:
            logger.warning("No URLs to scrape.")
            return []
        snapshot_id = self._trigger_scrape(urls)
        if not snapshot_id:
            logger.error("❌ Failed to trigger single-shot scrape")
            return []
        ok = self._wait_until_ready(snapshot_id, timeout=timeout_sec, interval=10)
        if not ok:
            logger.error("❌ Single-shot scrape failed or timed out")
            return []
        return self._fetch_results(snapshot_id, label="one-shot")

    # Compatibility for callers that expect batching
    def scrape_profiles_in_batches(self, urls: List[str], batch_size: int = 10, timeout_sec: int = 900) -> List[Dict[str, Any]]:
        urls = [u for u in urls if u]
        results: List[Dict[str, Any]] = []
        for i in range(0, len(urls), max(1, batch_size)):
            chunk = urls[i:i + max(1, batch_size)]
            try:
                chunk_data = self.scrape_profiles_one_shot(chunk, timeout_sec=timeout_sec)
                if chunk_data:
                    results.extend(chunk_data)
            except Exception as e:
                logger.warning(f"Batch scrape error on chunk {i//max(1,batch_size)}: {e}")
        # de-dupe by URL if present
        seen = set()
        deduped = []
        for p in results:
            url = (p or {}).get('url') or (p or {}).get('profile_url')
            key = url or json.dumps(p, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        return deduped

    def _trigger_scrape(self, urls: List[str]) -> Optional[str]:
        payload = [{"url": url} for url in urls]
        try:
            r = requests.post(self.trigger_url, headers=self.headers, json=payload, timeout=30)
            logger.info(f"🚀 Trigger response: {r.status_code}")
            if r.ok:
                js = r.json()
                return js.get("snapshot_id") or js.get("snapshot") or js.get("id")
            logger.error(f"Trigger error: {r.text}")
        except Exception as e:
            logger.error(f"Error triggering scrape: {e}")
        return None

    def _wait_until_ready(self, snapshot_id: str, timeout: int = 900, interval: int = 10) -> bool:
        logger.info(f"⏳ Waiting for snapshot {snapshot_id} to complete...")
        elapsed = 0
        while elapsed <= timeout:
            try:
                r = requests.get(self.status_url + snapshot_id, headers=self.headers, timeout=15)
                if r.ok:
                    js = r.json()
                    status = (js.get("status") or js.get("state") or "").lower()
                    logger.info(f"⏳ {elapsed}s - Status: {status}")
                    if status == "ready":
                        logger.info("✅ Snapshot ready!")
                        return True
                    if status == "error":
                        logger.error(f"❌ Snapshot error: {js}")
                        return False
                else:
                    logger.warning(f"Status check {r.status_code}: {r.text}")
            except Exception as e:
                logger.warning(f"Status check error: {e}")
            time.sleep(interval)
            elapsed += interval
        logger.error("❌ Timeout waiting for snapshot")
        return False

    def _fetch_results(self, snapshot_id: str, label: str = "") -> List[Dict[str, Any]]:
        url = self.result_url + snapshot_id
        try:
            r = requests.get(url, headers=self.headers, timeout=120)
            if r.ok:
                lines = [ln for ln in r.text.splitlines() if ln.strip()]
                data: List[Dict[str, Any]] = []
                for ln in lines:
                    try:
                        data.append(json.loads(ln))
                    except Exception:
                        pass
                logger.info(f"✅ Fetched {len(data)} profiles from {label or 'snapshot'}")
                return data
            logger.error(f"❌ Fetch results failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.error(f"Fetch error: {e}")
        return []


# -------------------------
# Company Research
# -------------------------
class CompanyReportGenerator:
    """Company research and analysis using web scraping and optional AI"""

    def __init__(self):
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.google_api_key = GOOGLE_API_KEY if HARDCODE else os.getenv("GOOGLE_API_KEY")
        self.google_cx      = GOOGLE_CSE_ID  if HARDCODE else os.getenv("GOOGLE_CSE_ID")

        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key or "",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
        }
        self.max_concurrent_requests = 5
        self.delay_between_requests = 1.0
        self.request_timeout = 15
        self.max_links_per_query = 5

    async def azure_chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000) -> str:
        if not (self.azure_api_key and self.azure_endpoint):
            logger.warning("Azure OpenAI not configured, skipping AI analysis")
            return ""
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment_id}/chat/completions?api-version={self.azure_api_version}"
        payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    logger.error(f"Azure API error: {response.status}")
        except Exception as e:
            logger.error(f"Azure API call failed: {e}")
        return ""

    def generate_search_queries(self, company_name: str) -> List[str]:
        return [
            f'"{company_name}" official website about company',
            f'"{company_name}" company information revenue financial results',
            f'"{company_name}" funding investment crunchbase',
            f'"{company_name}" news latest updates press releases',
            f'"{company_name}" business model products services',
            f'"{company_name}" employees team size headquarters location',
            f'"{company_name}" technology stack engineering',
            f'"{company_name}" industry analysis market position',
        ]

    async def google_search(self, query: str) -> List[str]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'q': query, 'key': self.google_api_key, 'cx': self.google_cx, 'num': self.max_links_per_query}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=self.request_timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [item['link'] for item in data.get('items', [])[:self.max_links_per_query]]
                    logger.error(f"Google Search API error: {response.status}")
        except Exception as e:
            logger.error(f"Google search failed for query '{query}': {e}")
        return []

    async def fetch_and_clean_page(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': self.headers['User-Agent']}, timeout=self.request_timeout) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch {url}: Status {response.status}")
                        return None
                    # Some pages may be binary (e.g., PDF). Try text safely.
                    try:
                        html = await response.text()
                    except Exception:
                        logger.warning(f"Non-text or undecodable content at {url}; skipping")
                        return None
                    soup = BeautifulSoup(html, 'html.parser')
                    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                        element.decompose()
                    title = soup.title.string.strip() if soup.title and soup.title.string else ""
                    meta_desc = soup.find("meta", attrs={"name": "description"})
                    description = meta_desc.get('content', '').strip() if meta_desc else ""
                    content_selectors = ['main', 'article', '.content', '.post-content', '#content', '.main-content', 'body']
                    content_text = ""
                    for selector in content_selectors:
                        element = soup.select_one(selector)
                        if element:
                            content_text = element.get_text(strip=True)
                            break
                    if not content_text:
                        content_text = soup.get_text(strip=True)
                    content_text = re.sub(r'\s+', ' ', content_text)
                    content_text = re.sub(r'[^\w\s.,;:!?()-]', '', content_text)
                    return {
                        'url': url,
                        'title': title,
                        'description': description,
                        'content': content_text[:15000],
                        'social_links': self.extract_social_links(soup),
                        'company_info': self.extract_company_info(soup, content_text),
                    }
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
        return None

    def extract_social_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        social_links = {k: '' for k in ['linkedin', 'twitter', 'facebook', 'instagram', 'youtube', 'crunchbase', 'github']}
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if 'linkedin.com' in href:
                social_links['linkedin'] = link['href']
            elif 'twitter.com' in href or 'x.com' in href:
                social_links['twitter'] = link['href']
            elif 'facebook.com' in href:
                social_links['facebook'] = link['href']
            elif 'instagram.com' in href:
                social_links['instagram'] = link['href']
            elif 'youtube.com' in href:
                social_links['youtube'] = link['href']
            elif 'crunchbase.com' in href:
                social_links['crunchbase'] = link['href']
            elif 'github.com' in href:
                social_links['github'] = link['href']
        return {k: v for k, v in social_links.items() if v}

    def extract_company_info(self, soup: BeautifulSoup, content: str) -> Dict[str, str]:
        info: Dict[str, str] = {}
        revenue_patterns = [
            r'\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)\s*(?:in\s*)?(?:revenue|sales|ARR|MRR)',
            r'revenue\s*of\s*\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)',
            r'(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)\s*(?:in\s*)?revenue',
        ]
        for pattern in revenue_patterns:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                info['revenue_estimate'] = m.group(0)
                break
        employee_patterns = [
            r'(\d+(?:,\d+)?)\s*employees',
            r'team\s*of\s*(\d+(?:,\d+)?)',
            r'(\d+(?:,\d+)?)\s*people\s*(?:work|employed)',
            r'workforce\s*of\s*(\d+(?:,\d+)?)',
        ]
        for pattern in employee_patterns:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                info['employee_estimate'] = m.group(0)
                break
        year_pattern = r'(?:founded|established|started|launched)(?:\s+in)?\s*(\d{4})'
        m = re.search(year_pattern, content, re.IGNORECASE)
        if m:
            info['founded_year'] = m.group(1)
        hq_patterns = [
            r'headquarters\s*(?:in|at|located)?\s*([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
            r'based\s*(?:in|at)\s*([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
            r'located\s*(?:in|at)\s*([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
        ]
        for pattern in hq_patterns:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                info['headquarters'] = m.group(1)
                break
        return info

    async def execute_parallel_searches(self, company_name: str) -> List[str]:
        queries = self.generate_search_queries(company_name)
        logger.info(f"🔍 Executing {len(queries)} parallel searches for {company_name}")
        tasks = [asyncio.create_task(self.google_search(q)) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_urls: List[str] = []
        for result in results:
            if isinstance(result, list):
                all_urls.extend(result)
            else:
                logger.error(f"Search failed: {result}")
        unique_urls = list(dict.fromkeys(all_urls))
        logger.info(f"📊 Collected {len(unique_urls)} unique URLs from parallel searches")
        return unique_urls

    async def process_urls_parallel(self, urls: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"🔄 Processing {len(urls)} URLs in parallel")
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def worker(u: str):
            async with semaphore:
                result = await self.fetch_and_clean_page(u)
                await asyncio.sleep(self.delay_between_requests)
                return result

        tasks = [worker(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, dict) and r]
        logger.info(f"✅ Successfully processed {len(valid)}/{len(urls)} URLs")
        return valid

    async def generate_company_report(self, company_name: str) -> CompanyData:
        logger.info(f"🏢 Generating company report for: {company_name}")
        urls = await self.execute_parallel_searches(company_name)
        if not urls:
            logger.warning("No URLs found from search queries")
            return CompanyData(name=company_name, timestamp=datetime.utcnow().isoformat())
        page_data = await self.process_urls_parallel(urls)
        if not page_data:
            logger.warning("No valid page data extracted")
            return CompanyData(name=company_name, timestamp=datetime.utcnow().isoformat())
        company_data = CompanyData(name=company_name, website=page_data[0]['url'] if page_data else "", timestamp=datetime.utcnow().isoformat())
        all_content: List[str] = []
        for page in page_data:
            all_content.append(
                f"Title: {page['title']}\nDescription: {page['description']}\nContent: {page['content'][:5000]}"
            )
            company_data.social_links.update(page['social_links'])
            info = page['company_info']
            if info.get('revenue_estimate') and not company_data.revenue_estimate:
                company_data.revenue_estimate = info['revenue_estimate']
            if info.get('employee_estimate') and not company_data.employee_estimate:
                company_data.employee_estimate = info['employee_estimate']
            if info.get('founded_year') and not company_data.founded_year:
                company_data.founded_year = info['founded_year']
            if info.get('headquarters') and not company_data.headquarters:
                company_data.headquarters = info['headquarters']
        combined_content = '\n\n'.join(all_content[:3])
        company_data.description = f"Company research based on web data analysis. Found {len(page_data)} relevant sources."
        if self.azure_api_key and self.azure_endpoint:
            try:
                analysis_prompt = (
                    f"""
                    Analyze the following content about {company_name} and provide a concise business summary.
                    Include: business model, industry, key products/services, market position, and any financial info.
                    Content:\n{combined_content[:8000]}
                    """
                )
                messages = [
                    {"role": "system", "content": "You are a business analyst. Provide clear, factual analysis."},
                    {"role": "user", "content": analysis_prompt},
                ]
                ai_analysis = await self.azure_chat_completion(messages, temperature=0.3, max_tokens=2000)
                if ai_analysis:
                    company_data.description = ai_analysis
                    company_data.business_model = ai_analysis
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
        logger.info(f"✅ Company report generated for {company_name}")
        return company_data


# -------------------------
# Comprehensive Data Writer
# -------------------------
class ComprehensiveDataManager:
    """Handles saving company + employee data in a single JSON"""

    def __init__(self, base_dir: str = "data"):
        self.data_dir = base_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def save_comprehensive_data(
        self,
        company_name: str,
        raw_employees: List[RawEmployee],
        detailed_profiles: Optional[List[Dict[str, Any]]] = None,
        company_data: Optional[CompanyData] = None,
    ) -> str:
        safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name.lower())
        filename = f"{safe_company_name}_complete_intelligence_report.json"
        filepath = os.path.join(self.data_dir, filename)

        comprehensive_data: Dict[str, Any] = {
            "Spectre_company": {
                "company_name": company_name,
                "report_type": "Complete Intelligence Report",
                "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_sources": ["Google Custom Search Engine"],
                "report_sections": ["company_intelligence", "employee_intelligence", "analytics"],
            },
            # Downstream compatibility for MIRAGE
            "report_metadata": {
                "company_name": company_name,
                "generated_at": datetime.utcnow().isoformat(),
            },
            "company_intelligence": {},
            "employee_intelligence": {
                "summary": {
                    "total_employees_found": len(raw_employees),
                    "detailed_profiles_scraped": len(detailed_profiles) if detailed_profiles else 0,
                    "scraping_success_rate": 0,
                    "scraping_completed": bool(detailed_profiles),
                },
                "employees": [],
            },
            "analytics": {},
            "executive_summary": "",
        }

        # Source credits
        meta = comprehensive_data.setdefault("Spectre_company", {})
        sources = meta.setdefault("data_sources", [])
        if detailed_profiles:
            sources.append("Bright Data LinkedIn Scraper")
        if company_data:
            sources.append("Company Web Research")
            if company_data.financial_data:
                sources.append("AI Analysis")

        # Success rate
        if detailed_profiles and raw_employees:
            success_rate = len(detailed_profiles) / len(raw_employees) * 100
            comprehensive_data["employee_intelligence"]["summary"]["scraping_success_rate"] = round(success_rate, 2)

        # Company intelligence
        if company_data:
            comprehensive_data["company_intelligence"] = {
                "basic_info": {
                    "name": company_data.name,
                    "website": company_data.website,
                    "industry": company_data.industry,
                    "headquarters": company_data.headquarters,
                    "founded_year": company_data.founded_year,
                    "employee_estimate": company_data.employee_estimate,
                    "revenue_estimate": company_data.revenue_estimate,
                },
                "business_analysis": {
                    "description": company_data.description,
                    "business_model": company_data.business_model,
                    "key_products": company_data.key_products,
                    "market_position": company_data.market_position,
                },
                "digital_presence": {
                    "social_links": company_data.social_links,
                    "tech_stack": company_data.tech_stack,
                },
                "financial_data": company_data.financial_data,
                "recent_news": company_data.recent_news,
            }
        else:
            comprehensive_data["company_intelligence"] = {
                "basic_info": {"name": company_name, "note": "Company research not available"}
            }

        # Map detailed profiles by URL for fast join
        detailed_map: Dict[str, Dict[str, Any]] = {}
        if detailed_profiles:
            for profile in detailed_profiles:
                url = (profile or {}).get('url', '')
                if url:
                    detailed_map[url] = profile

        # Merge raw + detailed
        for raw in raw_employees:
            emp_rec: Dict[str, Any] = {
                "basic_info": {
                    "name": raw.name,
                    "linkedin_url": raw.linkedin_url,
                    "company": raw.company,
                    "search_snippet": raw.snippet,
                },
                "detailed_profile": None,
                "data_status": {"found_in_search": True, "detailed_scraped": False, "scraping_error": None},
                "summary": {},
            }
            if raw.linkedin_url in detailed_map:
                prof = detailed_map[raw.linkedin_url]
                emp_rec["detailed_profile"] = prof
                emp_rec["data_status"]["detailed_scraped"] = True
                emp_rec["summary"] = self._extract_profile_summary(prof)
            else:
                emp_rec["summary"] = {
                    "full_name": raw.name,
                    "current_position": "Not available",
                    "location": "Not available",
                    "experience_years": 0,
                    "skills_count": 0,
                    "education_count": 0,
                    "connections": "Not available",
                }
                if detailed_profiles:
                    emp_rec["data_status"]["scraping_error"] = "Profile not found in scraped data"
            comprehensive_data["employee_intelligence"]["employees"].append(emp_rec)

        # Analytics
        comprehensive_data["analytics"] = self._generate_comprehensive_analytics(
            comprehensive_data["employee_intelligence"]["employees"], company_data
        )

        # Executive summary
        comprehensive_data["executive_summary"] = self._generate_executive_summary(
            company_name,
            comprehensive_data["company_intelligence"],
            comprehensive_data["employee_intelligence"]["summary"],
            comprehensive_data["analytics"],
        )

        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        logger.info(f"📁 Comprehensive intelligence report saved to: {filepath}")
        return filepath

    def _extract_profile_summary(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "full_name": str(profile.get("name") or "").strip(),
            "current_position": "",
            "location": str(profile.get("location") or "").strip(),
            "experience_years": 0,
            "skills_count": 0,
            "education_count": 0,
            "connections": str(profile.get("connections") or "").strip(),
        }
        experiences = profile.get("experience", [])
        if isinstance(experiences, list) and experiences:
            current_exp = experiences[0] if experiences else {}
            summary["current_position"] = str(current_exp.get("title", "")).strip()
            total_months = 0
            for exp in experiences:
                duration = str(exp.get("duration", ""))
                years = re.findall(r'(\d+)\s*yr', duration)
                months = re.findall(r'(\d+)\s*mo', duration)
                exp_months = 0
                if years:
                    exp_months += int(years[0]) * 12
                if months:
                    exp_months += int(months[0])
                total_months += exp_months
            summary["experience_years"] = round(total_months / 12, 1) if total_months > 0 else 0
        skills = profile.get("skills", [])
        summary["skills_count"] = len(skills) if isinstance(skills, list) else 0
        education = profile.get("education", [])
        summary["education_count"] = len(education) if isinstance(education, list) else 0
        return summary


    def _generate_comprehensive_analytics(self, employees: List[Dict[str, Any]], company_data: Optional[CompanyData] = None) -> Dict[str, Any]:
        total_employees = len(employees)
        scraped_count = sum(1 for e in employees if e["data_status"]["detailed_scraped"])
        analytics: Dict[str, Any] = {
            "employee_analytics": {
                "totals": {
                    "employees_found": total_employees,
                    "profiles_scraped": scraped_count,
                    "scraping_success_rate": round((scraped_count / total_employees * 100), 2) if total_employees else 0,
                },
                "top_positions": [],
                "top_skills": [],
                "top_locations": [],
                "experience_distribution": {"0-2_years": 0, "3-5_years": 0, "6-10_years": 0, "11+_years": 0},
            },
            "company_analytics": {},
            "data_quality": {"sources_analyzed": 0, "confidence_score": 0, "completeness_score": 0},
        }
        positions: List[str] = []
        skills: List[str] = []
        locations: List[str] = []
        for emp in employees:
            if emp["data_status"]["detailed_scraped"]:
                summary = emp.get("summary", {})
                pos = (summary.get("current_position") or "").strip()
                loc = (summary.get("location") or "").strip()
                if pos:
                    positions.append(pos)
                if loc:
                    locations.append(loc)
                exp_years = summary.get("experience_years", 0)
                if isinstance(exp_years, (int, float)):
                    if exp_years <= 2:
                        analytics["employee_analytics"]["experience_distribution"]["0-2_years"] += 1
                    elif exp_years <= 5:
                        analytics["employee_analytics"]["experience_distribution"]["3-5_years"] += 1
                    elif exp_years <= 10:
                        analytics["employee_analytics"]["experience_distribution"]["6-10_years"] += 1
                    else:
                        analytics["employee_analytics"]["experience_distribution"]["11+_years"] += 1
                prof = emp.get("detailed_profile") or {}
                prof_skills = prof.get("skills", [])
                if isinstance(prof_skills, list):
                    for s in prof_skills:
                        if isinstance(s, dict):
                            name = s.get("name", "")
                            if name:
                                skills.append(name)
        from collections import Counter
        analytics["employee_analytics"]["top_positions"] = [
            {"position": p, "count": c} for p, c in Counter(positions).most_common(10)
        ]
        analytics["employee_analytics"]["top_skills"] = [
            {"skill": s, "count": c} for s, c in Counter(skills).most_common(20)
        ]
        analytics["employee_analytics"]["top_locations"] = [
            {"location": l, "count": c} for l, c in Counter(locations).most_common(10)
        ]
        if company_data:
            analytics["company_analytics"] = {
                "web_presence": {
                    "social_platforms": len(company_data.social_links),
                    "platforms": list(company_data.social_links.keys()),
                },
                "data_richness": {
                    "has_revenue_data": bool(company_data.revenue_estimate),
                    "has_employee_estimate": bool(company_data.employee_estimate),
                    "has_founding_info": bool(company_data.founded_year),
                    "has_location_info": bool(company_data.headquarters),
                    "has_business_description": bool(company_data.description),
                },
            }
        analytics["data_quality"]["completeness_score"] = self._calculate_completeness_score(employees, company_data)
        analytics["data_quality"]["confidence_score"] = analytics["employee_analytics"]["totals"]["scraping_success_rate"]
        return analytics

    def _calculate_completeness_score(self, employees: List[Dict[str, Any]], company_data: Optional[CompanyData] = None) -> float:
        total_score = 0
        max_score = 0
        if employees:
            scraped = [e for e in employees if e["data_status"]["detailed_scraped"]]
            total_score += (len(scraped) / len(employees)) * 60
        max_score += 60
        if company_data:
            fields = [
                company_data.description,
                company_data.industry,
                company_data.headquarters,
                company_data.founded_year,
                company_data.employee_estimate,
                company_data.revenue_estimate,
                company_data.social_links,
                company_data.business_model,
            ]
            filled = sum(1 for f in fields if f)
            total_score += (filled / len(fields)) * 40
        max_score += 40
        return round((total_score / max_score) * 100, 2) if max_score else 0

    def _generate_executive_summary(
        self,
        company_name: str,
        company_intel: Dict[str, Any],
        employee_summary: Dict[str, Any],
        analytics: Dict[str, Any],
    ) -> str:
        parts: List[str] = []
        if company_intel.get("basic_info", {}).get("name"):
            info = company_intel["basic_info"]
            parts.append(f"Company Overview: {company_name}")
            if info.get("industry"):
                parts.append(f"operates in the {info['industry']} industry")
            if info.get("headquarters"):
                parts.append(f"headquartered in {info['headquarters']}")
            if info.get("founded_year"):
                parts.append(f"founded in {info['founded_year']}")
        emp_found = employee_summary.get("total_employees_found", 0)
        emp_scraped = employee_summary.get("detailed_profiles_scraped", 0)
        success_rate = employee_summary.get("scraping_success_rate", 0)
        parts.append(
            f"Employee Intelligence: Found {emp_found} employees on LinkedIn, "
            f"scraped {emp_scraped} detailed profiles ({success_rate}% success)"
        )
        top_positions = analytics.get("employee_analytics", {}).get("top_positions", [])
        if top_positions:
            parts.append(f"Most common position: {top_positions[0]['position']}")
        completeness = analytics.get("data_quality", {}).get("completeness_score", 0)
        parts.append(f"Overall data completeness: {completeness}%")
        return ". ".join(parts) + "."

# -------------------------
# Orchestrator adapter shims (keep signatures the orchestrator expects)
# -------------------------

async def run_company_intelligence_async(company: str, limit: int = 1):
    """
    Async: return a dict with 'employee_intelligence' so the orchestrator
    can normalize it without special-casing.
    """
    ctx = {"company_name": company, "spectre_n": int(limit), "run_dir": os.getcwd()}
    out = await _run_async(ctx)
    # Shape it like a full report dict; orchestrator can also save its own copy
    return {
        "report_metadata": {"company_name": company, "generated_at": datetime.utcnow().isoformat()},
        "employee_intelligence": {"employees": out.get("spectre_employees", [])},
        "intelligence_report_path": out.get("intelligence_report_path"),
        "shade_status": out.get("shade_status", "ok"),
    }

def run_company_intelligence(company: str, limit: int = 1):
    """
    Sync variant; orchestrator may try this too.
    """
    return run({"company_name": company, "spectre_n": int(limit), "run_dir": os.getcwd()})

async def discover_company_employees_async(company: str, limit: int = 1):
    """
    Minimal async option returning (employees_list, company_data_dict) as a tuple.
    """
    employees, company_data = await _gather_intel(company, int(limit))
    employees_list = [
        {"name": e.name, "linkedin_url": e.linkedin_url, "company": e.company, "snippet": e.snippet}
        for e in (employees or [])
    ]
    return employees_list, {
        "report_metadata": {"company_name": company, "generated_at": datetime.utcnow().isoformat()},
        "company_data": company_data.__dict__ if hasattr(company_data, "__dict__") else (company_data or {}),
    }
# =========================
# ADAPTERS + SAFE DEFAULTS
# =========================

def _resolve_limit(limit: Optional[int]) -> int:
    """Use orchestrator-provided limit if present; otherwise ENV; fallback 50."""
    if isinstance(limit, int) and limit > 0:
        return limit
    try:
        env_val = int(os.getenv("SPECTRE_EMPLOYEES", "50"))
        return max(1, env_val)
    except Exception:
        return 50

async def run_company_intelligence_async(company: str, limit: int = None):
    """
    Async adapter returning a dict with 'employee_intelligence'.
    Orchestrator often calls this directly.
    """
    lim = _resolve_limit(limit)
    ctx = {"company_name": company, "spectre_n": lim, "run_dir": os.getcwd()}
    out = await _run_async(ctx)
    return {
        "report_metadata": {"company_name": company, "generated_at": datetime.utcnow().isoformat()},
        "employee_intelligence": {"employees": out.get("spectre_employees", [])},
        "intelligence_report_path": out.get("intelligence_report_path"),
        "shade_status": out.get("shade_status", "ok"),
    }

def run_company_intelligence(company: str, limit: int = None):
    """
    Sync adapter; some orchestrators call this.
    """
    lim = _resolve_limit(limit)
    return run({"company_name": company, "spectre_n": lim, "run_dir": os.getcwd()})

async def discover_company_employees_async(company: str, limit: int = None):
    """
    Async minimal tuple (employees_list, company_data_dict); some orchestrators expect exactly this.
    """
    lim = _resolve_limit(limit)
    employees, company_data = await _gather_intel(company, lim)
    employees_list = [
        {"name": e.name, "linkedin_url": e.linkedin_url, "company": e.company, "snippet": e.snippet}
        for e in (employees or [])
    ]
    company_blob = {
        "report_metadata": {"company_name": company, "generated_at": datetime.utcnow().isoformat()},
        "company_data": company_data.__dict__ if hasattr(company_data, "__dict__") else (company_data or {}),
    }
    return employees_list, company_blob

def discover_company_employees(company: str, limit: int = None):
    """
    Sync wrapper of the above (older orchestrators sometimes call this one).
    """
    lim = _resolve_limit(limit)
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(discover_company_employees_async(company, lim))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"discover_company_employees error: {e}")
        return [], {"report_metadata": {"company_name": company}, "company_data": {}}

# Back-compat: some orchestrators used this exact name
async def run_parallel_intelligence_gathering(company: str, limit: int = None):
    """
    Return the same shape as discover_company_employees_async so legacy code works.
    """
    return await discover_company_employees_async(company, limit)

# -------------
# run(context) — accept dict OR legacy (company, limit[, run_dir])
# -------------
def run(context_or_company, limit: Optional[int] = None, run_dir: Optional[str] = None):
    """
    Flexible sync entrypoint:
      • run({"company_name": "...", "spectre_n": 75, "run_dir": "..."})
      • run("Acme Corp", 75)
      • run("Acme Corp", None, "/path/to/run_dir")
    """
    # Normalize context
    if isinstance(context_or_company, dict):
        ctx = dict(context_or_company)
        ctx.setdefault("company_name", ctx.get("spectre_company") or ctx.get("target_company"))
        ctx["spectre_n"] = _resolve_limit(ctx.get("spectre_n"))
        if run_dir:
            ctx["run_dir"] = run_dir
    else:
        # Legacy tuple-style: (company, limit, [run_dir])
        company = str(context_or_company or "").strip()
        if not company:
            raise ValueError("run(): company name is required")
        ctx = {
            "company_name": company,
            "spectre_n": _resolve_limit(limit),
            "run_dir": run_dir or os.getcwd(),
        }

    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run_async(ctx))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"SHADE.run error: {e}")
        ctx["shade_status"] = f"error: {e}"
        return ctx

# -------------------------
# Parallel (internal) helper used by run()
# -------------------------
async def _gather_intel(company_name: str, max_employees: int) -> tuple[List[RawEmployee], Optional[CompanyData]]:
    logger.info(f"🚀 Starting parallel intelligence gathering for {company_name}")
    employee_finder = GoogleCSEEmployeeFinder()
    company_researcher = CompanyReportGenerator()
    logger.info("📋 Task 1: Finding employees...")
    emp_task = asyncio.create_task(asyncio.to_thread(employee_finder.find_employees, company_name, max_employees))
    logger.info("📋 Task 2: Researching company...")
    comp_task = asyncio.create_task(company_researcher.generate_company_report(company_name))
    employees, company_data = await asyncio.gather(emp_task, comp_task, return_exceptions=False)
    return employees or [], company_data


# -------------------------
# Pipeline entrypoint: run(context) -> context
# -------------------------
async def _run_async(context: Dict[str, Any]) -> Dict[str, Any]:
    company_name = (
        context.get("company_name")
        or context.get("spectre_company")
        or context.get("target_company")
    )
    if not company_name:
        raise ValueError("SHADE.run: 'company_name' is required in context")

    max_employees = int(
    (context.get("spectre_n")
     or context.get("limit")
     or context.get("max_employees")
     or 50)
)

    run_dir = context.get("run_dir") or os.path.join(os.getcwd(), "runs", f"{re.sub(r'[^a-z0-9-]', '-', company_name.lower()).strip('-')}-{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    # Phase 1: discovery + company research
    employees, company_data = await _gather_intel(company_name, max_employees)

    # Phase 2: optional Bright Data scrape
    scraper = BrightDataScraper()
    linkedin_urls = [e.linkedin_url for e in employees]
    detailed_profiles: List[Dict[str, Any]] = await asyncio.to_thread(
        scraper.scrape_profiles_one_shot, linkedin_urls
    )

    # Phase 3: save comprehensive report (under run_dir/data)
    base_dir = os.path.join(run_dir, "data")
    manager = ComprehensiveDataManager(base_dir=base_dir)
    report_path = manager.save_comprehensive_data(company_name, employees, detailed_profiles, company_data)

    # Also copy to root ./data for legacy tools, if different
    try:
        legacy_dir = "data"
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_path = os.path.join(legacy_dir, os.path.basename(report_path))
        if os.path.abspath(legacy_path) != os.path.abspath(report_path):
            with open(report_path, 'r', encoding='utf-8') as src, open(legacy_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    except Exception as e:
        logger.warning(f"Legacy copy failed: {e}")

    # Prepare outputs for the orchestrator
    context.update({
        "intelligence_report_path": report_path,
        "spectre_employees": [
            {
                "name": e.name,
                "linkedin_url": e.linkedin_url,
                "company": e.company,
                "snippet": e.snippet,
            }
            for e in employees
        ],
        "company_data_available": bool(company_data),
        "shade_status": "ok",
    })
    return context


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous adapter for orchestrators."""
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run_async(context))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"SHADE.run error: {e}")
        context = dict(context or {})
        context["shade_status"] = f"error: {e}"
        return context


# -------------------------
# Optional CLI for standalone usage
# -------------------------
if __name__ == "__main__":
    print("🚀 SHADE — Company + Employee Discovery (Standalone)")
    try:
        company = input("Target company: ").strip()
        max_emp_s = input("How many employees to discover? [default 50]: ").strip() or "50"
        ctx: Dict[str, Any] = {"company_name": company, "spectre_n": int(max_emp_s)}
        out = run(ctx)
        print("\n✅ Done. Key outputs:")
        print(f" - intelligence_report_path: {out.get('intelligence_report_path')}")
        print(f" - employees found: {len(out.get('spectre_employees') or [])}")
        print(f" - company_data_available: {out.get('company_data_available')}")
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
