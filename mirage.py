#!/usr/bin/env python3
""" 
MIRAGE: GPT-Powered Competitive Intelligence System
==================================================
Clean production version with parallel processing and structured outputs.
Author: MIRAGE Intelligence System
Version: 3.0 - Clean Production
"""
# Use SHADE's Bright Data scraper

import sys, io, re
import os
import json
import time
import asyncio
import logging
import hashlib
import aiohttp
import requests
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Logging Configuration (Fixed for Windows)
# =============================================================================

def _force_utf8_console():
    """Force UTF-8 encoding for Windows console"""
    try:
        if sys.platform == "win32":
            # For Windows, change the console code page to UTF-8
            os.system("chcp 65001 > nul")
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Call this immediately
_force_utf8_console()

def setup_logging():
    """Configure comprehensive logging without emojis for Windows compatibility"""
    # Remove emojis from log format for Windows compatibility
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler('mirage_system.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger('MIRAGE')

logger = setup_logging()

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class CompetitorProfile:
    name: str
    industry: str
    similarity_score: float
    detection_method: str = "GPT Analysis"

@dataclass
class TargetEmployeeProfile:
    name: str
    title: str
    department: str
    experience_years: float
    key_skills: List[str]
    company: str

@dataclass
class CompetitorEmployee:
    name: str
    title: str
    company: str
    linkedin_url: str
    search_snippet: str

@dataclass
class EmployeeMatch:
    target_employee: str
    competitor_employee: str
    competitor_company: str
    similarity_score: float
    match_rationale: str
    linkedin_url: str

# =============================================================================
# Utility Functions
# =============================================================================

import math
from itertools import islice

def _chunks(iterable, size):
    it = iter(iterable)
    while True:
        block = list(islice(it, size))
        if not block:
            break
        yield block

async def _scrape_company_chunks(
    shade_scraper,
    company: str,
    urls: list[str],
    *,
    per_snapshot_cap: int = 100,      # Bright Data safe cap
    shade_batch_size: int = 8,        # your existing usage
    timeout_sec: int = 100000,
    max_chunk_parallel: int = 2       # keep 1–3 to respect rate limits
) -> tuple[str, dict[str, dict]]:
    """Run multiple Bright Data snapshots for ONE company, possibly in parallel."""
    # Normalize/uniq
    urls_norm = []
    seen = set()
    for u in (urls or []):
        if not u:
            continue
        u2 = u.strip().rstrip("/")
        if ("linkedin.com/in" in u2 or "linkedin.com/pub" in u2) and u2 not in seen:
            seen.add(u2); urls_norm.append(u2)

    # Split to ≤100 per snapshot
    url_chunks = list(_chunks(urls_norm, per_snapshot_cap))
    if not url_chunks:
        return company, {}

    sem = asyncio.Semaphore(max_chunk_parallel)

    async def _run_one(chunk):
        async with sem:
            # shade_scraper.scrape_profiles_in_batches is sync → run in thread
            profiles = await asyncio.to_thread(
                shade_scraper.scrape_profiles_in_batches,
                chunk,
                batch_size=shade_batch_size,
                timeout_sec=timeout_sec
            )
            url_map = {}
            for p in profiles or []:
                u = (p.get("url") or p.get("profile_url") or "").strip().rstrip("/")
                if u:
                    url_map[u] = p
            return url_map

    # Fire all company snapshots (limited by semaphore)
    maps = await asyncio.gather(*[_run_one(c) for c in url_chunks], return_exceptions=True)

    merged = {}
    for m in maps:
        if isinstance(m, dict):
            merged.update(m)
    return company, merged

async def scrape_matched_profiles_per_company_parallel(
    shade_scraper,
    matched_urls_per_company: dict[str, list[str]],
    *,
    per_snapshot_cap: int = 100,
    shade_batch_size: int = 8,
    timeout_sec: int = 100000,
    max_company_parallel: int = 3      # 2–4 is usually safe
) -> dict[str, dict[str, dict]]:
    """Kick off Bright Data snapshots in parallel PER company."""
    comp_sem = asyncio.Semaphore(max_company_parallel)

    async def _run_company(company, urls):
        async with comp_sem:
            return await _scrape_company_chunks(
                shade_scraper, company, urls,
                per_snapshot_cap=per_snapshot_cap,
                shade_batch_size=shade_batch_size,
                timeout_sec=timeout_sec,
                max_chunk_parallel=2  # per-company chunk parallelism
            )

    tasks = [_run_company(c, urls) for c, urls in (matched_urls_per_company or {}).items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = {}
    for r in results:
        if isinstance(r, tuple):
            company, url_map = r
            out[company] = url_map
    return out


def safe_json_parse(content: str) -> Optional[Dict]:
    """Safely parse JSON from GPT response with better error handling"""
    if not content:
        return None
        
    content = content.strip()
    
    # Remove code blocks if present
    if content.startswith('```'):
        lines = content.split('\n')
        if len(lines) > 2 and lines[0].strip() == '```' and lines[-1].strip() == '```':
            content = '\n'.join(lines[1:-1])
        elif lines[0].startswith('```json'):
            content = '\n'.join(lines[1:])
    
    # Try to find JSON object or array
    json_start = content.find('{')
    json_array_start = content.find('[')
    
    if json_start == -1 and json_array_start == -1:
        return None
        
    start_idx = json_start if json_start != -1 else json_array_start
    content = content[start_idx:]
    
    # Try to find the matching closing brace/bracket
    open_char = content[0]
    close_char = '}' if open_char == '{' else ']'
    
    depth = 0
    end_idx = -1
    
    for i, char in enumerate(content):
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    
    if end_idx != -1:
        content = content[:end_idx]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}. Content: {content[:200]}...")
        # Try to fix common JSON issues
        try:
            # Try to handle trailing commas
            fixed_content = re.sub(r',\s*}', '}', content)
            fixed_content = re.sub(r',\s*]', ']', fixed_content)
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            return None

def create_cache_key(*args) -> str:
    """Create cache key for GPT responses"""
    combined = '|'.join(str(arg) for arg in args)
    return hashlib.sha256(combined.encode()).hexdigest()

def safe_filename(name: str) -> str:
    """Create safe filename from company name"""
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip().lower())

# =============================================================================
# Azure OpenAI Client
# =============================================================================

class AzureGPTClient:
    """Centralized Azure OpenAI client"""
    
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        
        if not all([self.api_key, self.endpoint]):
            raise ValueError("Missing Azure OpenAI configuration")
            
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        self.cache = {}
        logger.info("Azure GPT Client initialized")
    
    async def chat_completion(self, system_prompt: str, user_prompt: str, 
                            temperature: float = 0.1, max_tokens: int = 1500) -> Optional[str]:
        """Make GPT chat completion request with caching"""
        
        cache_key = create_cache_key(system_prompt, user_prompt, temperature)
        if cache_key in self.cache:
            logger.debug("Using cached GPT response")
            return self.cache[cache_key]
        
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            logger.debug(f"Making GPT request (temp={temperature})")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        self.cache[cache_key] = content
                        logger.debug("GPT request successful")
                        return content
                    else:
                        error_text = await response.text()
                        logger.error(f"Azure API error {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error("GPT request timeout")
            return None
        except Exception as e:
            logger.error(f"GPT request failed: {e}")
            return None

# =============================================================================
# Step 1: Competitor Detection (STRICT, GPT-only, no fallbacks)
# =============================================================================
class CompetitorDetector:
    """
    GPT-only, tolerant parser:
    - Single GPT call.
    - Accepts minor schema drift (extra keys, score as string, etc.).
    - Ignores invalid rows instead of failing the whole response.
    - No hard exceptions on empty results.
    """

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        logger.info("CompetitorDetector initialized (GPT-only, tolerant)")

    async def detect_competitors(
        self,
        intelligence_data: Dict,
        max_competitors: int = 10
    ) -> List[CompetitorProfile]:
        logger.info(f"STEP 1: COMPETITOR DETECTION (limit={max_competitors})")

        company_name = self._extract_company_name(intelligence_data)
        ctx = self._extract_business_context(intelligence_data, company_name)

        logger.info(f"Analyzing competitors for: {company_name}")
        logger.info(f"Industry: {ctx.get('industry', 'Unknown')}")

        competitors = await self._gpt_competitor_analysis(ctx, max_competitors)

        # Keep model order and cap length without throwing
        competitors = competitors[:max_competitors]

        logger.info(f"Detected {len(competitors)} competitors:")
        for i, comp in enumerate(competitors, 1):
            try:
                score = float(comp.similarity_score)
            except Exception:
                score = 0.0
            logger.info(f"   {i}. {comp.name} (Score: {score:.1f})")

        return competitors

    # ---------- helpers ----------

    def _extract_company_name(self, data: Dict) -> str:
        for name in (
            data.get("report_metadata", {}).get("company_name"),
            data.get("company_intelligence", {}).get("basic_info", {}).get("name"),
            data.get("mission_metadata", {}).get("target_company"),
            data.get("company_name"),
        ):
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "Unknown Company"

    def _extract_business_context(self, data: Dict, company_name: str) -> Dict:
        ci = data.get("company_intelligence", {}) or {}
        basic = ci.get("basic_info", {}) or {}
        industry = (
            basic.get("industry")
            or ci.get("industry")
            or data.get("industry")
            or ("Fintech" if "fintech" in (company_name or "").lower() else "")
        )
        return {
            "company_name": company_name,
            "industry": industry,
            "description": (basic.get("description") or "")[:1200],
            "employee_count": basic.get("employee_estimate") or basic.get("employee_count") or "",
            "headquarters": basic.get("headquarters") or basic.get("hq") or "",
        }

    def _coerce_row(self, row: Any) -> Optional[CompetitorProfile]:
        """
        Accepts:
          - {"name": "...", "industry": "...", "similarity_score": <num|str>}
          - {"name": "..."}  (fills industry/score)
          - "Company Name"   (fills industry/score)
        Returns None for rows without a usable name.
        """
        if isinstance(row, str):
            name = row.strip()
            if not name:
                return None
            return CompetitorProfile(
                name=name,
                industry="",
                similarity_score=0.0,
                detection_method="GPT Analysis",
            )

        if isinstance(row, dict):
            name = (row.get("name") or "").strip()
            if not name:
                return None

            industry = (row.get("industry") or "").strip()

            score_raw = row.get("similarity_score", 0.0)
            try:
                # allow "9.1", "8", "7/10", "85%" → coerce to 0–10
                if isinstance(score_raw, str):
                    s = score_raw.strip()
                    if s.endswith("%"):
                        val = float(s[:-1].strip())
                        val = max(0.0, min(100.0, val))
                        score = round(val / 10.0, 2)
                    elif "/" in s:
                        # e.g., "7/10"
                        parts = s.split("/", 1)
                        num = float(parts[0].strip())
                        den = float(parts[1].strip())
                        score = 10.0 * (num / den) if den else 0.0
                    else:
                        score = float(s)
                else:
                    score = float(score_raw)
            except Exception:
                score = 0.0

            # clamp to [0, 10]
            if score < 0.0: score = 0.0
            if score > 10.0: score = 10.0

            return CompetitorProfile(
                name=name,
                industry=industry,
                similarity_score=score,
                detection_method="GPT Analysis",
            )

        return None

    def _extract_json_object_with_competitors(self, text: str) -> Optional[Dict]:
        """
        Tries to pull a JSON object (possibly inside code fences) that contains "competitors".
        This is still parsing the *same* GPT output (no external fallback).
        """
        if not text:
            return None

        # strip common code fences
        txt = text.strip()
        if txt.startswith("```"):
            # remove leading and trailing fences
            # e.g., ```json ... ```
            txt = re.sub(r"^```[\w-]*\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)

        # try full parse first
        try:
            data = safe_json_parse(txt)
            if isinstance(data, dict) and "competitors" in data:
                return data
        except Exception:
            pass

        # find the smallest object that has "competitors": [...]
        try:
            m = re.search(r"\{[^{}]*\"competitors\"[^{}]*\[[\s\S]*?\][\s\S]*?\}", txt)
            if m:
                data = safe_json_parse(m.group(0))
                if isinstance(data, dict) and "competitors" in data:
                    return data
        except Exception:
            pass

        return None

    async def _gpt_competitor_analysis(
        self,
        ctx: Dict,
        max_competitors: int
    ) -> List[CompetitorProfile]:
        """
        Single GPT call; tolerant parsing of its output.
        Accepts:
          - {"competitors": [ {name, industry?, similarity_score?}, ... ], ...}
          - ["Company A", "Company B", ...]
          - Any of the above inside code fences.
        """
        system_prompt = f"""You are a competitive-intelligence analyst.
Return competitors for the given company as JSON.

Preferred format (but minor variations are OK):
{{
  "competitors": [
    {{
      "name": "Competitor Company Name",
      "industry": "Industry (optional)",
      "similarity_score": 0.0
    }}
  ]
}}

Rules:
- Up to {max_competitors} items, most similar first.
- If unsure about industry or score, you may omit them or leave them blank/0.0.
- No prose needed, but it's OK if you include extra keys; the consumer will ignore them.
- If the comapany is Manipal Fintech always add Rupeek and Indiagold as the first 2 competitons
"""

        user_prompt = (
            f"Company: {ctx.get('company_name','')}\n"
            f"Industry: {ctx.get('industry','Not specified')}\n"
            f"Description: {ctx.get('description','Not specified')}\n"
            f"Employee Count: {ctx.get('employee_count','Not specified')}\n"
            f"Headquarters: {ctx.get('headquarters','Not specified')}"
        )

        logger.debug("Requesting GPT competitor analysis")
        raw = await self.gpt.chat_completion(
            system_prompt,
            user_prompt,
            temperature=0.2,
            max_tokens=800
        )

        if not raw:
            logger.warning("Empty response from GPT for competitors")
            return []

        # 1) Try object with "competitors"
        obj = self._extract_json_object_with_competitors(raw)
        items: List[Any] = []
        if obj and isinstance(obj.get("competitors"), list):
            items = obj["competitors"]
        else:
            # 2) Maybe GPT returned a bare list of names
            try:
                parsed = safe_json_parse(raw.strip())
                if isinstance(parsed, list):
                    items = parsed
            except Exception:
                items = []

        out: List[CompetitorProfile] = []
        for row in items:
            prof = self._coerce_row(row)
            if prof and prof.name:
                out.append(prof)

        # Trim to max, keep GPT order
        return out[:max_competitors]


# =============================================================================
# Step 2: Target Profile Building (FIXED VERSION)
# =============================================================================

class TargetProfileBuilder:
    """
    IMPROVED version with better error handling and fallback parsing
    """

    _DEPT_ENUM = {
        "Engineering","Sales","Marketing","Finance","Operations",
        "Product","Data","Design","HR","Legal","Support","Other"
    }

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        logger.info("TargetProfileBuilder initialized (IMPROVED with fallbacks)")

    async def build_target_profiles(self, employee_data: List[Dict]) -> List[TargetEmployeeProfile]:
        logger.info("STEP 2: TARGET PROFILE BUILDING")

        if not employee_data:
            logger.warning("No employee data provided")
            return []

        logger.info(f"Building profiles for {len(employee_data)} employees")

        all_profiles: List[TargetEmployeeProfile] = []

        for emp in employee_data:
            try:
                profile = await self._build_single_profile(emp)
                if profile:
                    all_profiles.append(profile)
                else:
                    # Try fallback method if GPT fails
                    fallback_profile = self._create_fallback_profile(emp)
                    if fallback_profile:
                        all_profiles.append(fallback_profile)
            except Exception as e:
                logger.warning(f"Profile building failed for employee: {e}")
                fallback_profile = self._create_fallback_profile(emp)
                if fallback_profile:
                    all_profiles.append(fallback_profile)

        logger.info(f"Built {len(all_profiles)} target profiles")
        return all_profiles

    async def _build_single_profile(self, employee: Dict) -> Optional[TargetEmployeeProfile]:
        raw = self._extract_employee_data(employee)
        if not raw.get("name"):
            return None

        system_prompt = """You are an HR analyst. Extract information from the provided employee data.

Return a JSON object with this structure:
{
  "name": "Full Name",
  "title": "Job Title",
  "department": "Department",
  "experience_years": 5.0,
  "key_skills": ["skill1", "skill2", "skill3"],
  "company": "Company Name"
}

Guidelines:
- If information is missing, make reasonable inferences
- For department, choose from: Engineering, Sales, Marketing, Finance, Operations, Product, Data, Design, HR, Legal, Support, Other
- For experience_years, estimate based on title (entry-level: 1-3, mid-level: 4-7, senior: 8+)
- For key_skills, suggest 3-5 relevant skills based on the role
- Keep responses concise and accurate"""

        user_prompt = f"""Please analyze this employee profile:

Name: {raw.get('name','Unknown')}
Title: {raw.get('title','Not specified')}
Company: {raw.get('company','Not specified')}
Location: {raw.get('location','Not specified')}

Provide the JSON analysis:"""

        try:
            content = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.1, max_tokens=500)
            
            if not content:
                return None

            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            data = safe_json_parse(content)
            if not data:
                return None

            # Validate required fields with fallbacks
            name = data.get("name") or raw.get("name") or "Unknown"
            title = data.get("title") or raw.get("title") or "Unknown Position"
            company = data.get("company") or raw.get("company") or "Unknown Company"
            
            # Department with validation
            dept = data.get("department", "").strip()
            if dept not in self._DEPT_ENUM:
                dept = self._infer_department(title)
            
            # Experience with validation
            try:
                exp = float(data.get("experience_years", 5.0))
            except (ValueError, TypeError):
                exp = self._estimate_experience(title)
            
            # Skills with validation
            skills = data.get("key_skills", [])
            if not skills or not isinstance(skills, list):
                skills = self._suggest_skills(title, dept)
            
            return TargetEmployeeProfile(
                name=name.strip(),
                title=title.strip(),
                department=dept,
                experience_years=exp,
                key_skills=[s.strip() for s in skills if s and isinstance(s, str)],
                company=company.strip()
            )
            
        except Exception as e:
            logger.warning(f"GPT profile analysis failed: {e}")
            return None

    def _create_fallback_profile(self, employee: Dict) -> Optional[TargetEmployeeProfile]:
        """Create a basic profile when GPT analysis fails"""
        raw = self._extract_employee_data(employee)
        if not raw.get("name"):
            return None

        name = raw.get("name", "Unknown Employee")
        title = raw.get("title", "Unknown Position")
        company = raw.get("company", "Unknown Company")
        
        return TargetEmployeeProfile(
            name=name,
            title=title,
            department=self._infer_department(title),
            experience_years=self._estimate_experience(title),
            key_skills=self._suggest_skills(title, self._infer_department(title)),
            company=company
        )

    def _infer_department(self, title: str) -> str:
        """Infer department from job title"""
        title_lower = title.lower()
        if any(x in title_lower for x in ["engineer", "developer", "tech"]):
            return "Engineering"
        elif any(x in title_lower for x in ["sale", "account", "business development"]):
            return "Sales"
        elif any(x in title_lower for x in ["market", "growth", "digital"]):
            return "Marketing"
        elif any(x in title_lower for x in ["data", "analyst", "scientist"]):
            return "Data"
        elif any(x in title_lower for x in ["product", "pm", "owner"]):
            return "Product"
        elif any(x in title_lower for x in ["design", "ux", "ui"]):
            return "Design"
        elif any(x in title_lower for x in ["finance", "accounting", "cfo"]):
            return "Finance"
        else:
            return "Other"

    def _estimate_experience(self, title: str) -> float:
        """Estimate experience based on title"""
        title_lower = title.lower()
        if any(x in title_lower for x in ["junior", "entry", "associate"]):
            return 2.0
        elif any(x in title_lower for x in ["senior", "lead", "principal", "manager"]):
            return 8.0
        elif any(x in title_lower for x in ["director", "head", "vp", "vice president"]):
            return 12.0
        else:
            return 5.0

    def _suggest_skills(self, title: str, department: str) -> List[str]:
        """Suggest skills based on title and department"""
        skills_map = {
            "Engineering": ["Python", "JavaScript", "AWS", "Docker", "Kubernetes"],
            "Sales": ["CRM", "Negotiation", "Relationship Building", "Salesforce"],
            "Marketing": ["SEO", "Content Marketing", "Social Media", "Analytics"],
            "Data": ["Python", "SQL", "Machine Learning", "Data Visualization"],
            "Product": ["Product Strategy", "Roadmapping", "User Research", "Agile"],
            "Design": ["Figma", "UI/UX Design", "Prototyping", "User Research"],
            "Finance": ["Financial Analysis", "Excel", "Accounting", "Financial Modeling"],
            "Other": ["Communication", "Project Management", "Problem Solving"]
        }
        return skills_map.get(department, ["Communication", "Problem Solving", "Teamwork"])

    def _extract_employee_data(self, employee: Dict) -> Dict:
        basic = employee.get("basic_info", {}) or {}
        detailed = employee.get("detailed_profile", {}) or {}
        return {
            "name": (employee.get("name") or basic.get("name") or "").strip(),
            "title": (employee.get("title") or employee.get("position") or basic.get("title") or detailed.get("position") or "").strip(),
            "company": (employee.get("company") or basic.get("company") or "").strip(),
            "location": (employee.get("location") or basic.get("location") or "").strip(),
        }
    
# =============================================================================
# Step 3: Competitor Employee Search
# =============================================================================

class CompetitorEmployeeFinder:
    """
    Finds competitor employees using Google Custom Search.

    Modes:
      - HARVEST_MODE=all  (default): harvest company-wide for EVERY detected competitor
      - HARVEST_MODE=none: use department-driven narrow queries (original behavior)
      - HARVEST_MODE=list: harvest only companies named in HARVEST_COMPANIES
                           (comma-separated)
    """

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        # Harvest controls (no hard-coded company names)
        self.harvest_mode = (os.getenv("HARVEST_MODE", "all") or "all").strip().lower()
        allow_raw = os.getenv("HARVEST_COMPANIES", "")
        self.harvest_allowlist = {x.strip().lower() for x in allow_raw.split(",") if x.strip()}

        # Pagination / throttling
        self.pages_per_query = int(os.getenv("PAGES_PER_QUERY", "15"))        # deep paging
        self.results_per_page = int(os.getenv("RESULTS_PER_PAGE", "10"))      # CSE max 10
        self.query_batch_sleep = float(os.getenv("QUERY_BATCH_SLEEP", "1.2")) # seconds

        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Missing Google API credentials - using mock data")
            self.use_mock_data = True
        else:
            self.use_mock_data = False

        logger.info(
            "CompetitorEmployeeFinder initialized | HARVEST_MODE=%s | allowlist=%s",
            self.harvest_mode, (", ".join(sorted(self.harvest_allowlist)) or "-")
        )

    async def find_competitor_employees(
        self,
        target_profiles: List[TargetEmployeeProfile],
        competitors: List[CompetitorProfile]
    ) -> Dict[str, List[CompetitorEmployee]]:
        """Find competitor employees for all companies (harvest or department-driven)."""
        logger.info("STEP 3: COMPETITOR EMPLOYEE SEARCH")

        if self.use_mock_data:
            return await self._mock_employee_search(competitors)

        tasks = []
        for comp in competitors:
            cname = (comp.name or "").strip()
            harvest = self._should_harvest(cname)
            tasks.append(self._find_employees_for_company(target_profiles, comp, harvest))

        logger.info("Processing %d companies in parallel", len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: Dict[str, List[CompetitorEmployee]] = {}
        for i, res in enumerate(results):
            cname = (competitors[i].name if i < len(competitors) else f"Company_{i}")
            if isinstance(res, dict):
                final_results.update(res)
            else:
                logger.error("Search failed for %s: %s", cname, res)

        total_found = sum(len(v) for v in final_results.values())
        logger.info("Total employees found: %d", total_found)
        return final_results

    def _should_harvest(self, company_name: str) -> bool:
        c = (company_name or "").strip().lower()
        if self.harvest_mode in ("all", "true", "1", "yes"):
            return True
        if self.harvest_mode == "list":
            return c in self.harvest_allowlist
        # "none" or anything else: disable harvest
        return False

    async def _find_employees_for_company(
        self,
        target_profiles: List[TargetEmployeeProfile],
        competitor: CompetitorProfile,
        harvest: bool
    ) -> Dict[str, List[CompetitorEmployee]]:
        company = (competitor.name or "").strip()
        logger.info("Searching employees at %s (%s)", company, "HARVEST" if harvest else "dept-queries")

        if harvest:
            harvested = await self._find_employees_companywide(company)
            unique = self._deduplicate_employees(harvested)
            logger.info("   Found %d employees at %s (company-wide)", len(unique), company)
            return {company: unique}

        # ---- department-driven (original) ----
        all_employees: List[CompetitorEmployee] = []
        departments = list(set(p.department for p in target_profiles))
        for department in departments[:3]:
            logger.debug("   Searching %s at %s", department, company)
            queries = await self._generate_search_queries(department, company)
            employees = await self._execute_searches(queries, company)
            all_employees.extend(employees)
            await asyncio.sleep(1)

        unique_employees = self._deduplicate_employees(all_employees)
        logger.info("   Found %d employees at %s", len(unique_employees), company)
        return {company: unique_employees}

    # -------------------- HARVEST helpers --------------------

    def _normalize_company_aliases(self, company_name: str) -> List[str]:
        c = (company_name or "").strip()
        # add known aliases here if you like; default to the literal brand only
        return [f'"{c}"']

    def _generate_broad_company_queries(self, company_name: str) -> List[str]:
        aliases = self._normalize_company_aliases(company_name)
        alias_expr = " OR ".join(aliases)

        title_buckets = [
            "(Head OR VP OR Director OR Lead OR Manager)",
            "(Engineer OR Developer OR Architect OR SDE OR DevOps OR QA)",
            "(Product OR PM OR Owner OR Growth)",
            "(Data OR Analyst OR Scientist OR BI OR ML OR AI)",
            "(Design OR UX OR UI OR Research)",
            "(Finance OR Accounting OR Controller OR Treasury)",
            "(HR OR Talent OR People OR Recruiter)",
            "(Operations OR Ops OR Supply OR Logistics)",
            "(Sales OR Business Development OR Partnerships OR Alliances)",
            "(Marketing OR Brand OR Performance OR SEO OR Content)",
        ]

        locations = [
            "Bengaluru OR Bangalore OR Karnataka",
            "Mumbai OR Maharashtra",
            "Delhi OR NCR OR Noida OR Gurgaon",
            "Hyderabad OR Telangana",
            "Chennai OR Tamil Nadu",
            "Pune",
            "India",
        ]

        base_sites = ["site:linkedin.com/in", "site:linkedin.com/pub"]

        queries: List[str] = []
        for site in base_sites:
            queries.append(f'{site} ({alias_expr}) -jobs -hiring -recruiter')
            for bucket in title_buckets:
                queries.append(f'{site} ({alias_expr}) {bucket} -jobs -hiring -recruiter')
            for bucket in title_buckets:
                for loc in locations:
                    queries.append(f'{site} ({alias_expr}) {bucket} ({loc}) -jobs -hiring -recruiter')

        # dedupe preserve order
        seen, out = set(), []
        for q in queries:
            if q not in seen:
                seen.add(q); out.append(q)
        logger.info("[HARVEST] Built %d broad queries for %s", len(out), company_name)
        return out

    async def _find_employees_companywide(self, company_name: str) -> List[CompetitorEmployee]:
        queries = self._generate_broad_company_queries(company_name)
        all_employees: List[CompetitorEmployee] = []
        total_hits = 0

        max_start = min(1 + (self.pages_per_query - 1) * self.results_per_page, 91)  # cap ~100

        for qi, query in enumerate(queries, 1):
            page_index = 0
            for start in range(1, max_start + 1, self.results_per_page):  # 1,11,21,...
                page_index += 1
                batch = await self._execute_search(query, company_name, start=start, num=self.results_per_page)
                if not batch:
                    logger.debug("[HARVEST] q%d/%d page %d: 0 hits (start=%d)", qi, len(queries), page_index, start)
                    break
                total_hits += len(batch)
                all_employees.extend(batch)
                logger.debug("[HARVEST] q%d/%d page %d: %d hits (start=%d)",
                             qi, len(queries), page_index, len(batch), start)
                await asyncio.sleep(self.query_batch_sleep)

        logger.info("[HARVEST] Raw hits for %s: %d | Collected: %d (pre-dedupe)",
                    company_name, total_hits, len(all_employees))
        return all_employees

    async def _execute_search(
        self,
        query: str,
        company_name: str,
        *,
        start: int = 1,
        num: int = 10
    ) -> List[CompetitorEmployee]:
        employees: List[CompetitorEmployee] = []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": max(1, min(10, int(num))),
                "start": max(1, int(start)),
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])
                        for item in items:
                            emp = self._parse_search_result(item, company_name)
                            if emp:
                                employees.append(emp)
                    else:
                        logger.warning("Search failed (%s) for start=%s", resp.status, start)
        except Exception as e:
            logger.warning("Search error for start=%s: %s", start, e)
        return employees

    # -------------------- original helpers (kept) --------------------

    async def _generate_search_queries(self, department: str, company_name: str) -> List[str]:
        system_prompt = """Generate Google search queries to find LinkedIn profiles of employees.

Return ONLY JSON array without any additional text:
["query1", "query2", "query3"]

Requirements:
- Include: site:linkedin.com/in
- Include company name in quotes
- Include department/role terms
- Add: -jobs -hiring -recruiter"""
        user_prompt = f"""Department: {department}
Company: {company_name}

Generate 3 targeted LinkedIn search queries."""
        response = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.2)
        if not response:
            return []
        queries = safe_json_parse(response)
        if not queries or not isinstance(queries, list):
            return []
        return queries[:3]

    async def _execute_searches(self, queries: List[str], company_name: str) -> List[CompetitorEmployee]:
        all_employees: List[CompetitorEmployee] = []
        for query in queries:
            try:
                logger.debug("Search: %s...", query[:60])
                url = "https://www.googleapis.com/customsearch/v1"
                params = {'key': self.google_api_key,'cx': self.google_cse_id,'q': query,'num': 10}
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            for item in data.get('items', []):
                                emp = self._parse_search_result(item, company_name)
                                if emp: all_employees.append(emp)
                        else:
                            logger.warning("Search failed: %s", response.status)
                await asyncio.sleep(1.2)
            except Exception as e:
                logger.warning("Search error: %s", e)
        return all_employees

    def _parse_search_result(self, item: Dict, company_name: str) -> Optional[CompetitorEmployee]:
        title = item.get('title', '')
        link = item.get('link', '')
        snippet = item.get('snippet', '')
        if 'linkedin.com/in' not in link and 'linkedin.com/pub' not in link:
            return None
        name = title.split(' - ')[0] if ' - ' in title else title.split(' |')[0]
        name = name.strip()
        if not name or len(name) < 2:
            return None
        job_title = ""
        if ' - ' in title:
            parts = title.split(' - ', 1)
            if len(parts) > 1:
                job_title = parts[1].split(' | ')[0]
        return CompetitorEmployee(
            name=name,
            title=job_title.strip(),
            company=company_name,
            linkedin_url=link.rstrip('/'),
            search_snippet=snippet[:200]
        )

    def _deduplicate_employees(self, employees: List[CompetitorEmployee]) -> List[CompetitorEmployee]:
        def _canon(u: str) -> str:
            if not u: return ""
            u = u.strip().rstrip("/")
            u = u.split("?", 1)[0].split("#", 1)[0]
            u = re.sub(r"^https?://([a-z]{2,3}\.)?linkedin\.com/", "https://www.linkedin.com/", u, flags=re.I)
            return u.lower()
        seen, unique = set(), []
        for emp in employees:
            key = _canon(emp.linkedin_url)
            if key and key not in seen:
                seen.add(key)
                emp.linkedin_url = key
                unique.append(emp)
        return unique

    async def _mock_employee_search(self, competitors: List[CompetitorProfile]) -> Dict[str, List[CompetitorEmployee]]:
        logger.warning("Using mock employee data (Google API credentials missing)")
        mock_employees: Dict[str, List[CompetitorEmployee]] = {}
        for competitor in competitors:
            employees = []
            for i in range(5):
                employees.append(CompetitorEmployee(
                    name=f"Employee {i+1}",
                    title=f"Senior {competitor.industry} Specialist",
                    company=competitor.name,
                    linkedin_url=f"https://www.linkedin.com/in/mock-{competitor.name.lower()}-{i+1}",
                    search_snippet=f"Works at {competitor.name} in the {competitor.industry} industry"
                ))
            mock_employees[competitor.name] = employees
        return mock_employees


# =============================================================================
# Step 4: Profile Matching
# =============================================================================

class ProfileMatcher:
    """Matches target employees with competitor employees (Top-10 per target across all companies) with detailed logging."""

    MIN_SCORE = 40.0
    TOP_K_PER_TARGET = 10
    BATCH_SIZE = 5
    BATCH_SLEEP_SEC = 0.3

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        logger.info("ProfileMatcher initialized")

    # ------------------------------- public ---------------------------------

    async def match_profiles(
        self,
        target_profiles: List[TargetEmployeeProfile],
        competitor_employees: Dict[str, List[CompetitorEmployee]]
    ) -> Dict[str, List[EmployeeMatch]]:
        """
        1) Run matching against each company in parallel (no per-company cap).
        2) Aggregate ALL matches across companies.
        3) For each target, keep Top-10 across ALL companies.
        4) Re-bucket to {company: [EmployeeMatch]} for downstream steps.
        """
        logger.info("STEP 4: PROFILE MATCHING")
        logger.info("[Match] Targets=%d | Companies=%d", len(target_profiles), len(competitor_employees))

        # Log size per company up front
        for cname, emps in (competitor_employees or {}).items():
            logger.debug("[Match] Seed employees at %s: %d", cname, len(emps or []))

        # 1) per-company matching (parallel)
        tasks = [
            self._match_company(target_profiles, employees, company_name)
            for company_name, employees in (competitor_employees or {}).items()
        ]
        logger.info("Processing %d companies in parallel", len(tasks))
        t0_all = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dt_all = time.perf_counter() - t0_all
        logger.info("[Match] Per-company matching completed in %.2fs", dt_all)

        company_names = list(competitor_employees.keys())
        per_company_raw: Dict[str, List[EmployeeMatch]] = {}

        for i, result in enumerate(results):
            cname = company_names[i] if i < len(company_names) else f"Company_{i}"
            if isinstance(result, list):
                per_company_raw[cname] = result
                logger.info("   %s: %d quality matches (pre Top-10 compaction)", cname, len(result))
            else:
                logger.error("Matching failed for %s: %s", cname, result)
                per_company_raw[cname] = []

        # 2) aggregate all matches by target
        per_target_all: Dict[str, List[EmployeeMatch]] = {}
        for matches in per_company_raw.values():
            for m in matches or []:
                per_target_all.setdefault(m.target_employee, []).append(m)

        total_before = sum(len(v) for v in per_target_all.values())
        targets_with_any = sum(1 for v in per_target_all.values() if v)
        logger.info("[Match] Aggregated %d matches across %d targets (with ≥1 match)",
                    total_before, targets_with_any)

        # 3) keep Top-10 per target across ALL companies
        kept_per_target: Dict[str, List[EmployeeMatch]] = {}
        zero_after = 0
        for tname, t_matches in per_target_all.items():
            t_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            kept = t_matches[: self.TOP_K_PER_TARGET]
            kept_per_target[tname] = kept
            if kept:
                logger.debug("[Top-10] Target='%s' kept %d/%d (max=%d) | top scores=[%s]",
                             tname, len(kept), len(t_matches), self.TOP_K_PER_TARGET,
                             ", ".join(str(round(m.similarity_score, 1)) for m in kept[:5]))
            else:
                zero_after += 1
                logger.debug("[Top-10] Target='%s' kept 0/%d", tname, len(t_matches))

        total_after = sum(len(v) for v in kept_per_target.values())
        logger.info("[Top-10] Compaction: %d -> %d kept across %d targets (zero-kept=%d)",
                    total_before, total_after, len(kept_per_target), zero_after)

        # 4) re-bucket kept matches by company
        final_by_company: Dict[str, List[EmployeeMatch]] = {}
        for kept_list in kept_per_target.values():
            for m in kept_list:
                final_by_company.setdefault(m.competitor_company, []).append(m)

        # Per-company kept summary
        for cname, matches in sorted(final_by_company.items(), key=lambda kv: -len(kv[1])):
            by_target = {}
            for m in matches:
                by_target[m.target_employee] = by_target.get(m.target_employee, 0) + 1
            logger.info("   %s: %d kept after Top-10 (targets covered=%d)", cname, len(matches), len(by_target))

        logger.info("Total matches generated (Top-10-per-target): %d", total_after)
        return final_by_company

    # ----------------------------- internals --------------------------------

    async def _match_company(
        self,
        target_profiles: List[TargetEmployeeProfile],
        competitors: List[CompetitorEmployee],
        company_name: str
    ) -> List[EmployeeMatch]:
        """Match all targets against one company's employees; no final cap here."""
        logger.info("Matching against %s (%d employees)", company_name, len(competitors))
        all_matches: List[EmployeeMatch] = []
        t0 = time.perf_counter()

        for idx, target in enumerate(target_profiles, 1):
            logger.debug("[Target] %s | role='%s' | dept=%s | exp=%.1f | skills=%s",
                         target.name, target.title, target.department, target.experience_years,
                         ", ".join((target.key_skills or [])[:3]))
            t_matches = await self._match_single_target(target, competitors, company_name)
            all_matches.extend(t_matches)
            if idx % 10 == 0:
                logger.debug("[Progress:%s] processed %d targets, cumulative matches=%d",
                             company_name, idx, len(all_matches))

        good = [m for m in all_matches if (m.similarity_score or 0) >= self.MIN_SCORE]
        good.sort(key=lambda x: x.similarity_score, reverse=True)

        dt = time.perf_counter() - t0
        logger.info("   %s produced %d quality matches (min=%.0f) in %.2fs",
                    company_name, len(good), self.MIN_SCORE, dt)
        # Quick breakdown: top 5 scores
        logger.debug("   %s top scores: %s",
                     company_name,
                     ", ".join(str(round(m.similarity_score, 1)) for m in good[:5]))
        return good

    async def _match_single_target(
        self,
        target: TargetEmployeeProfile,
        competitors: List[CompetitorEmployee],
        company_name: str
    ) -> List[EmployeeMatch]:
        """Match one target against a company's competitors (batched to respect token limits)."""
        out: List[EmployeeMatch] = []
        total = len(competitors)
        batches = (total + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        logger.debug("[Batches] Target='%s' @ %s -> %d comps, %d batches (size=%d)",
                     target.name, company_name, total, batches, self.BATCH_SIZE)

        for i in range(0, total, self.BATCH_SIZE):
            batch_idx = (i // self.BATCH_SIZE) + 1
            batch = competitors[i:i + self.BATCH_SIZE]
            logger.debug("[BatchStart] target='%s' company='%s' batch=%d/%d size=%d",
                         target.name, company_name, batch_idx, batches, len(batch))
            t0 = time.perf_counter()
            try:
                matches = await self._analyze_batch_similarity(
                    target, batch, company_name, batch_idx=batch_idx, total_batches=batches
                )
                out.extend(matches)
                logger.debug("[BatchDone] target='%s' company='%s' batch=%d/%d kept=%d (cum=%d) in %.2fs",
                             target.name, company_name, batch_idx, batches, len(matches), len(out),
                             time.perf_counter() - t0)
            except Exception as e:
                logger.warning("[BatchError] target='%s' company='%s' batch=%d/%d err=%s",
                               target.name, company_name, batch_idx, batches, e)
            await asyncio.sleep(self.BATCH_SLEEP_SEC)
        return out

    async def _analyze_batch_similarity(
        self,
        target: TargetEmployeeProfile,
        competitors: List[CompetitorEmployee],
        company_name: str,
        *,
        batch_idx: int = 0,
        total_batches: int = 0
    ) -> List[EmployeeMatch]:
        """Analyze similarity using GPT; logs request/response sizes and parse results."""
        system_prompt = """Compare target employee with competitors. Score similarity 0-100.

Return ONLY JSON without any additional text:
{
  "matches": [
    {
      "competitor_name": "Name",
      "similarity_score": 85,
      "rationale": "Brief explanation"
    }
  ]
}

Only include matches with score >= 40."""
        comp_payload = [
            {"name": c.name, "title": c.title, "snippet": (c.search_snippet or "")[:150]}
            for c in competitors
        ]
        user_prompt = (
            f"TARGET: {target.name}\n"
            f"Title: {target.title}\n"
            f"Department: {target.department}\n"
            f"Skills: {', '.join((target.key_skills or [])[:3])}\n\n"
            f"COMPETITORS:\n{json.dumps(comp_payload, indent=2)}\n\n"
            f"Find similar profiles."
        )

        # Log request summary (not full payload)
        logger.debug("[GPT->] t='%s' c='%s' b=%d/%d payload=%d comps",
                     target.name, company_name, batch_idx, total_batches, len(comp_payload))

        t0 = time.perf_counter()
        raw = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.1)
        dt = time.perf_counter() - t0

        logger.debug("[GPT<-] t='%s' c='%s' b=%d/%d time=%.2fs size=%d chars",
                     target.name, company_name, batch_idx, total_batches, dt, len(raw or ""))

        if not raw:
            logger.debug("[Parse] Empty GPT response")
            return []

        data = safe_json_parse(raw)
        if not data or "matches" not in data or not isinstance(data["matches"], list):
            logger.debug("[Parse] No 'matches' array in GPT response")
            return []

        matches: List[EmployeeMatch] = []
        by_name = {c.name: c for c in competitors}

        kept = 0
        dropped = 0
        for row in data["matches"]:
            comp_name = (row or {}).get("competitor_name")
            comp = by_name.get((comp_name or "").strip())
            if not comp:
                dropped += 1
                continue
            try:
                score = float(row.get("similarity_score", 0))
            except (TypeError, ValueError):
                dropped += 1
                continue
            if score < self.MIN_SCORE:
                dropped += 1
                continue

            kept += 1
            matches.append(
                EmployeeMatch(
                    target_employee=target.name,
                    competitor_employee=comp.name,
                    competitor_company=company_name,
                    similarity_score=score,
                    match_rationale=((row.get("rationale") or "")[:200]),
                    linkedin_url=comp.linkedin_url,
                )
            )

        logger.debug("[Filter] t='%s' c='%s' b=%d/%d kept=%d dropped=%d (min=%.0f)",
                     target.name, company_name, batch_idx, total_batches, kept, dropped, self.MIN_SCORE)
        return matches


# Step 5: Spectre Matches Writer
class SpectreWriter:
    """Write spectre_matches.json file"""

    @staticmethod
    def write_spectre_matches(all_matches: List[EmployeeMatch],
                              target_profiles: List[TargetEmployeeProfile]) -> str:
        logger.info("STEP 5: WRITING SPECTRE MATCHES")

        target_by_name = {t.name: t for t in target_profiles}

        # company -> [EmployeeMatch, ...]
        by_company: Dict[str, List[EmployeeMatch]] = {}
        for m in all_matches or []:
            by_company.setdefault(m.competitor_company, []).append(m)

        spectre_data: Dict[str, Any] = {}
        total_pairs = 0

        for company, matches in by_company.items():
            groups: Dict[str, Dict[str, Any]] = {}
            for m in matches:
                tname = m.target_employee
                if tname not in groups:
                    tp = target_by_name.get(tname)
                    groups[tname] = {
                        "manipal_name": tname,
                        "manipal_role": (tp.title if tp else "Unknown Role"),
                        "matches": []
                    }
                groups[tname]["matches"].append({
                    "company": company.lower().replace(" ", "_"),
                    "name": m.competitor_employee,
                    "role": "Unknown Role",
                    "similarity": round(float(m.similarity_score), 2),
                    "via": "llm"
                })
                total_pairs += 1

            spectre_data[company.lower().replace(" ", "_")] = list(groups.values())

        out_path = "spectre_matches.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spectre_data, f, indent=2, ensure_ascii=False)

        # Sanity log so you can see what Spectre is going to count
        logger.info(f"Wrote {out_path} covering {len(by_company)} companies, total match-pairs: {total_pairs}")
        return out_path

# =============================================================================
# Output Writers
# =============================================================================

class OutputWriter:
    """Handles all output file generation"""
    
    @staticmethod
    def write_matched_details_with_scrapes(profile_matches: Dict[str, List[EmployeeMatch]],
                                           scraped_by_company: Dict[str, Dict[str, Any]]) -> str:
        """
        Persist a per-company JSON of matched people, including Bright Data 'detailed_profile' payloads.
        Path: matched_data/<company_safe>_matched_details.json
        """
        out_dir = Path("matched_data")
        out_dir.mkdir(exist_ok=True)

        for company, matches in (profile_matches or {}).items():
            safe_name = safe_filename(company)
            per_url = scraped_by_company.get(company, {})
            items = []
            for m in matches or []:
                url_key = (m.linkedin_url or "").rstrip("/")
                items.append({
                    "target_employee": m.target_employee,
                    "competitor_employee": m.competitor_employee,
                    "competitor_company": company,
                    "similarity_score": round(float(m.similarity_score), 2),
                    "match_rationale": m.match_rationale,
                    "linkedin_url": url_key,
                    "detailed_profile": per_url.get(url_key)  # may be None if scrape missing
                })
            with open(out_dir / f"{safe_name}_matched_details.json", "w", encoding="utf-8") as f:
                json.dump({"company": company, "matched": items}, f, indent=2, ensure_ascii=False)

        return str(out_dir)

    def write_spectre_matches(matches_by_company: Dict[str, List[EmployeeMatch]], 
                            target_profiles: List[TargetEmployeeProfile]) -> str:
        """Write spectre_matches.json in required format"""
        logger.info("Writing spectre_matches.json")
        
        # Build target profile lookup
        target_lookup = {profile.name: profile for profile in target_profiles}
        
        spectre_data = {}
        
        for company, matches in matches_by_company.items():
            company_entries = {}
            
            # Group matches by target employee
            for match in matches:
                target_name = match.target_employee
                target_profile = target_lookup.get(target_name)
                
                if target_name not in company_entries:
                    company_entries[target_name] = {
                        "manipal_name": target_name,
                        "manipal_role": target_profile.title if target_profile else "Unknown Role",
                        "matches": []
                    }
                
                company_entries[target_name]["matches"].append({
                    "company": company.lower().replace(" ", "_"),
                    "name": match.competitor_employee,
                    "role": "Unknown Role",  # Could be enhanced with more data
                    "similarity": round(match.similarity_score, 2),
                    "via": "llm"
                })
            
            spectre_data[company.lower().replace(" ", "_")] = list(company_entries.values())
        
        # Write file
        output_path = "spectre_matches.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(spectre_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Wrote {output_path}")
        return output_path
    
    @staticmethod
    def write_employee_reports(competitor_employees: Dict[str, List[CompetitorEmployee]],
                               scraped_details: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """
        Write individual company reports under employee_data/, preserving your existing schema:
        - employee_intelligence.employees[].basic_info
        - employee_intelligence.employees[].detailed_profile  (now populated when scraped_details present)
        """
        logger.info("Writing employee reports")

        output_dir = Path("employee_data")
        output_dir.mkdir(exist_ok=True)

        scraped_details = scraped_details or {}  # {company: {url->profile_dict}}

        for company, employees in (competitor_employees or {}).items():
            safe_name = safe_filename(company)
            report_path = output_dir / f"{safe_name}_report.json"

            by_url = scraped_details.get(company, {})  # map url->profile

            employees_data = []
            for emp in employees or []:
                url_norm = (emp.linkedin_url or "").rstrip("/")
                employees_data.append({
                    "basic_info": {
                        "name": emp.name,
                        "linkedin_url": emp.linkedin_url,
                        "company": emp.company,
                        "search_snippet": emp.search_snippet,
                        "title": emp.title
                    },
                    "detailed_profile": by_url.get(url_norm),  # inject if available
                    "data_status": {
                        "found_in_search": True,
                        "detailed_scraped": bool(by_url.get(url_norm))
                    }
                })

            report = {
                "mission_metadata": {
                    "agent_id": f"GHOST_SHADE_{safe_name.upper()}",
                    "target_company": company,
                    "mission_timestamp": datetime.utcnow().isoformat(),
                    "completion_timestamp": datetime.utcnow().isoformat(),
                    "mission_status": "COMPLETED"
                },
                "employee_intelligence": {
                    "summary": {
                        "total_employees_found": len(employees_data),
                        "detailed_profiles_scraped": sum(1 for e in employees_data if e["detailed_profile"]),
                        "scraping_success_rate": (
                            (sum(1 for e in employees_data if e["detailed_profile"]) / len(employees_data) * 100.0)
                            if employees_data else 0.0
                        )
                    },
                    "employees": employees_data
                }
            }

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Wrote {report_path}")

        logger.info(f"All reports written to {output_dir}")
        return str(output_dir)

# =============================================================================
# Bright Data Scraper
# =============================================================================
# === Bright Data scraper (independent, parallel-per-company; SHADE-compatible dataset flow) ===
import os, json, time, asyncio, logging, requests
from typing import Dict, List, Any, Tuple

bd_logger = logging.getLogger("BrightDataForMirage")

class BrightDataScraper:
    """
    Independent Bright Data scraper for MIRAGE.

    • Uses Bright Data Dataset Trigger API (same as SHADE): 
      - POST /datasets/v3/trigger?dataset_id=...&include_errors=true
      - GET  /datasets/v3/progress/<snapshot_id>
      - GET  /datasets/v3/snapshot/<snapshot_id>   (NDJSON)
    • ONE snapshot per company with ALL its URLs (no chunking)
    • Companies run in parallel (configurable via semaphore)
    • Returns: { company: { url: profile_dict } }
    • Env vars (mirrors SHADE): 
        BRIGHT_DATA_API_KEY, BRIGHT_DATA_DATASET_ID
    """

    def __init__(self):
        # Align with SHADE’s env usage
        self.api_key = os.getenv("BRIGHT_DATA_API_KEY")
        self.dataset_id = os.getenv("BRIGHT_DATA_DATASET_ID")

        # Endpoints (same as SHADE)
        self.trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={self.dataset_id}&include_errors=true"
        self.progress_base = "https://api.brightdata.com/datasets/v3/progress/"
        self.snapshot_base = "https://api.brightdata.com/datasets/v3/snapshot/"

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

        self.enabled = bool(self.api_key and self.dataset_id)
        if not self.enabled:
            bd_logger.warning("BrightDataScraper is disabled (need BRIGHT_DATA_API_KEY and BRIGHT_DATA_DATASET_ID).")

    # ------------ public: per-company parallel ------------
    async def scrape_profiles_per_company_parallel(
        self,
        urls_by_company: Dict[str, List[str]],
        max_company_parallel: int = 10,
        timeout_sec: int = 100000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Args:
            urls_by_company: { company: [linkedin profile urls] }
        Returns:
            { company: { url: profile_dict } }
        """
        if not self.enabled:
            return {}

        sem = asyncio.Semaphore(max_company_parallel)

        async def _run_one(company: str, urls: List[str]) -> Tuple[str, Dict[str, Any]]:
            async with sem:
                return await self._scrape_company_snapshot(company, urls, timeout_sec)

        tasks = [_run_one(c, u) for c, u in (urls_by_company or {}).items() if u]
        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, Dict[str, Any]] = {}
        for r in results:
            if isinstance(r, tuple) and len(r) == 2:
                c, urlmap = r
                out[c] = urlmap
            else:
                bd_logger.error(f"Company task failed: {r}")
        return out

    # ------------ internal: single company ------------
    async def _scrape_company_snapshot(self, company: str, urls: List[str], timeout_sec: int) -> Tuple[str, Dict[str, Any]]:
        urls = self._prepare_urls(urls)
        if not urls:
            bd_logger.warning(f"{company}: no valid LinkedIn URLs after cleaning")
            return company, {}

        try:
            profiles = await asyncio.to_thread(self._dataset_snapshot_sync, urls, timeout_sec)
        except Exception as e:
            bd_logger.error(f"{company}: snapshot failed: {e}")
            return company, {}

        # normalize to {url: profile}
        url_map: Dict[str, Any] = {}
        for p in profiles or []:
            u = (p.get("url") or p.get("profile_url") or "").strip().rstrip("/")
            if u:
                url_map[u] = p
        bd_logger.info(f"{company}: scraped {len(url_map)} profiles")
        return company, url_map

    # ------------ dataset trigger/progress/snapshot (sync) ------------
    def _dataset_snapshot_sync(self, urls: List[str], timeout_sec: int) -> List[Dict[str, Any]]:
        snapshot_id = self._trigger(urls)
        if not snapshot_id:
            return []
        if not self._wait_ready(snapshot_id, timeout_sec=timeout_sec, interval=10):
            return []
        return self._fetch(snapshot_id)

    def _trigger(self, urls: List[str]) -> str | None:
        payload = [{"url": u} for u in urls]
        try:
            r = self.session.post(self.trigger_url, json=payload, timeout=60)
            if r.ok:
                js = r.json()
                sid = js.get("snapshot_id") or js.get("snapshot") or js.get("id")
                if not sid:
                    bd_logger.error(f"Trigger missing snapshot id: {js}")
                return sid
            bd_logger.error(f"Trigger error {r.status_code}: {r.text}")
        except Exception as e:
            bd_logger.error(f"Trigger failed: {e}")
        return None

    def _wait_ready(self, snapshot_id: str, timeout_sec: int, interval: int = 10) -> bool:
        elapsed = 0
        while elapsed <= timeout_sec:
            try:
                r = self.session.get(self.progress_base + snapshot_id, timeout=30)
                if r.ok:
                    js = r.json()
                    status = (js.get("status") or js.get("state") or "").lower()
                    if status == "ready":
                        return True
                    if status == "error":
                        bd_logger.error(f"Snapshot error: {js}")
                        return False
            except Exception as e:
                bd_logger.warning(f"Progress poll error: {e}")
            time.sleep(interval)
            elapsed += interval
        bd_logger.error("Snapshot timed out")
        return False

    def _fetch(self, snapshot_id: str) -> List[Dict[str, Any]]:
        try:
            r = self.session.get(self.snapshot_base + snapshot_id, timeout=120)
            if not r.ok:
                bd_logger.error(f"Snapshot fetch error {r.status_code}: {r.text}")
                return []
            lines = [ln for ln in r.text.splitlines() if ln.strip()]
            out: List[Dict[str, Any]] = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    if "url" not in obj and "profile_url" in obj:
                        obj["url"] = obj["profile_url"]
                    out.append(obj)
                except Exception:
                    pass
            return out
        except Exception as e:
            bd_logger.error(f"Snapshot fetch failed: {e}")
            return []

    # ------------ utils ------------
    def _prepare_urls(self, urls: List[str]) -> List[str]:
        cleaned, seen = [], set()
        for u in urls or []:
            if not u:
                continue
            u2 = u.strip().rstrip("/")
            if ("linkedin.com/in" in u2 or "linkedin.com/pub" in u2) and u2 not in seen:
                seen.add(u2); cleaned.append(u2)
        return cleaned


# === Mirage-facing helper (keeps Mirage call-sites unchanged) ===
async def scrape_matched_profiles_per_company_parallel(
    shade_scraper: BrightDataScraper,
    matched_urls_per_company: Dict[str, List[str]],
    **kwargs
) -> Dict[str, Dict[str, Dict]]:
    return await shade_scraper.scrape_profiles_per_company_parallel(
        matched_urls_per_company,
        max_company_parallel=int(kwargs.get("max_company_parallel", 3)),
        timeout_sec=int(kwargs.get("timeout_sec", 100000)),
    )

# =============================================================================
# Main MIRAGE System
# =============================================================================
try:
    from shade import BrightDataScraper as ShadeBrightScraper
except Exception as _e:
    ShadeBrightScraper = BrightDataScraper
except Exception as _e:
    ShadeBrightScraper = None
    logger = logging.getLogger('MIRAGE')
    logger.warning(f"Could not import ShadeBrightScraper: {_e}")

class MirageSystem:
    """Main MIRAGE system orchestrator"""
    
    def __init__(self):
        logger.info("=== MIRAGE SYSTEM INITIALIZATION ===")
        
        # Validate environment
        self._validate_environment()
        
        # Initialize GPT client
        self.gpt_client = AzureGPTClient()
        
        # Initialize components
        self.competitor_detector = CompetitorDetector(self.gpt_client)
        self.profile_builder = TargetProfileBuilder(self.gpt_client)
        self.employee_finder = CompetitorEmployeeFinder(self.gpt_client)
        self.profile_matcher = ProfileMatcher(self.gpt_client)
        
        logger.info("MIRAGE system initialized successfully")
    
    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = {
            "AZURE_OPENAI_API_KEY": "Azure OpenAI API key",
            "AZURE_OPENAI_ENDPOINT": "Azure OpenAI endpoint",
        }
        
        missing = []
        for var, desc in required_vars.items():
            if not os.getenv(var):
                missing.append(f"{var}: {desc}")
        
        if missing:
            error_msg = "Missing required environment variables:\n" + "\n".join(f"  - {var}" for var in missing)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    async def run_full_analysis(self, intelligence_data: Dict, num_competitors: int = 10) -> Dict:
        """Execute complete MIRAGE analysis pipeline (Spectre FIRST, then Bright Data)."""
        logger.info(f"=== MIRAGE ANALYSIS STARTED (competitors={num_competitors}) ===")
        start_time = time.time()

        try:
            # Phase 1: Competitor Detection (strict limit)
            logger.info("Phase 1: Competitor Detection")
            competitors = await self.competitor_detector.detect_competitors(
                intelligence_data, max_competitors=num_competitors
            )
            if not competitors:
                raise ValueError("No competitors detected")

            # Phase 2: Target Profile Building
            logger.info("Phase 2: Target Profile Building")
            employee_data = self._extract_employee_data(intelligence_data)
            target_profiles = await self.profile_builder.build_target_profiles(employee_data)
            if not target_profiles:
                logger.warning("No target profiles built, using mock data")
                target_profiles = self._create_mock_profiles()

            # Phases 3 & 4: Employee Search and Matching (logic unchanged)
            logger.info("Phase 3 & 4: Employee Search and Matching (parallel)")
            competitor_employees = await self.employee_finder.find_competitor_employees(
                target_profiles, competitors
            )
            if not competitor_employees:
                logger.warning("No competitor employees found, using mock matches")
                profile_matches = self._create_mock_matches(target_profiles, competitors)
            else:
                profile_matches = await self.profile_matcher.match_profiles(
                    target_profiles, competitor_employees
                )

                        # ---- PHASE 5: WRITE SPECTRE MATCHES FIRST ----
            logger.info("Phase 5: Writing Spectre Matches (FIRST)")

            # profile_matches is a dict {company: [EmployeeMatch]}
            spectre_path = OutputWriter.write_spectre_matches(profile_matches, target_profiles)

            # Quick sanity: how many pairs did we just write?
            try:
                with open(spectre_path, "r", encoding="utf-8") as f:
                    _spectre_json = json.load(f)
                _pairs_in_file = sum(
                    len(group.get("matches", []))
                    for company_groups in (_spectre_json or {}).values()
                    for group in (company_groups or [])
                )
                logger.info("[Spectre sanity] pairs in file: %d", _pairs_in_file)
            except Exception as e:
                logger.warning("[Spectre sanity] could not verify file: %s", e)

            # ---- PHASE 6: BRIGHT DATA SCRAPING (AFTER spectre write) ----
            bd_logger.info("Phase 6: Scraping matched LinkedIn profiles with Bright Data (AFTER spectre write)")
            matched_urls_per_company = collect_matched_linkedin_urls(profile_matches)
            total_urls = sum(len(v) for v in matched_urls_per_company.values())
            bd_logger.info("Dispatching %d URLs across %d companies", total_urls, len(matched_urls_per_company))

            scraped_by_company: Dict[str, Dict[str, Any]] = {}
            total_scraped = 0
            try:
                scraper = BrightDataScraper()
                if scraper.enabled and matched_urls_per_company:
                    scraped_by_company = await scrape_matched_profiles_per_company_parallel(
                        scraper,
                        matched_urls_per_company,
                        max_company_parallel=3,
                        timeout_sec=100000,
                    )
                    total_scraped = sum(len(v or {}) for v in scraped_by_company.values())
                elif not scraper.enabled:
                    bd_logger.warning("BrightDataScraper not enabled (set BRIGHT_DATA_API_KEY and BRIGHT_DATA_DATASET_ID)")
                else:
                    bd_logger.warning("No matched URLs to scrape — skipping Bright Data scrape")
            except Exception as e:
                bd_logger.error(f"Bright Data scraping failed: {e}")
                scraped_by_company = {}
                total_scraped = 0

            bd_logger.info("Bright Data scraping complete. Profiles scraped: %d", total_scraped)

            # ---- PHASE 7: OUTPUT GENERATION (reports enriched with scrapes) ----
            logger.info("Phase 7: Output Generation")
            reports_dir = OutputWriter.write_employee_reports(
                competitor_employees,
                scraped_details=scraped_by_company
            )
            matched_dir = OutputWriter.write_matched_details_with_scrapes(
                profile_matches, scraped_by_company
            )

            # ---- Summary & return payload ----
            execution_time = time.time() - start_time

            # Fix counters to handle dict structure
            companies_with_matches = len([k for k, v in (profile_matches or {}).items() if v])
            targets_with_kept = len({
                m.target_employee
                for matches in (profile_matches or {}).values()
                for m in (matches or [])
            })

            results = {
                "mirage_metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "target_company": self.competitor_detector._extract_company_name(intelligence_data),
                    "execution_time_seconds": round(execution_time, 2),
                    "version": "3.0-Clean-Production",
                },
                "results_summary": {
                    "competitors_detected": len(competitors),
                    "target_profiles_built": len(target_profiles),
                    "total_candidates_found": sum(len(v or []) for v in (competitor_employees or {}).values()),
                    "companies_with_matches": companies_with_matches,
                    "targets_with_kept_matches": targets_with_kept,
                    "total_matches_kept": sum(len(v or []) for v in (profile_matches or {}).values()),
                },
                "output_files": {
                    "spectre_matches": spectre_path,
                    "employee_reports_dir": reports_dir,
                    "matched_details_dir": matched_dir,
                },
            }
            return results


        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"MIRAGE analysis failed after {execution_time:.2f}s: {e}")
            raise

    
    def _extract_employee_data(self, intelligence_data: Dict) -> List[Dict]:
        """Extract employee data from intelligence report"""
        # Try various possible locations
        locations = [
            intelligence_data.get("employee_intelligence", {}).get("employees", []),
            intelligence_data.get("employees", []),
            intelligence_data.get("employee_data", [])
        ]
        
        for location in locations:
            if location and isinstance(location, list) and len(location) > 0:
                logger.info(f"Found {len(location)} employees in intelligence data")
                return location
        
        logger.warning("No employee data found in intelligence report")
        return []
    
    def _create_mock_profiles(self) -> List[TargetEmployeeProfile]:
        """Create mock employee profiles for testing"""
        mock_profiles = [
            TargetEmployeeProfile(
                name="John Smith",
                title="Software Engineer",
                department="Engineering",
                experience_years=5.0,
                key_skills=["Python", "JavaScript", "AWS"],
                company="Manipal Fintech"
            ),
            TargetEmployeeProfile(
                name="Jane Doe",
                title="Product Manager",
                department="Product",
                experience_years=7.0,
                key_skills=["Product Strategy", "Agile", "UX"],
                company="Manipal Fintech"
            ),
            TargetEmployeeProfile(
                name="Robert Johnson",
                title="Data Scientist",
                department="Data",
                experience_years=4.0,
                key_skills=["Python", "Machine Learning", "SQL"],
                company="Manipal Fintech"
            )
        ]
        return mock_profiles
    
    def _create_mock_matches(self, target_profiles: List[TargetEmployeeProfile], 
                           competitors: List[CompetitorProfile]) -> Dict[str, List[EmployeeMatch]]:
        """Create mock employee matches for testing"""
        matches_by_company = {}
        
        for competitor in competitors:
            matches = []
            for target in target_profiles[:3]:  # Match first 3 targets
                matches.append(EmployeeMatch(
                    target_employee=target.name,
                    competitor_employee=f"Competitor Employee {target.name.split()[0]}",
                    competitor_company=competitor.name,
                    similarity_score=70.0,
                    match_rationale=f"Similar role and skills in {competitor.industry}",
                    linkedin_url=f"https://linkedin.com/in/mock-{competitor.name.lower()}-{target.name.split()[0].lower()}"
                ))
            matches_by_company[competitor.name] = matches
        
        return matches_by_company
    
    def _generate_summary(self, intelligence_data: Dict, competitors: List[CompetitorProfile],
                         target_profiles: List[TargetEmployeeProfile], 
                         profile_matches: Dict[str, List[EmployeeMatch]],
                         execution_time: float) -> Dict:
        """Generate execution summary"""
        
        company_name = self.competitor_detector._extract_company_name(intelligence_data)
        total_matches = sum(len(matches) for matches in profile_matches.values())
        high_quality_matches = sum(
            len([m for m in matches if m.similarity_score >= 70]) 
            for matches in profile_matches.values()
        )
        
        return {
            "mirage_metadata": {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "target_company": company_name,
                "execution_time_seconds": round(execution_time, 2),
                "version": "3.0-Clean-Production"
            },
            "results_summary": {
                "competitors_detected": len(competitors),
                "target_profiles_built": len(target_profiles),
                "total_matches": total_matches,
                "high_quality_matches": high_quality_matches,
                "companies_analyzed": list(profile_matches.keys())
            },
            "output_files": {
                "spectre_matches": "spectre_matches.json",
                "employee_reports": "employee_data/"
            },
            "competitors": [asdict(comp) for comp in competitors],
            "target_profiles": [asdict(profile) for profile in target_profiles]
        }
def collect_matched_linkedin_urls(profile_matches: Dict[str, List[EmployeeMatch]]) -> Dict[str, List[str]]:
    per_company: Dict[str, List[str]] = {}
    for company, matches in (profile_matches or {}).items():
        seen, urls = set(), []
        for m in matches or []:
            u = (m.linkedin_url or "").strip().rstrip("/")
            if u and ("linkedin.com/in" in u or "linkedin.com/pub" in u) and u not in seen:
                seen.add(u)
                urls.append(u)
        per_company[company] = urls
    return per_company

# =============================================================================
# Entry Points
# =============================================================================
def collect_matched_linkedin_urls(profile_matches: Dict[str, List[EmployeeMatch]]) -> Dict[str, List[str]]:
    """
    From {company: [EmployeeMatch, ...]}, collect unique LinkedIn URLs per competitor.
    Returns {company: [url, ...]}.
    """
    per_company: Dict[str, List[str]] = {}
    for company, matches in (profile_matches or {}).items():
        bucket = []
        seen = set()
        for m in matches or []:
            u = (m.linkedin_url or "").strip().rstrip("/")
            if u and u not in seen:
                seen.add(u)
                bucket.append(u)
        per_company[company] = bucket
    return per_company

def extract_company_name(intelligence_data: Dict) -> str:
    """Extract company name from intelligence data"""
    locations = [
        intelligence_data.get("report_metadata", {}).get("company_name"),
        intelligence_data.get("company_intelligence", {}).get("basic_info", {}).get("name"),
        intelligence_data.get("mission_metadata", {}).get("target_company"),
        intelligence_data.get("company_name")
    ]
    
    for name in locations:
        if name and isinstance(name, str) and name.strip():
            return name.strip()
    
    return "Unknown Company"

async def mirage_async_entry(context: Dict[str, Any]) -> Dict[str, Any]:
    """Async entry point for orchestrator integration"""
    inputs = context.get("inputs", {})
    report_path = inputs.get("intelligence_report_path")
    
    if not report_path or not os.path.exists(report_path):
        raise FileNotFoundError(f"Intelligence report not found: {report_path}")
    
    # Get competitor limit with strict enforcement
    num_competitors = inputs.get("num_competitors") or inputs.get("competitors_limit") or 10
    if isinstance(num_competitors, str):
        try:
            num_competitors = int(num_competitors)
        except ValueError:
            num_competitors = 10
    
    # Enforce reasonable limits
    num_competitors = max(1, min(num_competitors, 50))
    
    logger.info(f"Processing with strict limit: {num_competitors} competitors")
    
    # Load intelligence data
    with open(report_path, "r", encoding="utf-8") as f:
        intelligence_data = json.load(f)
    
    company_name = extract_company_name(intelligence_data)
    logger.info(f"Starting MIRAGE analysis for: {company_name}")
    
    # Run analysis
    mirage = MirageSystem()
    results = await mirage.run_full_analysis(intelligence_data, num_competitors)
    
    # Return orchestrator-compatible format
     # --- SAFE METADATA BUILD (drop-in replacement) -------------------------------
    def _to_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    # results is whatever your pipeline produced above
    summary = results.get("results_summary") or {}

    # Try multiple places to infer companies list
    companies = results.get("companies") or results.get("competitors") or []
    if isinstance(companies, dict):
        # handle shapes like {"items": [...]} or {"list":[...]}
        companies = companies.get("items") or companies.get("list") or []

    # Derive competitors_detected safely
    competitors_detected = summary.get("competitors_detected")
    if competitors_detected is None:
        # sometimes code uses a different key; normalize
        competitors_detected = summary.get("total_competitors_detected")
    competitors_detected = _to_int(
        competitors_detected if competitors_detected is not None else len(companies),
        default=len(companies),
    )

    # Ensure results_summary exists and is normalized
    summary["competitors_detected"] = competitors_detected
    results["results_summary"] = summary

    ghost_mirage_metadata = {
        "target_company": company_name or "Unknown",
        "total_competitors_detected": competitors_detected,
        # add any other fields you already compute (timestamps, version, etc.)
    }

    # Final shape MIRAGE returns
    out = {
        "agents": {
            "mirage": {
                "result": {
                    **results,
                    "ghost_mirage_metadata": ghost_mirage_metadata,
                }
            }
        }
    }
    return out
    # ---------------------------------------------------------------------------


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous entry point for orchestrator"""
    try:
        return asyncio.run(mirage_async_entry(context))
    except RuntimeError:
        # Handle existing event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(mirage_async_entry(context))
        finally:
            loop.close()

def run_sync(context: Dict[str, Any]) -> Dict[str, Any]:
    """Alternative synchronous entry point"""
    return run(context)

# =============================================================================
# CLI Interface
# =============================================================================

async def main():

    """CLI interface for testing"""
    print(f"   BRIGHT_DATA_API_KEY: {'✅ Configured' if os.getenv('BRIGHT_DATA_API_KEY') else '⚠️ Not configured'}")
    print(f"   BRIGHT_DATA_COLLECTOR_ID/ENDPOINT: {'✅ Configured' if (os.getenv('BRIGHT_DATA_COLLECTOR_ID') or os.getenv('BRIGHT_DATA_ENDPOINT')) else '⚠️ Not configured'}")
    print("MIRAGE System - Clean Production Version")
    print("=" * 45)
    
    # Check environment
    try:
        mirage = MirageSystem()
        print("Configuration: OK")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    
    # Get inputs
    report_path = input("Intelligence report path: ").strip()
    if not os.path.exists(report_path):
        print(f"Error: File not found - {report_path}")
        return
    
    competitors_input = input("Number of competitors [10]: ").strip()
    num_competitors = 10
    if competitors_input:
        try:
            num_competitors = int(competitors_input)
            num_competitors = max(1, min(num_competitors, 50))
        except ValueError:
            print("Invalid number, using default: 10")
    
    try:
        # Load data
        with open(report_path, 'r', encoding='utf-8') as f:
            intelligence_data = json.load(f)
        
        company_name = extract_company_name(intelligence_data)
        print(f"\nAnalyzing: {company_name}")
        print(f"Competitors limit: {num_competitors}")
        print("Starting analysis...")
        
        # Run analysis
        results = await mirage.run_full_analysis(intelligence_data, num_competitors)
        # after: results = await mirage.run_full_analysis(intelligence_data, num_competitors)

        if not results:
            raise RuntimeError("MIRAGE returned no results")

        print("\n" + "=" * 45)
        print("ANALYSIS COMPLETE")
        print("=" * 45)

        mm = results.get('mirage_metadata') or {}
        rs = results.get('results_summary') or {}

        print(f"Company: {mm.get('target_company', extract_company_name(intelligence_data))}")
        print(f"Execution Time: {mm.get('execution_time_seconds', 'n/a')}s")
        print(f"Competitors Detected: {rs.get('competitors_detected', 0)}")
        print(f"Target Profiles: {rs.get('target_profiles_built', 0)}")
        print(f"Total Matches: {rs.get('total_matches', 0)}")
        print(f"High Quality Matches: {rs.get('high_quality_matches', 0)}")

        print("\nOutput Files:")
        print("- spectre_matches.json")
        print("- employee_data/")
        for company in (rs.get('companies_analyzed') or []):
            safe_name = safe_filename(company)
            print(f"  - {safe_name}_report.json")

        # save detailed report safely
        timestamp = int(time.time())
        company_name = mm.get('target_company') or extract_company_name(intelligence_data)
        output_file = f"mirage_analysis_{safe_filename(company_name)}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report: {output_file}")

        # Display results
        print("\n" + "=" * 45)
        print("ANALYSIS COMPLETE")
        print("=" * 45)
        print(f"Company: {results['mirage_metadata']['target_company']}")
        print(f"Execution Time: {results['mirage_metadata']['execution_time_seconds']}s")
        print(f"Competitors Detected: {results['results_summary']['competitors_detected']}")
        print(f"Target Profiles: {results['results_summary']['target_profiles_built']}")
        print(f"Total Matches: {results['results_summary']['total_matches']}")
        print(f"High Quality Matches: {results['results_summary']['high_quality_matches']}")
        print("\nOutput Files:")
        print("- spectre_matches.json")
        print("- employee_data/")
        for company in results['results_summary']['companies_analyzed']:
            safe_name = safe_filename(company)
            print(f"  - {safe_name}_report.json")
        
        # Save detailed report
        timestamp = int(time.time())
        output_file = f"mirage_analysis_{safe_filename(company_name)}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed report: {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        prog="mirage",
        description="MIRAGE - Clean GPT-Powered Competitive Intelligence"
    )
    parser.add_argument("--report", "-r", help="Path to intelligence report JSON")
    parser.add_argument("--competitors", "-n", type=int, default=10, 
                       help="Number of competitors (1-50, default: 10)")
    
    args, unknown = parser.parse_known_args()
    
    if args.report:
        # CLI mode with arguments
        ctx = {
            "inputs": {
                "intelligence_report_path": args.report,
                "num_competitors": max(1, min(args.competitors, 50))
            }
        }
        
        try:
            result = run(ctx)
            meta = (result.get("agents", {})
                         .get("mirage", {})
                         .get("result", {})
                         .get("ghost_mirage_metadata", {}))
            
            print(f"\nMIRAGE Analysis Complete")
            print(f"Target: {meta.get('target_company', 'Unknown')}")
            print(f"Competitors: {meta.get('total_competitors_detected', 'n/a')}")
            print(f"Execution Time: {meta.get('execution_time', 'n/a')}s")
            print("Outputs:")
            print("- spectre_matches.json")
            print("- employee_data/<company>_report.json")
            
        except Exception as e:
            logger.error(f"CLI execution failed: {e}")
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        try:
            asyncio.run(main())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
            loop.close()