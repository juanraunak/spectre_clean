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
    timeout_sec: int = 900,
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
    timeout_sec: int = 900,
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
    """Finds competitor employees using Google Custom Search"""
    
    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Missing Google API credentials - using mock data")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
            
        logger.info("CompetitorEmployeeFinder initialized")
    
    async def find_competitor_employees(self, target_profiles: List[TargetEmployeeProfile], 
                                      competitors: List[CompetitorProfile]) -> Dict[str, List[CompetitorEmployee]]:
        """Find competitor employees for all companies"""
        logger.info("STEP 3: COMPETITOR EMPLOYEE SEARCH")
        
        if self.use_mock_data:
            return await self._mock_employee_search(competitors)
        
        # Process all competitors in parallel
        tasks = []
        for competitor in competitors:
            task = self._find_employees_for_company(target_profiles, competitor)
            tasks.append(task)
        
        logger.info(f"Processing {len(competitors)} companies in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        final_results = {}
        for i, result in enumerate(results):
            if isinstance(result, dict):
                final_results.update(result)
            elif isinstance(result, Exception):
                company_name = competitors[i].name if i < len(competitors) else "Unknown"
                logger.error(f"Search failed for {company_name}: {result}")
        
        total_found = sum(len(employees) for employees in final_results.values())
        logger.info(f"Total employees found: {total_found}")
        
        return final_results
    
    async def _find_employees_for_company(self, target_profiles: List[TargetEmployeeProfile], 
                                        competitor: CompetitorProfile) -> Dict[str, List[CompetitorEmployee]]:
        """Find employees for a single company"""
        logger.info(f"Searching employees at {competitor.name}")
        
        all_employees = []
        
        # Generate searches for different departments
        departments = list(set(profile.department for profile in target_profiles))
        
        for department in departments[:3]:  # Limit departments
            logger.debug(f"   Searching {department} at {competitor.name}")
            
            queries = await self._generate_search_queries(department, competitor.name)
            employees = await self._execute_searches(queries, competitor.name)
            all_employees.extend(employees)
            
            await asyncio.sleep(1)  # Rate limiting
        
        # Deduplicate
        unique_employees = self._deduplicate_employees(all_employees)
        logger.info(f"   Found {len(unique_employees)} employees at {competitor.name}")
        
        return {competitor.name: unique_employees}
    
    async def _generate_search_queries(self, department: str, company_name: str) -> List[str]:
        """Generate search queries for department and company"""
        
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
        """Execute Google search queries"""
        all_employees = []
        
        for query in queries:
            try:
                logger.debug(f"Search: {query[:60]}...")
                
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.google_api_key,
                    'cx': self.google_cse_id,
                    'q': query,
                    'num': 10
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            items = data.get('items', [])
                            
                            for item in items:
                                employee = self._parse_search_result(item, company_name)
                                if employee:
                                    all_employees.append(employee)
                        else:
                            logger.warning(f"Search failed: {response.status}")
                
                await asyncio.sleep(1.2)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Search error: {e}")
        
        return all_employees
    
    def _parse_search_result(self, item: Dict, company_name: str) -> Optional[CompetitorEmployee]:
        """Parse search result to CompetitorEmployee"""
        title = item.get('title', '')
        link = item.get('link', '')
        snippet = item.get('snippet', '')
        
        if 'linkedin.com/in' not in link:
            return None
        
        # Extract name from title
        name = title.split(' - ')[0] if ' - ' in title else title.split(' |')[0]
        name = name.strip()
        
        if not name or len(name) < 2:
            return None
        
        # Extract job title
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
        """Remove duplicates by LinkedIn URL"""
        seen_urls = set()
        unique = []
        
        for emp in employees:
            url_key = emp.linkedin_url.lower().rstrip('/')
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique.append(emp)
        
        return unique
    
    async def _mock_employee_search(self, competitors: List[CompetitorProfile]) -> Dict[str, List[CompetitorEmployee]]:
        """Mock employee search for testing when API credentials are missing"""
        logger.warning("Using mock employee data (Google API credentials missing)")
        
        mock_employees = {}
        for competitor in competitors:
            employees = []
            # Generate some mock employees for each competitor
            for i in range(5):
                employees.append(CompetitorEmployee(
                    name=f"Employee {i+1}",
                    title=f"Senior {competitor.industry} Specialist",
                    company=competitor.name,
                    linkedin_url=f"https://linkedin.com/in/mock-{competitor.name.lower()}-{i+1}",
                    search_snippet=f"Works at {competitor.name} in the {competitor.industry} industry"
                ))
            mock_employees[competitor.name] = employees
        
        return mock_employees

# =============================================================================
# Step 4: Profile Matching
# =============================================================================

class ProfileMatcher:
    """Matches target employees with competitor employees"""
    
    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        logger.info("ProfileMatcher initialized")
    
    async def match_profiles(self, target_profiles: List[TargetEmployeeProfile],
                           competitor_employees: Dict[str, List[CompetitorEmployee]]) -> Dict[str, List[EmployeeMatch]]:
        """Match profiles across all companies in parallel"""
        logger.info("STEP 4: PROFILE MATCHING")
        
        # Process all companies in parallel
        tasks = []
        for company_name, employees in competitor_employees.items():
            task = self._match_company(target_profiles, employees, company_name)
            tasks.append(task)
        
        logger.info(f"Processing {len(competitor_employees)} companies in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        all_matches = {}
        company_names = list(competitor_employees.keys())
        
        for i, result in enumerate(results):
            if isinstance(result, list):
                company_name = company_names[i] if i < len(company_names) else f"Company_{i}"
                all_matches[company_name] = result
            elif isinstance(result, Exception):
                company_name = company_names[i] if i < len(company_names) else f"Company_{i}"
                logger.error(f"Matching failed for {company_name}: {result}")
        
        total_matches = sum(len(matches) for matches in all_matches.values())
        logger.info(f"Total matches generated: {total_matches}")
        
        return all_matches
    
    async def _match_company(self, target_profiles: List[TargetEmployeeProfile],
                           competitors: List[CompetitorEmployee], company_name: str) -> List[EmployeeMatch]:
        """Match target profiles against one company's employees"""
        logger.info(f"Matching against {company_name} ({len(competitors)} employees)")
        
        all_matches = []
        
        # Process targets in smaller batches
        for target in target_profiles[:10]:  # Limit targets
            matches = await self._match_single_target(target, competitors, company_name)
            all_matches.extend(matches)
        
        # Filter and sort by similarity
        good_matches = [m for m in all_matches if m.similarity_score >= 40]
        good_matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(f"   {len(good_matches)} quality matches for {company_name}")
        return good_matches[:15]  # Keep top 15
    
    async def _match_single_target(self, target: TargetEmployeeProfile, 
                                 competitors: List[CompetitorEmployee],
                                 company_name: str) -> List[EmployeeMatch]:
        """Match one target against competitors"""
        
        # Process in batches to avoid token limits
        batch_size = 5
        all_matches = []
        
        for i in range(0, len(competitors), batch_size):
            batch = competitors[i:i+batch_size]
            matches = await self._analyze_batch_similarity(target, batch, company_name)
            all_matches.extend(matches)
            
            await asyncio.sleep(0.3)  # Brief pause
        
        return all_matches
    
    async def _analyze_batch_similarity(self, target: TargetEmployeeProfile,
                                      competitors: List[CompetitorEmployee],
                                      company_name: str) -> List[EmployeeMatch]:
        """Analyze similarity using GPT"""
        
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
        
        competitor_info = []
        for comp in competitors:
            competitor_info.append({
                "name": comp.name,
                "title": comp.title,
                "snippet": comp.search_snippet[:150]
            })
        
        user_prompt = f"""TARGET: {target.name}
Title: {target.title}
Department: {target.department}
Skills: {', '.join(target.key_skills[:3])}

COMPETITORS:
{json.dumps(competitor_info, indent=2)}

Find similar profiles."""
        
        response = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.1)
        
        if not response:
            return []
        
        analysis = safe_json_parse(response)
        if not analysis or 'matches' not in analysis:
            return []
        
        # Convert to EmployeeMatch objects
        matches = []
        competitor_map = {comp.name: comp for comp in competitors}
        
        for match_data in analysis['matches']:
            competitor_name = match_data.get('competitor_name', '')
            competitor = competitor_map.get(competitor_name)
            
            if not competitor:
                continue
            
            try:
                match = EmployeeMatch(
                    target_employee=target.name,
                    competitor_employee=competitor.name,
                    competitor_company=company_name,
                    similarity_score=float(match_data.get('similarity_score', 0)),
                    match_rationale=match_data.get('rationale', '')[:200],
                    linkedin_url=competitor.linkedin_url
                )
                
                if match.similarity_score >= 40:
                    matches.append(match)
            except (ValueError, TypeError):
                continue
        
        return matches

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

# =============================================================================
# Bright Data Scraper - Optimized for Per-Company Parallel Snapshots
# =============================================================================

# =============================================================================
# Bright Data Scraper - Optimized for Per-Company Parallel Snapshots
# =============================================================================

class BrightDataScraper:
    """
    Optimized Bright Data scraper that:
    - Creates one snapshot per company with ALL URLs for that company
    - Runs company snapshots in parallel
    - No URL chunking - sends complete employee list per snapshot
    """

    def __init__(self):
        self.api_key = os.getenv("BRIGHT_DATA_API_KEY")
        self.collector_id = os.getenv("BRIGHT_DATA_COLLECTOR_ID")
        self.endpoint = os.getenv("BRIGHT_DATA_ENDPOINT")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.enabled = bool(self.api_key and (self.collector_id or self.endpoint))

    def _prepare_urls(self, urls: List[str]) -> List[str]:
        """Clean and deduplicate LinkedIn profile URLs"""
        cleaned = []
        seen = set()
        
        for url in urls:
            if not url:
                continue
            url = url.strip().rstrip("/")
            if ("linkedin.com/in" in url or "linkedin.com/pub" in url) and url not in seen:
                seen.add(url)
                cleaned.append(url)
        
        return cleaned

    async def scrape_profiles_per_company_parallel(
        self,
        urls_by_company: Dict[str, List[str]],
        max_company_parallel: int = 3,
        timeout_sec: int = 900
    ) -> Dict[str, Dict[str, Any]]:
        """
        Main entry point: scrape profiles with one snapshot per company, run in parallel
        
        Args:
            urls_by_company: {company_name: [list_of_urls]}
            max_company_parallel: Max companies to process simultaneously
            timeout_sec: Timeout for each snapshot
            
        Returns:
            {company_name: {url: profile_data}}
        """
        if not self.enabled:
            logging.warning("BrightDataScraper disabled (missing API key/endpoint)")
            return {}

        # Create semaphore to limit parallel company processing
        semaphore = asyncio.Semaphore(max_company_parallel)
        
        async def _process_company(company: str, urls: List[str]):
            async with semaphore:
                return await self._scrape_company_snapshot(company, urls, timeout_sec)

        # Launch all company tasks in parallel
        tasks = [
            _process_company(company, urls) 
            for company, urls in urls_by_company.items() 
            if urls  # Only process companies with URLs
        ]
        
        if not tasks:
            logging.warning("No companies with URLs to process")
            return {}

        logging.info(f"Starting {len(tasks)} company snapshots in parallel (max {max_company_parallel} concurrent)")
        
        # Execute all company snapshots in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        final_results = {}
        company_names = [comp for comp, urls in urls_by_company.items() if urls]
        
        for i, result in enumerate(results):
            company_name = company_names[i] if i < len(company_names) else f"Company_{i}"
            
            if isinstance(result, tuple) and len(result) == 2:
                company, url_profiles = result
                final_results[company] = url_profiles
                logging.info(f"✅ {company}: {len(url_profiles)} profiles scraped")
            elif isinstance(result, Exception):
                logging.error(f"❌ {company_name} failed: {result}")
                final_results[company_name] = {}
            else:
                logging.warning(f"⚠️ {company_name}: unexpected result format")
                final_results[company_name] = {}

        total_scraped = sum(len(profiles) for profiles in final_results.values())
        logging.info(f"Scraping complete: {total_scraped} total profiles across {len(final_results)} companies")
        
        return final_results

    async def _scrape_company_snapshot(
        self, 
        company: str, 
        urls: List[str], 
        timeout_sec: int
    ) -> tuple[str, Dict[str, Any]]:
        """
        Create and execute a single Bright Data snapshot for one company with ALL its URLs
        
        Returns:
            (company_name, {url: profile_data})
        """
        urls = self._prepare_urls(urls)
        if not urls:
            logging.warning(f"No valid URLs for {company}")
            return company, {}

        logging.info(f"Creating snapshot for {company}: {len(urls)} URLs")

        try:
            # Run the snapshot (sync operation) in a thread to avoid blocking
            profiles = await asyncio.to_thread(
                self._execute_snapshot_sync, urls, timeout_sec
            )
            
            # Convert list of profiles to URL-keyed dict
            url_map = {}
            for profile in profiles or []:
                profile_url = (profile.get("url") or profile.get("profile_url") or "").strip().rstrip("/")
                if profile_url:
                    url_map[profile_url] = profile
            
            logging.info(f"{company}: snapshot completed, {len(url_map)} profiles returned")
            return company, url_map
            
        except Exception as e:
            logging.error(f"{company}: snapshot failed - {e}")
            return company, {}

    def _execute_snapshot_sync(self, urls: List[str], timeout_sec: int) -> List[Dict[str, Any]]:
        """
        Execute a single Bright Data snapshot synchronously
        This method handles the actual API interaction
        """
        if self.collector_id:
            return self._collector_snapshot(urls, timeout_sec)
        elif self.endpoint:
            return self._endpoint_snapshot(urls, timeout_sec)
        else:
            raise ValueError("No collector_id or endpoint configured")

    def _collector_snapshot(self, urls: List[str], timeout_sec: int) -> List[Dict[str, Any]]:
        """Execute snapshot using Bright Data Collector API"""
        # Start the collector run
        start_url = f"https://api.brightdata.com/collectors/{self.collector_id}/start"
        payload = {"start_urls": [{"url": url} for url in urls]}
        
        logging.debug(f"Starting collector with {len(urls)} URLs")
        response = self.session.post(start_url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        status_url = data.get("status_url") or data.get("url")
        
        if not status_url:
            logging.error("No status URL returned from collector start")
            return []

        # Poll for results with exponential backoff
        poll_interval = 3
        max_polls = timeout_sec // poll_interval
        
        for attempt in range(max_polls):
            try:
                status_response = self.session.get(status_url, timeout=60)
                
                if status_response.status_code == 200:
                    result_data = status_response.json()
                    items = result_data.get("items") or result_data.get("results") or []
                    
                    if items:
                        # Normalize profile data
                        normalized = []
                        for item in items:
                            if item and isinstance(item, dict):
                                # Ensure URL field exists
                                if "url" not in item and "profile_url" in item:
                                    item["url"] = item["profile_url"]
                                normalized.append(item)
                        
                        logging.debug(f"Collector returned {len(normalized)} profiles")
                        return normalized
                    
                    # Check if still processing
                    status = result_data.get("status", "").lower()
                    if status in ["completed", "finished", "done"]:
                        logging.warning("Collector completed but returned no items")
                        return []
                
                # Wait before next poll, with exponential backoff
                wait_time = min(poll_interval * (1.2 ** (attempt // 5)), 30)
                time.sleep(wait_time)
                
            except Exception as e:
                logging.warning(f"Polling attempt {attempt + 1} failed: {e}")
                time.sleep(poll_interval)

        logging.error(f"Collector timeout after {timeout_sec}s")
        return []

    def _endpoint_snapshot(self, urls: List[str], timeout_sec: int) -> List[Dict[str, Any]]:
        """Execute snapshot using direct endpoint"""
        payload = {"urls": urls}
        
        logging.debug(f"Sending {len(urls)} URLs to endpoint")
        response = self.session.post(self.endpoint, json=payload, timeout=timeout_sec)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items") or data.get("results") or []
        
        # Normalize profile data
        normalized = []
        for item in items:
            if item and isinstance(item, dict):
                # Ensure URL field exists
                if "url" not in item and "profile_url" in item:
                    item["url"] = item["profile_url"]
                normalized.append(item)
        
        logging.debug(f"Endpoint returned {len(normalized)} profiles")
        return normalized

# =============================================================================
# Updated helper function for the main pipeline
# =============================================================================

async def scrape_matched_profiles_per_company_parallel(
    shade_scraper: BrightDataScraper,
    matched_urls_per_company: Dict[str, List[str]],
    **kwargs  # Accept any extra params but ignore them
) -> Dict[str, Dict[str, Dict]]:
    """
    Updated interface that uses the new per-company snapshot approach
    
    Args:
        shade_scraper: BrightDataScraper instance
        matched_urls_per_company: {company: [urls]}
        **kwargs: Ignored (for backward compatibility)
    
    Returns:
        {company: {url: profile_data}}
    """
    return await shade_scraper.scrape_profiles_per_company_parallel(
        matched_urls_per_company,
        max_company_parallel=3,  # Can be made configurable
        timeout_sec=900
    )

# Also add the old method name for backward compatibility
class BrightDataScraperCompat(BrightDataScraper):
    """Compatibility wrapper that maintains the old interface"""
    
    def scrape_profiles_in_batches(self, urls: List[str], batch_size: int = 10, 
                                   timeout_sec: int = 900) -> List[Dict[str, Any]]:
        """Legacy sync method - converts to async and runs single company"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run as single company
        company_data = {"single_company": urls}
        result = loop.run_until_complete(
            self.scrape_profiles_per_company_parallel(company_data, timeout_sec=timeout_sec)
        )
        
        # Return flat list for compatibility
        profiles = []
        for company_profiles in result.values():
            profiles.extend(company_profiles.values())
        return profiles

# =============================================================================
# Updated helper function for the main pipeline
# =============================================================================

async def scrape_matched_profiles_per_company_parallel(
    shade_scraper: BrightDataScraper,
    matched_urls_per_company: Dict[str, List[str]],
    **kwargs  # Accept any extra params but ignore them
) -> Dict[str, Dict[str, Dict]]:
    """
    Updated interface that uses the new per-company snapshot approach
    
    Args:
        shade_scraper: BrightDataScraper instance
        matched_urls_per_company: {company: [urls]}
        **kwargs: Ignored (for backward compatibility)
    
    Returns:
        {company: {url: profile_data}}
    """
    return await shade_scraper.scrape_profiles_per_company_parallel(
        matched_urls_per_company,
        max_company_parallel=3,  # Can be made configurable
        timeout_sec=900
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
        """Execute complete MIRAGE analysis pipeline"""
        logger.info(f"=== MIRAGE ANALYSIS STARTED (competitors={num_competitors}) ===")
        start_time = time.time()
        
        try:
            # Step 1: Detect Competitors (with strict limit)
            logger.info("Phase 1: Competitor Detection")
            competitors = await self.competitor_detector.detect_competitors(
                intelligence_data, max_competitors=num_competitors
            )
            
            if not competitors:
                raise ValueError("No competitors detected")
            
            # Step 2: Build Target Profiles
            logger.info("Phase 2: Target Profile Building")
            employee_data = self._extract_employee_data(intelligence_data)
            target_profiles = await self.profile_builder.build_target_profiles(employee_data)
            
            if not target_profiles:
                logger.warning("No target profiles built, using mock data")
                target_profiles = self._create_mock_profiles()
            
            # Steps 3 & 4: Find Employees and Match Profiles (parallel)
            logger.info("Phase 3 & 4: Employee Search and Matching (parallel)")
            
            # Run steps 3 and 4 in sequence but with parallel processing within each
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
            # === Phase 4.5: Scrape matched LinkedIn profiles using SHADE Bright scraper (PARALLEL PER COMPANY) ===
            logger.info("Phase 4.5: Scraping matched LinkedIn profiles with SHADE Bright scraper (parallel per company)")
            matched_urls_per_company = collect_matched_linkedin_urls(profile_matches)

            scraped_by_company: Dict[str, Dict[str, Any]] = {}
            total_scraped = 0

            if ShadeBrightScraper is None:
                logger.warning("SHADE scraper not importable — skipping matched profile scraping")
            else:
                shade_scraper = ShadeBrightScraper()
                logger.info("SHADE scraper enabled? True")

                # Fire multiple Bright Data snapshots in parallel PER company,
                # auto-splitting each company’s list into ≤100-url chunks
                scraped_by_company = await scrape_matched_profiles_per_company_parallel(
                    shade_scraper,
                    matched_urls_per_company,
                    per_snapshot_cap=100,       # Bright Data cap
                    shade_batch_size=8,         # your existing value
                    timeout_sec=900,
                    max_company_parallel=3      # tune if you hit rate limits
                )

                total_scraped = sum(len(v or {}) for v in scraped_by_company.values())

            logger.info(f"SHADE scraping complete. Profiles scraped: {total_scraped}")



            # === Phase 5: Output Generation (now with scraped profiles) ===
            logger.info("Phase 5: Output Generation")
            spectre_path = OutputWriter.write_spectre_matches(profile_matches, target_profiles)

            # employee reports now include 'detailed_profile' where available
            reports_dir = OutputWriter.write_employee_reports(
                competitor_employees,
                scraped_details=scraped_by_company
            )

            # also write per-company matched details bundle
            matched_dir = OutputWriter.write_matched_details_with_scrapes(
                profile_matches, scraped_by_company
            )

            
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