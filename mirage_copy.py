#!/usr/bin/env python3
""" 
MIRAGE: GPT-Powered Competitive Intelligence System
==================================================
FIXED VERSION - Correct flow per requirements:
- Per TARGET flow:
  1) Build target profiles via GPT.
  2) For each target -> query every competitor for same-department candidates.
  3) GPT filters to same department & ranks by experience/seniority ONLY (Top-5/target).
  4) Write ALL matches to spectre_matches.json FIRST.
  5) Pool by company and send to Bright Data.
"""

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
# Logging Configuration
# =============================================================================

def _force_utf8_console():
    """Force UTF-8 encoding for Windows console"""
    try:
        if sys.platform == "win32":
            os.system("chcp 65001 > nul")
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

_force_utf8_console()

def setup_logging():
    """Configure comprehensive logging"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    file_handler = logging.FileHandler('mirage_system.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

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

def safe_json_parse(content: str) -> Optional[Dict]:
    """Safely parse JSON (tolerant of code fences, trailing commas, partial blocks)."""
    if not content:
        return None

    txt = content.strip()

    # Strip code fences like ```json ... ``` or ``` ... ```
    if txt.startswith("```"):
        lines = txt.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        txt = "\n".join(lines).strip()

    # Find the first JSON object or array and trim to balanced braces
    start_obj = txt.find("{")
    start_arr = txt.find("[")
    if start_obj == -1 and start_arr == -1:
        return None

    start = start_obj if start_obj != -1 else start_arr
    block = txt[start:]
    open_ch = block[0]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    end_idx = -1
    for i, ch in enumerate(block):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx != -1:
        block = block[:end_idx]

    try:
        return json.loads(block)
    except json.JSONDecodeError:
        block2 = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            return json.loads(block2)
        except json.JSONDecodeError:
            return None

def create_cache_key(*args) -> str:
    combined = '|'.join(str(arg) for arg in args)
    return hashlib.sha256(combined.encode()).hexdigest()

def safe_filename(name: str) -> str:
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
# Step 1: Competitor Detection
# =============================================================================

class CompetitorDetector:
    """GPT-only competitor detection"""

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        logger.info("CompetitorDetector initialized")

    async def detect_competitors(self, intelligence_data: Dict, max_competitors: int = 10) -> List[CompetitorProfile]:
        logger.info(f"STEP 1: COMPETITOR DETECTION (limit={max_competitors})")

        company_name = self._extract_company_name(intelligence_data)
        ctx = self._extract_business_context(intelligence_data, company_name)

        logger.info(f"Analyzing competitors for: {company_name}")
        logger.info(f"Industry: {ctx.get('industry', 'Unknown')}")

        competitors = await self._gpt_competitor_analysis(ctx, max_competitors)
        competitors = competitors[:max_competitors]

        logger.info(f"Detected {len(competitors)} competitors:")
        for i, comp in enumerate(competitors, 1):
            logger.info(f"   {i}. {comp.name} (Score: {comp.similarity_score:.1f})")

        return competitors

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

    async def _gpt_competitor_analysis(self, ctx: Dict, max_competitors: int) -> List[CompetitorProfile]:
        """Single GPT call for competitor analysis"""
        system_prompt = f"""You are a competitive-intelligence analyst.
Return competitors for the given company as JSON.

Format:
{{
  "competitors": [
    {{
      "name": "Competitor Company Name",
      "industry": "Industry",
      "similarity_score": 8.5
    }}
  ]
}}

Rules:
- Return exactly {max_competitors} competitors, most similar first
- If company is Manipal Fintech, always include Rupeek and India Gold as top competitors
- Similarity score should be 0-10 (higher = more similar)
"""

        user_prompt = (
            f"Company: {ctx.get('company_name','')}\n"
            f"Industry: {ctx.get('industry','Not specified')}\n"
            f"Description: {ctx.get('description','Not specified')}\n"
            f"Employee Count: {ctx.get('employee_count','Not specified')}\n"
            f"Headquarters: {ctx.get('headquarters','Not specified')}"
        )

        raw = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.2, max_tokens=800)
        
        if not raw:
            logger.warning("Empty response from GPT for competitors")
            return []

        try:
            data = safe_json_parse(raw)
            if data and "competitors" in data:
                competitors = []
                for item in data["competitors"]:
                    if isinstance(item, dict) and item.get("name"):
                        competitors.append(CompetitorProfile(
                            name=item["name"].strip(),
                            industry=item.get("industry", "").strip(),
                            similarity_score=float(item.get("similarity_score", 0)),
                            detection_method="GPT Analysis"
                        ))
                return competitors
        except Exception as e:
            logger.error(f"Failed to parse competitor response: {e}")
        
        return []

# =============================================================================
# Step 2: Target Profile Building (GPT-only)
# =============================================================================

class TargetProfileBuilder:
    """Build target employee profiles (GPT-only)."""

    _VALID_DEPTS = {
        "Engineering","Sales","Marketing","Finance","Operations",
        "Product","Data","Design","HR","Legal","Support","Other"
    }

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        logger.info("TargetProfileBuilder initialized (GPT-only)")

    async def build_target_profiles(self, employee_data: List[Dict]) -> List[TargetEmployeeProfile]:
        logger.info("STEP 2: TARGET PROFILE BUILDING")
        if not employee_data:
            return []

        profiles: List[TargetEmployeeProfile] = []
        for emp in employee_data:
            p = await self._build_single_profile(emp)  # GPT only
            if p:
                profiles.append(p)

        logger.info(f"Built {len(profiles)} target profiles (GPT only)")
        return profiles

    async def _build_single_profile(self, employee: Dict) -> Optional[TargetEmployeeProfile]:
        raw = self._extract_employee_data(employee)
        if not raw.get("name"):
            return None

        system_prompt = """You are an HR analyst. Extract information from the provided employee data.
Return JSON ONLY in this exact shape:
{
  "name": "Full Name",
  "title": "Job Title",
  "department": "Engineering|Sales|Marketing|Finance|Operations|Product|Data|Design|HR|Legal|Support|Other",
  "experience_years": 5.0,
  "key_skills": ["skill1","skill2","skill3"],
  "company": "Company Name"
}
Rules:
- Always fill all fields.
- "experience_years" MUST be a number (not null/string).
- "department" MUST be one of the allowed enums. If unclear, use "Other".
- "key_skills" MUST be a short list (3–6) of plain strings.
- Output JSON only (no prose, no code fences).
"""

        user_prompt = (
            f"Employee:\n"
            f"Name: {raw.get('name','Unknown')}\n"
            f"Title: {raw.get('title','Not specified')}\n"
            f"Company: {raw.get('company','Not specified')}\n"
            f"Location: {raw.get('location','Not specified')}"
        )

        try:
            content = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.1, max_tokens=500)
            data = safe_json_parse(content or "")
            if not isinstance(data, dict):
                logger.info(f"Profile build skipped (no JSON) for {raw.get('name')}")
                return None

            name = (data.get("name") or raw.get("name") or "Unknown").strip()
            title = (data.get("title") or raw.get("title") or "Unknown").strip()
            company = (data.get("company") or raw.get("company") or "Unknown Company").strip()

            dept = (data.get("department") or "Other").strip()
            if dept not in self._VALID_DEPTS:
                dept = "Other"

            try:
                exp = float(data.get("experience_years"))
            except (TypeError, ValueError):
                exp = 5.0

            ks = data.get("key_skills")
            if not isinstance(ks, list) or not ks:
                ks = ["Communication", "Problem Solving", "Teamwork"]
            key_skills = [str(s).strip() for s in ks if s]

            # 🔎 Log what GPT inferred so you can see dept/exp directly
            logger.info(
                f"[TargetProfile] Name='{name}' | Title='{title}' | Dept='{dept}' | Exp={exp} yrs | Company='{company}'"
            )

            return TargetEmployeeProfile(
                name=name,
                title=title,
                department=dept,
                experience_years=exp,
                key_skills=key_skills,
                company=company
            )

        except Exception as e:
            logger.warning(f"GPT profile analysis failed: {e}")
            return None

    def _extract_employee_data(self, employee: Dict) -> Dict:
        basic = employee.get("basic_info", {}) or {}
        detailed = employee.get("detailed_profile", {}) or {}
        return {
            "name": (employee.get("name") or basic.get("name") or "").strip(),
            "title": (
                employee.get("title") or employee.get("position")
                or basic.get("title") or detailed.get("position") or ""
            ).strip(),
            "company": (employee.get("company") or basic.get("company") or "").strip(),
            "location": (employee.get("location") or basic.get("location") or "").strip(),
        }

# =============================================================================
# Step 3: Competitor Employee Search
# =============================================================================

class CompetitorEmployeeFinder:
    """Find competitor employees using Google Search"""
    
    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Missing Google API credentials")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
            
        logger.info("CompetitorEmployeeFinder initialized")
    
    async def collect_candidates_for_target(
        self,
        target: TargetEmployeeProfile,
        competitor_names: List[str],
        per_company_limit: int = 150
    ) -> List[CompetitorEmployee]:
        """
        For ONE target employee:
          - For each competitor, generate role-aware queries (dept + target title).
          - Execute searches and pool candidates.
          - Return combined (deduped) candidate list.
        """
        if self.use_mock_data:
            mocked = []
            for cname in competitor_names:
                for i in range(5):
                    mocked.append(CompetitorEmployee(
                        name=f"Employee {i+1} {cname}",
                        title=f"{target.department} Specialist",
                        company=cname,
                        linkedin_url=f"https://linkedin.com/in/mock-{safe_filename(cname)}-{i+1}",
                        search_snippet=f"Works at {cname}"
                    ))
            return self._deduplicate_employees(mocked)

        tasks = []
        for company in competitor_names:
            tasks.append(self._find_employees_for_company_and_dept(
                company_name=company,
                department=target.department,
                per_company_limit=per_company_limit,
                target_title=target.title
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        pooled: List[CompetitorEmployee] = []
        for r in results:
            if isinstance(r, list):
                pooled.extend(r)

        pooled = self._deduplicate_employees(pooled)
        logger.info(f"[SearchPool] Target='{target.name}' pooled unique candidates: {len(pooled)}")
        return pooled

    async def find_all_competitor_employees(self, competitors: List[CompetitorProfile]) -> List[CompetitorEmployee]:
        logger.info("STEP 3: COMPETITOR EMPLOYEE SEARCH (bulk/all)")
        if self.use_mock_data:
            return await self._mock_employee_search(competitors)
        
        all_employees = []
        tasks = []
        for competitor in competitors:
            task = self._find_employees_for_company(competitor)
            tasks.append(task)
        
        logger.info(f"Processing {len(competitors)} companies in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_employees.extend(result)
            elif isinstance(result, Exception):
                company_name = competitors[i].name if i < len(competitors) else "Unknown"
                logger.error(f"Search failed for {company_name}: {result}")
        
        unique_employees = self._deduplicate_employees(all_employees)
        logger.info(f"Total unique employees found across all competitors: {len(unique_employees)}")
        return unique_employees
    
    async def _find_employees_for_company(self, competitor: CompetitorProfile) -> List[CompetitorEmployee]:
        logger.info(f"Searching employees at {competitor.name}")
        all_employees = []
        departments = ["Engineering", "Sales", "Marketing", "Product", "Data"]
        for department in departments:
            logger.debug(f"   Searching {department} at {competitor.name}")
            queries = await self._generate_search_queries(department, competitor.name)
            employees = await self._execute_searches(queries, competitor.name)
            all_employees.extend(employees)
            await asyncio.sleep(1)
        logger.info(f"   Found {len(all_employees)} employees at {competitor.name}")
        return all_employees

    async def _find_employees_for_company_and_dept(
        self,
        company_name: str,
        department: str,
        per_company_limit: int,
        target_title: Optional[str] = None
    ) -> List[CompetitorEmployee]:
        """Find employees for a single company and ONE department (used in per-target flow)."""
        logger.debug(f"Target-wise search: dept={department} | title={target_title} @ {company_name}")
        # Role-aware first
        queries = await self._generate_search_queries_role_aware(department, company_name, target_title)
        if not queries:
            queries = await self._generate_search_queries(department, company_name)
        employees = await self._execute_searches(queries, company_name)
        if per_company_limit and len(employees) > per_company_limit:
            employees = employees[:per_company_limit]
        return employees
    
    async def _generate_search_queries(self, department: str, company_name: str) -> List[str]:
        system_prompt = """Generate Google search queries to find LinkedIn profiles.
Return JSON array: ["query1", "query2", "query3"]

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

    async def _generate_search_queries_role_aware(
        self,
        department: str,
        company_name: str,
        target_title: Optional[str]
    ) -> List[str]:
        """Role-aware queries: department + target title synonyms/seniority."""
        system_prompt = """Generate Google queries (JSON array) to find LinkedIn profiles.
Return ONLY a JSON array of strings: ["q1", "q2", "q3"].
Rules:
- MUST include: site:linkedin.com/in
- MUST include the company name in quotes
- SHOULD include the target role/title and common synonyms/seniority variants
- SHOULD include department hints when sensible
- ALWAYS add: -jobs -hiring -recruiter
- Prefer 3 concise, distinct queries."""

        role_hint = (target_title or "").strip()
        user_prompt = f"""Company: {company_name}
Department: {department}
Target Title: {role_hint if role_hint else "Unknown/None"}

Generate 3 targeted LinkedIn search queries that prioritize similar roles (use synonyms like Sr/Lead/Head/Manager etc.)."""

        response = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.2, max_tokens=300)
        if not response:
            return []
        arr = safe_json_parse(response)
        if not isinstance(arr, list):
            return []

        norm = []
        for q in arr[:3]:
            q = str(q).strip()
            if not q:
                continue
            if "site:linkedin.com/in" not in q.lower():
                q = f"site:linkedin.com/in {q}"
            if "-jobs" not in q:
                q += " -jobs"
            if "-hiring" not in q:
                q += " -hiring"
            if "-recruiter" not in q:
                q += " -recruiter"
            if f'"{company_name}"' not in q and company_name not in q:
                q = q + f' "{company_name}"'
            norm.append(q)
        return norm[:3]
    
    async def _execute_searches(self, queries: List[str], company_name: str) -> List[CompetitorEmployee]:
        all_employees = []
        for query in queries:
            try:
                logger.debug(f"Search: {query[:80]}...")
                if self.use_mock_data:
                    all_employees.append(CompetitorEmployee(
                        name="Mock Person",
                        title=f"{company_name} {query[:20]}",
                        company=company_name,
                        linkedin_url=f"https://linkedin.com/in/mock-{safe_filename(company_name)}-{hash(query)%10000}",
                        search_snippet=f"Works at {company_name}"
                    ))
                    continue

                url = "https://www.googleapis.com/customsearch/v1"
                params = {'key': self.google_api_key, 'cx': self.google_cse_id, 'q': query, 'num': 10}
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
                await asyncio.sleep(1.2)
            except Exception as e:
                logger.warning(f"Search error: {e}")
        return all_employees
    
    def _parse_search_result(self, item: Dict, company_name: str) -> Optional[CompetitorEmployee]:
        title = item.get('title', '')
        link = item.get('link', '')
        snippet = item.get('snippet', '')
        if 'linkedin.com/in' not in link:
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
        seen_urls = set()
        unique = []
        for emp in employees:
            url_key = emp.linkedin_url.lower().rstrip('/')
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique.append(emp)
        return unique
    
    async def _mock_employee_search(self, competitors: List[CompetitorProfile]) -> List[CompetitorEmployee]:
        logger.warning("Using mock employee data")
        mock_employees = []
        for competitor in competitors:
            for i in range(5):
                mock_employees.append(CompetitorEmployee(
                    name=f"Employee {i+1}",
                    title=f"Senior {competitor.industry} Specialist",
                    company=competitor.name,
                    linkedin_url=f"https://linkedin.com/in/mock-{competitor.name.lower().replace(' ', '-')}-{i+1}",
                    search_snippet=f"Works at {competitor.name}"
                ))
        return mock_employees


# =============================================================================
# Step 4: Profile Matching (GPT-only: dept inference + exp-only similarity)
# =============================================================================
class ProfileMatcher:
    """
    GPT owns:
      1) Infer target department (and candidate departments).
      2) Keep ONLY candidates in SAME department (unless target dept is 'Other'/unclear).
      3) Rank by ROLE/TITLE similarity + seniority/ scope (ignore specific tech/education).
    """

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        logger.info("ProfileMatcher initialized (role/title similarity)")

    async def rank_target_against_candidates(
        self,
        target: TargetEmployeeProfile,
        candidates: List[CompetitorEmployee],
        *,
        per_target_cap: int = 5,
        min_score: float = 30.0,  # softened default
        batch_size: int = 20,
        max_tokens: int = 1200,
    ) -> List[EmployeeMatch]:

        batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
        target_matches: List[EmployeeMatch] = []

        async def _rank_batch(batch: List[CompetitorEmployee]) -> List[EmployeeMatch]:
            system_prompt = """You are an HR matcher.

GOAL:
Find candidates whose ROLES are similar to the TARGET's role within the SAME department.
Use TITLE/ROLE semantics and seniority LEVEL/SCOPE to assess similarity.

PROCESS:
1) Infer TARGET's department from title/context (if 'Other' or unclear, keep it as 'Other').
2) For each CANDIDATE, infer department from title/snippet.
3) Keep ONLY candidates in the SAME department as TARGET.
   IF TARGET department is 'Other' or unclear, DO NOT filter by department.
4) Score similarity on a 0–100 scale using:
   - Role/title semantic similarity (primary)
   - Seniority level alignment (IC vs. Manager vs. Head/Director/VP)
   - Scope/ownership signals in title/snippet (team, product, region, P&L)
   Ignore specific tech stacks or education; prefer role/level closeness.

OUTPUT:
Return ONLY a JSON array of:
{
  "competitor_name": "...",
  "competitor_company": "...",
  "similarity_score": 0-100,
  "match_rationale": "1-2 lines on role/level similarity and scope",
  "linkedin_url": "https://..."
}

RULES:
- Exclude candidates not in the same department unless TARGET dept is 'Other'/unclear.
- Include ONLY items with similarity_score >= MIN_SCORE.
- Keep the array concise and relevant (no duplicates).
"""

            comp_payload = [{
                "name": c.name,
                "title": c.title,
                "company": c.company,
                "linkedin_url": c.linkedin_url,
                "snippet": c.search_snippet
            } for c in batch]

            user_prompt = (
                f"MIN_SCORE: {min_score}\n\n"
                f"TARGET:\n"
                f"- name: {target.name}\n"
                f"- title: {target.title}\n"
                f"- stated_department: {target.department}\n"
                f"- years_of_experience: {target.experience_years}\n\n"
                f"CANDIDATES_JSON:\n{json.dumps(comp_payload, ensure_ascii=False)}\n\n"
                f"Return ONLY the JSON array."
            )

            raw = await self.gpt.chat_completion(system_prompt, user_prompt, temperature=0.1, max_tokens=max_tokens)
            data = safe_json_parse(raw or "")
            if not isinstance(data, list):
                return []

            out: List[EmployeeMatch] = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                try:
                    score = float(row.get("similarity_score", 0))
                    if score < min_score:
                        continue
                    out.append(
                        EmployeeMatch(
                            target_employee=target.name,
                            competitor_employee=(row.get("competitor_name") or "").strip() or "Unknown",
                            competitor_company=(row.get("competitor_company") or "").strip() or "Unknown",
                            similarity_score=score,
                            match_rationale=row.get("match_rationale", "") or "",
                            linkedin_url=(row.get("linkedin_url") or "").strip(),
                        )
                    )
                except Exception:
                    continue
            return out

        results = await asyncio.gather(*[_rank_batch(b) for b in batches], return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                target_matches.extend(r)

        target_matches.sort(key=lambda m: m.similarity_score, reverse=True)
        kept = target_matches[:per_target_cap]
        logger.info(f"[Match] Target='{target.name}' kept {len(kept)} (Top {per_target_cap}) from {len(candidates)} candidates")
        return kept


# =============================================================================
# Step 5: Spectre Matches Writer
# =============================================================================

class SpectreWriter:
    """Write spectre_matches.json file"""

    @staticmethod
    def write_spectre_matches(all_matches: List[EmployeeMatch], target_profiles: List[TargetEmployeeProfile]) -> str:
        logger.info("STEP 5: WRITING SPECTRE MATCHES")

        # target lookup for roles
        target_by_name = {t.name: t for t in target_profiles}

        # company -> [EmployeeMatch,...]
        by_company: Dict[str, List[EmployeeMatch]] = {}
        for m in all_matches or []:
            by_company.setdefault(m.competitor_company, []).append(m)

        spectre_data: Dict[str, Any] = {}

        for company, matches in by_company.items():
            # Group by target employee
            target_groups: Dict[str, Dict[str, Any]] = {}
            for m in matches:
                tname = m.target_employee
                if tname not in target_groups:
                    tp = target_by_name.get(tname)
                    target_groups[tname] = {
                        "manipal_name": tname,
                        "manipal_role": (tp.title if tp else "Unknown Role"),
                        "matches": []
                    }
                target_groups[tname]["matches"].append({
                    "company": company.lower().replace(" ", "_"),
                    "name": m.competitor_employee,
                    "role": "Unknown Role",   # Could be enhanced from search title if stored
                    "similarity": round(float(m.similarity_score), 2),
                    "via": "llm"
                })

            spectre_data[company.lower().replace(" ", "_")] = list(target_groups.values())

        # write file
        output_path = "spectre_matches.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(spectre_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote {output_path} (companies: {len(by_company)})")
        return output_path

# =============================================================================
# Step 6: Bright Data Scraper (AFTER spectre_matches.json is written)
# =============================================================================

class BrightDataScraper:
    """Bright Data scraper for LinkedIn profiles"""
    
    def __init__(self):
        self.api_key = os.getenv("BRIGHT_DATA_API_KEY")
        self.dataset_id = os.getenv("BRIGHT_DATA_DATASET_ID")
        
        self.trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={self.dataset_id}&include_errors=true"
        self.progress_base = "https://api.brightdata.com/datasets/v3/progress/"
        self.snapshot_base = "https://api.brightdata.com/datasets/v3/snapshot/"
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        
        self.enabled = bool(self.api_key and self.dataset_id)
        if not self.enabled:
            logger.warning("BrightDataScraper disabled (missing credentials)")
    
    async def scrape_profiles_by_company(
        self,
        urls_by_company: Dict[str, List[str]],
        max_company_parallel: int = 3,
        timeout_sec: int = 100000
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scrape LinkedIn profiles grouped by company
        Args:
            urls_by_company: {company: [linkedin_urls]}
        Returns:
            {company: {url: profile_data}}
        """
        if not self.enabled:
            logger.warning("Bright Data scraping skipped (not configured)")
            return {}
        
        logger.info(f"STEP 6: BRIGHT DATA SCRAPING")
        logger.info(f"Companies to scrape: {len(urls_by_company)}")
        
        semaphore = asyncio.Semaphore(max_company_parallel)
        
        async def _scrape_company(company: str, urls: List[str]) -> tuple[str, Dict[str, Any]]:
            async with semaphore:
                return await self._scrape_company_snapshot(company, urls, timeout_sec)
        
        tasks = []
        for company, urls in urls_by_company.items():
            if urls:
                tasks.append(_scrape_company(company, urls))
        
        if not tasks:
            logger.warning("No URLs to scrape")
            return {}
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scraped_by_company = {}
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                company, url_profiles = result
                scraped_by_company[company] = url_profiles
            else:
                logger.error(f"Company scraping failed: {result}")
        
        total_scraped = sum(len(profiles) for profiles in scraped_by_company.values())
        logger.info(f"Bright Data scraping complete: {total_scraped} profiles scraped")
        
        return scraped_by_company
    
    async def _scrape_company_snapshot(self, company: str, urls: List[str], timeout_sec: int) -> tuple[str, Dict[str, Any]]:
        """Scrape one company's URLs using Bright Data dataset"""
        urls = self._prepare_urls(urls)
        if not urls:
            logger.warning(f"{company}: no valid LinkedIn URLs")
            return company, {}
        
        logger.info(f"Scraping {len(urls)} profiles for {company}")
        
        try:
            profiles = await asyncio.to_thread(self._dataset_snapshot_sync, urls, timeout_sec)
            
            url_map = {}
            for profile in profiles or []:
                url = (profile.get("url") or profile.get("profile_url") or "").strip().rstrip("/")
                if url:
                    url_map[url] = profile
            
            logger.info(f"{company}: scraped {len(url_map)} profiles successfully")
            return company, url_map
            
        except Exception as e:
            logger.error(f"{company}: scraping failed: {e}")
            return company, {}
    
    def _dataset_snapshot_sync(self, urls: List[str], timeout_sec: int) -> List[Dict[str, Any]]:
        """Synchronous dataset snapshot workflow"""
        snapshot_id = self._trigger_snapshot(urls)
        if not snapshot_id:
            return []
        
        if not self._wait_for_completion(snapshot_id, timeout_sec):
            return []
        
        return self._fetch_snapshot_data(snapshot_id)
    
    def _trigger_snapshot(self, urls: List[str]) -> Optional[str]:
        """Trigger Bright Data dataset snapshot"""
        payload = [{"url": url} for url in urls]
        
        try:
            response = self.session.post(self.trigger_url, json=payload, timeout=60)
            if response.ok:
                data = response.json()
                snapshot_id = data.get("snapshot_id") or data.get("snapshot") or data.get("id")
                if snapshot_id:
                    logger.debug(f"Triggered snapshot: {snapshot_id}")
                    return snapshot_id
                else:
                    logger.error(f"No snapshot ID in response: {data}")
            else:
                logger.error(f"Trigger failed {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Trigger error: {e}")
        
        return None
    
    def _wait_for_completion(self, snapshot_id: str, timeout_sec: int, interval: int = 10) -> bool:
        """Wait for snapshot completion"""
        elapsed = 0
        while elapsed <= timeout_sec:
            try:
                response = self.session.get(self.progress_base + snapshot_id, timeout=30)
                if response.ok:
                    data = response.json()
                    status = (data.get("status") or data.get("state") or "").lower()
                    
                    if status == "ready":
                        logger.debug(f"Snapshot {snapshot_id} ready")
                        return True
                    elif status == "error":
                        logger.error(f"Snapshot error: {data}")
                        return False
                else:
                    logger.warning(f"Progress check failed: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Progress poll error: {e}")
            
            time.sleep(interval)
            elapsed += interval
        
        logger.error(f"Snapshot {snapshot_id} timed out")
        return False
    
    def _fetch_snapshot_data(self, snapshot_id: str) -> List[Dict[str, Any]]:
        """Fetch snapshot data (NDJSON format)"""
        try:
            response = self.session.get(self.snapshot_base + snapshot_id, timeout=120)
            if not response.ok:
                logger.error(f"Snapshot fetch failed {response.status_code}: {response.text}")
                return []
            
            profiles = []
            for line in response.text.splitlines():
                line = line.strip()
                if line:
                    try:
                        profile = json.loads(line)
                        if "url" not in profile and "profile_url" in profile:
                            profile["url"] = profile["profile_url"]
                        profiles.append(profile)
                    except json.JSONDecodeError:
                        continue
            
            logger.debug(f"Fetched {len(profiles)} profiles from snapshot")
            return profiles
            
        except Exception as e:
            logger.error(f"Snapshot fetch error: {e}")
            return []
    
    def _prepare_urls(self, urls: List[str]) -> List[str]:
        """Clean and deduplicate LinkedIn URLs"""
        cleaned = []
        seen = set()
        
        for url in urls or []:
            if not url:
                continue
            
            url = url.strip().rstrip("/")
            if ("linkedin.com/in" in url or "linkedin.com/pub" in url) and url not in seen:
                seen.add(url)
                cleaned.append(url)
        
        return cleaned

# =============================================================================
# Step 7: Output Writers
# =============================================================================

class OutputWriter:
    """Handle output file generation"""
    
    @staticmethod
    def write_employee_reports_with_scraped_data(
        all_matches: List[EmployeeMatch],
        scraped_by_company: Dict[str, Dict[str, Any]]
    ) -> str:
        """Write employee reports grouped by company with scraped data"""
        logger.info("STEP 7: WRITING EMPLOYEE REPORTS")
        
        output_dir = Path("employee_data")
        output_dir.mkdir(exist_ok=True)
        
        matches_by_company = {}
        for match in all_matches:
            company = match.competitor_company
            if company not in matches_by_company:
                matches_by_company[company] = []
            matches_by_company[company].append(match)
        
        for company, matches in matches_by_company.items():
            safe_name = safe_filename(company)
            report_path = output_dir / f"{safe_name}_report.json"
            
            company_scraped_data = scraped_by_company.get(company, {})
            
            employees_data = []
            for match in matches:
                url_norm = match.linkedin_url.rstrip("/")
                scraped_profile = company_scraped_data.get(url_norm)
                
                employees_data.append({
                    "basic_info": {
                        "name": match.competitor_employee,
                        "linkedin_url": match.linkedin_url,
                        "company": match.competitor_company,
                        "title": "Unknown",
                        "match_info": {
                            "target_employee": match.target_employee,
                            "similarity_score": match.similarity_score,
                            "match_rationale": match.match_rationale
                        }
                    },
                    "detailed_profile": scraped_profile,
                    "data_status": {
                        "found_in_search": True,
                        "detailed_scraped": bool(scraped_profile)
                    }
                })
            
            report = {
                "mission_metadata": {
                    "agent_id": f"MIRAGE_{safe_name.upper()}",
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
            
            logger.info(f"Wrote {report_path} with {len(employees_data)} employees")
        
        logger.info(f"All reports written to {output_dir}")
        return str(output_dir)

# =============================================================================
# Helper Functions
# =============================================================================

def collect_urls_by_company(all_matches: List[EmployeeMatch]) -> Dict[str, List[str]]:
    """Group LinkedIn URLs by company from matches"""
    logger.info("Collecting LinkedIn URLs by company for Bright Data")
    
    urls_by_company = {}
    for match in all_matches:
        company = match.competitor_company
        url = match.linkedin_url.strip().rstrip("/")
        
        if not url or ("linkedin.com/in" not in url and "linkedin.com/pub" not in url):
            continue
        
        if company not in urls_by_company:
            urls_by_company[company] = []
        
        if url not in urls_by_company[company]:
            urls_by_company[company].append(url)
    
    for company, urls in urls_by_company.items():
        logger.info(f"   {company}: {len(urls)} unique URLs")
    
    total_urls = sum(len(urls) for urls in urls_by_company.values())
    logger.info(f"Total URLs to scrape: {total_urls}")
    
    return urls_by_company

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

# =============================================================================
# Main MIRAGE System (PER-TARGET FLOW)
# =============================================================================

class MirageSystem:
    """Main MIRAGE system with corrected per-target flow"""
    
    def __init__(self):
        logger.info("=== MIRAGE SYSTEM INITIALIZATION ===")
        
        self._validate_environment()
        
        self.gpt_client = AzureGPTClient()
        self.competitor_detector = CompetitorDetector(self.gpt_client)
        self.profile_builder = TargetProfileBuilder(self.gpt_client)
        self.employee_finder = CompetitorEmployeeFinder(self.gpt_client)
        self.profile_matcher = ProfileMatcher(self.gpt_client)
        self.bright_data_scraper = BrightDataScraper()
        
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
        """Execute complete MIRAGE analysis pipeline with PER-TARGET FLOW"""
        logger.info(f"=== MIRAGE ANALYSIS STARTED (competitors={num_competitors}) ===")
        start_time = time.time()
        
        try:
            # Phase 1: Detect Competitors
            logger.info("Phase 1: Competitor Detection")
            competitors = await self.competitor_detector.detect_competitors(
                intelligence_data, max_competitors=num_competitors
            )
            competitor_names = [c.name for c in competitors]
            if not competitors:
                raise ValueError("No competitors detected")
            
            # Phase 2: Build Target Profiles
            logger.info("Phase 2: Target Profile Building")
            employee_data = self._extract_employee_data(intelligence_data)
            target_profiles = await self.profile_builder.build_target_profiles(employee_data)
            
            if not target_profiles:
                logger.warning("No target profiles built, using mock data")
                target_profiles = self._create_mock_profiles()

            # Phase 3+4: PER-TARGET search + GPT-only matching
            logger.info("Phase 3+4: Target-wise search + GPT-only matching (Top 5 per target)")
            all_matches: List[EmployeeMatch] = []
            for idx, target in enumerate(target_profiles, 1):
                logger.info(f"[Target {idx}/{len(target_profiles)}] {target.name} — dept={target.department} | exp={target.experience_years}")
                
                # Per-target candidate collection
                candidates = await self.employee_finder.collect_candidates_for_target(
                    target=target,
                    competitor_names=competitor_names,
                    per_company_limit=150
                )
                logger.info(f"Collected {len(candidates)} raw candidates across {len(competitor_names)} companies for {target.name}")

                # GPT-only dept filter + exp/seniority ranking
                top_matches = await self.profile_matcher.rank_target_against_candidates(
                    target=target,
                    candidates=candidates,
                    per_target_cap=5,
                    min_score=40.0,
                    batch_size=20,
                    max_tokens=1200,
                )
                all_matches.extend(top_matches)
                logger.info(f"Target '{target.name}': kept {len(top_matches)} matches (Top 5)")

            # Phase 5: Write Spectre Matches FIRST
            logger.info("Phase 5: Writing Spectre Matches")
            spectre_path = SpectreWriter.write_spectre_matches(all_matches, target_profiles)
            
            # Phase 6: Group URLs by Company and Send to Bright Data
            logger.info("Phase 6: Bright Data Scraping (after spectre_matches written)")
            urls_by_company = collect_urls_by_company(all_matches)
            
            scraped_by_company = await self.bright_data_scraper.scrape_profiles_by_company(
                urls_by_company,
                max_company_parallel=3,
                timeout_sec=900
            )
            
            # Phase 7: Write Employee Reports with Scraped Data
            logger.info("Phase 7: Writing Employee Reports")
            reports_dir = OutputWriter.write_employee_reports_with_scraped_data(
                all_matches, scraped_by_company
            )
            
            # Summary
            execution_time = time.time() - start_time
            results = self._generate_summary(
                intelligence_data, competitors, target_profiles, 
                all_matches, scraped_by_company, execution_time
            )
            
            logger.info(f"=== MIRAGE ANALYSIS COMPLETED in {execution_time:.2f}s ===")
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"MIRAGE analysis failed after {execution_time:.2f}s: {e}")
            raise
    
    def _extract_employee_data(self, intelligence_data: Dict) -> List[Dict]:
        """Extract employee data from intelligence report"""
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
        return [
            TargetEmployeeProfile(
                name="John Smith", title="Software Engineer", department="Engineering",
                experience_years=5.0, key_skills=["Python", "JavaScript", "AWS"], company="Manipal Fintech"
            ),
            TargetEmployeeProfile(
                name="Jane Doe", title="Product Manager", department="Product",
                experience_years=7.0, key_skills=["Product Strategy", "Agile", "UX"], company="Manipal Fintech"
            ),
            TargetEmployeeProfile(
                name="Robert Johnson", title="Data Scientist", department="Data",
                experience_years=4.0, key_skills=["Python", "Machine Learning", "SQL"], company="Manipal Fintech"
            )
        ]
    
    async def _create_mock_competitor_employees(self, competitors: List[CompetitorProfile]) -> List[CompetitorEmployee]:
        """Create mock competitor employees"""
        mock_employees = []
        for competitor in competitors:
            for i in range(5):
                mock_employees.append(CompetitorEmployee(
                    name=f"Mock Employee {i+1}",
                    title=f"Senior {competitor.industry} Professional",
                    company=competitor.name,
                    linkedin_url=f"https://linkedin.com/in/mock-{competitor.name.lower().replace(' ', '-')}-{i+1}",
                    search_snippet=f"Professional at {competitor.name}"
                ))
        return mock_employees
    
    def _generate_summary(
        self, 
        intelligence_data: Dict, 
        competitors: List[CompetitorProfile],
        target_profiles: List[TargetEmployeeProfile], 
        all_matches: List[EmployeeMatch],
        scraped_by_company: Dict[str, Dict[str, Any]],
        execution_time: float
    ) -> Dict:
        """Generate execution summary"""
        
        company_name = self.competitor_detector._extract_company_name(intelligence_data)
        total_scraped = sum(len(profiles) for profiles in scraped_by_company.values())
        high_quality_matches = len([m for m in all_matches if m.similarity_score >= 70])
        
        companies_with_matches = set(match.competitor_company for match in all_matches)
        
        return {
            "mirage_metadata": {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "target_company": company_name,
                "execution_time_seconds": round(execution_time, 2),
                "version": "3.2-PerTarget-Flow"
            },
            "results_summary": {
                "competitors_detected": len(competitors),
                "target_profiles_built": len(target_profiles),
                "total_matches": len(all_matches),
                "high_quality_matches": high_quality_matches,
                "companies_with_matches": len(companies_with_matches),
                "total_profiles_scraped": total_scraped
            },
            "flow_summary": {
                "step_1": "Competitor detection completed",
                "step_2": "Target profiles built",
                "step_3_4": "Per-target search + GPT-only Top-5 matches complete",
                "step_5": "Spectre matches written FIRST",
                "step_6": "Bright Data scraping completed",
                "step_7": "Employee reports written with scraped data"
            },
            "output_files": {
                "spectre_matches": "spectre_matches.json",
                "employee_reports": "employee_data/",
                "companies_analyzed": list(companies_with_matches)
            },
            "competitors": [asdict(comp) for comp in competitors],
            "target_profiles": [asdict(profile) for profile in target_profiles],
            "ghost_mirage_metadata": {
                "target_company": company_name,
                "total_competitors_detected": len(competitors),
                "total_matches_generated": len(all_matches),
                "scraping_success": total_scraped > 0
            }
        }

# =============================================================================
# Entry Points
# =============================================================================

async def mirage_async_entry(context: Dict[str, Any]) -> Dict[str, Any]:
    """Async entry point for orchestrator integration"""
    inputs = context.get("inputs", {})
    report_path = inputs.get("intelligence_report_path")
    
    if not report_path or not os.path.exists(report_path):
        raise FileNotFoundError(f"Intelligence report not found: {report_path}")
    
    num_competitors = inputs.get("num_competitors") or inputs.get("competitors_limit") or 10
    if isinstance(num_competitors, str):
        try:
            num_competitors = int(num_competitors)
        except ValueError:
            num_competitors = 10
    
    num_competitors = max(1, min(num_competitors, 50))
    logger.info(f"Processing with limit: {num_competitors} competitors")
    
    with open(report_path, "r", encoding="utf-8") as f:
        intelligence_data = json.load(f)
    
    company_name = extract_company_name(intelligence_data)
    logger.info(f"Starting MIRAGE analysis for: {company_name}")
    
    mirage = MirageSystem()
    results = await mirage.run_full_analysis(intelligence_data, num_competitors)
    
    return {
        "agents": {
            "mirage": {
                "result": results
            }
        }
    }

def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous entry point for orchestrator"""
    try:
        return asyncio.run(mirage_async_entry(context))
    except RuntimeError:
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
    print("MIRAGE System - Per-Target Flow Version")
    print("=" * 45)
    print("Flow: Target → per-competitor search (dept) → GPT Top-5 → Spectre → Bright")
    print("=" * 45)
    
    try:
        mirage = MirageSystem()
        print("Configuration: OK")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    
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
        with open(report_path, 'r', encoding='utf-8') as f:
            intelligence_data = json.load(f)
        
        company_name = extract_company_name(intelligence_data)
        print(f"\nAnalyzing: {company_name}")
        print(f"Competitors limit: {num_competitors}")
        print("Starting analysis with PER-TARGET FLOW...")
        
        results = await mirage.run_full_analysis(intelligence_data, num_competitors)
        
        print("\n" + "=" * 45)
        print("ANALYSIS COMPLETE")
        print("=" * 45)
        
        mm = results.get('mirage_metadata', {})
        rs = results.get('results_summary', {})
        fs = results.get('flow_summary', {})
        
        print(f"Company: {mm.get('target_company', company_name)}")
        print(f"Execution Time: {mm.get('execution_time_seconds', 0)}s")
        print(f"Competitors Detected: {rs.get('competitors_detected', 0)}")
        print(f"Target Profiles: {rs.get('target_profiles_built', 0)}")
        print(f"Total Matches (Top-5/target): {rs.get('total_matches', 0)}")
        print(f"High Quality Matches: {rs.get('high_quality_matches', 0)}")
        print(f"Companies with Matches: {rs.get('companies_with_matches', 0)}")
        print(f"Profiles Scraped: {rs.get('total_profiles_scraped', 0)}")
        
        print("\nFlow Execution:")
        for step, status in fs.items():
            print(f"  {step}: {status}")
        
        print("\nOutput Files:")
        print("- spectre_matches.json (written FIRST)")
        print("- employee_data/ (written AFTER Bright Data scraping)")
        
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
    
    parser = argparse.ArgumentParser(
        prog="mirage",
        description="MIRAGE - Per-Target Competitive Intelligence"
    )
    parser.add_argument("--report", "-r", help="Path to intelligence report JSON")
    parser.add_argument("--competitors", "-n", type=int, default=10, 
                       help="Number of competitors (1-50, default: 10)")
    
    args, unknown = parser.parse_known_args()
    
    if args.report:
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
                          .get("mirage_metadata", {}))
            outputs = (result.get("agents", {})
                             .get("mirage", {})
                             .get("result", {})
                             .get("output_files", {}))

            print("\n=== MIRAGE (non-interactive) complete ===")
            print(f"Target Company      : {meta.get('target_company', 'Unknown')}")
            print(f"Exec Time (seconds) : {meta.get('execution_time_seconds', 'n/a')}")
            print("Outputs:")
            print(f"- spectre_matches   : {outputs.get('spectre_matches', 'spectre_matches.json')}")
            print(f"- employee_reports  : {outputs.get('employee_reports', 'employee_data/')}")
        except Exception as e:
            logger.error(f"CLI execution failed: {e}")
            print(f"Error: {e}")
            sys.exit(1)
    else:
        try:
            asyncio.run(main())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
            loop.close
