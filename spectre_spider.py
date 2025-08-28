import os
import json
import re
import time
import math
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup

###############################################
# ENHANCED SPECTRE COURSE BUILDER WITH DATALOADER INTEGRATION
# - Uses DataLoader to extract existing skills from skills directory
# - Reads final gaps JSON for missing skills analysis
# - Creates comprehensive personalized courses based on prior knowledge
# - Merges existing skills data with gap analysis for intelligent course generation
###############################################

###############################################
# CONFIG
###############################################
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBsa_JCmZy5cJANA3-ksT3sPvwYqhuUQ4s")
GOOGLE_CX = os.getenv("GOOGLE_CX", "55d9d391fe2394876")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "2be1544b3dc14327b60a870fe8b94f35")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://notedai.openai.azure.com")  
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")

MAX_CONCURRENT_FETCHES = int(os.getenv("MAX_CONCURRENT_FETCHES", "3"))
MAX_CONCURRENT_EMPLOYEES = int(os.getenv("MAX_CONCURRENT_EMPLOYEES", "3"))  # Process N employees in parallel
MAX_CONCURRENT_SKILLS = int(os.getenv("MAX_CONCURRENT_SKILLS", "2"))  # Process N skills per employee in parallel
WEBSITES_PER_QUERY = int(os.getenv("WEBSITES_PER_QUERY", "5"))
DELAY_BETWEEN_SEARCHES_MS = int(os.getenv("DELAY_BETWEEN_SEARCHES_MS", "800"))  # Reduced delay
DELAY_BETWEEN_FETCH_BATCHES_MS = int(os.getenv("DELAY_BETWEEN_FETCH_BATCHES_MS", "400"))  # Reduced delay
DELAY_BETWEEN_EMPLOYEES_MS = int(os.getenv("DELAY_BETWEEN_EMPLOYEES_MS", "1000"))  # Delay between employee batches

PROCESS_LIMIT = int(os.getenv("PROCESS_LIMIT", "3"))  # Limit number of courses to generate
EMPLOYEE_SELECTION = os.getenv("EMPLOYEE_SELECTION", )  # Options: "random", "first", "specific", "interactive"
# Robust env parsing (no crash when var is missing)
SPECIFIC_EMPLOYEES = [
    s.strip() for s in (os.getenv("SPECIFIC_EMPLOYEES") or "").split(",") if s.strip()
]

INPUT_JSON = os.getenv("INPUT_JSON", "final_skill_gaps_detailed_gpt.json")

SKILLS_FILE = os.getenv("SKILLS_FILE", "Xto10X_skills.json")  # Direct skills file path
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "spectre_courses.json")

REQUIRED_ENVS = [
    ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    ("GOOGLE_CX", GOOGLE_CX),
    ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
    ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
    ("AZURE_OPENAI_DEPLOYMENT_ID", AZURE_OPENAI_DEPLOYMENT_ID),
]


###############################################
# CONFIG CLASS
###############################################
class Config:
    # File paths
    SKILLS_FILE = SKILLS_FILE
    
    # Company filtering
    SPECTRE_COMPANY = "Manipal"
    TARGET_COMPANIES = []  # empty means all companies
    EXCLUDE_COMPANIES = []
    
    # Employee filtering
    MIN_SKILLS_FOR_ANALYSIS = 1
    INCLUDE_EMPTY_SKILL_EMPLOYEES = True
    MAX_EMPLOYEES_TO_ANALYZE = None
    MAX_SKILLS_TO_SHOW = None
    
    # Matching parameters
    FUZZY_NAME_THRESHOLD = 0.7
    
    # Debug
    DEBUG_MODE = True

###############################################
# ENHANCED DATALOADER CLASS
###############################################
class DataLoader:
    def __init__(self, skills_file: str = None) -> None:
        self.skills_file = skills_file or Config.SKILLS_FILE
        # Outputs
        self.skills_by_name: Dict[str, List[str]] = {}  # {normalized_name: [skills,...]}
        self.skills_by_id: Dict[str, List[str]] = {}   # {employee_id: [skills,...]}
        self.name_to_id_map: Dict[str, str] = {}       # {normalized_name: employee_id}

    def run(self) -> None:
        """Load skills data from the skills file."""
        self._load_skills()

    @staticmethod
    def load_json(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def norm_name(s: str) -> str:
        """Normalize names for better matching"""
        if not s:
            return ""
        # Remove special characters and extra spaces, convert to lowercase
        normalized = re.sub(r"[^\w\s]", "", s.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _load_skills(self) -> None:
        """Load skills from the skills file"""
        print(f"=== Loading Skills Data from {self.skills_file} ===")
        
        if not os.path.exists(self.skills_file):
            print(f"⚠️  Skills file not found: {self.skills_file}")
            return

        try:
            data = self.load_json(self.skills_file)
            
            # Handle different JSON structures
            employees = []
            if isinstance(data, list):
                employees = data
            elif isinstance(data, dict):
                # Try different possible keys
                for key in ['employees', 'data', 'skills_data', 'employee_skills']:
                    if key in data and isinstance(data[key], list):
                        employees = data[key]
                        break
                else:
                    # If it's a dict but no list found, maybe it's direct mapping
                    if all(isinstance(v, (list, dict)) for v in data.values()):
                        # Convert dict format to list format
                        employees = []
                        for name_or_id, skills_data in data.items():
                            if isinstance(skills_data, list):
                                employees.append({
                                    "name": name_or_id,
                                    "employee_id": name_or_id,
                                    "skills": skills_data
                                })
                            elif isinstance(skills_data, dict) and "skills" in skills_data:
                                employee_data = skills_data.copy()
                                employee_data["name"] = employee_data.get("name", name_or_id)
                                employee_data["employee_id"] = employee_data.get("employee_id", name_or_id)
                                employees.append(employee_data)

            if not employees:
                print(f"⚠️  No employee data found in {self.skills_file}")
                return

            # Process employees
            for emp in employees:
                if not isinstance(emp, dict):
                    continue
                
                # Extract name, ID, and skills
                name = emp.get("name", "")
                emp_id = emp.get("employee_id", emp.get("id", ""))
                skills = emp.get("skills", [])
                
                # Handle case where skills might be a string
                if isinstance(skills, str):
                    skills = [skills]
                elif not isinstance(skills, list):
                    skills = []
                
                # Skip empty records
                if not name and not emp_id:
                    continue
                
                # Normalize and store
                normalized_name = self.norm_name(name) if name else ""
                
                if normalized_name:
                    self.skills_by_name[normalized_name] = skills
                    if emp_id:
                        self.name_to_id_map[normalized_name] = str(emp_id)
                
                if emp_id:
                    self.skills_by_id[str(emp_id)] = skills

            print(f"✅ Loaded skills for {len(self.skills_by_name)} employees by name")
            print(f"✅ Loaded skills for {len(self.skills_by_id)} employees by ID")
            
            if Config.DEBUG_MODE and self.skills_by_name:
                sample_names = list(self.skills_by_name.keys())[:5]
                print(f"📋 Sample employee names: {sample_names}")

        except Exception as e:
            print(f"❌ Failed to load skills file {self.skills_file}: {e}")
            import traceback
            traceback.print_exc()

    def find_employee_skills(self, employee_name: str) -> List[str]:
        """Find existing skills for an employee by name"""
        if not employee_name:
            return []
            
        normalized_name = self.norm_name(employee_name)
        
        # Direct match
        if normalized_name in self.skills_by_name:
            return self.skills_by_name[normalized_name]
        
        # Fuzzy match
        best_match = self._fuzzy_match_name(normalized_name, list(self.skills_by_name.keys()))
        if best_match:
            if Config.DEBUG_MODE:
                print(f"   🔍 Fuzzy matched '{employee_name}' to '{best_match}'")
            return self.skills_by_name[best_match]
        
        # Try partial matches (first name, last name combinations)
        name_parts = normalized_name.split()
        if len(name_parts) >= 2:
            # Try first + last name
            first_last = f"{name_parts[0]} {name_parts[-1]}"
            if first_last in self.skills_by_name:
                return self.skills_by_name[first_last]
            
            # Try partial matches in existing names
            for existing_name in self.skills_by_name.keys():
                existing_parts = existing_name.split()
                if len(existing_parts) >= 2:
                    # Check if first and last names match
                    if (name_parts[0] == existing_parts[0] and 
                        name_parts[-1] == existing_parts[-1]):
                        if Config.DEBUG_MODE:
                            print(f"   🔍 Partial matched '{employee_name}' to '{existing_name}'")
                        return self.skills_by_name[existing_name]
        
        return []

    def _fuzzy_match_name(self, target_name: str, available_names: List[str]) -> Optional[str]:
        """Find best fuzzy match for employee name"""
        if not target_name or not available_names:
            return None
            
        best_match = None
        best_score = 0.0
        
        for name in available_names:
            score = SequenceMatcher(None, target_name, name).ratio()
            if score > best_score and score >= Config.FUZZY_NAME_THRESHOLD:
                best_score = score
                best_match = name
        
        return best_match

###############################################
# UTILS
###############################################
async def sleep_ms(ms: int):
    await asyncio.sleep(ms / 1000.0)

async def with_backoff(coro_fn, attempts: int = 4, base_ms: int = 800):
    last_err = None
    for i in range(attempts):
        try:
            return await coro_fn()
        except Exception as e:
            last_err = e
            wait = base_ms * (2 ** i) + int(200 * (os.urandom(1)[0] / 255))
            await sleep_ms(wait)
    raise last_err

def ensure_env():
    missing = [k for k, v in REQUIRED_ENVS if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")

def normalize_skill(s: str) -> str:
    """Normalize skill names for consistency"""
    if not s:
        return ""
    m = s.lower().strip()
    mapping = {
        "ci cd": "CI/CD",
        "cicd": "CI/CD", 
        "microservices": "Microservices",
        "microservices architecture": "Microservices",
        "devops": "DevOps",
        "data analysis": "Data Analysis",
        "data analytics": "Data Analytics",
        "sql": "SQL",
        "ui ux design": "UI/UX Design",
        "ui/ux": "UI/UX Design",
        "financial analysis": "Financial Analysis",
        "distributed systems": "Distributed Systems",
        "kubernetes": "Kubernetes",
        "docker": "Docker",
        "linux": "Linux",
        "cloud computing": "Cloud Computing",
        "aws": "AWS",
        "azure": "Azure",
        "python": "Python",
        "javascript": "JavaScript",
        "react": "React",
        "nodejs": "Node.js",
        "node js": "Node.js",
        "machine learning": "Machine Learning",
        "ml": "Machine Learning",
        "artificial intelligence": "AI",
        "ai": "AI",
    }
    if m in mapping:
        return mapping[m]
    return re.sub(r"\b(\w)", lambda mo: mo.group(1).upper(), re.sub(r"\s+", " ", s.strip()))

def categorize_missing_skills(missing_skills: List[str], skill_importance: Dict[str, str]) -> Dict[str, List[str]]:
    """Categorize missing skills by importance level"""
    categories = {
        "critical": [],
        "important": [],
        "nice_to_have": []
    }
    
    for skill in missing_skills:
        normalized_skill = normalize_skill(skill)
        importance = skill_importance.get(skill, "").lower()
        
        if importance == "critical":
            categories["critical"].append(normalized_skill)
        elif importance == "important":
            categories["important"].append(normalized_skill)
        else:
            categories["nice_to_have"].append(normalized_skill)
    
    return categories

###############################################
# AZURE OPENAI (chat/completions)
###############################################
async def azure_chat_completion(session: ClientSession, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 1200) -> str:
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_ID}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async def do_post():
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"Azure OpenAI error {resp.status}: {text}")
            data = await resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return await with_backoff(do_post)

###############################################
# SSR FOUNDATION AGENT (Enhanced with Prior Knowledge)
###############################################
class SSRFoundationAgent:
    def __init__(self, session: ClientSession):
        self.session = session

    async def google_search(self, query: str) -> List[str]:
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={aiohttp.helpers.quote(query, safe='')}"
        )
        print(f"🔍 search: {query}")

        async def do_get():
            async with self.session.get(url, timeout=15) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"CSE error {resp.status}: {text}")
                data = await resp.json()
                items = data.get("items", [])
                return [it.get("link") for it in items[:WEBSITES_PER_QUERY] if it.get("link")]

        return await with_backoff(do_get)

    async def fetch_and_clean(self, url: str) -> Optional[str]:
        headers = {
            "User-Agent": "SpectreSpider/1.0 (+contact@example.com)"
        }
        async def do_get():
            async with self.session.get(url, headers=headers, timeout=15) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Fetch error {resp.status} for {url}")
                ct = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" not in ct:
                    return None
                html = await resp.text(errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
                    tag.decompose()
                for ad in soup.select(".ad, .ads, .advertisement"):
                    ad.decompose()
                node = None
                for sel in ["main", "article", ".content", ".post-content", ".entry-content", "body"]:
                    node = soup.select_one(sel)
                    if node:
                        break
                text = node.get_text(" ") if node else soup.get_text(" ")
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) < 300:
                    return None
                return text[:20000]
        return await with_backoff(do_get)

    async def summarize(self, content: str, topic: str, intent: str, existing_skills: List[str]) -> Optional[str]:
        prior_knowledge = f"Learner's existing skills: {', '.join(existing_skills[:10])}" if existing_skills else "New learner with limited background"
        
        sys = {
            "role": "system", 
            "content": f"Extract a 200-300 word educational summary for personalized course building. Consider the learner's existing knowledge and build upon it appropriately. {prior_knowledge}. Focus on bridging knowledge gaps. If content is off-topic/marketing, reply IRRELEVANT."
        }
        usr = {
            "role": "user", 
            "content": f"Topic: {topic}\nLearning Intent: {intent}\nExisting Skills: {existing_skills}\n\nCONTENT:\n{content[:12000]}"
        }
        out = await azure_chat_completion(self.session, [sys, usr], temperature=0.2, max_tokens=900)
        if re.search(r"\bIRRELEVANT\b|OFF_TOPIC", out, re.I):
            return None
        return out.strip() if out else None

    async def generate_queries(self, topic: str, intent: str, existing_skills: List[str]) -> List[str]:
        skill_context = f"Building on: {', '.join(existing_skills[:5])}" if existing_skills else "Starting from basics"
        
        sys = {
            "role": "system", 
            "content": f"Generate 5-7 targeted Google search queries (3–7 words each) for personalized learning. {skill_context}. Focus on educational content that bridges from existing knowledge to new skills. One query per line, no numbering."
        }
        usr = {
            "role": "user", 
            "content": f"New Skill to Learn: {topic}\nLearning Intent: {intent}\nExisting Skills: {existing_skills}"
        }
        
        out = await azure_chat_completion(self.session, [sys, usr], temperature=0.2, max_tokens=500)
        qs = [q.strip() for q in out.split("\n") if q.strip()]
        
        if not qs:
            # Enhanced fallback queries based on existing skills
            base_queries = [
                f"{topic} fundamentals tutorial",
                f"{topic} beginner guide",
                f"{topic} best practices",
                f"{topic} practical examples",
                f"learn {topic} step by step"
            ]
            
            if existing_skills:
                # Add advanced queries if they have related skills
                related_skills = [s for s in existing_skills if any(word in s.lower() for word in topic.lower().split())]
                if related_skills:
                    base_queries.extend([
                        f"{topic} for {related_skills[0]} developers",
                        f"advanced {topic} techniques"
                    ])
            
            return base_queries[:7]
        return qs[:7]

    async def execute_searches(self, queries: List[str]) -> List[str]:
        urls: List[str] = []
        for q in queries:
            try:
                results = await self.google_search(q)
                urls.extend(results)
            except Exception as e:
                print(f"Search failed for '{q}': {e}")
            await sleep_ms(DELAY_BETWEEN_SEARCHES_MS)
        
        # Dedupe while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        return unique_urls

    async def process_urls(self, urls: List[str], topic: str, intent: str, existing_skills: List[str]) -> List[str]:
        print(f"🔄 Processing {len(urls)} URLs (concurrency={MAX_CONCURRENT_FETCHES})")
        summaries: List[str] = []
        
        # Process URLs in batches
        idx = 0
        while idx < len(urls):
            batch = urls[idx: idx + MAX_CONCURRENT_FETCHES]
            tasks = [self._fetch_then_summarize(u, topic, intent, existing_skills) for u in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for r in results:
                if isinstance(r, str) and r:
                    summaries.append(r)
            
            idx += MAX_CONCURRENT_FETCHES
            if idx < len(urls):
                await sleep_ms(DELAY_BETWEEN_FETCH_BATCHES_MS)
        
        print(f"✅ Generated {len(summaries)}/{len(urls)} summaries")
        return summaries

    async def _fetch_then_summarize(self, url: str, topic: str, intent: str, existing_skills: List[str]) -> Optional[str]:
        try:
            content = await self.fetch_and_clean(url)
            if not content:
                return None
            return await self.summarize(content, topic, intent, existing_skills)
        except Exception:
            return None

    async def build_web_of_truth(self, topic: str, intent: str, existing_skills: List[str]) -> str:
        print(f"\n===== Building Web of Truth for {topic} =====")
        queries = await self.generate_queries(topic, intent, existing_skills)
        urls = await self.execute_searches(queries)
        
        if not urls:
            return "NO_URLS"
        
        summaries = await self.process_urls(urls, topic, intent, existing_skills)
        if not summaries:
            return "NO_SUMMARIES"
        
        combined = ("\n\n").join(summaries)[:18000]
        prior_knowledge_context = f"Learner's existing skills: {', '.join(existing_skills)}" if existing_skills else "New learner"
        
        sys = {
            "role": "system", 
            "content": f"Create a comprehensive Web of Truth for personalized curriculum design. Build upon the learner's existing knowledge to create appropriate learning progression. {prior_knowledge_context}. Use ONLY the provided text sources."
        }
        usr = {
            "role": "user", 
            "content": f"Skill to Learn: {topic}\nLearning Intent: {intent}\nExisting Skills: {existing_skills}\nSources Used: {len(summaries)}\n\nCOMBINED CONTENT:\n{combined}"
        }
        
        report = await azure_chat_completion(self.session, [sys, usr], temperature=0.2, max_tokens=1800)
        return report

###############################################
# ENHANCED SPIDER KING
###############################################
class SpiderKing:
    def __init__(self, session: ClientSession):
        self.session = session

    def _parse_course_json(self, s: str) -> Optional[Dict[str, Any]]:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', s, re.DOTALL)
        if not json_match:
            return None
        
        try:
            obj = json.loads(json_match.group(0))
            if obj.get("courseName") and isinstance(obj.get("course"), list):
                return obj
        except Exception:
            return None
        return None

    def _merge_skill_courses(self, courses: List[Dict[str, Any]], employee_name: str, comprehensive_intent: str) -> Dict[str, Any]:
        """Merge multiple skill-specific courses into one comprehensive curriculum"""
        merged_chapters = []
        seen_topics = set()
        
        for course in courses:
            course_chapters = course.get("course", [])
            for chapter in course_chapters:
                chapter_name = chapter.get("chapterName", "Learning Module")
                topics = []
                
                for topic in chapter.get("chapter", []):
                    topic_name = topic.get("topicName", "").strip()
                    details = topic.get("details", "").strip()
                    
                    # Avoid duplicate topics
                    topic_key = topic_name.lower()
                    if topic_name and details and topic_key not in seen_topics:
                        seen_topics.add(topic_key)
                        topics.append({
                            "topicName": topic_name,
                            "details": details
                        })
                
                if topics:
                    merged_chapters.append({
                        "chapterName": chapter_name,
                        "chapter": topics
                    })
        
        return {
            "courseName": f"Personalized Learning Path for {employee_name}",
            "description": comprehensive_intent,
            "skillsCovered": len(courses),
            "totalTopics": len(seen_topics),
            "course": merged_chapters
        }

    async def create_skill_specific_course(self, web_of_truth: str, skill: str, intent: str, existing_skills: List[str]) -> Optional[Dict[str, Any]]:
        """Create a course for a specific skill considering existing knowledge"""
        prior_knowledge = f"Building on existing skills: {', '.join(existing_skills[:8])}" if existing_skills else "Starting from fundamentals"
        
        sys = {
            "role": "system",
            "content": f"""YOU ARE SPIDER KING - Expert Course Architect. Create a structured JSON course that builds upon existing knowledge.
            {prior_knowledge}
            
            Return ONLY JSON with structure:
            {{
                "courseName": "...",
                "course": [
                    {{
                        "chapterName": "...",
                        "chapter": [
                            {{"topicName": "...", "details": "..."}}
                        ]
                    }}
                ]
            }}
            
            Create progressive learning that bridges from existing skills to new skill."""
        }
        
        usr = {
            "role": "user",
            "content": f"""LEARNING CONTEXT:
Skill to Master: {skill}
Learning Intent: {intent}
Existing Skills: {existing_skills}

WEB OF TRUTH:
{web_of_truth}

Create a personalized course that leverages existing knowledge."""
        }
        
        out = await azure_chat_completion(self.session, [sys, usr], temperature=0.2, max_tokens=1500)
        return self._parse_course_json(out)

    async def create_comprehensive_course(self, skill_courses: List[Dict[str, Any]], employee_name: str, comprehensive_intent: str, existing_skills: List[str]) -> Dict[str, Any]:
        """Create final comprehensive course from individual skill courses"""
        merged_course = self._merge_skill_courses(skill_courses, employee_name, comprehensive_intent)
        
        # Add learning progression recommendations
        if existing_skills:
            merged_course["learningPrerequisites"] = existing_skills[:10]
            merged_course["recommendedPreparation"] = f"This course builds upon your existing knowledge in {', '.join(existing_skills[:3])}"
        else:
            merged_course["recommendedPreparation"] = "This course starts from fundamentals - no prior experience required"
            
        return merged_course

###############################################
# ENHANCED EMPLOYEE SKILLS RESOLVER
###############################################
class EmployeeSkillsResolver:
    """Enhanced class to resolve employee skills using DataLoader and gaps data"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def resolve_employee_skills(self, employee_name: str) -> List[str]:
        """Resolve existing skills for an employee using multiple strategies"""
        # Strategy 1: Direct name lookup in skills data
        existing_skills = self.data_loader.find_employee_skills(employee_name)
        
        if existing_skills:
            print(f"   📚 Found {len(existing_skills)} existing skills for {employee_name}")
            return [normalize_skill(skill) for skill in existing_skills]
        
        print(f"   ⚠️  No existing skills found for {employee_name}")
        return []

###############################################
# ENHANCED INTENT GENERATION WITH CONTEXT
###############################################
async def infer_comprehensive_intent_with_context(session: ClientSession, employee_name: str, role: str, 
                                                existing_skills: List[str], missing_skills_by_category: Dict[str, List[str]]) -> str:
    """Generate comprehensive learning intent considering existing skills and gaps"""
    
    all_missing = []
    for category, skills in missing_skills_by_category.items():
        all_missing.extend(skills)
    
    existing_context = f"Strong foundation in: {', '.join(existing_skills[:8])}" if existing_skills else "Building from fundamentals"
    critical_skills = missing_skills_by_category.get("critical", [])
    important_skills = missing_skills_by_category.get("important", [])
    
    sys = {
        "role": "system", 
        "content": """Create a comprehensive, personalized learning intent that:
1. Acknowledges existing strengths and builds upon them
2. Prioritizes critical skill gaps for immediate career impact
3. Creates logical learning progression 
4. Focuses on practical, job-relevant outcomes
5. Motivates the learner with clear value proposition

Keep it concise but inspiring (2-3 sentences)."""
    }
    
    usr = {
        "role": "user", 
        "content": f"""Employee Profile:
• Name: {employee_name}
• Role: {role}
• Existing Strengths: {existing_context}
• Critical Gaps: {', '.join(critical_skills[:5]) if critical_skills else 'None identified'}
• Important Gaps: {', '.join(important_skills[:5]) if important_skills else 'None identified'}
• Total Skills to Develop: {len(all_missing)}

Generate a personalized learning intent that bridges from their current skills to career advancement."""
    }
    
    out = await azure_chat_completion(session, [sys, usr], temperature=0.3, max_tokens=400)
    return out.strip() if out else f"Comprehensive professional development for {role}, building upon existing expertise to master {len(all_missing)} essential skills."

###############################################
# PARALLEL PROCESSING UTILITIES
###############################################
class ProgressTracker:
    """Track progress of parallel processing"""
    def __init__(self, total_employees: int, total_skills_per_employee: int = 6):
        self.total_employees = total_employees
        self.total_skills_per_employee = total_skills_per_employee
        self.completed_employees = 0
        self.completed_skills = 0
        self.lock = asyncio.Lock()
    
    async def update_skill_progress(self, employee_name: str, skill: str, success: bool):
        async with self.lock:
            self.completed_skills += 1
            status = "✅" if success else "❌"
            print(f"   {status} {employee_name}: {skill} ({self.completed_skills} skills completed)")
    
    async def update_employee_progress(self, employee_name: str, skills_processed: int):
        async with self.lock:
            self.completed_employees += 1
            print(f"🎓 Completed {employee_name}: {skills_processed} modules ({self.completed_employees}/{self.total_employees} employees)")

###############################################
# PARALLEL EMPLOYEE PROCESSOR
###############################################
class ParallelEmployeeProcessor:
    def __init__(self, session: ClientSession, progress_tracker: ProgressTracker):
        self.session = session
        self.ssr = SSRFoundationAgent(session)
        self.spider_king = SpiderKing(session)
        self.progress_tracker = progress_tracker
    
    async def process_single_employee(self, employee_data: Dict[str, Any], 
                                    skills_resolver: EmployeeSkillsResolver, 
                                    employee_index: int, total_employees: int) -> Dict[str, Any]:
        """Process a single employee with parallel skill processing"""
        employee_name = employee_data.get("manipal_employee", "Unknown")
        role = employee_data.get("role", "Unknown")
        company = employee_data.get("company", "manipal")
        
        try:
            # Resolve existing skills
            existing_skills = skills_resolver.resolve_employee_skills(employee_name)
            
            # Get missing skills
            missing_skills_raw = employee_data.get("missing_skills") or []
            skill_importance = employee_data.get("skill_importance") or {}
            
            if not missing_skills_raw:
                return {
                    "employeeName": employee_name,
                    "role": role,
                    "company": company,
                    "existingSkills": existing_skills,
                    "missingSkills": [],
                    "course": None,
                    "reason": "No skill gaps identified"
                }
            
            # Categorize missing skills
            missing_skills_categorized = categorize_missing_skills(missing_skills_raw, skill_importance)
            all_missing_skills = []
            for category_skills in missing_skills_categorized.values():
                all_missing_skills.extend(category_skills)
            
            # Generate comprehensive learning intent
            comprehensive_intent = await infer_comprehensive_intent_with_context(
                self.session, employee_name, role, existing_skills, missing_skills_categorized
            )
            
            # Select priority skills
            priority_skills = (
                missing_skills_categorized["critical"][:3] + 
                missing_skills_categorized["important"][:3] + 
                missing_skills_categorized["nice_to_have"][:2]
            )[:6]  # Max 6 skills
            
            # Process skills in parallel (with limited concurrency)
            skill_courses = await self.process_skills_parallel(
                priority_skills, comprehensive_intent, existing_skills, employee_name
            )
            
            # Create comprehensive course
            if skill_courses:
                comprehensive_course = await self.spider_king.create_comprehensive_course(
                    skill_courses, employee_name, comprehensive_intent, existing_skills
                )
            else:
                comprehensive_course = {
                    "courseName": f"Learning Path for {employee_name}",
                    "description": comprehensive_intent,
                    "course": [],
                    "note": "No course content could be generated - insufficient web resources"
                }
            
            # Update progress
            await self.progress_tracker.update_employee_progress(employee_name, len(skill_courses))
            
            return {
                "employeeName": employee_name,
                "role": role,
                "company": company,
                "existingSkills": existing_skills,
                "missingSkillsCategories": missing_skills_categorized,
                "prioritySkillsSelected": priority_skills,
                "comprehensiveIntent": comprehensive_intent,
                "course": comprehensive_course,
                "skillModulesProcessed": len(skill_courses),
                "totalMissingSkills": len(all_missing_skills),
                "processingStatus": "complete"
            }
            
        except Exception as e:
            print(f"❌ Error processing {employee_name}: {e}")
            return {
                "employeeName": employee_name,
                "role": role,
                "company": company,
                "existingSkills": [],
                "course": None,
                "error": str(e),
                "processingStatus": "failed"
            }
    
    async def process_skills_parallel(self, priority_skills: List[str], comprehensive_intent: str, 
                                    existing_skills: List[str], employee_name: str) -> List[Dict[str, Any]]:
        """Process multiple skills in parallel for a single employee"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SKILLS)
        
        async def process_single_skill(skill: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    # Build Web of Truth for this skill
                    web_of_truth = await self.ssr.build_web_of_truth(skill, comprehensive_intent, existing_skills)
                    
                    if web_of_truth not in ["NO_URLS", "NO_SUMMARIES"]:
                        # Create course for this skill
                        skill_course = await self.spider_king.create_skill_specific_course(
                            web_of_truth, skill, comprehensive_intent, existing_skills
                        )
                        
                        success = skill_course is not None
                        await self.progress_tracker.update_skill_progress(employee_name, skill, success)
                        return skill_course
                    else:
                        await self.progress_tracker.update_skill_progress(employee_name, skill, False)
                        return None
                        
                except Exception as e:
                    await self.progress_tracker.update_skill_progress(employee_name, skill, False)
                    return None
        
        # Process all skills in parallel
        tasks = [process_single_skill(skill) for skill in priority_skills]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        skill_courses = [result for result in results 
                        if isinstance(result, dict) and result is not None]
        
        return skill_courses
import random

def select_employees_interactive(employees: List[Dict[str, Any]]) -> List[int]:
    """Interactive employee selection"""
    print(f"\n📋 Available Employees ({len(employees)} total):")
    print("-" * 80)
    
    for i, emp in enumerate(employees):
        name = emp.get("manipal_employee", "Unknown")
        role = emp.get("role", "Unknown")
        missing_count = len(emp.get("missing_skills", []))
        print(f"{i+1:2d}. {name:<30} | {role:<25} | {missing_count} missing skills")
    
    print("-" * 80)
    print(f"Enter your selection (max {PROCESS_LIMIT}):")
    print("• Enter numbers separated by commas (e.g., 1,3,5)")
    print("• Enter 'r' for random selection")
    print("• Enter 'f' for first N employees")
    print("• Enter 'q' to quit")
    
    while True:
        selection = input("\nYour choice: ").strip().lower()
        
        if selection == 'q':
            return []
        elif selection == 'r':
            return random.sample(range(len(employees)), min(PROCESS_LIMIT, len(employees)))
        elif selection == 'f':
            return list(range(min(PROCESS_LIMIT, len(employees))))
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(",")]
                indices = [i for i in indices if 0 <= i < len(employees)]
                if len(indices) > PROCESS_LIMIT:
                    print(f"⚠️  Too many selections. Limiting to first {PROCESS_LIMIT}.")
                    indices = indices[:PROCESS_LIMIT]
                if indices:
                    return indices
                else:
                    print("❌ No valid selections found. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter numbers separated by commas.")

def select_employees_by_names(employees: List[Dict[str, Any]], target_names: List[str]) -> List[int]:
    """Select employees by specific names"""
    selected_indices = []
    
    for target_name in target_names:
        target_name = target_name.strip()
        if not target_name:
            continue
            
        # Try exact match first
        for i, emp in enumerate(employees):
            emp_name = emp.get("manipal_employee", "").strip()
            if emp_name.lower() == target_name.lower():
                selected_indices.append(i)
                break
        else:
            # Try fuzzy match
            best_match_idx = None
            best_score = 0.0
            
            for i, emp in enumerate(employees):
                emp_name = emp.get("manipal_employee", "").strip()
                score = SequenceMatcher(None, target_name.lower(), emp_name.lower()).ratio()
                if score > best_score and score >= 0.7:
                    best_score = score
                    best_match_idx = i
            
            if best_match_idx is not None:
                selected_indices.append(best_match_idx)
                print(f"   🔍 Fuzzy matched '{target_name}' to '{employees[best_match_idx].get('manipal_employee')}'")
            else:
                print(f"   ⚠️  Employee '{target_name}' not found")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    
    return unique_indices[:PROCESS_LIMIT]

def select_employees(employees: List[Dict[str, Any]], selection_mode: str) -> List[int]:
    """Main employee selection function"""
    if not employees:
        return []
    
    if selection_mode == "interactive":
        return select_employees_interactive(employees)
    elif selection_mode == "first":
        return list(range(min(PROCESS_LIMIT, len(employees))))
    elif selection_mode == "random":
        return random.sample(range(len(employees)), min(PROCESS_LIMIT, len(employees)))
    elif selection_mode == "specific" and SPECIFIC_EMPLOYEES:
        return select_employees_by_names(employees, SPECIFIC_EMPLOYEES)
    else:
        # Default to first N employees
        print(f"⚠️  Unknown selection mode '{selection_mode}', defaulting to first {PROCESS_LIMIT} employees")
        return list(range(min(PROCESS_LIMIT, len(employees))))

###############################################
# EMPLOYEE SELECTION UTILITIES  
###############################################
###############################################
# ENHANCED MAIN FUNCTION WITH PARALLEL PROCESSING
###############################################
async def build_comprehensive_courses_with_prior_knowledge():
    ensure_env()
    
    # Check input files
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Gap analysis file not found: {INPUT_JSON}")
    
    print("🚀 Starting Enhanced SPECTRE Course Builder with Parallel Processing")
    
    # Initialize DataLoader to get existing skills
    print("\n" + "="*60)
    print("STAGE 1: Loading Employee Skills Data")
    print("="*60)
    
    data_loader = DataLoader(skills_file=SKILLS_FILE)
    data_loader.run()
    
    skills_resolver = EmployeeSkillsResolver(data_loader)
    
    # Load gap analysis data
    print("\n" + "="*60)
    print("STAGE 2: Loading Gap Analysis Data")
    print("="*60)
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        gap_analysis_data: List[Dict[str, Any]] = json.load(f)
    
    print(f"Loaded gap analysis for {len(gap_analysis_data)} employees")
    
    # Employee Selection
    print("\n" + "="*60)
    print("STAGE 2.5: Employee Selection")
    print("="*60)
    
    selected_indices = select_employees(gap_analysis_data, EMPLOYEE_SELECTION)
    
    if not selected_indices:
        print("❌ No employees selected. Exiting.")
        return
    
    selected_employees = [gap_analysis_data[i] for i in selected_indices]
    
    print(f"📝 Selected {len(selected_employees)} employees for course generation:")
    for i, emp in enumerate(selected_employees):
        name = emp.get("manipal_employee", "Unknown")
        role = emp.get("role", "Unknown") 
        missing_count = len(emp.get("missing_skills", []))
        print(f"   {i+1}. {name} ({role}) - {missing_count} missing skills")
    
    # Parallel Processing Setup
    print("\n" + "="*60)
    print("STAGE 3: Parallel Course Generation")
    print("="*60)
    print(f"🔄 Processing {len(selected_employees)} employees with:")
    print(f"   • Max {MAX_CONCURRENT_EMPLOYEES} employees in parallel")
    print(f"   • Max {MAX_CONCURRENT_SKILLS} skills per employee in parallel") 
    print(f"   • Max {MAX_CONCURRENT_FETCHES} web fetches in parallel")
    print("="*60)
    
    progress_tracker = ProgressTracker(len(selected_employees))
    
    async with aiohttp.ClientSession() as session:
        processor = ParallelEmployeeProcessor(session, progress_tracker)
        
        # Process employees in parallel batches
        outputs: List[Dict[str, Any]] = []
        employee_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMPLOYEES)
        
        async def process_employee_with_semaphore(emp_data, emp_index):
            async with employee_semaphore:
                result = await processor.process_single_employee(
                    emp_data, skills_resolver, emp_index, len(selected_employees)
                )
                # Small delay between employee batches to avoid overwhelming APIs
                await sleep_ms(DELAY_BETWEEN_EMPLOYEES_MS)
                return result
        
        # Create tasks for all employees
        tasks = [
            process_employee_with_semaphore(emp_data, i) 
            for i, emp_data in enumerate(selected_employees)
        ]
        
        # Execute all tasks in parallel
        print(f"🚀 Starting parallel processing of {len(tasks)} employees...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                emp_name = selected_employees[i].get("manipal_employee", "Unknown")
                print(f"❌ Failed to process {emp_name}: {result}")
                outputs.append({
                    "employeeName": emp_name,
                    "error": str(result),
                    "processingStatus": "failed"
                })
            else:
                outputs.append(result)
        
        elapsed = time.time() - start_time
        print(f"⚡ Parallel processing completed in {elapsed:.1f} seconds")
    
    # Save results
    print("\n" + "="*60)
    print("STAGE 4: Saving Results")
    print("="*60)
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    # Generate summary statistics
    successful = [emp for emp in outputs if emp.get("processingStatus") == "complete"]
    failed = [emp for emp in outputs if emp.get("processingStatus") == "failed"]
    total_modules = sum(emp.get("skillModulesProcessed", 0) for emp in successful)
    
    print(f"💾 Saved comprehensive course data for {len(outputs)} employees")
    print(f"   ✅ Successfully processed: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")
    print(f"   📚 Total course modules generated: {total_modules}")
    print(f"   💾 Output file: {OUTPUT_JSON}")
    print(f"   🕒 Total processing time: {elapsed:.1f} seconds")
    print(f"   ⚡ Average time per employee: {elapsed/len(outputs):.1f} seconds")
    
    # Detailed summary
    if successful:
        print(f"\n📊 Success Summary:")
        for emp in successful:
            modules = emp.get('skillModulesProcessed', 0)
            missing = emp.get('totalMissingSkills', 0)
            print(f"   ✅ {emp['employeeName']}: {modules} modules, {missing} total gaps")
    
    if failed:
        print(f"\n⚠️  Failed Employees:")
        for emp in failed:
            print(f"   ❌ {emp['employeeName']}: {emp.get('error', 'Unknown error')}")

###############################################
# JSON UTILITIES
###############################################
def load_json(path: str) -> Any:
    """Load JSON file with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON from {path}: {e}")
        return {}

def save_json(data: Any, path: str) -> None:
    """Save JSON file with error handling"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved JSON to {path}")
    except Exception as e:
        print(f"❌ Failed to save JSON to {path}: {e}")
async def run_spectre_spider_with_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Function interface for orchestrator integration
    """
    # Set environment variables from config
    os.environ["INPUT_JSON"] = config.get("input_json", "final_skill_gaps_detailed_gpt.json")
    os.environ["SKILLS_FILE"] = config.get("skills_file", "Xto10X_skills.json")  
    os.environ["OUTPUT_JSON"] = config.get("output_json", "spectre_courses.json")
    os.environ["PROCESS_LIMIT"] = str(config.get("process_limit", 3))
    os.environ["EMPLOYEE_SELECTION"] = config.get("employee_selection", "first")
    
    # Set API keys
    if config.get("google_api_key"):
        os.environ["GOOGLE_API_KEY"] = config["google_api_key"]
    if config.get("google_cx"):
        os.environ["GOOGLE_CX"] = config["google_cx"]
    
    # Set Azure config
    azure_config = config.get("azure_config", {})
    for key, env_var in [
        ("api_key", "AZURE_OPENAI_API_KEY"),
        ("endpoint", "AZURE_OPENAI_ENDPOINT"), 
        ("api_version", "AZURE_OPENAI_API_VERSION"),
        ("deployment_id", "AZURE_OPENAI_DEPLOYMENT_ID")
    ]:
        if azure_config.get(key):
            os.environ[env_var] = azure_config[key]
    
    # Run the main function
    await build_comprehensive_courses_with_prior_knowledge()
    
    # Return result summary
    output_file = config.get("output_json", "spectre_courses.json")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            courses_data = json.load(f)
        return {"courses_generated": len(courses_data), "output_file": output_file}
    else:
        return {"courses_generated": 0, "error": "Output file not created"}
###############################################
# MAIN EXECUTION
###############################################
if __name__ == "__main__":
    start_time = time.time()
    print("\n" + "="*60)
    print("🚀 SPECTRE COURSE BUILDER")
    print("="*60)
    print(f"• Input: {INPUT_JSON}")
    print(f"• Skills File: {SKILLS_FILE}")
    print(f"• Output: {OUTPUT_JSON}")
    print(f"• Process Limit: {PROCESS_LIMIT}")
    print(f"• Selection Mode: {EMPLOYEE_SELECTION}")
    if EMPLOYEE_SELECTION == "specific" and SPECIFIC_EMPLOYEES:
        print(f"• Specific Employees: {', '.join(SPECIFIC_EMPLOYEES)}")
    print(f"• Max Concurrent Employees: {MAX_CONCURRENT_EMPLOYEES}")
    print(f"• Max Concurrent Skills per Employee: {MAX_CONCURRENT_SKILLS}")
    print(f"• Max Concurrent Fetches: {MAX_CONCURRENT_FETCHES}")
    print(f"• Websites per Query: {WEBSITES_PER_QUERY}")
    print("="*60)
    print("\n🚀 Parallel Processing Configuration:")
    print("• Multiple employees processed simultaneously")
    print("• Multiple skills per employee processed in parallel")
    print("• Intelligent progress tracking and error handling")
    print("• Optimized delays to respect API rate limits")
    print("="*60)
    
    try:
        asyncio.run(build_comprehensive_courses_with_prior_knowledge())
        print("\n✅ Processing completed successfully!")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total execution time: {math.ceil(elapsed)} seconds")