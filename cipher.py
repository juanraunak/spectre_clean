# agent3_fractal.py
# ------------------------------------------------------------
# Agent 3 — FRACTAL: company-by-company skills extraction.
# Core logic preserved:
# - Walk JSON files per company
# - Extract basic employee profiles
# - Use Azure OpenAI for skill inference (fallback to keywords)
# - Save per-company skills JSON
# Orchestrator adapters:
# - async run(context) -> context
# - run_sync(context)
# ------------------------------------------------------------

import json
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    # Azure OpenAI SDK (v1.x)
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None  # will be checked at runtime

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent3_fractal")


# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class EmployeeProfile:
    """Basic employee profile structure"""
    name: str
    current_position: str
    company_name: str
    employee_id: str
    skills: List[str] = None


@dataclass
class CompanySkillProfile:
    """Company-wise skill profile"""
    company_name: str
    employees: List[EmployeeProfile]


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _safe_json_array(s: str) -> Optional[List[str]]:
    """
    Try very hard to parse a JSON array from a model response.
    - Strips code fences
    - Extracts first [...] block
    Returns None if parsing fails.
    """
    if not s:
        return None
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    # quick path
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # find first [...] segment
    import re as _re
    m = _re.search(r"\[(?:.|\s)*\]", t, flags=_re.MULTILINE)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            return None
    return None


# ------------------------------------------------------------
# Fractal Skill Extractor
# ------------------------------------------------------------
class FractalSkillExtractor:
    """Extracts skills company by company"""

    def __init__(self, azure_config: Dict[str, str] = None, max_workers: int = 5):
        self.max_workers = max_workers
        self.company_profiles: Dict[str, CompanySkillProfile] = {}

        # Azure config (no hard-coded keys; use env or provided dict)
        azure_config = azure_config or {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY","2be1544b3dc14327b60a870fe8b94f35"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "https://notedai.openai.azure.com"),
            "deployment_id": os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o"),
        }

        self.deployment_id = azure_config.get("deployment_id")
        self._client = None
        if AzureOpenAI and all([azure_config.get("api_key"), azure_config.get("endpoint"), azure_config.get("api_version")]):
            try:
                self._client = AzureOpenAI(
                    api_key=azure_config["api_key"],
                    api_version=azure_config["api_version"],
                    azure_endpoint=azure_config["endpoint"],
                )
                logger.info("✅ Azure OpenAI client configured")
            except Exception as e:
                logger.warning(f"⚠️ Azure OpenAI client init failed: {e}")
        else:
            logger.info("ℹ️ Azure OpenAI not configured; will use keyword fallback")

    # ------------ Connection Test ------------
    def test_openai_connection(self) -> bool:
        if not self._client:
            logger.error("❌ OpenAI client is None")
            return False
        try:
            r = self._client.chat.completions.create(
                model=self.deployment_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0.0,
            )
            ok = bool(getattr(r, "choices", None))
            logger.info("✅ OpenAI connection test successful" if ok else "⚠️ OpenAI test returned no choices")
            return ok
        except Exception as e:
            logger.error(f"❌ OpenAI connection test failed: {e}")
            return False

    # ------------ Main: per-company JSON files ------------
    async def process_company_files(self, data_directory: str, output_dir: str = "output/company_skills"):
        """Process each company file separately (per-file extraction + save)."""
        data_path = Path(data_directory)
        json_files = sorted(list(data_path.glob("*.json")))
        logger.info(f"📂 Found {len(json_files)} JSON files in {data_directory}")

        for json_file in json_files:
            company_name = self._extract_company_name(json_file)
            logger.info(f"🏢 Processing {company_name} ({json_file.name})")

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to read {json_file.name}: {e}")
                continue

            # Extract employees list from flexible shapes
            employees_data: List[Dict[str, Any]] = []
            if isinstance(data, dict):
                if "employee_intelligence" in data and "employees" in data["employee_intelligence"]:
                    employees_data = data["employee_intelligence"]["employees"]
                elif "employees" in data:
                    employees_data = data["employees"]

            employees: List[EmployeeProfile] = []
            for i, emp_data in enumerate(employees_data):
                emp = self._extract_basic_profile(emp_data, f"{company_name[:3]}_{i+1:03d}")
                if emp:
                    employees.append(emp)

            # Extract skills
            await self.extract_skills_for_company(company_name, employees)

            # Save promptly
            self.save_company_profile(company_name, output_dir)

        logger.info("🏁 Completed company-wise skill extraction.")

    # ------------ Helpers ------------
    def _extract_company_name(self, json_file: Path) -> str:
        """Derive a human-friendly company name from filename."""
        filename = json_file.stem
        clean_name = re.sub(r"(_report|_intelligence|_complete|_fintech)", "", filename, flags=re.IGNORECASE)
        clean_name = clean_name.replace("_", " ").strip()
        return clean_name.title() if clean_name else filename.title()

    def _extract_basic_profile(self, emp_data: Dict[str, Any], employee_id: str) -> Optional[EmployeeProfile]:
        """Extract basic employee info from varied shapes (detailed_profile or flat)."""
        try:
            profile = emp_data.get("detailed_profile") or emp_data.get("basic_info") or emp_data

            name = (
                profile.get("name")
                or profile.get("full_name")
                or f'{profile.get("first_name", "")} {profile.get("last_name", "")}'.strip()
                or (emp_data.get("basic_info", {}) or {}).get("name")
                or f"Employee_{employee_id}"
            )

            position = (
                profile.get("current_position")
                or profile.get("position")
                or profile.get("title")
                or (emp_data.get("basic_info", {}) or {}).get("title", "")
            )
            name = (
                profile.get("name")
                or profile.get("full_name")
                or f'{profile.get("first_name", "")} {profile.get("last_name", "")}'.strip()
                or f"Employee_{employee_id}"
            )
            position = profile.get("current_position") or profile.get("position") or profile.get("title") or ""
            # NOTE: preserve your original logic: company_name derived from employee_id prefix
            company_from_id = employee_id.split("_")[0]
            return EmployeeProfile(
                name=name.strip(),
                current_position=position,
                company_name=company_from_id,
                employee_id=employee_id,
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract basic profile: {e}")
            return None

    # ------------ Skill extraction ------------
    async def extract_skills_for_company(self, company_name: str, employees: List[EmployeeProfile]):
        """Extract skills (GPT → keyword fallback) and build the company profile."""
        logger.info(f"🧠 Extracting skills for {company_name} ({len(employees)} employees)")
        logger.info(f"OpenAI enabled: {self._client is not None} | Deployment: {self.deployment_id}")

        # prepare container
        company_profile = CompanySkillProfile(company_name=company_name, employees=[])

        if not self._client or not self.deployment_id:
            logger.warning("⚠️ No GPT client configured; using keyword fallback for all employees")
            self._extract_skills_keyword_fallback(employees)
            company_profile.employees.extend(employees)
            self.company_profiles[company_name] = company_profile
            return

        batch_size = 10
        total = len(employees)
        if total == 0:
            self.company_profiles[company_name] = company_profile
            return

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = employees[start:end]
            logger.info(f"🔄 Processing batch {start//batch_size + 1}/{(total + batch_size - 1)//batch_size} for {company_name}")

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(self._extract_skills_gpt, e) for e in batch]

                for i, fut in enumerate(futures):
                    try:
                        emp = fut.result()
                        if emp:
                            company_profile.employees.append(emp)
                    except Exception as e:
                        logger.warning(f"⚠️ Skill extraction error: {e}")
                        fb = self._create_fallback_skill_profile(batch[i])
                        company_profile.employees.append(fb)

            if end < total:
                await asyncio.sleep(1)  # gentle pacing

        self.company_profiles[company_name] = company_profile
        logger.info(f"✅ Completed skill extraction for {company_name}")

    def _extract_skills_gpt(self, employee: EmployeeProfile) -> Optional[EmployeeProfile]:
        """Single-employee skill extraction via Azure OpenAI (synchronous, safe-parsing)."""
        if not self._client or not self.deployment_id:
            return self._create_fallback_skill_profile(employee)

        prompt = (
            "Extract technical and professional skills from the following employee profile.\n"
            "Return ONLY a JSON array of strings (no extra text).\n\n"
            f"Name: {employee.name}\n"
            f"Position: {employee.current_position}\n"
            f"Company: {employee.company_name}\n\n"
            'Example: ["Python", "Data Analysis", "Financial Modeling", "Risk Management"]'
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": "You are a professional skills extraction assistant. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip()
            arr = _safe_json_array(text)
            if isinstance(arr, list) and arr:
                employee.skills = arr
                logger.info(f"✅ Extracted {len(arr)} skills for {employee.name}")
                return employee
            logger.warning(f"⚠️ Non-list/empty skills for {employee.name}; using fallback")
            return self._create_fallback_skill_profile(employee)
        except Exception as e:
            logger.warning(f"❌ GPT extraction failed for {employee.name}: {e}")
            return self._create_fallback_skill_profile(employee)

    # ------------ Fallbacks & Saving ------------
    def _extract_skills_keyword_fallback(self, employees: List[EmployeeProfile]):
        """Assign basic skills heuristically to a list of employees."""
        for e in employees:
            self._create_fallback_skill_profile(e)

    def _create_fallback_skill_profile(self, employee: EmployeeProfile) -> EmployeeProfile:
        """Keyword-based fallback from title text."""
        position = (employee.current_position or "").lower()
        if "data" in position:
            skills = ["SQL", "Python", "Data Analysis", "Machine Learning"]
        elif "risk" in position:
            skills = ["Risk Assessment", "Compliance", "Regulatory Frameworks"]
        elif "manager" in position:
            skills = ["Leadership", "Project Management", "Strategic Planning"]
        else:
            skills = ["Financial Analysis", "Communication", "Problem Solving"]
        employee.skills = skills
        return employee

    def save_company_profile(self, company_name: str, output_dir: str):
        """Save company skill profile to a JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        profile = self.company_profiles.get(company_name)
        if not profile:
            logger.info(f"ℹ️ No profile to save for {company_name}")
            return

        out_path = Path(output_dir) / f"{company_name.replace(' ', '_')}_skills.json"
        payload = {
            "company_name": company_name,
            "employees": [
                {
                    "name": e.name,
                    "position": e.current_position,
                    "employee_id": e.employee_id,
                    "skills": e.skills or [],
                }
                for e in profile.employees
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved {company_name} skills → {out_path}")


# ------------------------------------------------------------
# Orchestrator adapters
# ------------------------------------------------------------
async def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrator entrypoint for Agent 3 (Fractal).
    Expects:
      context = {
        "inputs": {
            "data_directory": "employee_data",        # folder of JSON files to process
            "output_dir": "output/company_skills",    # optional
            "azure_config": {                         # optional (else env vars)
                "api_key": "...",
                "api_version": "2024-06-01",
                "endpoint": "https://<your>.openai.azure.com",
                "deployment_id": "gpt-4o"
            },
            "test_connection": true                   # optional
        },
        ...
      }
    Returns context with results under context["agents"]["agent3"].
    """
    inputs = (context or {}).get("inputs", {})
    data_directory = inputs.get("data_directory") or "employee_data"
    output_dir = inputs.get("output_dir") or "output/company_skills"
    azure_config = inputs.get("azure_config")
    test_connection = bool(inputs.get("test_connection", False))

    extractor = FractalSkillExtractor(azure_config=azure_config)

    if test_connection:
        extractor.test_openai_connection()

    await extractor.process_company_files(data_directory=data_directory, output_dir=output_dir)

    # attach minimal summary to context
    context.setdefault("agents", {})
    context["agents"]["agent3"] = {
        "data_directory": data_directory,
        "output_dir": output_dir,
        "companies_processed": list(extractor.company_profiles.keys()),
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    }
    return context


def run_sync(context: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous helper for orchestrators that aren't async-aware."""
    return asyncio.run(run(context))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent 3 — Fractal Skill Extractor")
    parser.add_argument("--data_directory", default="employee_data", help="Directory of input JSON files")
    parser.add_argument("--output_dir", default="output/company_skills", help="Directory to write results")
    parser.add_argument("--test_connection", action="store_true", help="Ping Azure OpenAI before processing")
    args = parser.parse_args()

    cfg = {
        "inputs": {
            "data_directory": args.data_directory,
            "output_dir": args.output_dir,
            "test_connection": args.test_connection,
            # Optionally pass azure_config here; otherwise rely on env vars.
        }
    }
    run_sync(cfg)
