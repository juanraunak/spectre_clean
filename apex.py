#!/usr/bin/env python3
"""
unified_course_workflow.py

Unified course processing workflow that combines all three steps:
1. AI-first course transformation (from spectre_courses_pruned.json)
2. Text to JSON conversion (matching exact schema requirements)
3. Sending to API endpoint

Features:
- Process all employees or select specific ones
- Exclude specific employees
- Full workflow automation
- Individual step control
- Progress tracking and error handling
- Exact schema matching for API compatibility

Usage:
  # Process all employees
  python unified_course_workflow.py spectre_courses_pruned.json

  # Process specific employees only
  python unified_course_workflow.py spectre_courses_pruned.json --include "John Doe" "Jane Smith"

  # Process all except specific employees
  python unified_course_workflow.py spectre_courses_pruned.json --exclude "John Doe"

  # Run only specific steps
  python unified_course_workflow.py spectre_courses_pruned.json --steps transform convert --no-send

  # Custom output directory and API settings
  python unified_course_workflow.py spectre_courses_pruned.json --output-dir ./courses --base-url https://custom-api.com
"""

import json
import os
import sys
import time
import argparse
import requests
import urllib.parse
import re
from openai import AzureOpenAI
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path


class UnifiedCourseProcessor:
    def __init__(self, output_dir: str = "./output", base_url: str = None, route: str = None):
        """Initialize the unified processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # AI Configuration
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", "2be1544b3dc14327b60a870fe8b94f35"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://notedai.openai.azure.com")
        )
        self.deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")
        
        # API Configuration
        self.base_url = base_url or "https://mynotedbe-guh7ekdxajcddvd2.southindia-01.azurewebsites.net"
        self.route = route or "/api/trpc/addSyncFlowJob"
        self.api_url, self.batch_url = self._make_urls()
        
        # AI System Prompt
        self.system_prompt = """
Role
You are a senior curriculum architect specializing in AI-first course design. You design executive-ready courses that turn topic lists into complete AI-foundational courses.

Task
From the JSON I provide, create a separate course for every topic. Keep the original intent and wording, but AI is the foundation - build the course around AI and include any AI skills around this topic. AI is not an addon but the foundation of how the course is designed.

Context
• The JSON may include fields like courseName, course[], chapterName, topicName, details, and optional skills (existingSkills, missingSkillsCategories).
• Center all content on the JSON provided but make sure to change those topics to be AI foundational (e.g., Topic: Risk Management → New AI foundational topic: AI-First Risk Management).
• Maintain a unified tone and structure.

Reasoning (do this internally before writing the answer)
1. For each AI-topic, design a course with:
   - Objective (1–2 sentences, plain, business-aligned)
   - 3–4 Modules (progressive, practical, with AI embedded)
   - Capstone Project (realistic, applies the topic to a transformation scenario)
2. If skills categories are provided in the JSON, append a consolidated Skills Roll-Up (critical, important, nice_to_have) without altering wording.
3. Internally check: every AI-topic must have an Objective, 3–4 Modules, Capstone. Courses must feel unified with ~90% overlap in structure and AI foundation.

Output format
Return plain text only, structured exactly like this per topic:

Course: <topicName>
Objective: <1–2 sentences>

Modules:
1. — <module 1>
2. — <module 2>
3. — <module 3>
4. — <module 4 (if needed)>

Capstone Project: <project>



After all topics:

Skills Roll-Up (from input, unchanged)
• Critical: …
• Important: …
• Nice to have: …

Stop conditions
• Do not output JSON.
• Do not rename or invent titles/names.
• Do not add fluff; every line must be practical and have the data from the JSON.

Completion criteria
• Every topic = full course (Objective, Modules, Capstone).
• All courses align to a unified learning path
• AI integration is the core and foundational.
"""
        
    def _make_urls(self) -> Tuple[str, str]:
        """Create API URLs."""
        base = self.base_url.rstrip("/")
        route = self.route if self.route.startswith("/") else "/" + self.route
        url = f"{base}{route}"
        batch_url = f"{url}?batch=1"
        return url, batch_url

    # ==================== STEP 1: AI TRANSFORMATION ====================
    
    def load_course_data(self, file_path: str) -> Dict[str, Any]:
        """Load course data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"✅ Successfully loaded course data from {file_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading file '{file_path}': {e}")
            raise

    # --- inside UnifiedCourseProcessor.filter_employees ---

    def filter_employees(self, course_data: Any, include_employees: List[str] = None, 
                        exclude_employees: List[str] = None) -> List[Dict[str, Any]]:
        """Filter employees based on include/exclude lists."""
        # NEW: unwrap top-level {"employees": [...]} shape
        if isinstance(course_data, dict):
            if isinstance(course_data.get("employees"), list):
                course_data = course_data["employees"]
            else:
                course_data = [course_data]

        if not isinstance(course_data, list):
            raise ValueError("Course data must be a list or single object")

        # Normalize helper
        def norm(s: str) -> str:
            return (s or "").strip().lower()

        include_norm = {norm(n) for n in (include_employees or [])}
        exclude_norm = {norm(n) for n in (exclude_employees or [])}

        employees = []
        for item in course_data:
            if not isinstance(item, dict):
                continue
            emp_name = item.get("employeeName", "Unknown")
            emp_norm = norm(emp_name)

            if include_norm and emp_norm not in include_norm:
                continue
            if exclude_norm and emp_norm in exclude_norm:
                continue

            employees.append(item)

        if not employees:
            # Helpful debug: show available names to spot typos/whitespace
            available = ", ".join([str(e.get("employeeName", "Unknown")) for e in course_data])
            raise ValueError(f"No employees match the filtering criteria. Available: {available}")

        return employees


    def transform_course_with_ai(self, course_data: Dict[str, Any]) -> str:
        """Send JSON to Azure OpenAI and return plain text content."""
        emp_name = course_data.get('employeeName', 'Unknown')
        print(f"🤖 Transforming courses for {emp_name} with Azure OpenAI...")
        
        input_json = json.dumps(course_data, ensure_ascii=False, indent=2)

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Transform this course data:\n\n{input_json}"}
                ],
                temperature=0.3,
                max_tokens=4000
            )
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI request failed: {e}")

        if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            raise RuntimeError("Azure OpenAI returned an empty response")

        content = response.choices[0].message.content.strip()
        if not content:
            raise RuntimeError("Azure OpenAI returned empty content")

        return content

    # ==================== STEP 2: TEXT TO JSON CONVERSION ====================
    
    def text_to_json(self, text_content: str, employee_name: str) -> Dict[str, Any]:
        """Convert AI-generated text to structured JSON matching the required schema."""
        print(f"📝 Converting text to JSON for {employee_name}...")
        
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        courses = []
        current_course = None
        current_modules = []
        in_skills_rollup = False
        skills_rollup = {}
        
        # Parse the AI text content
        for line in lines:
            if line.startswith('Course:'):
                # Save previous course if exists
                if current_course:
                    current_course['modules'] = current_modules
                    courses.append(current_course)
                
                # Start new course
                course_name = line.replace('Course:', '').strip()
                current_course = {
                    'courseName': course_name,
                    'objective': '',
                    'modules': [],
                    'capstoneProject': ''
                }
                current_modules = []
                
            elif line.startswith('Objective:'):
                if current_course:
                    current_course['objective'] = line.replace('Objective:', '').strip()
                    
            elif line.startswith('Modules:'):
                continue  # Header line, skip
                
            elif re.match(r'^\d+\.\s*—\s*', line):
                # Module line
                module = re.sub(r'^\d+\.\s*—\s*', '', line).strip()
                current_modules.append(module)
                
            elif line.startswith('Capstone Project:'):
                if current_course:
                    current_course['capstoneProject'] = line.replace('Capstone Project:', '').strip()
                    
            elif line.startswith('Skills Roll-Up'):
                in_skills_rollup = True
                # Save last course
                if current_course:
                    current_course['modules'] = current_modules
                    courses.append(current_course)
                    current_course = None
                    
            elif in_skills_rollup and line.startswith('•'):
                # Parse skills
                if ':' in line:
                    skill_type, skills = line.split(':', 1)
                    skill_type = skill_type.replace('•', '').strip().lower().replace(' ', '_')
                    skills_list = [s.strip() for s in skills.split(',') if s.strip()]
                    skills_rollup[skill_type] = skills_list

        # Don't forget the last course
        if current_course:
            current_course['modules'] = current_modules
            courses.append(current_course)

        # Transform to required schema format
        course_chapters = []
        total_topics = 0
        
        for course in courses:
            # Create chapter structure for each course
            chapter_topics = []
            for i, module in enumerate(course['modules'], 1):
                chapter_topics.append({
                    "topicName": module,
                    "details": course['objective']  # Use objective as details
                })
                total_topics += 1
            
            course_chapters.append({
                "chapterName": course['courseName'],
                "chapter": chapter_topics,
                "capstoneProject": course['capstoneProject']
            })

        # Build final structure matching the required schema
        result = {
            "courseName": f"Learning AI Path for {employee_name}",
            "description": f"This learning path equips {employee_name} with AI-powered skills to drive informed decision-making and business growth.",
            "skillsCovered": len(skills_rollup.get('critical', [])) + len(skills_rollup.get('important', [])),
            "totalTopics": total_topics,
            "course": course_chapters
        }
        
        # Add skills rollup if available
        if skills_rollup:
            result['skillsRollUp'] = skills_rollup
        else:
            # Default skills structure
            result['skillsRollUp'] = {
                "critical": [],
                "important": [],
                "nice_to_have": []
            }
            
        return result

    # ==================== STEP 3: API SENDING ====================
    
    def send_course_to_api(self, course_json: Dict[str, Any], employee_name: str) -> bool:
        """Send only the SyncFlowJobInput-shaped JSON to the API endpoint."""
        print(f"🚀 Sending course data for {employee_name} to API...")

        # --- Build strict SyncFlow payload ---
        def _to_syncflow_payload(data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "courseName": data.get("courseName", "Untitled Course"),
                "course": [
                    {
                        "chapterName": ch.get("chapterName", "Untitled Chapter"),
                        "chapter": [
                            {"topicName": t.get("topicName", "Untitled Topic")}
                            for t in (ch.get("chapter") or [])
                            if isinstance(t, dict) and t.get("topicName")
                        ],
                    }
                    for ch in (data.get("course") or [])
                    if isinstance(ch, dict) and ch.get("chapterName")
                ],
            }

        trimmed = _to_syncflow_payload(course_json)

        # Optional: quick sanity checks
        if not trimmed.get("courseName") or not isinstance(trimmed.get("course"), list):
            print("  ❌ Invalid payload: missing courseName or course list")
            return False

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "unified-course-processor/1.0",
        }

        # Try a few reasonable shapes with the TRPC endpoint
        shapes = [
            ("Direct JSON", {"json": trimmed}),
            ("Input wrapper", {"input": trimmed}),
            ("Raw body", trimmed),
            ("Batch format", [{"id": 0, "json": trimmed}]),
        ]

        for shape_name, payload in shapes:
            try:
                print(f"  Trying {shape_name}...")
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=180,
                    allow_redirects=False,
                )

                if 200 <= resp.status_code < 300:
                    print(f"  ✅ Success with {shape_name}")
                    try:
                        data = resp.json()
                        job_id = None
                        if isinstance(data, dict):
                            result = data.get("result", {}).get("data", data)
                            if isinstance(result, dict):
                                job_id = result.get("JobId") or result.get("jobId") or result.get("id")
                        print(f"  📋 Job ID: {job_id}" if job_id else "  ✅ Request successful (no Job ID found)")
                    except Exception:
                        print("  ✅ Request successful (non-JSON response)")
                    return True

                print(f"  ❌ Failed with status {resp.status_code}")
                if resp.status_code == 403 and "web app is stopped" in resp.text.lower():
                    print("  🛑 Azure Web App appears to be stopped")
                    return False

            except Exception as e:
                print(f"  ❌ Error with {shape_name}: {e}")

        print(f"  ❌ All request shapes failed for {employee_name}")
        return False

    # ==================== MAIN WORKFLOW ====================
    
    def process_employee(self, employee_data: Dict[str, Any], steps: List[str]) -> Dict[str, Any]:
        """Process a single employee through the workflow."""
        employee_name = employee_data.get('employeeName', 'Unknown')
        results = {
            'employee': employee_name,
            'steps_completed': [],
            'outputs': {},
            'success': False
        }
        
        try:
            # Step 1: AI Transformation
            if 'transform' in steps:
                print(f"\n📚 Step 1: AI Transformation for {employee_name}")
                ai_text = self.transform_course_with_ai(employee_data)
                
                # Save text output
                text_file = self.output_dir / f"{employee_name.replace(' ', '_').lower()}_transformed.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(ai_text)
                
                results['outputs']['ai_text'] = str(text_file)
                results['steps_completed'].append('transform')
                print(f"  ✅ Saved AI output: {text_file}")
            else:
                # Load existing text file if not transforming
                text_file = self.output_dir / f"{employee_name.replace(' ', '_').lower()}_transformed.txt"
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        ai_text = f.read()
                    print(f"  📖 Loaded existing text: {text_file}")
                else:
                    raise FileNotFoundError(f"Text file not found: {text_file}")

            # Step 2: Convert to JSON
            if 'convert' in steps:
                print(f"\n🔄 Step 2: Text to JSON conversion for {employee_name}")
                course_json = self.text_to_json(ai_text, employee_name)
                
                # Save JSON output
                json_file = self.output_dir / f"{employee_name.replace(' ', '_').lower()}_courses.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(course_json, f, ensure_ascii=False, indent=2)
                
                results['outputs']['course_json'] = str(json_file)
                results['course_json_data'] = course_json  # Store the JSON data for unified output
                results['steps_completed'].append('convert')
                print(f"  ✅ Saved JSON: {json_file}")
                print(f"  📊 Generated {course_json['totalTopics']} topics across {len(course_json['course'])} chapters")
            else:
                # Load existing JSON file if not converting
                json_file = self.output_dir / f"{employee_name.replace(' ', '_').lower()}_courses.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        course_json = json.load(f)
                    results['course_json_data'] = course_json  # Store the JSON data
                    print(f"  📖 Loaded existing JSON: {json_file}")
                else:
                    raise FileNotFoundError(f"JSON file not found: {json_file}")

            # Step 3: Send to API
            if 'send' in steps:
                print(f"\n🌐 Step 3: Send to API for {employee_name}")
                api_success = self.send_course_to_api(course_json, employee_name)
                
                if api_success:
                    results['steps_completed'].append('send')
                    print(f"  ✅ Successfully sent to API")
                else:
                    print(f"  ❌ Failed to send to API")

            results['success'] = True
            print(f"\n🎉 Completed processing for {employee_name}")
            
        except Exception as e:
            print(f"\n❌ Error processing {employee_name}: {e}")
            results['error'] = str(e)
            
        return results

    def create_unified_json(self, employee_results: List[Dict[str, Any]]) -> str:
        """Create a unified JSON file containing all processed employees' course data."""
        print("\n📦 Creating unified JSON file...")
        
        # Create unified directory
        unified_dir = self.output_dir / 'unified'
        unified_dir.mkdir(parents=True, exist_ok=True)
        
        unified_data = {
            'metadata': {
                'generatedAt': datetime.now().isoformat(),
                'totalEmployees': len(employee_results),
                'successfulEmployees': sum(1 for r in employee_results if r['success']),
                'totalTopicsGenerated': sum(r.get('course_json_data', {}).get('totalTopics', 0) 
                                          for r in employee_results if r['success']),
                'generator': 'unified_course_workflow.py'
            },
            'employeeCourses': []
        }
        
        for result in employee_results:
            if result['success'] and 'course_json_data' in result:
                unified_data['employeeCourses'].append(result['course_json_data'])
        
        # Save unified JSON
        unified_file = unified_dir / 'all_employees_courses.json'
        with open(unified_file, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ Unified JSON saved: {unified_file}")
        print(f"  📊 Contains {len(unified_data['employeeCourses'])} employee course sets")
        print(f"  🎯 Total topics generated: {unified_data['metadata']['totalTopicsGenerated']}")
        
        return str(unified_file)

    def run_workflow(self, input_file: str, include_employees: List[str] = None, 
                    exclude_employees: List[str] = None, steps: List[str] = None, 
                    delay: float = 2.0) -> Dict[str, Any]:
        """Run the complete workflow."""
        if steps is None:
            steps = ['transform', 'convert', 'send']
            
        print("🚀 Starting Unified Course Processing Workflow")
        print("=" * 60)
        print(f"📁 Input file: {input_file}")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"🔧 Steps to run: {', '.join(steps)}")
        if include_employees:
            print(f"👥 Include employees: {', '.join(include_employees)}")
        if exclude_employees:
            print(f"🚫 Exclude employees: {', '.join(exclude_employees)}")
        print("=" * 60)
        
        # Load and filter data
        course_data = self.load_course_data(input_file)
        employees = self.filter_employees(course_data, include_employees, exclude_employees)
        
        print(f"👥 Processing {len(employees)} employee(s)")
        
        # Process each employee
        results = {
            'total_employees': len(employees),
            'successful': 0,
            'failed': 0,
            'employee_results': [],
            'unified_json': None
        }
        
        for i, employee_data in enumerate(employees):
            employee_name = employee_data.get('employeeName', f'Employee_{i}')
            print(f"\n{'='*20} [{i+1}/{len(employees)}] Processing {employee_name} {'='*20}")
            
            employee_result = self.process_employee(employee_data, steps)
            results['employee_results'].append(employee_result)
            
            if employee_result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
            
            # Delay between employees (except for the last one)
            if i < len(employees) - 1 and delay > 0:
                print(f"⏱️  Waiting {delay} seconds before next employee...")
                time.sleep(delay)
        
        # Create unified JSON if any conversions were successful
        if 'convert' in steps and results['successful'] > 0:
            unified_json_path = self.create_unified_json(results['employee_results'])
            results['unified_json'] = unified_json_path
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"✅ Successful: {results['successful']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📁 Output directory structure:")
        print(f"   📂 {self.output_dir}/")
        print(f"   ├── 📝 *_transformed.txt    # AI-generated course text")
        print(f"   ├── 📋 *_courses.json       # Schema-formatted JSON")  
        print(f"   └── 📦 unified/             # Combined JSON for all employees")
        
        # Show individual employee files
        for emp_result in results['employee_results']:
            status = "✅" if emp_result['success'] else "❌"
            steps_done = ", ".join(emp_result['steps_completed']) or "none"
            print(f"{status} {emp_result['employee']}: {steps_done}")
        
        # Show unified JSON
        if results['unified_json']:
            print(f"\n📦 Unified JSON: {results['unified_json']}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified course processing workflow: AI transformation → JSON conversion → API sending",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all employees through complete workflow
  python unified_course_workflow.py spectre_courses_pruned.json
  
  # Process only specific employees
  python unified_course_workflow.py spectre_courses_pruned.json --include "John Doe" "Jane Smith"
  
  # Exclude specific employees
  python unified_course_workflow.py spectre_courses_pruned.json --exclude "John Doe"
  
  # Run only transformation and conversion (skip API sending)
  python unified_course_workflow.py spectre_courses_pruned.json --steps transform convert
  
  # Custom settings
  python unified_course_workflow.py spectre_courses_pruned.json \\
    --output-dir ./my_courses \\
    --base-url https://my-api.com \\
    --delay 5 \\
    --include "Employee A" "Employee B"
        """
    )
    
    # Positional arguments
    parser.add_argument("input_file", help="Path to spectre_courses_pruned.json or similar input file")
    
    # Employee filtering
    parser.add_argument("--include", "-i", nargs="+", metavar="EMPLOYEE",
                       help="Process only these employees (by name)")
    parser.add_argument("--exclude", "-e", nargs="+", metavar="EMPLOYEE", 
                       help="Exclude these employees (by name)")
    
    # Step control
    parser.add_argument("--steps", nargs="+", choices=["transform", "convert", "send"],
                       default=["transform", "convert", "send"],
                       help="Steps to run (default: all steps)")
    
    # Output and API settings
    parser.add_argument("--output-dir", "-o", default="./output",
                       help="Output directory for generated files (default: ./output)")
    parser.add_argument("--base-url", "-b", 
                       default="https://mynotedbe-guh7ekdxajcddvd2.southindia-01.azurewebsites.net",
                       help="API base URL")
    parser.add_argument("--route", "-r", default="/api/trpc/addSyncFlowJob",
                       help="API route path")
    
    # Processing options
    parser.add_argument("--delay", "-d", type=float, default=2.0,
                       help="Delay between employee processing (seconds, default: 2.0)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually doing it")
    
    args = parser.parse_args()
    
    # Dry run mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - showing what would be processed")
        try:
            processor = UnifiedCourseProcessor(args.output_dir, args.base_url, args.route)
            course_data = processor.load_course_data(args.input_file)
            employees = processor.filter_employees(course_data, args.include, args.exclude)
            
            print(f"📁 Input: {args.input_file}")
            print(f"📂 Output: {args.output_dir}")
            print(f"🔧 Steps: {', '.join(args.steps)}")
            print(f"👥 Would process {len(employees)} employee(s):")
            for emp in employees:
                print(f"  - {emp.get('employeeName', 'Unknown')}")
        except Exception as e:
            print(f"❌ Error in dry run: {e}")
        return
    
    # Run the actual workflow
    try:
        processor = UnifiedCourseProcessor(args.output_dir, args.base_url, args.route)
        results = processor.run_workflow(
            args.input_file,
            include_employees=args.include,
            exclude_employees=args.exclude,
            steps=args.steps,
            delay=args.delay
        )
        
        # Exit with error code if any processing failed
        if results['failed'] > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Workflow error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()