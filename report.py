import json
import os
from pathlib import Path
import time
from typing import Dict, Any
from openai import AzureOpenAI
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn

class SkillBoostPlanGenerator:
    def __init__(self):
        """Initialize with Azure OpenAI configuration (hard-coded)"""
        # Hardcoded Azure OpenAI configuration
        self.api_key = "2be1544b3dc14327b60a870fe8b94f35"
        self.endpoint = "https://notedai.openai.azure.com"
        self.api_version = "2024-06-01"
        self.deployment_id = "gpt-4o"
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def load_course_data(self, file_path: str) -> list:
        """Load course data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
    
    def generate_skill_boost_plan(self, employee_data: Dict[str, Any]) -> str:
        """Generate skill boost plan using Azure OpenAI GPT for a single employee"""
        
        prompt = f"""
        Convert the following employee course data into a structured 0-3 Month Skill Boost Plan report format.

        Employee Data:
        Name: {employee_data['employeeName']}
        Role: {employee_data['role']}
        Company: {employee_data['company']}
        Existing Skills: {', '.join(employee_data['existingSkills'])}
        Priority Skills: {', '.join(employee_data['prioritySkillsSelected'])}
        
        Course Information:
        Course Name: {employee_data['course']['courseName']}
        Description: {employee_data['course']['description']}
        Skills Covered: {employee_data['course']['skillsCovered']}
        Total Topics: {employee_data['course']['totalTopics']}

        Create a report following this EXACT format:

        # {employee_data['employeeName']} -- 0-3 Month Skill Boost Plan

        This document outlines the 0-3 month skill upgrade plan for {employee_data['employeeName']}, focusing on closing immediate skill gaps, recommended courses, proof-of-concept (POC) deliverables, and the direct business impact for {employee_data['company']}.

        | Timeline | Skill Gap | Recommended Skills/Tools | Course Recommendation | Proposed POC | Business Impact for {employee_data['company']} |
        |----------|-----------|-------------------------|----------------------|--------------|-------------------------------------------|
        | 0-1 Month | [First critical skill gap] | [Specific tools/technologies] | [Course name and key modules] | [Practical deliverable] | [Direct business value] |
        | 0-1 Month | [Second critical skill gap] | [Specific tools/technologies] | [Course name and key modules] | [Practical deliverable] | [Direct business value] |
        | 1-3 Months | [Important skill gap] | [Advanced tools/technologies] | [Advanced course modules] | [Complex deliverable] | [Strategic business value] |
        | 1-3 Months | [Another important skill] | [Specialized tools] | [Specialized modules] | [Industry-specific POC] | [Long-term business impact] |

        ## Execution Priority

        1. Weeks 1-4 → Focus on [priority skills] → Deliver [key deliverables].
        2. Weeks 5-12 → Advanced skills in [advanced areas] → Complete [strategic projects].

        Instructions:
        1. Analyze the employee's missing skills from prioritySkillsSelected
        2. Map course chapters to specific skill gaps
        3. Create realistic POCs based on their role and existing skills
        4. Focus on business impact relevant to their company/industry
        5. Ensure timeline is practical and progressive
        6. Use technical tools and frameworks appropriate for their field
        7. Make POCs specific and measurable
        8. Connect learning to direct business value
        """

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": "You are an expert learning and development specialist who creates detailed skill development plans. Always follow the exact format requested and provide practical, actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating plan for {employee_data['employeeName']}: {e}")
            return None
    
    def markdown_to_word(self, markdown_content: str, employee_name: str) -> Document:
        """Convert markdown content to Word document with proper formatting"""
        doc = Document()
        
        lines = markdown_content.split('\n')
        table_rows = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Handle headers
            if line.startswith('# '):
                heading = doc.add_heading(line[2:], 1)
                heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif line.startswith('## '):
                doc.add_heading(line[3:], 2)
            
            # Handle table
            elif line.startswith('|') and 'Timeline' in line:
                # Table header found
                in_table = True
                headers = [cell.strip() for cell in line.split('|')[1:-1]]
                continue
            elif line.startswith('|') and '---' in line:
                # Table separator, skip
                continue
            elif line.startswith('|') and in_table:
                # Table row
                row_data = [cell.strip() for cell in line.split('|')[1:-1]]
                table_rows.append(row_data)
            elif in_table and not line.startswith('|'):
                # End of table, create it
                if table_rows:
                    table = doc.add_table(rows=1, cols=len(headers))
                    table.style = 'Table Grid'
                    
                    # Add headers
                    header_cells = table.rows[0].cells
                    for i, header in enumerate(headers):
                        header_cells[i].text = header
                        # Make header bold
                        for paragraph in header_cells[i].paragraphs:
                            for run in paragraph.runs:
                                run.bold = True
                    
                    # Add data rows
                    for row_data in table_rows:
                        row_cells = table.add_row().cells
                        for i, cell_data in enumerate(row_data):
                            if i < len(row_cells):
                                row_cells[i].text = cell_data
                
                table_rows = []
                in_table = False
                
                # Continue with current line (probably paragraph text)
                if line and not line.startswith('#'):
                    doc.add_paragraph(line)
            
            # Handle regular paragraphs
            elif not line.startswith('#') and not line.startswith('|') and not in_table:
                if line.startswith('1.') or line.startswith('2.'):
                    # Numbered list item
                    doc.add_paragraph(line, style='List Number')
                else:
                    doc.add_paragraph(line)
        
        # Handle any remaining table data
        if table_rows and in_table:
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = 'Table Grid'
            
            # Add headers
            header_cells = table.rows[0].cells
            for i, header in enumerate(headers):
                header_cells[i].text = header
                for paragraph in header_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
            
            # Add data rows
            for row_data in table_rows:
                row_cells = table.add_row().cells
                for i, cell_data in enumerate(row_data):
                    if i < len(row_cells):
                        row_cells[i].text = cell_data
        
        return doc
    
    def save_word_report(self, employee_name: str, report_content: str, output_dir: str = "reports"):
        """Save the report as a Word document"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Clean employee name for filename
        clean_name = "".join(c for c in employee_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_')
        
        filename = f"{clean_name}_0-3_Month_Skill_Boost_Plan.docx"
        filepath = Path(output_dir) / filename
        
        try:
            # Convert markdown to Word document
            doc = self.markdown_to_word(report_content, employee_name)
            
            # Save the document
            doc.save(filepath)
            print(f"✓ Word report saved: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"Error saving Word report for {employee_name}: {e}")
            return None
    
    def process_all_employees(self, json_file_path: str, output_dir: str = "reports"):
        """Process all employees and generate individual Word reports"""
        
        # Load the course data
        course_data = self.load_course_data(json_file_path)
        
        if not course_data:
            print("No data loaded. Please check your file path and format.")
            return
        
        total_employees = len(course_data)
        print(f"Found {total_employees} employees to process")
        
        successful_reports = []
        failed_reports = []
        
        for i, employee in enumerate(course_data, 1):
            employee_name = employee.get('employeeName', f'Employee_{i}')
            print(f"\n[{i}/{total_employees}] Processing: {employee_name}")
            
            # Generate the report
            report_content = self.generate_skill_boost_plan(employee)
            
            if report_content:
                # Save the report as Word document
                saved_path = self.save_word_report(employee_name, report_content, output_dir)
                if saved_path:
                    successful_reports.append((employee_name, saved_path))
                else:
                    failed_reports.append(employee_name)
            else:
                failed_reports.append(employee_name)
            
            # Add delay to avoid rate limiting
            if i < total_employees:
                time.sleep(2)
        
        # Summary
        print(f"\n" + "="*60)
        print(f"PROCESSING COMPLETE")
        print(f"="*60)
        print(f"Total Employees: {total_employees}")
        print(f"Successful Reports: {len(successful_reports)}")
        print(f"Failed Reports: {len(failed_reports)}")
        
        if successful_reports:
            print(f"\nSuccessfully generated Word reports:")
            for name, path in successful_reports:
                print(f"  • {name}: {path}")
        
        if failed_reports:
            print(f"\nFailed to generate reports for:")
            for name in failed_reports:
                print(f"  • {name}")

def main():
    """Main function to run the script"""
    
    print("Course to Skill Boost Plan Generator (Azure OpenAI + Word Export)")
    print("="*65)
   
    
    # Get input file path
    input_file = input("Enter path to JSON file (or press Enter for 'spectre_courses_cleaned.json'): ").strip()
    if not input_file:
        input_file = "spectre_courses.json"
    
    # Get output directory
    output_dir = input("Enter output directory (or press Enter for 'reports'): ").strip()
    if not output_dir:
        output_dir = "reports"
    
    try:
        # Initialize generator
        generator = SkillBoostPlanGenerator()
        
        # Process all employees
        generator.process_all_employees(input_file, output_dir)
        
    except Exception as e:
        print(f"Error initializing generator: {e}")

if __name__ == "__main__":
    main()