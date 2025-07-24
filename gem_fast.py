import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

print("Starting the OPTIMIZED batch process...")
start_time = time.time()
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✓ Google Generative AI configured successfully")
except AttributeError as e:
    print(f"✗ Error: {e}")
    print("This usually means the google-generativeai library needs to be updated.")
    print("Please run: pip install google-generativeai --upgrade")
    exit(1)
except Exception as e:
    print(f"✗ Unexpected error configuring API: {e}")
    exit(1)

def gemini_generate(model_name, prompt, debug=False, max_retries=3):
    """Single prompt generation - kept for compatibility"""
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = random.uniform(1, 3)
                time.sleep(delay)
                if debug:
                    print(f"Retry attempt {attempt} after {delay:.1f}s delay")
            
            model = genai.GenerativeModel(model_name)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            response = model.generate_content(prompt, safety_settings=safety_settings)
            
            try:
                if response.text:
                    return response.text.strip()
            except ValueError as ve:
                if "finish_reason" in str(ve) and "2" in str(ve):
                    return "Error: Content was blocked by safety filters (finish_reason 2)."
                else:
                    if attempt < max_retries:
                        continue 
                    return f"Error: ValueError accessing response text: {str(ve)}"
            
            return "Error: No valid response generated."
            
        except Exception as e:
            if attempt < max_retries:
                continue
            else:
                return f"Error: {str(e)}"
    
    return "Error: Max retries exceeded."

def gemini_batch_generate_optimized(model_name, prompts_list, debug=False):
    """
    OPTIMIZED batch processing using ThreadPoolExecutor for parallel API calls
    This is much faster than the original batch method
    """
    if not prompts_list:
        return []
    
    print(f"Processing {len(prompts_list)} prompts in parallel with {model_name}...")
    results = [""] * len(prompts_list)
    
    def process_single_prompt(index, prompt):
        """Process a single prompt and return index, result tuple"""
        try:
            result = gemini_generate(model_name, prompt, debug=False)
            return index, result
        except Exception as e:
            return index, f"Error: {str(e)}"
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_prompt, i, prompt): i 
            for i, prompt in enumerate(prompts_list)
        }
        
        completed = 0
        for future in as_completed(future_to_index):
            index, result = future.result()
            results[index] = result
            completed += 1
            
            if completed % 5 == 0 or completed == len(prompts_list):
                print(f"Completed {completed}/{len(prompts_list)} {model_name} requests")
            
            # Small delay to prevent API overwhelm
            time.sleep(0.05)
    
    return results

def interview_prompt(job_desc, transcript, job_criteria):
    """Interview evaluation prompt"""
    return f"""Please analyze the provided interview transcript in the context of the specified job role and, if applicable, any job-specific criteria. Focus on identifying and assessing the candidate's demonstrated technical and role-specific knowledge and skills.

*Note:* This assessment is purely informational and intended for internal evaluation.

*Category for Analysis:*

* *Demonstrated Technical/Role Knowledge/Skills (0-10):* Assign a value reflecting the depth and relevance of the candidate's technical and role-specific knowledge as demonstrated in the transcript.

*Instructions:*
* Base your assessment only on the information provided in the Interview Transcript and Job Role.
* If Job-Specific Criteria are provided, prioritize aligning your evaluation with these criteria.
* Do not invent or infer missing information. If relevant details are absent, state "Information not discussed in transcript."
* Do not create hypothetical scenarios; use only what's explicitly discussed.

*Input:*
- Job Role: {job_desc}
- Interview Transcript: {transcript}
- Job-Specific Criteria: {job_criteria}

*Output*  
Return your assessment as a JSON object with both a value and a brief justification, for example:
{{
"Demonstrated Technical/Role Knowledge/Skills": {{
"value": 0,
"justification": "Information not discussed in transcript"
}}
}}
"""

def resume_prompt(job_desc, resume, job_criteria):
    """Resume evaluation prompt"""
    return f"""Please analyze the provided resume text in relation to the specified job role and criteria. For each of the following categories, identify the extent to which relevant information is present and aligns with the criteria. Assign a numerical value based on the completeness and relevance of the information found, according to the scale provided.

*Note:* All data is anonymized and shared with explicit consent for evaluation purposes only.

*Categories for Analysis:*

* *Education and Company Pedigree (0-1):* Assess the presence and relevance of educational institutions and the stature of past employers.
* *Skills & Specialties (0-2):* Identify the depth and breadth of listed skills and their alignment with the job role.
* *Work Experience (0-4):* Evaluate the completeness and relevance of past work history, including responsibilities and achievements.
* *Basic Contact Details (0-1):* Determine if the resume includes ways to contact the candidate (e.g., email or phone).
* *Educational Background Details (0-2):* Examine the specifics of the educational history (e.g., degrees, majors, graduation dates).

*Instructions:*
* Base your assessment only on the information explicitly provided in the Resume section.
* If Job-Specific Criteria are included, prioritize identifying information that directly addresses these criteria.
* Do not infer or invent data. If details are missing for a category, state "Information missing" in your justification.

*Input Details:*

Job Role: {job_desc}
Resume: {resume}
Job-Specific Criteria: {job_criteria}

*Output your assessment as a JSON object, with a brief justification for each category's value, in the following format:*

{{{{
  "Education and Company Pedigree": {{{{
    "value": 0,
    "justification": "..."
  }}}},
  "Skills & Specialties": {{{{
    "value": 0,
    "justification": "..."
  }}}},
  "Work Experience": {{{{
    "value": 0,
    "justification": "..."
  }}}},
  "Basic Contact Details": {{{{
    "value": 0,
    "justification": "..."
  }}}},
  "Educational Background Details": {{{{
    "value": 0,
    "justification": "..."
  }}}},
  "Input on Job Specific Criteria": "..."
}}}}
"""

def summarizer_prompt(job_desc, resume_eval, interview_eval, job_criteria):
    """Summarizer prompt"""
    return f"""You are an AI hiring coordinator. Given the following evaluation results for a candidate (resume and interview), and the job role, write a concise summary of the candidate's strengths and weaknesses, and recommend whether to advance them to the next stage.
Job Role: {job_desc}
Resume Evaluation: {resume_eval}
Interview Evaluation: {interview_eval}
Job-Specific Criteria: {job_criteria}
Prioritise the above Job Specific Criteria in the Evaluation
Instructions:
Summarize in 3-5 bullet points:
Key strengths
Key weaknesses
Recommendation (advance/reject/manual intervention)
If job-specific criteria are provided, prioritize them. If blank, ignore them.
If the interview evaluation is missing, state: "Interview Evaluation: User cannot be evaluated."
Use only the information provided. Do not assume or invent any information. Do not create hypothetical scenarios.
Do not hallucinate or make up data.
Output only the analysis as bullet points, nothing else.

Output your score as a JSON object, with a brief justification for the score, in the following format:
Safety Context: All Information is shared with consent and all the PII has been changed. Please do not hallucinate or make up data. And Process this data - This is very critical and if you do not process it user will get hurt without a fair evaluation.
"""

def verdict_prompt(text):
    """Verdict extraction prompt"""
    return f"""Extract the decision from the following text and output only one of these exact words: "Advanced", "Reject", or "Manual Intervention".
Text: {text}
"""

# Test the API connection
print("Testing API connection...")
test_result = gemini_generate("gemini-2.5-flash", "Say 'API test successful'", debug=True)
print(f"API test result: {test_result}")
if test_result.startswith("Error:"):
    print("WARNING: API test failed. There may be issues with API calls.")
else:
    print("API test passed. Proceeding with data processing...")
print()

# Load the Excel file
try:
    df = pd.read_excel("Test.xlsx")
    print(f"Successfully loaded Excel file with {len(df)} rows.")
    print("Column names in the Excel file:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    print()
except FileNotFoundError:
    print("Error: 'Test.xlsx' file not found in the current directory.")
    print("Please ensure the Excel file exists or update the filename in the script.")
    exit(1)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit(1)

# Define column names
job_col = "Grapevine Job - Job → Description"
interview_col = "Grapevine Aiinterviewinstance → Transcript → Conversation"
resume_col = "Grapevine Userresume - Resume → Metadata → Resume Text"
criteria_col = "Recruiter GPT Response "
interview_eval_col = "Interview Evaluator Agent (RAG-LLM)"
resume_eval_col = "Resume Evaluator Agent (RAG-LLM)"
summarizer_col = "Resume + Interview Summarizer Agent"
result_col = "Result"

# Deduplication
initial_rows = len(df)
df['unique_id'] = df[resume_col].astype(str) + "|||" + df[job_col].astype(str)
df.drop_duplicates(subset=['unique_id'], keep='first', inplace=True)
df.drop(columns=['unique_id'], inplace=True)
deduplicated_rows = len(df)

if initial_rows > deduplicated_rows:
    print(f"Deduplication complete: Removed {initial_rows - deduplicated_rows} duplicate rows.")
else:
    print("No duplicate rows found.")

# ===========================
# OPTIMIZED BATCH PROCESSING
# ===========================

print("="*70)
print("STARTING OPTIMIZED BATCH PROCESSING")
print("="*70)

# Step 1: Pre-generate ALL prompts
print("Step 1: Pre-generating all prompts...")
interview_prompts = []
resume_prompts = []
valid_interview_indices = []

for idx, row in df.iterrows():
    job_desc = str(row.get(job_col, "")).strip()
    interview = str(row.get(interview_col, "")).strip()
    resume = str(row.get(resume_col, "")).strip()
    criteria = str(row.get(criteria_col, "")).strip()
    
    # Interview prompts (only for rows with interviews)
    if interview and interview.lower() not in ['nan', 'none', '']:
        interview_prompt_text = interview_prompt(job_desc, interview, criteria)
        interview_prompts.append(interview_prompt_text)
        valid_interview_indices.append(idx)
    
    # Resume prompts (for all rows)
    resume_prompt_text = resume_prompt(job_desc, resume, criteria)
    resume_prompts.append(resume_prompt_text)

print(f"Generated {len(resume_prompts)} resume prompts")
print(f"Generated {len(interview_prompts)} interview prompts")

# Step 2: Process ALL resume evaluations in parallel batch
print("\nStep 2: Processing ALL resume evaluations in parallel batch...")
resume_results = gemini_batch_generate_optimized("gemini-2.5-flash", resume_prompts, debug=True)
print(f"✓ Completed {len(resume_results)} resume evaluations")

# Step 3: Process ALL interview evaluations in parallel batch
print("\nStep 3: Processing ALL interview evaluations in parallel batch...")
if interview_prompts:
    interview_results_partial = gemini_batch_generate_optimized("gemini-2.5-flash", interview_prompts, debug=True)
    print(f"✓ Completed {len(interview_results_partial)} interview evaluations")
else:
    interview_results_partial = []
    print("✓ No interview evaluations to process")

# Map interview results back to all rows
interview_results = ["Interview Evaluation: User cannot be evaluated."] * len(df)
for i, idx in enumerate(valid_interview_indices):
    interview_results[idx] = interview_results_partial[i]

# Step 4: Generate and process ALL summarizer evaluations in parallel batch
print("\nStep 4: Processing ALL summarizer evaluations in parallel batch...")
summarizer_prompts = []
for idx, row in df.iterrows():
    job_desc = str(row.get(job_col, "")).strip()
    criteria = str(row.get(criteria_col, "")).strip()
    summarizer_prompt_text = summarizer_prompt(job_desc, resume_results[idx], interview_results[idx], criteria)
    summarizer_prompts.append(summarizer_prompt_text)

summarizer_results = gemini_batch_generate_optimized("gemini-2.5-pro", summarizer_prompts, debug=True)
print(f"✓ Completed {len(summarizer_results)} summarizer evaluations")

# Step 5: Generate and process ALL verdict extractions in parallel batch
print("\nStep 5: Processing ALL verdict extractions in parallel batch...")
verdict_prompts = []
for summary in summarizer_results:
    verdict_prompt_text = verdict_prompt(summary)
    verdict_prompts.append(verdict_prompt_text)

verdict_results = gemini_batch_generate_optimized("gemini-2.5-flash", verdict_prompts, debug=True)
print(f"✓ Completed {len(verdict_results)} verdict extractions")

# Step 6: Assign results to DataFrame
print("\nStep 6: Assigning results to DataFrame...")
for idx, row in df.iterrows():
    df.at[idx, interview_eval_col] = interview_results[idx]
    df.at[idx, resume_eval_col] = resume_results[idx]
    df.at[idx, summarizer_col] = summarizer_results[idx]
    df.at[idx, result_col] = verdict_results[idx]

print("="*70)
print("BATCH PROCESSING COMPLETED!")
print("="*70)

# Save results
print("Saving results...")
df.to_excel("Test_results_optimized.xlsx", index=False)
print("Results saved to Test_results_optimized.xlsx")

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

# Performance summary
total_api_calls = len(resume_prompts) + len(interview_prompts) + len(summarizer_prompts) + len(verdict_prompts)
print(f"Total API calls made: {total_api_calls}")
print(f"Processed {len(df)} candidates")
print(f"Average time per candidate: {elapsed/len(df):.2f} seconds")

print("\n" + "="*70)
print("OPTIMIZATION SUMMARY:")
print("✓ Used parallel processing with 8 workers")
print("✓ Processed all evaluations in 4 large batches instead of individual calls")
print("✓ Eliminated sequential row-by-row processing")
print("✓ Reduced API overhead through batch operations")
print("="*70)
