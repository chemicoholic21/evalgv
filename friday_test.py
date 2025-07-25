import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

print("Starting the process...")
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
    """
    Generates content using the specified Gemini model (for single prompts).
    This function is primarily kept for compatibility or testing single calls,
    but batching is preferred for speed.
    """
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
            
            if debug and attempt == 0:
                print(f"--- Single Prompt Sample ({len(prompt)} chars) ---")
                print(f"{prompt[:500]}...")
                print(f"------------------------------------")
            
            response = model.generate_content(
                prompt, 
                safety_settings=safety_settings
            )
            
            try:
                if response.text:
                    return response.text.strip()
            except ValueError as ve:
                if "finish_reason" in str(ve) and "2" in str(ve):
                    if debug:
                        print(f"Response blocked: {str(ve)}")
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                if hasattr(candidate, 'safety_ratings'):
                                    print(f"Safety ratings detail: {candidate.safety_ratings}")
                    return "Error: Content was blocked by safety filters (finish_reason 2)."
                else:
                    if debug:
                        print(f"ValueError accessing response text: {str(ve)}")
                    if attempt < max_retries:
                        continue 
                    return f"Error: ValueError accessing response text: {str(ve)}"
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if debug:
                        print(f"Finish reason from candidate: {finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            print(f"Safety ratings detail: {candidate.safety_ratings}")
                    
                    if finish_reason == 2:
                        return "Error: Content was blocked due to safety filters."
                    elif finish_reason == 3:
                        return "Error: Content was blocked due to recitation concerns."
                    elif finish_reason == 4:
                        return "Error: Content was blocked for other reasons."
                    else:
                        return f"Error: Generation finished with reason {finish_reason}."
            
            return "Error: No valid response generated."
            
        except Exception as e:
            if debug:
                print(f"Exception during single generation: {type(e).__name__}: {str(e)}")
            
            if attempt < max_retries:
                continue
            else:
                return f"Error: {str(e)}"
    
    return "Error: Max retries exceeded."


def gemini_batch_generate(model_name, prompts_list, debug=False, max_retries=3, batch_delay=1):
    """
    Generates content using the specified Gemini model for a list of prompts (batch processing).

    Args:
        model_name (str): The name of the Gemini model to use.
        prompts_list (list): A list of input prompts.
        debug (bool): Whether to print debug information.
        max_retries (int): Maximum number of retries on failure.
        batch_delay (int): Delay in seconds between batch retries.

    Returns:
        list: A list of generated texts, or error messages for each.
    """
    # Initialize results with errors to ensure all positions are filled
    results = ["Error: Initial batch processing failure"] * len(prompts_list) 
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(batch_delay * random.uniform(0.8, 1.2))
                if debug:
                    print(f"Retry attempt {attempt} for batch after {batch_delay:.1f}s delay.")
            
            model = genai.GenerativeModel(model_name)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            if debug and attempt == 0:
                print(f"--- Batch Processing: Sending {len(prompts_list)} prompts to {model_name} ---")
                for i, p in enumerate(prompts_list[:min(3, len(prompts_list))]):
                    print(f"  Prompt {i+1} ({len(p)} chars): {p[:200]}...")
                print(f"--------------------------------------------------")
            
            responses = model.generate_content(
                prompts_list,
                safety_settings=safety_settings
            )
            
            processed_responses = []
            for i, response in enumerate(responses):
                try:
                    if response.text:
                        processed_responses.append(response.text.strip())
                    else:
                        processed_responses.append(f"Error: No text in response for prompt {i}.")
                except ValueError as ve:
                    if "finish_reason" in str(ve) and "2" in str(ve):
                        processed_responses.append("Error: Content was blocked by safety filters (finish_reason 2).")
                    else:
                        processed_responses.append(f"Error: ValueError accessing text for prompt {i}: {str(ve)}")
                except Exception as e:
                    processed_responses.append(f"Error processing response {i}: {str(e)}")
            
            # Check if any errors occurred in the processed responses
            if any("Error:" in r for r in processed_responses):
                results = processed_responses # Update results with current (potentially error-filled) batch
                if debug:
                    print(f"Batch completed with some errors. Retrying whole batch if attempts remain.")
                continue # Go to next attempt
            else:
                return processed_responses # All responses successful

        except Exception as e:
            if debug:
                print(f"Exception during batch generation: {type(e).__name__}: {str(e)}")
            if attempt < max_retries:
                continue
            else:
                # If max retries exhausted for a general exception, fill all with this error
                return ["Error: Batch API call failed: " + str(e)] * len(prompts_list)
    
    return results # Return the last set of results (might contain errors) after max retries


def anonymize_text(text: str) -> str:
    """
    Anonymizes sensitive information in the input text (resume or transcript).
    This function is designed to be robust but may need further tuning
    based on the specific patterns in your data.
    """
    if not isinstance(text, str):
        return "[INVALID_INPUT_TYPE]"

    # 1. Anonymize Names: More comprehensive patterns for full names and common first/last names
    text = re.sub(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', '[PERSON_NAME]', text)
    text = re.sub(r'\bDr\.\s+[A-Z][a-z]+\b', '[PERSON_NAME]', text)

    # 2. Anonymize Contact Information
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL_ADDRESS]', text)
    text = re.sub(r'(\+?\d{1,3}[-.●\s]?)?(\(?\d{2,4}\)?[-.\s]?){2}\d{4,6}\b', '[PHONE_NUMBER]', text)
    text = re.sub(r'https?://(?:www\.)?\S+\.\S+(?:/\S*)?', '[URL]', text)
    text = re.sub(r'\bwww\.\S+\.\S+(?:/\S*)?', '[URL]', text)

    text = re.sub(r'\d+\s+(?:[A-Za-z]+\s?){1,5}(?:Street|Road|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Terrace|St|Rd|Ave|Blvd|Ln|Dr|Ct|Pl|Sq|Ter)\b', '[STREET_ADDRESS]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Apt|Apartment|Unit|Suite|Ste)\s+\w+\b', '[UNIT_NUMBER]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', '[STATE_ZIP]', text)
    text = re.sub(r'\b[A-Z][a-z]+(?:,\s*[A-Z]{2})?\s+\d{5}\b', '[CITY_STATE_ZIP]', text)
    text = re.sub(r'\b(?:City|Town|Village) of [A-Z][a-z]+\b', '[CITY_NAME]', text)

    # 3. Anonymize Sensitive Identifiers
    text = re.sub(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[SSN]', text)
    text = re.sub(r'\b(?:[A-Z]{1,3}\d{6,9}|[A-Z]{2}\s?\d{7})\b', '[ID_NUMBER]', text)

    # 4. Anonymize Dates related to Age
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b', '[SPECIFIC_DATE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[SPECIFIC_DATE]', text)

    # 5. Anonymize Sensitive Categories (if explicitly mentioned)
    text = re.sub(r'\b(?:male|female|non-binary|gender-fluid)\b', '[GENDER]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:black|white|asian|hispanic|caucasian|african american)\b', '[ETHNICITY]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:christian|muslim|jewish|hindu|buddhist|atheist)\b', '[RELIGION]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:disabled|handicapped|medical condition|therapy|diagnosis|illness)\b', '[HEALTH_INFO]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpregnancy|maternity leave\b', '[FAMILY_INFO]', text, flags=re.IGNORECASE)

    # If the text becomes too short or entirely placeholders, replace with a general flag
    anonymized_length = len(re.sub(r'\[.*?\]', '', text).strip())
    if anonymized_length < 20 and len(text) > 0:
        return "[HIGHLY_ANONYMIZED_CONTENT_REDUCED]"
    elif not text.strip():
        return "[EMPTY_ORIGINAL_TEXT]"

    return text



def interview_prompt(job_desc, transcript, job_criteria):
    """
    Constructs the prompt for interview evaluation, focusing on technical/role knowledge and skills.

    Args:
        job_desc (str): Description of the job role.
        transcript (str): Interview transcript.
        job_criteria (str): Job-specific criteria.

    Returns:
        str: The formatted prompt for interview evaluation.
    """
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
“””
"""

def resume_prompt(job_desc, resume, job_criteria):
    """
    Constructs the prompt for resume evaluation.

    Args:
        job_desc (str): Description of the job role.
        resume (str): Candidate's resume text.
        job_criteria (str): Job-specific criteria.

    Returns:
        str: The formatted prompt for resume evaluation.
    """
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
    """
    Constructs the prompt for summarizing candidate evaluations.

    Args:
        job_desc (str): Description of the job role.
        resume_eval (str): Result of resume evaluation.
        interview_eval (str): Result of interview evaluation.
        job_criteria (str): Job-specific criteria.

    Returns:
        str: The formatted prompt for summarization.
    """
    return f"""You are an AI hiring coordinator. Given the following evaluation results for a candidate (resume and interview), and the job role, write a concise summary of the candidate’s strengths and weaknesses, and recommend whether to advance them to the next stage.
Job Role: {job_desc}
Resume Evaluation: {resume_eval}
Interview Evaluation: {interview_eval}
Job-Specific Criteria: {job_criteria}
Priortise the above Job Specific Criteria in the Evaluation
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
    """
    Constructs the prompt for extracting the final verdict.

    Args:
        text (str): The summary text from which to extract the verdict.

    Returns:
        str: The formatted prompt for verdict extraction.
    """
    return f"""Extract the decision from the following text and output only one of these exact words: "Advanced", "Reject", or "Manual Intervention".
Text: {text}
"""

# Test the API connection with a simple prompt
print("Testing API connection...")
test_result = gemini_generate("gemini-2.5-flash", "Say 'API test successful'", debug=True)
print(f"API test result: {test_result}")
if test_result.startswith("Error:"):
    print("WARNING: API test failed. There may be issues with API calls.")
else:
    print("API test passed. Proceeding with data processing...")
print()

def convert_excel_to_csv(excel_file, csv_file):
    """
    Convert Excel file to CSV format
    """
    try:
        df = pd.read_excel(excel_file)
        df.to_csv(csv_file, index=False)
        print(f"✓ Converted {excel_file} to {csv_file}")
        return True
    except Exception as e:
        print(f"✗ Error converting {excel_file} to CSV: {e}")
        return False

# Load the CSV file
try:
    df = pd.read_csv("friday_test.csv")
    print(f"Successfully loaded CSV file with {len(df)} rows.")
    print("Column names in the CSV file:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    print()
except FileNotFoundError:
    print("Error: 'friday_test.csv' file not found in the current directory.")
    print("Attempting to find and convert Excel files...")
    
    # Try to find Excel files and convert them
    import glob
    excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
    
    if excel_files:
        print("Excel files found:")
        for i, file in enumerate(excel_files):
            print(f"  {i+1}. {file}")
        
        print("\nAttempting to convert the first Excel file to CSV...")
        excel_file = excel_files[0]
        csv_file = excel_file.replace('.xlsx', '.csv').replace('.xls', '.csv')
        
        if convert_excel_to_csv(excel_file, csv_file):
            print(f"Now trying to load {csv_file}...")
            try:
                df = pd.read_csv(csv_file)
                print(f"Successfully loaded converted CSV file with {len(df)} rows.")
                print("Column names in the CSV file:")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i}. {col}")
                print()
            except Exception as e:
                print(f"Error loading converted CSV: {e}")
                exit(1)
        else:
            exit(1)
    else:
        print("No Excel or CSV files found in current directory.")
        print("Please ensure you have a data file (CSV or Excel) to process.")
        exit(1)
        
except pd.errors.EmptyDataError:
    print("Error: The CSV file appears to be empty or has no columns to parse.")
    print("Please check that friday_test.csv contains valid data with headers.")
    exit(1)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    print("This could be due to:")
    print("  - File is corrupted or empty")
    print("  - File encoding issues")
    print("  - Invalid CSV format")
    print("Please check the file content and format.")
    exit(1)

# Define column names for input and output
job_col = "Grapevine Job - Job → Description"
interview_col = "Grapevine Aiinterviewinstance → Transcript → Conversation"
resume_col = "Grapevine Userresume - Resume → Metadata → Resume Text"
criteria_col = "Recruiter GPT Response "
# Assuming a column for candidate name exists, adjust this if your column name is different
candidate_name_col = "Candidate Name" # <--- ADDED: Column for candidate name
interview_eval_col = "Interview Evaluator Agent (RAG-LLM)"
resume_eval_col = "Resume Evaluator Agent (RAG-LLM)"
summarizer_col = "Resume + Interview Summarizer Agent"
result_col = "Result"

# --- ADDED: Deduplication Step ---
# Deduplication Logic
initial_rows = len(df)
# Create a unique identifier for each candidate-job application combination
# Assuming a combination of resume text and job description uniquely identifies an application
df['unique_id'] = df[resume_col].astype(str) + "|||" + df[job_col].astype(str)

# Drop duplicates based on this unique_id, keeping the first occurrence
df.drop_duplicates(subset=['unique_id'], keep='first', inplace=True)

# Remove the temporary unique_id column
df.drop(columns=['unique_id'], inplace=True)

deduplicated_rows = len(df)
if initial_rows > deduplicated_rows:
    print(f"Deduplication complete: Removed {initial_rows - deduplicated_rows} duplicate rows.")
else:
    print("No duplicate rows found.")
# --- END ADDED: Deduplication Step ---

def process_row(idx, row):
    """
    Processes a single row of the DataFrame, performing evaluations and summarization.

    Args:
        idx (int): The index of the current row.
        row (pd.Series): The row data from the DataFrame.

    Returns:
        tuple: A tuple containing the row index and the results of evaluations and summarization.
    """
    print(f"[Row {idx}] Processing started.")
    job_desc = str(row.get(job_col, "")).strip()
    interview = str(row.get(interview_col, "")).strip()
    resume = str(row.get(resume_col, "")).strip()
    criteria = str(row.get(criteria_col, "")).strip()

    # Interview Evaluation (using gemini-2.5-flash)
    if interview:
        print(f"[Row {idx}] Running interview evaluation...")
        interview_prompt_text = interview_prompt(job_desc, interview, criteria)
        interview_eval = gemini_generate("gemini-2.5-flash", interview_prompt_text)
        if interview_eval.startswith("Error:"):
            print(f"[Row {idx}] Interview evaluation failed: {interview_eval}")
        else:
            print(f"[Row {idx}] Interview evaluation complete.")
    else:
        interview_eval = "Interview Evaluation: User cannot be evaluated."
        print(f"[Row {idx}] No interview transcript. Skipping interview evaluation.")

    # Resume Evaluation (using gemini-2.5-flash)
    print(f"[Row {idx}] Running resume evaluation...")
    resume_prompt_text = resume_prompt(job_desc, resume, criteria)
    resume_eval = gemini_generate("gemini-2.5-flash", resume_prompt_text)
    if resume_eval.startswith("Error:"):
        print(f"[Row {idx}] Resume evaluation failed: {resume_eval}")
    else:
        print(f"[Row {idx}] Resume evaluation complete.")

    # Summarizer (using gemini-2.5-pro)
    print(f"[Row {idx}] Running summarizer...")
    summarizer_prompt_text = summarizer_prompt(job_desc, resume_eval, interview_eval, criteria)
    summarizer = gemini_generate("gemini-2.5-pro", summarizer_prompt_text)
    if summarizer.startswith("Error:"):
        print(f"[Row {idx}] Summarizer failed: {summarizer}")
    else:
        print(f"[Row {idx}] Summarizer complete.")

    # Final Verdict (using gemini-2.5-flash)
    print(f"[Row {idx}] Running verdict extraction...")
    verdict_prompt_text = verdict_prompt(summarizer)
    verdict = gemini_generate("gemini-2.5-flash", verdict_prompt_text)
    if verdict.startswith("Error:"):
        print(f"[Row {idx}] Verdict extraction failed: {verdict}")
        verdict = "Manual Intervention"  # Default fallback
    else:
        print(f"[Row {idx}] Verdict extraction complete.")

    print(f"[Row {idx}] Processing finished.\n")
    return idx, interview_eval, resume_eval, summarizer, verdict

# Set maximum workers for ThreadPoolExecutor (reduced to avoid API rate limiting)
max_workers = 8  # Sequential processing to avoid rate limits
print(f"Submitting {len(df)} rows for processing with {max_workers} worker (sequential processing)...")

# Process rows in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_row, idx, row) for idx, row in df.iterrows()]
    for future in as_completed(futures):
        idx, interview_eval, resume_eval, summarizer, verdict = future.result()
        print(f"[Row {idx}] Writing results to DataFrame.")
        df.at[idx, interview_eval_col] = interview_eval
        df.at[idx, resume_eval_col] = resume_eval
        df.at[idx, summarizer_col] = summarizer
        df.at[idx, result_col] = verdict

print("All rows processed. Saving results...")

# Save the updated DataFrame to a new CSV file
df.to_csv("friday_test_results.csv", index=False)
print("Results saved to friday_test_results.csv.")

end_time = time.time()
elapsed = end_time - start_time
print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")