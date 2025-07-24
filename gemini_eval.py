import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Starting the process...")
start_time = time.time()
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

def gemini_generate(model_name, prompt, max_tokens=1024, debug=False, max_retries=2):
    """
    Generates content using the specified Gemini model.

    Args:
        model_name (str): The name of the Gemini model to use (e.g., "gemini-2.5-flash", "gemini-2.5-pro").
        prompt (str): The input prompt for the model.
        max_tokens (int): The maximum number of output tokens.
        debug (bool): Whether to print debug information.
        max_retries (int): Maximum number of retries on failure.

    Returns:
        str: The generated text, or an error message if generation fails.
    """
    import time
    import random
    
    for attempt in range(max_retries + 1):
        try:
            # Add a small random delay to avoid overwhelming the API
            if attempt > 0:
                delay = random.uniform(1, 3)  # 1-3 second delay on retry
                time.sleep(delay)
                if debug:
                    print(f"Retry attempt {attempt} after {delay:.1f}s delay")
            
            model = genai.GenerativeModel(model_name)
            
            
            # Configure safety settings to be very permissive (similar to n8n)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            if debug and attempt == 0:  # Only show prompt details on first attempt
                print(f"Prompt length: {len(prompt)} characters")
                print(f"First 200 chars of prompt: {prompt[:200]}...")
            
            response = model.generate_content(
                prompt, 
                generation_config={"max_output_tokens": max_tokens},
                safety_settings=safety_settings
            )
            
            # Handle the specific case where response.text access fails
            try:
                if hasattr(response, "parts") and response.parts:
                    # Safely access the first part's text if available
                    part_text = getattr(response.parts[0], "text", None)
                    if part_text:
                        return part_text.strip()
                elif hasattr(response, "text") and response.text:
                    return response.text.strip()
            except ValueError as ve:
                # This catches the specific "response.text quick accessor" error
                if "finish_reason" in str(ve):
                    if debug:
                        print(f"Response blocked: {str(ve)}")
                    # Don't retry on safety blocks, return immediately
                    return "Error: Content was blocked by safety filters (finish_reason 2)."
                else:
                    if attempt < max_retries:
                        continue  # Retry on other ValueError cases
                    return f"Error: ValueError accessing response text: {str(ve)}"
            # Handle cases where response is blocked or invalid
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if debug:
                        print(f"Finish reason: {finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            print(f"Safety ratings: {candidate.safety_ratings}")
                    
                    if finish_reason == 2:  # SAFETY
                        return "Error: Content was blocked due to safety filters."
                    elif finish_reason == 3:  # RECITATION
                        return "Error: Content was blocked due to recitation concerns."
                    elif finish_reason == 4:  # OTHER
                        return "Error: Content was blocked for other reasons."
                    else:
                        return f"Error: Generation finished with reason {finish_reason}."
            
            return "Error: No valid response generated."
            
        except Exception as e:
            if debug:
                print(f"Exception details: {type(e).__name__}: {str(e)}")
            
            # Retry on certain exceptions, but not on safety-related ones
            if attempt < max_retries and "safety" not in str(e).lower():
                continue
            else:
                return f"Error: {str(e)}"
    
    return "Error: Max retries exceeded."

def interview_prompt(job_desc, transcript, job_criteria):
    """
    Constructs a prompt for summarizing technical/role knowledge and skills from an interview transcript.

    Args:
        job_desc (str): Description of the job role.
        transcript (str): Interview transcript.
        job_criteria (str): Job-specific criteria.

    Returns:
        str: The formatted prompt for interview summary.
    """
    return f"""You are an AI assistant helping to organize information from an interview transcript for a specific job role.

Job Role: {job_desc}
Interview Transcript: {transcript}
Job-Specific Criteria: {job_criteria}

Instructions:
- Summarize the candidate's demonstrated technical and role-specific knowledge and skills, as shown in the transcript.
- If job-specific criteria are provided, focus on those in your summary.
- If relevant information is missing or not discussed, state: "Relevant information not discussed in transcript."
- Use only the information provided. Do not add, assume, or invent any details.
- Output your summary as 3-5 bullet points, nothing else.
"""

#def interview_prompt(job_desc, transcript, job_criteria):
#     """
#     Please analyze the provided interview transcript in the context of the given job role and any specified job-specific criteria. Focus on identifying and assessing the candidate's demonstrated **Technical/Role Knowledge/Skills**.

#     **Important Note:** This analysis is for informational purposes only, to help identify the presence and depth of specific knowledge and skills within the interview. It is not intended for making hiring decisions or for any form of discrimination.

#     **Category for Analysis:**

#     * **Demonstrated Technical/Role Knowledge/Skills (0-10):** Assess the candidate's understanding and application of technical concepts, industry knowledge, and role-specific abilities as evidenced in the transcript.

#     **Instructions:**

#     * Base your assessment **only** on the information explicitly provided in the `Interview Transcript` and `Job Role`.
#     * If `Job-Specific Criteria` are provided, prioritize identifying how the candidate's responses align with or demonstrate these specific criteria. If blank, disregard this aspect.
#     * **Do not** invent, assume, or infer any information. **Do not** create hypothetical scenarios. Evaluate the candidate solely based on the provided transcript.
#     * If information relevant to this category is missing or not discussed in the transcript, clearly state "Information not discussed in transcript" or "Limited information available" in your justification.
#     * **Do not** hallucinate or provide any information not directly supported by the transcript.

#     **Input Details:**

#     Job Role: {job_desc}
#     Interview Transcript: {transcript}
#     Job-Specific Criteria: {job_criteria}

#     **Prioritize the `Job-Specific Criteria` in your analysis.**

#     **Output your assessment as a JSON object, including a brief justification for the assigned value, in the following format:**

#     ```json
#     {
#       "Demonstrated Technical/Role Knowledge/Skills": {
#         "value": 0,
#         "justification": "Information not discussed in transcript"
#       }
#     }
#     ```
#     """
#     return f"""Please analyze the provided interview transcript in the context of the given job role and any specified job-specific criteria. Focus on identifying and assessing the candidate's demonstrated **Technical/Role Knowledge/Skills**.

# **Important Note:** This analysis is for informational purposes only, to help identify the presence and depth of specific knowledge and skills within the interview. It is not intended for making hiring decisions or for any form of discrimination.

# **Category for Analysis:**

# * **Demonstrated Technical/Role Knowledge/Skills (0-10):** Assess the candidate's understanding and application of technical concepts, industry knowledge, and role-specific abilities as evidenced in the transcript.

# **Instructions:**

# * Base your assessment **only** on the information explicitly provided in the `Interview Transcript` and `Job Role`.
# * If `Job-Specific Criteria` are provided, prioritize identifying how the candidate's responses align with or demonstrate these specific criteria. If blank, disregard this aspect.
# * **Do not** invent, assume, or infer any information. **Do not** create hypothetical scenarios. Evaluate the candidate solely based on the provided transcript.
# * If information relevant to this category is missing or not discussed in the transcript, clearly state "Information not discussed in transcript" or "Limited information available" in your justification.
# * **Do not** hallucinate or provide any information not directly supported by the transcript.

# **Input Details:**

# Job Role: {job_desc}
# Interview Transcript: {transcript}
# Job-Specific Criteria: {job_criteria}

# **Prioritize the `Job-Specific Criteria` in your analysis.**

# **Output your assessment as a JSON object, including a brief justification for the assigned value, in the following format:**
# """

# def resume_prompt(job_desc, resume, job_criteria):
#     """
#     # Constructs the prompt for resume evaluation.

   
#     Constructs the prompt for resume evaluation.

#     Args:
#         job_desc (str): Description of the job role.
#         resume (str): Candidate's resume text.
#         job_criteria (str): Job-specific criteria.

#     Returns:
#         str: The formatted prompt for resume evaluation.
#     """
#     return f"""Please analyze the provided resume text in relation to the specified job role and criteria. For each of the following categories, identify the extent to which relevant information is present and aligns with the criteria. Assign a numerical value based on the completeness and relevance of the information found, according to the scale provided.

# **Important Note:** This analysis is for informational purposes only, to help identify the presence of specific data points within the resume. It is not intended to be used for making hiring decisions or for any form of discrimination.

# **Categories for Analysis:**

# * **Education and Company Pedigree (0-1):** Assess the presence and relevance of educational institutions and the stature of past employers.
# * **Skills & Specialties (0-2):** Identify the depth and breadth of listed skills and their alignment with the job role.
# * **Work Experience (0-4):** Evaluate the completeness and relevance of past work history, including responsibilities and achievements.
# * **Basic Contact Information (0-1):** Determine if essential contact details are present.
# * **Educational Background Details (0-2):** Examine the specifics of the educational history (e.g., degrees, majors, graduation dates).

# **Instructions:**

# * Base your assessment **only** on the information explicitly provided in the `Resume` section.
# * If `Job-Specific Criteria` are provided, prioritize identifying information that directly addresses these criteria. If blank, ignore this aspect.
# * **Do not** invent or infer any information. If details are missing for a category, clearly state "Information missing" in your justification.

# **Input Details:**

# Job Role: {job_desc}
# Resume: {resume}
# Job-Specific Criteria: {job_criteria}

# **Output your assessment as a JSON object, including a brief justification for each category's value, in the following format:**

# ```json
# {
#   "Education and Company Pedigree": {
#     "value": 0,
#     "justification": "..."
#   },
#   "Skills & Specialties": {
#     "value": 0,
#     "justification": "..."
#   },
#   "Work Experience": {
#     "value": 0,
#     "justification": "..."
#   },
#   "Basic Contact Information": {
#     "value": 0,
#     "justification": "..."
#   },
#   "Educational Background Details": {
#     "value": 0,
#     "justification": "..."
#   },
#   "Input on Job Specific Criteria": "..."
# }
# ```
# def interview_prompt(job_desc, transcript, job_criteria):
    """
    Constructs a prompt for evaluating an interview transcript, focusing on technical/role knowledge and skills.

    Args:
        job_desc (str): Description of the job role.
        transcript (str): Interview transcript.
        job_criteria (str): Job-specific criteria.

    Returns:
        str: The formatted prompt for interview evaluation.
    """
    return f"""
You are an expert interview evaluator.

*Task:*  
Analyze the provided interview transcript in the context of the specified job role and any job-specific criteria. Your focus is to identify and assess the candidate's *Technical/Role Knowledge and Skills*.

*Important:*  
- This analysis is for informational purposes only, to help identify the presence and depth of specific knowledge and skills within the interview.  
- It is not intended for making hiring decisions or for any form of discrimination.

---

*Category for Analysis:*

- *Demonstrated Technical/Role Knowledge/Skills (0-10):*  
  Assess the candidate's understanding and application of technical concepts, industry knowledge, and role-specific abilities as evidenced in the transcript.

---

*Instructions:*

- Base your assessment *only* on the information explicitly provided in the Interview Transcript and Job Role.
- If Job-Specific Criteria are provided, prioritize identifying how the candidate's responses align with or demonstrate these criteria. If blank, disregard this aspect.
- *Do not* invent, assume, or infer any information. *Do not* create hypothetical scenarios. Evaluate the candidate solely based on the provided transcript.
- If information relevant to this category is missing or not discussed in the transcript, clearly state "Information not discussed in transcript" or "Limited information available" in your justification.
- *Do not* hallucinate or provide any information not directly supported by the transcript.

---

*Input Details:*

- *Job Role:* {job_desc}
- *Interview Transcript:* {transcript}
- *Job-Specific Criteria:* {job_criteria}

---

**Prioritize the Job-Specific Criteria in your analysis.**

---

*Output Format:*  
Return your assessment as a JSON object, including a brief justification for the assigned value, in the following format:

```json
{{
  "Demonstrated Technical/Role Knowledge/Skills": {{
    "value": 0,
    "justification": "Information not discussed in transcript"
  }}
}}
"""
def skills_analysis_prompt(job_desc, transcript, job_criteria):
    """
    Constructs a prompt for analyzing a conversation for technical and role-related knowledge.

    Args:
        job_desc (str): Description of the job role.
        transcript (str): Conversational transcript.
        job_criteria (str): Specific capabilities or expectations for the role.

    Returns:
        str: Reformatted prompt suitable for Gemini API safety filters.
    """
    return f"""
You are a domain expert analyzing a professional conversation for evidence of relevant technical and role-specific knowledge.

**Task:**  
Carefully review the provided conversation to identify demonstrations of understanding, skills, or knowledge related to the specified job description and criteria.

**Note:**  
- This is a neutral content analysis task, not intended for decision-making or evaluation of individuals.
- Focus only on the presence of relevant subject-matter knowledge within the discussion.

---

**Area of Interest:**

- **Technical/Role Knowledge and Skills (0-10):**  
  Based on the conversation, assign a score representing how clearly and accurately relevant concepts, skills, or domain knowledge were communicated.

---

**Instructions:**

- Use only the content found directly in the transcript and job description.
- If job-specific criteria are listed, highlight how the conversation aligns with them.
- Avoid assumptions, inferences, or imagined scenarios. Stick strictly to the data presented.
- If content related to the category is not clearly discussed, respond with "Not covered in transcript" or "Minimal relevant information".

---

**Input Details:**

- **Job Description:** {job_desc}
- **Conversation Transcript:** {transcript}
- **Focus Criteria:** {job_criteria}

---

**Output Format (JSON):**

```json
{{
  "Technical/Role Knowledge and Skills": {{
    "value": 0,
    "justification": "Not covered in transcript"
  }}
}}
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
CRITICAL: DO NOT HALLUCINATE OR MAKE UP DATA.
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
test_result = gemini_generate("gemini-2.5-flash", "Say 'API test successful'", max_tokens=50, debug=True)
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
    verdict = gemini_generate("gemini-2.5-flash", verdict_prompt_text, max_tokens=20)
    if verdict.startswith("Error:"):
        print(f"[Row {idx}] Verdict extraction failed: {verdict}")
        verdict = "Manual Intervention"  # Default fallback
    else:
        print(f"[Row {idx}] Verdict extraction complete.")

    print(f"[Row {idx}] Processing finished.\n")
    return idx, interview_eval, resume_eval, summarizer, verdict

# Set maximum workers for ThreadPoolExecutor (reduced to avoid API rate limiting)
max_workers = 1  # Sequential processing to avoid rate limits
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

# Save the updated DataFrame to a new Excel file
df.to_excel("Test_results.xlsx", index=False)
print("Results saved to Test_results.xlsx.")

end_time = time.time()
elapsed = end_time - start_time
print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
# Program to print "Hello World" and the Gemini API key from .env

# import os
# from dotenv import load_dotenv

# load_dotenv()
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# print("Hello World")
# import google.generativeai as genai

# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash")
# response = model.generate_content("Where is Antarctica?")
# print("Gemini response:", response.text.strip() if hasattr(response, "text") and response.text else "")