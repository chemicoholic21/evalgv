import os
import pandas as pd
import openai
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Starting the process...")
start_time = time.time()
# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Prompts

def interview_prompt(job_desc, transcript, job_criteria):
    return f"""You are an AI interviewer. Given the following interview transcript and job role, score the candidate on the following aspect:
Technical/Role Knowledge/Skills (0-10)
Instructions:
Use only the information provided in the interview transcript and job role.
If job-specific criteria are provided, prioritize them. If they are blank, ignore them.
Do not assume or invent any information. Do not create hypothetical scenarios. Evaluate the candidate only based on the provided transcript.
If information is missing, state this in your justification.
Do not hallucinate or make up data.
Job Role: {job_desc}
Interview Transcript: {transcript}
Job-Specific Criteria: {job_criteria}
Priortise the above Job Specific Criteria in the Evaluation
Output your score as a JSON object, with a brief justification for the score, in the following format:
"""


def resume_prompt(job_desc, resume, job_criteria):
    return f"""Given the following candidate resume and job role, score the candidate on these five aspects:
Education and Company Pedigree (0-1)
Skills & Specialties (0-2)
Work Experience (0-4)
Basic Information (0-1)
Education Background (0-2)
Use only the information provided. If job-specific criteria are provided, prioritize them; if blank, ignore. Do not hallucinate or make up data. If information is missing, state so in your justification.
Job Role: {job_desc}
Resume: {resume}
Job-Specific Criteria:{job_criteria}
Output your scores as a JSON object, with a brief justification for each score, in this format in addition to your Input on Job Specific Criteria:
"""


def summarizer_prompt(job_desc, resume_eval, interview_eval, job_criteria):
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
    return f"""Extract the decision from the following text and output only one of these exact words: "Advanced", "Reject", or "Manual Intervention".
Text: {text}
"""


def call_openai(prompt, max_tokens=512, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2
    )
    content = response.choices[0].message.content
    if content is not None:
        return content.strip()
    else:
        return ""

# Load Excel
df = pd.read_excel("Toast.xlsx")

# Column names (adjust if needed)
job_col = "Grapevine Job - Job → Description"
interview_col = "Grapevine Aiinterviewinstance → Transcript → Conversation"
resume_col = "Grapevine Userresume - Resume → Metadata → Resume Text"
criteria_col = "Recruiter GPT Response "
interview_eval_col = "Interview Evaluator Agent (RAG-LLM)"
resume_eval_col = "Resume Evaluator Agent (RAG-LLM)"
summarizer_col = "Resume + Interview Summarizer Agent"
result_col = "Result"

# Function to process a single row
def process_row(idx, row):
    print(f"[Row {idx}] Processing started.")
    job_desc = str(row.get(job_col, "")).strip()
    interview = str(row.get(interview_col, "")).strip()
    resume = str(row.get(resume_col, "")).strip()
    criteria = str(row.get(criteria_col, "")).strip()

    # Interview Evaluation
    if interview:
        print(f"[Row {idx}] Running interview evaluation...")
        interview_prompt_text = interview_prompt(job_desc, interview, criteria)
        interview_eval = call_openai(interview_prompt_text)
        print(f"[Row {idx}] Interview evaluation complete.")
    else:
        interview_eval = "Interview Evaluation: User cannot be evaluated."
        print(f"[Row {idx}] No interview transcript. Skipping interview evaluation.")

    # Resume Evaluation
    print(f"[Row {idx}] Running resume evaluation...")
    resume_prompt_text = resume_prompt(job_desc, resume, criteria)
    resume_eval = call_openai(resume_prompt_text)
    print(f"[Row {idx}] Resume evaluation complete.")

    # Summarizer
    print(f"[Row {idx}] Running summarizer...")
    summarizer_prompt_text = summarizer_prompt(job_desc, resume_eval, interview_eval, criteria)
    summarizer = call_openai(summarizer_prompt_text)
    print(f"[Row {idx}] Summarizer complete.")

    # Final Verdict
    print(f"[Row {idx}] Running verdict extraction...")
    verdict_prompt_text = verdict_prompt(summarizer)
    verdict = call_openai(verdict_prompt_text, max_tokens=10)
    print(f"[Row {idx}] Verdict extraction complete.")

    print(f"[Row {idx}] Processing finished.\n")
    return idx, interview_eval, resume_eval, summarizer, verdict

# Set the number of workers (adjust based on your API quota and system)
max_workers = 8
print(f"Submitting {len(df)} rows for processing with {max_workers} workers...")
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
df.to_excel("Toast_results.xlsx", index=False)
print("Results saved to Toast_results.xlsx.")
end_time = time.time()
elapsed = end_time - start_time
print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")