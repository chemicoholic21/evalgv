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
            
            # print(f"Response received: {response.parts}")
            
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
            
            # # Retry on certain exceptions, but not on safety-related ones
            if attempt < max_retries and "safety" not in str(e).lower():
                continue
            else:
                return f"Error: {str(e)}"
    
    return "Error: Max retries exceeded."


print(gemini_generate("gemini-2.5-flash", "Hello, world!", max_tokens=50, debug=True))