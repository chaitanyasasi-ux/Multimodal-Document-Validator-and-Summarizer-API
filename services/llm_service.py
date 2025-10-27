import os
import google.genai as genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors # Import the errors module
from dotenv import load_dotenv
import time # For implementing simple backoff

# Load environment variables
load_dotenv()

try:
    client = genai.Client()
except Exception as e:
    print(f"WARNING: Gemini Client failed to initialize. Check GEMINI_API_KEY in .env. Error: {e}")
    client = None


class LLMService:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        """
        Helper for simple, synchronous LLM call using the Gemini SDK with retry logic for 503 UNAVAILABLE.
        """
        if client is None:
            raise Exception("LLM Client is not initialized. Cannot make API call.")
        
        # Combine system instruction and user content into one prompt string
        combined_prompt = f"{system_prompt}\n\n[DOCUMENT CONTENT START]\n{user_content}\n[DOCUMENT CONTENT END]"
        
        contents = [
            genai_types.Content(
                role="user", 
                parts=[genai_types.Part(text=combined_prompt)]
            )
        ]

        # --- RETRY LOGIC IMPLEMENTATION ---
        MAX_RETRIES = 3
        RETRY_DELAY = 2  # Start with 2 seconds

        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.0
                    )
                )
                # If successful, return the result immediately
                return response.text.strip()

            except genai_errors.ClientError as e:
                # Check for 503 UNAVAILABLE specifically
                if e.status_code == 503 and attempt < MAX_RETRIES - 1:
                    print(f"INFO: Attempt {attempt + 1} failed with 503 UNAVAILABLE. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    RETRY_DELAY *= 2  # Exponential backoff
                    continue
                
                # Re-raise any other critical error (400, 403, 429) or final 503 error
                raise e

        # Should be unreachable due to 'raise e' in the loop, but kept for safety.
        raise Exception("LLM call failed after all retries.")

    def validate_content(self, text: str) -> tuple[bool, str]:
        """Chain 1: Guardrail to check for professional relevance and safety."""
        validation_prompt = (
            "You are a document content classifier. Your task is to check if the following text "
            "is a professional, academic, or work-related document (e.g., contract, report, academic paper). "
            "You must also check if the content contains any explicit, harmful, or inappropriate language. "
            "Respond ONLY with 'PASS' if the content is professional AND safe. "
            "Otherwise, respond ONLY with a single word reason like 'HARMFUL', 'IRRELEVANT', or 'EMPTY'."
        )
        
        validation_input = text[:4000] 
        
        # Handle the LLM error in the outer layer to keep core logic clean
        try:
            result = self._call_llm(validation_prompt, validation_input).upper()
            return result == "PASS", result
        except genai_errors.ClientError as e:
             # If validation fails due to API error (e.g., 503, 429), block the content but report the technical error.
             error_message = f"API_ERROR: {e.status_code} - {e.error_message}"
             print(f"FATAL VALIDATION ERROR: {error_message}")
             return False, error_message # Block the summary if validation itself fails due to service outage

    def summarize_content(self, text: str) -> str:
        """Chain 2: Summarizes the text if the guardrail passes."""
        summary_prompt = (
            "You are an expert summarization bot. Summarize the following document content "
            "in 3 concise, professional bullet points. The content has already been vetted for safety."
        )
        
        # This function relies on the automatic retry in _call_llm
        return self._call_llm(summary_prompt, text)
