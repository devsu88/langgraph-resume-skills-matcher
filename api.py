import os
import time
from google import genai
from langfuse.openai import OpenAI


def call_gemini_api(prompt, api_key=None, max_retries=3, delay=2):
    """
    Call Gemini API with retry logic and exception handling.

    Args:
    prompt (str): The prompt to send to the API
    api_key (str): Optional API key. If None, uses GEMINI_API_KEY env variable
    max_retries (int): Maximum number of retry attempts
    delay (int): Delay in seconds between retries

    Returns:
    str: The response text from the API, or None if all retries failed
    """
    try:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        client = genai.Client(api_key=api_key)

    except Exception as e:
        print(f"Error initializing client: {e}")
        print("Make sure GEMINI_API_KEY is set in your environment or .env file.")
        return None

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            print("Success!")
            return response.text

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {type(e).__name__} - {e}")
            if attempt == max_retries - 1:
                print("All retry attempts failed.")
                return None

            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff


def call_openai_api(prompt, api_key=None, model="gpt-4o-mini", max_retries=3, delay=2):
    """
    Call OpenAI API with retry logic and exception handling.

    Args:
    prompt (str): The prompt to send to the API
    api_key (str): Optional API key. If None, uses OPENAI_API_KEY env variable
    model (str): Model name (default: gpt-4o)
    max_retries (int): Maximum number of retry attempts
    delay (int): Delay in seconds between retries

    Returns:
    str: The response text from the API, or None if all retries failed
    """
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        client = OpenAI(api_key=api_key)

    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY is set in your environment or .env file.")
        return None

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} (OpenAI)...")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            print("Success!")
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {type(e).__name__} - {e}")
            if attempt == max_retries - 1:
                print("All retry attempts failed.")
                return None

            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff


def call_llm(prompt, provider="gemini", **kwargs):
    """
    Chiama l'API del provider indicato (gemini o openai).

    Args:
    prompt (str): Il prompt da inviare
    provider (str): "gemini" o "openai"
    **kwargs: argomenti aggiuntivi (api_key, model per openai, max_retries, delay)

    Returns:
    str: Testo della risposta, o None in caso di errore
    """
    if provider.lower() == "gemini":
        return call_gemini_api(prompt, **kwargs)
    return call_openai_api(prompt, **kwargs)
