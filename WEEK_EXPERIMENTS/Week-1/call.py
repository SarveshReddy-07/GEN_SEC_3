import json
import os
import urllib.request
from dotenv import load_dotenv

load_dotenv()


def call_ollama(prompt, model="llama3.2"):
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")

        request = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("response", "").strip()
    except Exception:
        return None


prompt = "What is Artificial Intelligence? Explain in simple words."
api_key = os.getenv("GEMINI_API_KEY")

if api_key and api_key.strip() and api_key != "your_actual_gemini_api_key":
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    print("\nLLM Response:")
    print(response.text)
else:
    answer = call_ollama(prompt)
    if answer:
        print("\nLLM Response (local Ollama):")
        print(answer)
    else:
        raise ValueError(
            "Gemini API key not found and Ollama is not available. "
            "Start Ollama and run 'ollama pull llama3.2'."
        )
