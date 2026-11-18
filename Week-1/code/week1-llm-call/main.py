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


openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
prompt = "What is Artificial Intelligence? Explain in simple words."

if gemini_key and gemini_key.strip() and gemini_key != "your_actual_gemini_api_key":
    from google import genai

    client = genai.Client(api_key=gemini_key)

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    print("\nLLM Response:")
    print(response.text)

elif openai_key and openai_key.strip() and openai_key != "your_actual_openai_api_key":
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        print("\nLLM Response:")
        print(response.output_text)
    except Exception:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        print("\nLLM Response:")
        print(response.choices[0].message.content)

else:
    answer = call_ollama(prompt)
    if answer:
        print("\nLLM Response (local Ollama):")
        print(answer)
    else:
        raise ValueError(
            "No API key was found, and the local Ollama model is not available. "
            "Start Ollama and run 'ollama pull llama3.2' or provide an API key."
        )
