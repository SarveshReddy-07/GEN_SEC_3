import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
RELEASE = "v0.1"
PASS_THRESHOLD = 60.0
REQUEST_TIMEOUT_SECONDS = 120
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "golden_set.json"
OUTPUT_FILE = BASE_DIR / "baseline_results.csv"
CSV_FIELDS = [
    "Question_ID",
    "Question",
    "Expected_Answer",
    "LLM_Answer",
    "Matched_Keywords",
    "Missing_Keywords",
    "Score",
    "Status",
    "Source",
    "Model",
    "Release",
]


class GoldenQuestion(TypedDict):
    id: str
    question: str
    expected_answer: str
    keywords: list[str]
    source: str


def load_golden_questions(file_path: Path = INPUT_FILE) -> list[GoldenQuestion]:
    with file_path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The golden question set must be a JSON array.")

    questions: list[GoldenQuestion] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")

        question_id = item.get("id")
        question = item.get("question")
        expected_answer = item.get("expected_answer")
        keywords = item.get("keywords")
        source = item.get("source")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (question_id, question, expected_answer, source)
        ):
            raise ValueError(
                f"Question {index} must have non-empty id, question, "
                "expected_answer, and source strings."
            )
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) and keyword.strip()
            for keyword in keywords
        ):
            raise ValueError(f"Question {index} must have a list of keywords.")

        questions.append(
            {
                "id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "keywords": keywords,
                "source": source,
            }
        )

    return questions


def get_llm_answer(question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()

    if not isinstance(result, dict):
        raise ValueError("Ollama returned an unexpected response format.")

    answer = result.get("response")
    if not isinstance(answer, str):
        raise ValueError("Ollama response did not contain a text answer.")
    return answer.strip()


def calculate_score(
    answer: str, expected_keywords: list[str]
) -> tuple[float, list[str], list[str]]:
    answer_lower = answer.casefold()
    matched = [
        keyword for keyword in expected_keywords
        if keyword.casefold() in answer_lower
    ]
    missing = [
        keyword for keyword in expected_keywords
        if keyword.casefold() not in answer_lower
    ]
    score = (
        len(matched) / len(expected_keywords) * 100
        if expected_keywords
        else 0.0
    )
    return round(score, 2), matched, missing


def evaluate_question(item: GoldenQuestion) -> dict[str, str | float]:
    try:
        answer = get_llm_answer(item["question"])
    except (requests.RequestException, ValueError) as error:
        print(f"Error for {item['id']}: {error}")
        return {
            "Question_ID": item["id"],
            "Question": item["question"],
            "Expected_Answer": item["expected_answer"],
            "LLM_Answer": "",
            "Matched_Keywords": "",
            "Missing_Keywords": ", ".join(item["keywords"]),
            "Score": 0.0,
            "Status": "ERROR",
            "Source": item["source"],
            "Model": MODEL_NAME,
            "Release": RELEASE,
        }

    score, matched, missing = calculate_score(answer, item["keywords"])
    status = "PASS" if score >= PASS_THRESHOLD else "REVIEW"
    print(f"\nQuestion {item['id']}: {item['question']}")
    print("LLM Answer:", answer)
    print("Expected Answer:", item["expected_answer"])
    print("Matched Keywords:", matched)
    print("Missing Keywords:", missing)
    print(f"Score: {score}%")
    print("Status:", status)

    return {
        "Question_ID": item["id"],
        "Question": item["question"],
        "Expected_Answer": item["expected_answer"],
        "LLM_Answer": answer,
        "Matched_Keywords": ", ".join(matched),
        "Missing_Keywords": ", ".join(missing),
        "Score": score,
        "Status": status,
        "Source": item["source"],
        "Model": MODEL_NAME,
        "Release": RELEASE,
    }


def run_baseline() -> None:
    questions = load_golden_questions()
    print("======================================")
    print("GOLDEN QUESTION SET")
    print("======================================")
    print("Total Questions:", len(questions))
    print("Model:", MODEL_NAME)
    print(f"Release: {RELEASE}")
    print("\n======================================")
    print("BASELINE RUN STARTED")
    print("======================================")

    results = [evaluate_question(item) for item in questions]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    successful_scores = [
        float(result["Score"])
        for result in results
        if result["Status"] != "ERROR"
    ]
    average_score = (
        sum(successful_scores) / len(successful_scores)
        if successful_scores
        else 0.0
    )
    passed = sum(result["Status"] == "PASS" for result in results)
    reviewed = sum(result["Status"] == "REVIEW" for result in results)
    errors = sum(result["Status"] == "ERROR" for result in results)

    print("\n======================================")
    print("BASELINE EVALUATION SUMMARY")
    print("======================================")
    print("Release Version:", RELEASE)
    print("Model:", MODEL_NAME)
    print("Total Questions:", len(questions))
    print("Successful Responses:", len(successful_scores))
    print("Passed Questions:", passed)
    print("Questions for Review:", reviewed)
    print("Failed Requests:", errors)
    print("Average Keyword Coverage:", f"{average_score:.2f}%")
    print("Evaluation Date:", datetime.now().astimezone().isoformat(timespec="seconds"))
    print("\nResults saved to:", OUTPUT_FILE)
    print(f"\nBASELINE RUN COMPLETED — Release {RELEASE}.")


if __name__ == "__main__":
    run_baseline()
