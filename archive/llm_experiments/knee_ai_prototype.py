#!/usr/bin/env python3

import argparse
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "Qwen3-4B-Q4_K_M.gguf"

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

SYSTEM_PROMPT = r"""
You are a medical radiology report extraction system for the RSNA Knee Abnormality Detection task.

Your ONLY task is to determine the presence or absence of exactly these 12 abnormalities: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

MULTILINGUAL PROCESSING: Reports may be written in English, Spanish, Turkish, Greek, Croatian, Dutch, German, Bulgarian, or other languages. Before making any label decision, internally normalize the report into clear English medical terminology. Do NOT output the translation. Preserve the original meaning exactly. Do not add findings that are not present. Carefully interpret medical terminology, anatomy, negation, and grammatical context in the original language.

LABELS: 0 = absent, 1 = present. Use ONLY information contained in the report.

EXPLICIT POSITIVE: Explicitly described abnormality = 1.
EXPLICIT NEGATIVE: Intact, normal, preserved, absent, no evidence, no signs, or equivalent = 0. Negation always has priority.
NOT MENTIONED: If a target is not mentioned and there is no supporting evidence, output 0. Never invent findings.

ANATOMICAL SPECIFICITY: ACL = anterior cruciate ligament. MCL = medial collateral ligament. Medial/Lateral Meniscus = corresponding meniscus. Medial OA = medial femorotibial compartment. Lateral OA = lateral femorotibial compartment. PF OA = patellofemoral compartment including patella/trochlea when the described abnormality represents PF degeneration.

CRITICAL DISTINCTIONS: Effusion is NOT Synovitis. Meniscal degeneration is NOT automatically a tear. Bone marrow edema is NOT automatically Contusion. Tendinosis is NOT ligament tear. Bone abnormality is NOT automatically fracture.

OA COMPARTMENTS: Do NOT treat the word medial or lateral alone as proof of Medial/Lateral OA. For example, medial femoral trochlea is patellofemoral anatomy, NOT the medial femorotibial compartment. A trochlear cartilage defect may support PF OA but does not automatically support Medial OA.

MENISCUS: Label 1 only when tear, rupture, or equivalent meniscal injury is actually described.
SYNOVITIS: Label 1 only with synovial inflammation, thickening/hypertrophy, or explicit synovitis. Effusion alone = Synovitis 0.
CONTUSION: Label 1 only for explicit bone contusion/contusional injury or when the report clearly identifies marrow edema as contusion.
FRACTURE: Label 1 only when fracture is actually described. Trauma history alone does not establish fracture.

CLINICAL HISTORY: Do not label an abnormality merely because it appears in the clinical history. MRI Findings/Impression must support it.

Use the entire report, including Findings and Impression. Reports may be multilingual. Interpret clearly equivalent medical terminology in other languages. Do not guess when meaning is uncertain.

For every target provide a binary label, confidence from 0.0 to 1.0, and short evidence grounded in the report. If not mentioned, evidence must be "Not mentioned in the report." Do not fabricate evidence.

Return ONLY valid JSON with exactly these top-level keys: labels, confidence, evidence. Every label must be exactly 0 or 1. Every confidence must be between 0.0 and 1.0. Every evidence value must be a string.

Required JSON structure:
{"labels":{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":0,"Baker's":0,"Contusion":0,"Fracture":0},"confidence":{"ACL":0.0,"MCL":0.0,"Medial Meniscus":0.0,"Lateral Meniscus":0.0,"Medial OA":0.0,"Lateral OA":0.0,"PF OA":0.0,"Effusion":0.0,"Synovitis":0.0,"Baker's":0.0,"Contusion":0.0,"Fracture":0.0},"evidence":{"ACL":"","MCL":"","Medial Meniscus":"","Lateral Meniscus":"","Medial OA":"","Lateral OA":"","PF OA":"","Effusion":"","Synovitis":"","Baker's":"","Contusion":"","Fracture":""}}
"""


def extract_json(text):
    """Extract a JSON object even if the model accidentally surrounds it with text."""
    text = text.strip()

    # First try the complete response directly.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Otherwise locate the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}") from e


def validate(result):
    if not isinstance(result, dict):
        raise ValueError("Top-level response is not a JSON object")

    required_top = {"labels", "confidence", "evidence"}

    if set(result.keys()) != required_top:
        raise ValueError(
            "Top-level JSON keys are incorrect. "
            f"Expected {sorted(required_top)}, got {sorted(result.keys())}"
        )

    expected = set(LABELS)

    for section in ("labels", "confidence", "evidence"):
        if not isinstance(result[section], dict):
            raise ValueError(f"{section} must be an object")

        actual = set(result[section].keys())

        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)

            msg = f"{section} keys are incorrect."
            if missing:
                msg += f" Missing: {missing}."
            if extra:
                msg += f" Extra: {extra}."

            raise ValueError(msg)

    # Validate labels.
    for label in LABELS:
        value = result["labels"][label]

        # bool is technically an int subclass in Python, so explicitly reject it.
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ValueError(
                f"Invalid label for {label}: {value!r}. "
                "Must be exactly integer 0 or 1."
            )

    # Validate confidence.
    for label in LABELS:
        value = result["confidence"][label]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid confidence for {label}: {value!r}. "
                "Must be a number from 0.0 to 1.0."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"Invalid confidence for {label}: {value!r}. "
                "Must be between 0.0 and 1.0."
            )

    # Validate evidence.
    for label in LABELS:
        value = result["evidence"][label]

        if not isinstance(value, str):
            raise ValueError(
                f"Invalid evidence for {label}: must be a string."
            )

    return result


def call_model(report):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "Knee MRI report:\n\n" + report,
            },
        ],
        "temperature": 0,
        "max_tokens": 1800,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }

    data = json.dumps(payload).encode("utf-8")

    request = Request(
        SERVER,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"llama.cpp HTTP {e.code}: {body}"
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"Cannot connect to llama.cpp at {SERVER}: {e}"
        ) from e

    try:
        api_response = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid JSON returned by llama.cpp API: {e}"
        ) from e

    try:
        content = api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            "Unexpected llama.cpp response structure:\n" + raw
        ) from e

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "Model returned empty content."
        )

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Extract 12 knee MRI findings using local Qwen3."
    )

    parser.add_argument(
        "report",
        help="Path to a text file containing the radiology report.",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw model content before validation.",
    )

    args = parser.parse_args()

    report_path = args.report

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read().strip()
    except OSError as e:
        print(f"ERROR: cannot read report: {e}", file=sys.stderr)
        return 1

    if not report:
        print("ERROR: report is empty", file=sys.stderr)
        return 1

    try:
        raw_content = call_model(report)

        if args.raw:
            print("=== RAW MODEL RESPONSE ===", file=sys.stderr)
            print(raw_content, file=sys.stderr)
            print("=== END RAW RESPONSE ===", file=sys.stderr)

        result = extract_json(raw_content)
        result = validate(result)

        # Output only machine-readable JSON to stdout.
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
