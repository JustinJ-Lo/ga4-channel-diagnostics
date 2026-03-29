import json
import os
import time
from typing import Any, Dict, Optional, Tuple


def _get_client() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """
    Returns (client, provider, model_name)
    Supports OpenAI or Groq depending on available env vars.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return client, "openai", model
        except Exception as e:
            print(f"OpenAI client init failed: {e}")

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            return client, "groq", model
        except Exception as e:
            print(f"Groq client init failed: {e}")

    return None, None, None


def llm_status() -> Dict[str, Any]:
    client, provider, model = _get_client()
    return {
        "available": client is not None,
        "provider": provider,
        "model": model,
    }


def _build_messages(context: Dict[str, Any]) -> list[dict]:
    system_prompt = """
You are an analytics explanation assistant.

Rules:
- Use only the facts provided by the user/context.
- Do not invent metrics, channels, causes, dates, or recommendations.
- Keep the explanation concise and business-friendly.
- Name the biggest driver clearly.
- If a weakest channel is provided, mention it exactly.
- If there is uncertainty, say so plainly.
- Prefer direct language over hype.
""".strip()

    user_prompt = f"""
Return a JSON object with these keys:
- summary
- main_driver
- weakest_channel
- next_check

Context:
{json.dumps(context, indent=2)}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_openai_json(client: Any, model: str, messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _call_groq_json(client: Any, model: str, messages: list[dict]) -> str:
    # Groq may not reliably enforce JSON schema like OpenAI, so we still ask for JSON.
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # try to extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def generate_summary_structured(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    client, provider, model = _get_client()
    if client is None:
        return None

    messages = _build_messages(context)

    try:
        raw = (
            _call_openai_json(client, model, messages)
            if provider == "openai"
            else _call_groq_json(client, model, messages)
        )
        parsed = _extract_json(raw)
        if not parsed:
            return None

        return {
            "summary": str(parsed.get("summary", "")).strip(),
            "main_driver": str(parsed.get("main_driver", "")).strip(),
            "weakest_channel": str(parsed.get("weakest_channel", "")).strip(),
            "next_check": str(parsed.get("next_check", "")).strip(),
            "provider": provider,
            "model": model,
            "raw": raw,
        }
    except Exception as e:
        print(f"LLM structured call failed: {e}")
        return None


def generate_summary(context: Dict[str, Any]) -> Optional[str]:
    """
    Backward-compatible plain-text wrapper.
    """
    result = generate_summary_structured(context)
    if not result:
        return None
    return result.get("summary") or None


def audit_faithfulness(llm_result, context):
    """
    Scored faithfulness audit (0.0 – 1.0).

    Replaces the old binary pass/fail with a weighted score across five checks.
    The LLM response is accepted only if score >= PASS_THRESHOLD (0.70).
    This means a response that gets the main driver and direction right but
    misses the exact channel name by phrasing still passes, instead of being
    silently discarded.

    Weights:
        has_summary       0.20  — response must contain text at all
        direction_correct 0.30  — highest weight; direction wrong = misleading
        main_driver       0.25  — key driver correctly named
        weakest_channel   0.20  — weakest channel named (skipped for compare)
        magnitude_ok      0.05  — bonus: key percentage appears in text
    """
    PASS_THRESHOLD = 0.70

    if not llm_result:
        return {
            "used_llm": False,
            "score": 0.0,
            "passed": False,
            "checks": {
                "has_summary": False,
                "direction_correct": None,
                "main_driver_matches": False,
                "weakest_channel_matches": None,
                "magnitude_ok": None,
            },
            "notes": ["LLM result missing; fallback used."],
        }

    summary = (llm_result.get("summary") or "").lower()
    question_type = str(context.get("question_type", "")).strip().lower()
    main_driver_expected = str(context.get("main_driver", "")).strip().lower()
    weakest_channel_expected = str(context.get("weakest_channel", "")).strip().lower()
    rev_chg = context.get("rev_chg")

    # --- Check 1: summary is non-empty (weight 0.20) ---
    has_summary = bool(summary.strip())

    # --- Check 2: direction correct (weight 0.30) ---
    # Does the summary correctly say whether revenue went up or down?
    direction_correct = None
    if rev_chg is not None:
        negative_words = {"declin", "drop", "fell", "fall", "decreas", "down", "lost", "loss", "worse"}
        positive_words = {"increas", "grew", "rose", "gain", "up", "improv", "better"}
        if rev_chg < 0:
            direction_correct = any(w in summary for w in negative_words)
        elif rev_chg > 0:
            direction_correct = any(w in summary for w in positive_words)
        else:
            direction_correct = True  # flat revenue; don't penalize

    # --- Check 3: main driver mentioned (weight 0.25) ---
    main_driver_matches = False
    if main_driver_expected:
        main_driver_matches = (
            main_driver_expected in summary
            or main_driver_expected == str(llm_result.get("main_driver", "")).strip().lower()
        )

    # --- Check 4: weakest channel named (weight 0.20, skipped for compare) ---
    weakest_channel_matches = None
    if question_type != "compare" and weakest_channel_expected:
        weakest_channel_matches = (
            weakest_channel_expected in summary
            or weakest_channel_expected == str(llm_result.get("weakest_channel", "")).strip().lower()
        )

    # --- Check 5: magnitude in ballpark (weight 0.05, bonus) ---
    # Just checks that the rounded percentage string appears somewhere in the summary.
    # e.g. if rev_chg = -0.032, looks for "3.2%" in the text.
    magnitude_ok = None
    expected_pct = None
    if rev_chg is not None:
        expected_pct = f"{abs(rev_chg):.1%}"   # e.g. "3.2%"
        magnitude_ok = expected_pct in summary

    # --- Score ---
    score = 0.0
    score += 0.20 if has_summary else 0.0
    score += 0.30 if direction_correct else 0.0
    score += 0.25 if main_driver_matches else 0.0
    # For compare questions weakest_channel is N/A — don't penalize
    if weakest_channel_matches is True:
        score += 0.20
    elif weakest_channel_matches is None:
        score += 0.20
    score += 0.05 if magnitude_ok else 0.0

    passed = score >= PASS_THRESHOLD

    # --- Notes for the audit expander in the UI ---
    notes = []
    if not has_summary:
        notes.append("Summary text was empty.")
    if direction_correct is False:
        label = "decline" if (rev_chg is not None and rev_chg < 0) else "increase"
        notes.append(f"Direction mismatch: expected {label} language but summary may not reflect it.")
    if main_driver_expected and not main_driver_matches:
        notes.append(f"Main driver '{context.get('main_driver')}' not clearly preserved.")
    if weakest_channel_matches is False:
        notes.append(f"Weakest channel '{context.get('weakest_channel')}' not clearly preserved.")
    if magnitude_ok is False and expected_pct:
        notes.append(f"Expected magnitude '{expected_pct}' not found in summary (minor).")
    if passed:
        notes.append(f"Score {score:.2f} ≥ {PASS_THRESHOLD} — response accepted.")
    else:
        notes.append(f"Score {score:.2f} < {PASS_THRESHOLD} — fallback used.")

    return {
        "used_llm": True,
        "score": round(score, 2),
        "passed": passed,
        "checks": {
            "has_summary": has_summary,
            "direction_correct": direction_correct,
            "main_driver_matches": main_driver_matches,
            "weakest_channel_matches": weakest_channel_matches,
            "magnitude_ok": magnitude_ok,
        },
        "notes": notes,
    }
