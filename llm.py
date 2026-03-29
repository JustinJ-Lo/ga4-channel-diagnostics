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
    if not llm_result:
        return {
            "used_llm": False,
            "passed": False,
            "checks": {
                "has_summary": False,
                "main_driver_matches": False,
                "weakest_channel_matches": None,
            },
            "notes": ["LLM result missing; fallback likely used."],
        }

    summary = (llm_result.get("summary") or "").lower()
    main_driver_expected = str(context.get("main_driver", "")).strip().lower()
    weakest_channel_expected = str(context.get("weakest_channel", "")).strip().lower()
    question_type = str(context.get("question_type", "")).strip().lower()

    has_summary = bool(summary.strip())

    main_driver_matches = (
        bool(main_driver_expected) and main_driver_expected in summary
    ) or (
        bool(main_driver_expected)
        and main_driver_expected == str(llm_result.get("main_driver", "")).strip().lower()
    )

    if question_type == "compare" or not weakest_channel_expected:
        weakest_channel_matches = None
    else:
        weakest_channel_matches = (
            weakest_channel_expected in summary
        ) or (
            weakest_channel_expected
            == str(llm_result.get("weakest_channel", "")).strip().lower()
        )

    passed = has_summary and (
        not main_driver_expected or main_driver_matches
    ) and (
        weakest_channel_matches in [True, None]
    )

    notes = []
    if not has_summary:
        notes.append("Summary text was empty.")
    if main_driver_expected and not main_driver_matches:
        notes.append(f"Expected main driver '{context.get('main_driver')}' was not clearly preserved.")
    if weakest_channel_matches is False:
        notes.append(f"Expected weakest channel '{context.get('weakest_channel')}' was not clearly preserved.")
    if passed:
        notes.append("Structured explanation is consistent with the key supplied fields.")

    return {
        "used_llm": True,
        "passed": passed,
        "checks": {
            "has_summary": has_summary,
            "main_driver_matches": main_driver_matches,
            "weakest_channel_matches": weakest_channel_matches,
        },
        "notes": notes,
    }