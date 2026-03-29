def classify_question(question):
    q = question.lower().strip()

    if not q:
        return None, "Please type a question first."

    compare_terms = ["compare", " vs ", "versus"]
    underperform_terms = ["underperform", "underperforming", "worst", "weakest"]
    diagnose_terms = [
        "why", "drop", "decline", "fell", "fall",
        "what caused", "what happened"
    ]
    summary_terms = [
        "summary", "summarize", "what changed", "key changes",
        "overview", "how did we do", "how are we doing",
        "last 7 days", "prior 7 days", "performance"
    ]

    known_terms = [
        "revenue", "sessions", "conversion", "channel",
        "paid search", "email", "social", "organic",
        "spend", "roas", "cpa"
    ]

    if any(term in q for term in compare_terms):
        return "compare", None

    if any(term in q for term in underperform_terms):
        return "underperform", None

    if any(term in q for term in diagnose_terms):
        return "diagnose", None

    if any(term in q for term in summary_terms):
        return "summary", None

    if any(term in q for term in known_terms):
        return "summary", None

    return None, (
        "I can help with revenue, sessions, conversions, and channel performance. "
        "Try: 'Why did revenue drop?' or 'Compare email and paid search.'"
    )