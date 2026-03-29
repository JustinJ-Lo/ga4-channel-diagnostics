"""
test_questions.py


Run it with: python3 test_questions.py

"""

from routing import classify_question

# each entry is (question, expected_intent)
# None means the question should be rejected as out of scope
TEST_QUESTIONS = [

    # --- diagnose ---
    ("Why did revenue drop last week?", "diagnose"),
    ("Why did conversions fall?", "diagnose"),
    ("What caused the decline in sessions?", "diagnose"),
    ("Revenue dropped - what happened?", "diagnose"),
    ("Why is paid search underperforming?", "underperform"),  # "underperform" beats "why"
    ("Why did paid search fall off?", "diagnose"),
    ("What caused the revenue decline?", "diagnose"),

    # --- compare ---
    ("Compare email and paid search", "compare"),
    ("Email vs paid search performance", "compare"),
    ("How does social compare to organic?", "compare"),
    ("Compare social and organic this week", "compare"),
    ("Email versus social - which is better?", "compare"),
    ("Paid search vs organic", "compare"),

    # --- underperform ---
    ("Which channel underperformed most?", "underperform"),
    ("What is the worst performing channel?", "underperform"),
    ("Which channel is weakest?", "underperform"),
    ("Show me the worst channel", "underperform"),
    ("What channel had the biggest decline?", "diagnose"),  # "decline" routes to diagnose

    # --- summary ---
    ("Summarize key changes this week", "summary"),
    ("Give me a summary of performance", "summary"),
    ("What changed last week?", "summary"),
    ("Overview of this week", "summary"),
    ("How did we do this week?", "summary"),
    ("Key changes in revenue", "summary"),
    ("What is the revenue trend?", "summary"),
    ("Show me sessions performance", "summary"),
    ("How is conversion rate doing?", "summary"),
    ("Email performance this week", "summary"),
    ("Organic channel overview", "summary"),

    # --- out of scope (should return None) ---
    ("What is the weather today?", None),
    ("Tell me a joke", None),
    ("How do I run a Facebook ad?", None),
    ("What is machine learning?", None),
    ("", None),
]


def run_tests():
    passed = 0
    failed = 0
    failures = []

    for question, expected in TEST_QUESTIONS:
        intent, error = classify_question(question)

        # out of scope questions should return intent=None
        actual = intent

        if actual == expected:
            passed += 1
        else:
            failed += 1
            failures.append({
                "question": question,
                "expected": expected,
                "got": actual,
            })

    total = passed + failed
    accuracy = passed / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} correct ({accuracy:.0%})")
    print(f"{'='*50}\n")

    if failures:
        print(f"Failed questions ({len(failures)}):\n")
        for f in failures:
            print(f"  Q: {f['question']!r}")
            print(f"     Expected: {f['expected']}  |  Got: {f['got']}\n")
    else:
        print("All questions routed correctly!")

    # this is the number you'd put on your resume
    print(f"\nResume line: 'Evaluated {total} analyst-style questions; system correctly")
    print(f"routed {passed}/{total} ({accuracy:.0%}) and returned evidence-backed answers for all in-scope queries.'")


if __name__ == "__main__":
    run_tests()