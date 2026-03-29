# GA4 Channel Diagnostics Assistant

A Streamlit analytics assistant that turns plain-English questions about GA4 channel performance into structured, evidence-backed answers.

Instead of making a stakeholder click through dashboards to figure out what changed, the app routes questions like:

- Why did revenue drop last week?
- Which channel underperformed most?
- Compare email and paid search

It then runs the relevant analysis, returns the answer in business-friendly language, and shows the supporting evidence.

The core logic goes through these steps: question routing, period comparisons, revenue-driver decomposition, channel mix analysis, and weekday-aware anomaly detection. An optional LLM layer can rewrite the explanation in plain English, but only if it passes a faithfulness audit against the computed results.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)

## Why I built it

I believe most dashboards are good at showing what happened, but fewer are good at explaining why it happened.

I wanted to build something closer to the way a business user actually asks questions. A stakeholder usually asks:

- What changed?
- Why did it change?
- Which channel is the problem?
- What should I check first?

I moved the project to try to answer those questions with structured analytics rather than open-ended text generation.

## What the app does

The app supports four workflows:

### 1. Summary
Returns a period-over-period view of:
- revenue
- sessions
- conversions

### 2. Diagnose
Breaks revenue change into three drivers:
- traffic effect
- conversion effect
- value per conversion effect

### 3. Underperform
Finds the weakest channel and checks whether the issue looks more like:
- traffic loss
- conversion weakness
- lower value per conversion

### 4. Compare
Compares two channels side by side across:
- sessions
- conversions
- conversion rate
- revenue
- spend and ROAS, when available

Each answer is paired with an evidence table so the output stays tied to the underlying numbers.

## Core ideas behind the project

### Revenue bridge decomposition

The diagnosis workflow decomposes revenue change into three parts:

- Traffic effect: did revenue move because session volume changed?
- Conversion effect: did revenue move because conversion efficiency changed?
- Value effect: did revenue move because revenue per conversion changed?

A revenue decline becomes much easier to investigate once it is broken into those components.

### Mix effect

Aggregate revenue can fall even when nothing changes much within each channel if traffic shifts away from stronger channels and toward weaker ones.

The app estimates a mix effect by holding prior-period efficiency fixed and measuring how changing session share across channels affected revenue. This, in turn, helps separate “a channel got worse” from “traffic composition changed.”

### Weekday-aware anomaly detection

A naive rolling z-score can flag false positives when weekly seasonality is strong. To reduce the issue, the anomaly logic compares each day only to prior observations from the same weekday:
- Monday vs past Mondays
- Saturday vs past Saturdays

This logic makes the anomaly scan more believable for ecommerce-style traffic patterns.

### Controlled LLM explanation layer

The LLM is optional.When an API key is available, the app sends structured context to OpenAI or Groq and asks for a concise explanation. The model does not decide the analysis. It only rewrites computed results into plain English.

Before the app shows the LLM answer, it runs a faithfulness audit to check whether the response preserved:
- the direction of change
- the main driver
- the weakest channel
- the approximate magnitude

If the audit fails, the app falls back to a deterministic template, which keeps the explanation layer constrained by the actual numbers.

## How it works

### 1. Question routing

A keyword-based intent classifier maps the user’s question to one of four handlers:
- `summary`
- `diagnose`
- `underperform`
- `compare`

The routing layer keeps the app simple and predictable for common analyst-style questions.

### 2. Period aggregation

The app summarizes channel performance across the latest 7 days and the prior 7 days.

Metrics include:
- sessions
- conversions
- revenue
- conversion rate
- revenue per conversion
- spend and ROAS, if spend exists in the dataset

I used DuckDB inside the app for period aggregation so the same logic scales better than repeated in-memory groupbys as data grows.

### 3. Analytical handlers

Each routed question triggers a different analysis:

- `summary_analysis`: period-over-period overview
- `diagnose_analysis`: bridge decomposition plus mix effect
- `underperform_analysis`: weakest channel diagnosis
- `compare_analysis`: side-by-side channel comparison

### 4. Evidence and explanation

Each handler returns:
- a business-friendly answer
- a structured evidence table
- an optional LLM explanation audit

So, the numbers come first, and then the wording comes second.

### Benchmark and evaluation

I tested the project on 500 simulated scenarios with seeded shocks in:
- sessions
- conversion rate
- revenue per conversion

The goal was to see whether the system could correctly identify the channel and driver behind a performance problem.

#### Diagnosis performance

| Metric | Accuracy |
|---|---:|
| Channel accuracy | 80.6% |
| Metric accuracy | 85.8% |
| Exact root-cause accuracy | 79.2% |

Diagnosis accuracy stays stable across anomaly thresholds because diagnosis is driven by bridge decomposition, not by anomaly flagging alone.

#### Anomaly detection tradeoff

| z threshold | Exact accuracy | False positive rate |
|---|---:|---:|
| 2.0 | 38.1% | 3.5% |

Higher thresholds reduced false positives, but they also reduced recall enough that `z = 2.0` was the most useful balance in this version.

### Project structure

```text
├── app.py
├── detect_recent_anomalies.py
├── llm.py
├── routing.py
├── simulation.py
├── benchmark.py
├── test_questions.py
├── requirements.txt
└── data/
    └── processed/
        └── ga4_daily_channel_metrics.csv
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/JustinJ-Lo/ga4-channel-diagnostics.git
cd ga4-channel-diagnostics
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your data

Place your GA4 daily channel metrics CSV here:

```text
data/processed/ga4_daily_channel_metrics.csv
```

Required columns:
- `date`
- `channel`
- `sessions`
- `conversions`
- `revenue`

Optional:
- `spend`

The app runs without spend. Compare outputs adjust automatically when spend is missing.

### 5. Configure the LLM layer if wanted

The app works without an API key.

Set one of the following only if you want model-written explanations:

For OpenAI:
```bash
OPENAI_API_KEY=sk-...
```

For Groq:
```bash
GROQ_API_KEY=gsk_...
```

Provider order:
1. OpenAI
2. Groq

Default models:
- OpenAI: `gpt-4o-mini`
- Groq: `llama-3.3-70b-versatile`

### 6. Run the app

```bash
streamlit run app.py
```

### 7. Run the benchmark

```bash
python3 benchmark.py
```

### 8. Run the routing tests

```bash
python3 test_questions.py
```

## Example questions

Some prompts to try:
- Why did revenue drop in the last 7 days?
- Which channel underperformed most?
- Compare email and paid search
- Compare direct and referral
- Summarize key changes in the last 7 days

## What matters most in this project

A few parts are more important than the rest:

- weekday-aware anomaly detection, because it reduces false positives from weekly seasonality
- bridge decomposition, because it turns a vague revenue change into specific drivers
- mix effect analysis, because it separates within-channel weakness from composition shifts
- LLM faithfulness auditing, because it keeps generated explanations tied to computed evidence
- benchmarking, because it tests whether the diagnosis logic is useful instead of only sounding convincing

## Limits

A few limits to consider:

- Mix effect is a decomposition, not proof of causality
- Anomaly detection still relies on a fairly small same-weekday history
- Routing is rule-based, not a learned NLP classifier
- Benchmark results come from simulated shocks, not labeled real-world incidents
- Output quality depends on the quality of the input GA4 export
- The LLM layer is optional and never replaces the underlying analysis

## Why I think the project is useful

My goal was not to build another dashboard. It was to build a lightweight analytics assistant that helps a user to see not just that "Revenue has gone down", but to see that “Paid Search drove the decline, and the biggest issue was traffic volume”.


