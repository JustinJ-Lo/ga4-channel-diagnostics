# GA4 Channel Diagnostics Assistant

I built this project to make GA4 channel performance easier to investigate in plain English.

A user can ask questions like:

- Why did revenue drop last week?
- Which channel underperformed most?
- Compare email and paid search

The app routes the question to the right analysis, runs the numbers, and returns a structured answer with evidence. If an LLM key is available, it can also write the explanation in plain English. If not, the app falls back to a deterministic template.

![Python](https://img.shields.io/badge/python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-app-red)

## Why I made it

Most dashboards are good at showing what happened, but they do not do much to explain why it happened.

I wanted to build something closer to how a stakeholder actually asks questions. Instead of clicking through charts, a user can type a question like "Why did revenue drop?" and get a structured answer backed by calculations.

## What the app does

The app supports four main question types:

- Summary: gives a period over period view of revenue, sessions, and conversions
- Diagnose: breaks revenue change into traffic, conversion rate, and revenue per conversion effects
- Underperform: identifies the weakest channel and suggests what to check next
- Compare: compares two channels side by side

## How it works

### Routing

routing.py uses a keyword-based intent classifier to map a question to one of four handlers:

- summary
- diagnose
- underperform
- compare

### Core analysis

app.py contains the main analytics logic.

Each handler runs a different analysis:

- summary_analysis: period over period summary of revenue, sessions, and conversions
- diagnose_analysis: bridge decomposition plus a mix effect layer
- underperform_analysis: identifies the weakest channel and checks whether the main issue looks more like traffic or conversion rate
- compare_analysis: side by side comparison of two channels

### Revenue bridge

The diagnosis flow decomposes revenue change into three effects:

- Traffic effect
- Conversion effect
- Value effect

A drop in revenue can come from fewer sessions, worse conversion efficiency, lower revenue per conversion, or some combination of all three.

### Mix effect

I also added a mix effect layer.

A business can lose revenue even when nothing changes much within each channel if traffic shifts away from stronger channels and toward weaker ones. The mix effect estimates the impact of that composition shift while holding prior-period efficiency fixed.

### Weekday-aware anomaly detection

detect_recent_anomalies.py compares each day only to prior observations from the same weekday.

Monday gets compared to past Mondays. Saturday gets compared to past Saturdays.

A normal rolling window can overreact to weekly seasonality, so I changed the anomaly logic to make it more believable for ecommerce data.

### LLM explanation layer

llm.py sends structured context to OpenAI or Groq and asks for a plain-English explanation.

I did not want the model to just freestyle an answer, so I added a faithfulness audit. The app checks whether the LLM preserved the key driver and weakest channel before using the response. If the audit fails, the app falls back to a deterministic template.

## Benchmark results

I evaluated the system across 500 simulated scenarios with seeded shocks in:

- sessions
- conversion rate
- revenue per conversion

### Diagnosis performance

| Metric | Accuracy |
|---|---:|
| Channel accuracy | 80.6% |
| Metric accuracy | 85.8% |
| Exact root-cause accuracy | 79.2% |

Diagnosis accuracy stays stable across anomaly thresholds because the diagnosis module is driven by the bridge decomposition, not by anomaly flagging.

### Anomaly detection

| z threshold | Exact accuracy | False positive rate |
|---|---:|---:|
| 2.0 | 38.1% | 3.5% |

Higher thresholds reduced false positives, but they also reduced recall enough that z = 2.0 was the most useful setting in this version.

## Project structure

```
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

git clone https://github.com/JustinJ-Lo/ga4-channel-diagnostics.git
cd ga4-channel-diagnostics

### 2. Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

For Windows:

.venv\\Scripts\\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Add your data

Put your GA4 daily channel metrics CSV here:

data/processed/ga4_daily_channel_metrics.csv

Required columns:

- date
- channel
- sessions
- conversions
- revenue

Optional columns:

- spend

The app can still run without spend. Compare outputs will adjust based on what is available in the dataset.

### 5. Configure the LLM layer if you want

The app works without an API key.

An API key is only needed if you want model-written explanations.

For OpenAI:

OPENAI_API_KEY=sk-...

For Groq:

GROQ_API_KEY=gsk_...

The app checks for OpenAI first, then Groq.

### 6. Run the app

streamlit run app.py

### 7. Run the benchmark

python3 benchmark.py

### 8. Run the routing test suite

python3 test_questions.py

## LLM providers

| Provider | Env var | Default model |
|---|---|---|
| OpenAI | OPENAI_API_KEY | gpt-4o-mini |
| Groq | GROQ_API_KEY | llama-3.3-70b-versatile |

## What I think is most useful here

A few parts of the project matter more than the rest:

- weekday-aware anomaly detection, because it handles weekly seasonality better than a naive rolling window
- bridge decomposition, because it turns a vague revenue drop into specific drivers
- the LLM audit, because it keeps the explanation layer tied to the actual numbers
- the benchmark harness, because it gives me a way to evaluate whether the system is doing something useful instead of just sounding smart

## Limits

A few caveats are worth keeping in mind:

- Mix effect is a decomposition, not proof of causality
- Anomaly detection still depends on a fairly small same-weekday history
- Real performance depends on the quality of the input GA4 export
- The LLM layer is optional and only used when its response passes the audit checks
