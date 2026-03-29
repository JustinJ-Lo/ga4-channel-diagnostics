import streamlit as st
from llm import generate_summary_structured, audit_faithfulness, llm_status
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

ANOMALY_Z_THRESHOLD = 2.0

st.set_page_config(page_title="GA4 Channel Diagnostics Assistant", layout="wide")

CHANNEL_MAP = {
    "direct": "Direct",
    "organic": "Organic Search",
    "paid search": "Paid Search",
    "cpc": "Paid Search",
    "referral": "Referral",
    "email": "Email",
}


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/ga4_daily_channel_metrics.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["channel", "date"])

    df["conversion_rate"] = np.where(
        df["sessions"] > 0,
        df["conversions"] / df["sessions"],
        np.nan
    )
    df["revenue_per_conversion"] = np.where(
        df["conversions"] > 0,
        df["revenue"] / df["conversions"],
        np.nan
    )
    return df


def finalize_llm_answer(context: dict, fallback: str, use_llm: bool = True):
    if not use_llm:
        return fallback, {
            "used_llm": False,
            "passed": False,
            "notes": ["LLM toggle was off; deterministic fallback used."]
        }, False

    llm_result = generate_summary_structured(context)
    audit = audit_faithfulness(llm_result, context)

    if llm_result and llm_result.get("summary") and audit and audit["passed"]:
        answer = llm_result["summary"]
        used_llm = True
    else:
        answer = fallback
        used_llm = False

    return answer, audit, used_llm

def get_periods(df):
    end = df["date"].max()
    start = end - pd.Timedelta(days=6)
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=6)
    return start, end, prev_start, prev_end


def summarize_period(df, start_date, end_date):
    """
    Aggregates channel metrics for a date range using DuckDB SQL.

    Why DuckDB: pandas loads and scans the entire DataFrame in memory for
    every period. DuckDB pushes the WHERE + GROUP BY into a query engine,
    returning only the ~5 channel rows Streamlit actually needs. For the
    current CSV this is negligible, but it scales cleanly to multi-million
    row GA4 BigQuery exports without changing any downstream code.
    """
    has_spend = "spend" in df.columns
    spend_agg = "SUM(spend) AS spend," if has_spend else ""

    query = f"""
        SELECT
            channel,
            SUM(sessions)    AS sessions,
            SUM(conversions) AS conversions,
            SUM(revenue)     AS revenue,
            {spend_agg}
            1 AS _dummy
        FROM df
        WHERE date >= '{start_date}'
          AND date <= '{end_date}'
        GROUP BY channel
        ORDER BY channel
    """

    # DuckDB can reference local pandas DataFrames by variable name.
    # We use a fresh in-process connection so there's no state leakage
    # between Streamlit reruns.
    con = duckdb.connect()
    by_channel = con.execute(query).df().drop(columns=["_dummy"])
    con.close()

    by_channel["conversion_rate"] = np.where(
        by_channel["sessions"] > 0,
        by_channel["conversions"] / by_channel["sessions"],
        np.nan
    )
    by_channel["revenue_per_conversion"] = np.where(
        by_channel["conversions"] > 0,
        by_channel["revenue"] / by_channel["conversions"],
        np.nan
    )
    if has_spend:
        by_channel["roas"] = np.where(
            by_channel["spend"] > 0,
            by_channel["revenue"] / by_channel["spend"],
            np.nan
        )

    total = {
        "channel": "Total",
        "sessions": by_channel["sessions"].sum(),
        "conversions": by_channel["conversions"].sum(),
        "revenue": by_channel["revenue"].sum(),
    }
    if has_spend:
        total["spend"] = by_channel["spend"].sum()
        total["roas"] = (
            total["revenue"] / total["spend"] if total["spend"] > 0 else np.nan
        )
    total["conversion_rate"] = (
        total["conversions"] / total["sessions"] if total["sessions"] > 0 else np.nan
    )
    total["revenue_per_conversion"] = (
        total["revenue"] / total["conversions"] if total["conversions"] > 0 else np.nan
    )

    return by_channel, total


def pct_change(current, previous):
    if previous == 0 or pd.isna(previous):
        return np.nan
    return (current - previous) / previous


def extract_channels(question):
    q = question.lower()
    found = []
    for key, label in CHANNEL_MAP.items():
        if key in q and label not in found:
            found.append(label)
    return found


def format_signed_money(x):
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.0f}"


def fmt_pct(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}%}"


def fmt_money(x, decimals=0):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.{decimals}f}"


def revenue_bridge(curr_row, prev_row):
    s0, s1 = prev_row["sessions"], curr_row["sessions"]
    cr0, cr1 = prev_row["conversion_rate"], curr_row["conversion_rate"]
    rpc0, rpc1 = prev_row["revenue_per_conversion"], curr_row["revenue_per_conversion"]

    traffic_effect = (s1 - s0) * cr0 * rpc0
    conversion_effect = s1 * (cr1 - cr0) * rpc0
    value_effect = s1 * cr1 * (rpc1 - rpc0)

    return {
        "traffic_effect": traffic_effect,
        "conversion_effect": conversion_effect,
        "value_effect": value_effect,
        "total_delta": traffic_effect + conversion_effect + value_effect,
    }


def pick_main_driver(bridge):
    effects = {
        "traffic volume": bridge["traffic_effect"],
        "conversion efficiency": bridge["conversion_effect"],
        "value per conversion": bridge["value_effect"],
    }
    if bridge["total_delta"] < 0:
        return min(effects, key=effects.get)
    return max(effects, key=effects.get)


def compute_mix_effect(curr_by_channel, prev_by_channel):
    """
    Mix effect: how much revenue changed just because the share
    of sessions shifted across channels — even if per-channel
    conversion rates and revenue per conversion stayed the same.

    Example: if Email (high CR) loses share to Social (low CR),
    aggregate revenue falls even if nothing changed within each channel.

    Formula per channel:
        mix_effect = total_curr_sessions
                     * (curr_share - prev_share)
                     * prev_CR
                     * prev_RPC
    """
    total_curr = curr_by_channel["sessions"].sum()
    total_prev = prev_by_channel["sessions"].sum()

    merged = curr_by_channel[["channel", "sessions"]].merge(
        prev_by_channel[["channel", "sessions", "conversion_rate", "revenue_per_conversion"]],
        on="channel",
        suffixes=("_curr", "_prev"),
    )

    merged["share_curr"] = merged["sessions_curr"] / total_curr
    merged["share_prev"] = merged["sessions_prev"] / total_prev
    merged["share_shift"] = merged["share_curr"] - merged["share_prev"]

    # Hold prior CR and RPC constant — only the composition is moving
    merged["mix_effect"] = (
        total_curr
        * merged["share_shift"]
        * merged["conversion_rate"]          # prior period CR
        * merged["revenue_per_conversion"]   # prior period RPC
    )

    return merged[[
        "channel", "share_prev", "share_curr", "share_shift", "mix_effect"
    ]]


def detect_recent_anomalies(
    df,
    metrics=("sessions", "revenue", "conversion_rate", "revenue_per_conversion"),
    same_weekday_lookback=4,
    min_history_points=3,
    z_thresh=ANOMALY_Z_THRESHOLD
):
    """
    Weekday-aware anomaly detection.

    For each channel, metric, and date in the latest 7 days:
    - compare the observed value to prior values from the SAME weekday
    - compute a z-score from that weekday-specific baseline

    Goal is to reduce false positives caused by normal weekly seasonality.
    """
    start, end, _, _ = get_periods(df)
    base = df.sort_values(["channel", "date"]).copy()
    base["weekday"] = base["date"].dt.day_name()

    all_results = []

    for metric in metrics:
        metric_df = base[["date", "channel", "weekday", metric]].copy()
        metric_df = metric_df.rename(columns={metric: "value"})

        rows = []

        for channel, grp in metric_df.groupby("channel"):
            grp = grp.sort_values("date").copy()

            for idx, row in grp.iterrows():
                current_date = row["date"]
                current_weekday = row["weekday"]
                current_value = row["value"]

                # Only score dates in the latest 7 days
                if current_date < start or current_date > end:
                    continue

                history = grp[
                    (grp["date"] < current_date) &
                    (grp["weekday"] == current_weekday)
                ].sort_values("date")

                history_values = history["value"].dropna().tail(same_weekday_lookback)

                if len(history_values) < min_history_points:
                    continue

                baseline_mean = history_values.mean()
                baseline_std = history_values.std(ddof=1)

                if pd.isna(baseline_std) or baseline_std == 0:
                    continue

                z_score = (current_value - baseline_mean) / baseline_std

                if abs(z_score) >= z_thresh:
                    rows.append({
                        "date": current_date,
                        "channel": channel,
                        "metric": metric,
                        "value": current_value,
                        "weekday": current_weekday,
                        "baseline_mean": baseline_mean,
                        "baseline_std": baseline_std,
                        "history_points": int(len(history_values)),
                        "z_score": z_score,
                        "direction": "High" if z_score > 0 else "Low",
                    })

        if rows:
            all_results.append(pd.DataFrame(rows))

    if not all_results:
        return pd.DataFrame(columns=[
            "date", "channel", "metric", "value", "weekday",
            "baseline_mean", "baseline_std", "history_points",
            "z_score", "direction"
        ])

    out = pd.concat(all_results, ignore_index=True)

    # Keep biggest anomalies first
    out = out.sort_values("z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    return out[[
        "date", "channel", "metric", "value", "weekday",
        "baseline_mean", "baseline_std", "history_points",
        "z_score", "direction"
    ]]


def format_anomaly_value(metric, value):
    if pd.isna(value):
        return "N/A"
    if metric == "conversion_rate":
        return f"{value:.2%}"
    if metric == "revenue_per_conversion":
        return f"${value:,.2f}"
    if metric == "revenue":
        return f"${value:,.0f}"
    return f"{value:,.0f}"


def get_top_anomaly_text(df):
    anomalies = detect_recent_anomalies(df)

    if anomalies.empty:
        return "No statistically unusual channel-level movements detected in the latest 7 days relative to same-weekday history."

    top = anomalies.iloc[0]
    value_str = format_anomaly_value(top["metric"], top["value"])
    baseline_str = format_anomaly_value(top["metric"], top["baseline_mean"])

    return (
        f"Top anomaly: {top['channel']} had a {top['direction'].lower()} "
        f"{top['metric'].replace('_', ' ')} reading on {top['date'].date()} "
        f"({top['weekday']}; observed {value_str} vs same-weekday baseline {baseline_str}; "
        f"z={top['z_score']:.2f}, n={top['history_points']})."
    )


from routing import classify_question


def summary_analysis(df, use_llm=True):
    start, end, prev_start, prev_end = get_periods(df)

    curr_by_channel, curr_total = summarize_period(df, start, end)
    prev_by_channel, prev_total = summarize_period(df, prev_start, prev_end)

    rev_chg = pct_change(curr_total["revenue"], prev_total["revenue"])
    sess_chg = pct_change(curr_total["sessions"], prev_total["sessions"])
    conv_chg = pct_change(curr_total["conversions"], prev_total["conversions"])

    merged = curr_by_channel.merge(
        prev_by_channel[["channel", "revenue", "sessions", "conversions"]],
        on="channel",
        suffixes=("_curr", "_prev")
    )
    merged["rev_chg_pct"] = (merged["revenue_curr"] - merged["revenue_prev"]) / merged["revenue_prev"]

    worst = merged.sort_values("rev_chg_pct").iloc[0]
    best = merged.sort_values("rev_chg_pct", ascending=False).iloc[0]

    context = {
        "question_type": "summary",
        "rev_chg": float(rev_chg),
        "sess_chg": float(sess_chg),
        "conv_chg": float(conv_chg),
        "main_driver": "overall performance summary",
        "weakest_channel": worst["channel"],
        "best_channel": best["channel"],
        "worst_channel_change": float(worst["rev_chg_pct"]),
        "best_channel_change": float(best["rev_chg_pct"]),
    }

    fallback = f"""
### Summary
Revenue changed **{rev_chg:.1%}** vs the prior 7-day period.

- Sessions: **{sess_chg:.1%}**
- Conversions: **{conv_chg:.1%}**
- Biggest decline: **{worst['channel']}** ({worst['rev_chg_pct']:.1%})
- Biggest gain: **{best['channel']}** ({best['rev_chg_pct']:.1%})

### What to look at next
Start with the weakest channel and check whether the issue is traffic volume, conversion rate, or value per conversion.
"""

    answer, audit, used_llm = finalize_llm_answer(context, fallback, use_llm=use_llm)

    evidence = merged[[
        "channel", "revenue_curr", "revenue_prev", "rev_chg_pct",
        "sessions_curr", "sessions_prev"
    ]].copy()
    evidence.columns = [
        "Channel", "Revenue (Last 7d)", "Revenue (Prior 7d)",
        "Revenue Change", "Sessions (Last 7d)", "Sessions (Prior 7d)"
    ]
    evidence["Revenue (Last 7d)"] = evidence["Revenue (Last 7d)"].map(lambda x: f"${x:,.0f}")
    evidence["Revenue (Prior 7d)"] = evidence["Revenue (Prior 7d)"].map(lambda x: f"${x:,.0f}")
    evidence["Revenue Change"] = evidence["Revenue Change"].map(lambda x: f"{x:.1%}")

    return answer, evidence, audit, used_llm


def diagnose_analysis(df, use_llm=True):
    start, end, prev_start, prev_end = get_periods(df)

    curr_by_channel, curr_total = summarize_period(df, start, end)
    prev_by_channel, prev_total = summarize_period(df, prev_start, prev_end)

    rev_chg = pct_change(curr_total["revenue"], prev_total["revenue"])

    merged = curr_by_channel.merge(
        prev_by_channel[
            ["channel", "sessions", "revenue", "conversion_rate", "revenue_per_conversion"]
        ],
        on="channel",
        suffixes=("_curr", "_prev"),
    )

    # Standard 3-way bridge decomposition
    merged["traffic_effect"] = (
        (merged["sessions_curr"] - merged["sessions_prev"])
        * merged["conversion_rate_prev"]
        * merged["revenue_per_conversion_prev"]
    )

    merged["conversion_effect"] = (
        merged["sessions_curr"]
        * (merged["conversion_rate_curr"] - merged["conversion_rate_prev"])
        * merged["revenue_per_conversion_prev"]
    )

    merged["value_effect"] = (
        merged["sessions_curr"]
        * merged["conversion_rate_curr"]
        * (merged["revenue_per_conversion_curr"] - merged["revenue_per_conversion_prev"])
    )

    merged["rev_delta"] = (
        merged["traffic_effect"]
        + merged["conversion_effect"]
        + merged["value_effect"]
    )

    worst = merged.sort_values("rev_delta").iloc[0]

    effect_to_metric = {
        "traffic_effect": "sessions",
        "conversion_effect": "conversion_rate",
        "value_effect": "revenue_per_conversion",
    }

    worst_effect_name = min(
        ["traffic_effect", "conversion_effect", "value_effect"],
        key=lambda col: worst[col],
    )

    main_driver_metric = effect_to_metric[worst_effect_name]

    # Mix effect: did sessions shift toward lower-value channels?
    # This catches cases where aggregate revenue fell because users moved
    # away from high-CR channels (e.g. Email) toward lower-CR ones (e.g. Social),
    # even if nothing changed within each channel.
    mix_df = compute_mix_effect(curr_by_channel, prev_by_channel)
    total_mix_effect = mix_df["mix_effect"].sum()

    # Which channel's shift hurt the most?
    worst_mix_channel = mix_df.sort_values("mix_effect").iloc[0]

    # Only surface the mix note if it is material — at least 1% of prior revenue.
    # A fixed dollar threshold would be arbitrary across different data scales.
    mix_note = ""
    prior_revenue = prev_total["revenue"]
    mix_is_material = (
        prior_revenue > 0
        and abs(total_mix_effect) / prior_revenue >= 0.01
    )
    if mix_is_material:
        direction = "away from" if worst_mix_channel["share_shift"] < 0 else "toward"
        mix_note = (
            f"\n\n**Mix effect:** Sessions shifted {direction} "
            f"**{worst_mix_channel['channel']}**, contributing an estimated "
            f"**{format_signed_money(total_mix_effect)}** revenue impact from composition alone."
        )

    context = {
        "question_type": "diagnose",
        "rev_chg": float(rev_chg),
        "main_driver": main_driver_metric,
        "weakest_channel": worst["channel"],
        "traffic_effect": float(worst["traffic_effect"]),
        "conversion_effect": float(worst["conversion_effect"]),
        "value_effect": float(worst["value_effect"]),
        "rev_delta": float(worst["rev_delta"]),
        "total_mix_effect": float(total_mix_effect),
    }

    fallback = f"""
### Why did it change?
Overall revenue changed **{rev_chg:.1%}** vs the prior 7 days.

The weakest channel was **{worst['channel']}**, with a total revenue impact of **${worst['rev_delta']:,.0f}**.

The strongest negative driver was **{main_driver_metric}**.{mix_note}

### What to check
Start with **{worst['channel']}** and verify whether the issue came from traffic loss, conversion weakness, or lower revenue per conversion.
"""

    answer, audit, used_llm = finalize_llm_answer(context, fallback, use_llm=use_llm)

    # Bridge evidence table
    bridge_evidence = merged[[
        "channel", "traffic_effect", "conversion_effect",
        "value_effect", "rev_delta"
    ]].copy().sort_values("rev_delta")

    bridge_evidence.columns = [
        "Channel", "Traffic Effect", "Conversion Effect",
        "Value Effect", "Revenue Delta",
    ]

    for col in ["Traffic Effect", "Conversion Effect", "Value Effect", "Revenue Delta"]:
        bridge_evidence[col] = bridge_evidence[col].map(lambda x: f"${x:,.0f}")

    # Mix effect evidence table
    mix_evidence = mix_df.copy()
    mix_evidence["share_prev"] = mix_evidence["share_prev"].map(lambda x: f"{x:.1%}")
    mix_evidence["share_curr"] = mix_evidence["share_curr"].map(lambda x: f"{x:.1%}")
    mix_evidence["share_shift"] = mix_evidence["share_shift"].map(lambda x: f"{x:+.1%}")
    mix_evidence["mix_effect"] = mix_evidence["mix_effect"].map(lambda x: f"${x:,.0f}")
    mix_evidence.columns = [
        "Channel", "Session Share (Prior)", "Session Share (Current)",
        "Share Shift", "Mix Effect ($)",
    ]

    return answer, bridge_evidence, mix_evidence, audit, used_llm


def underperform_analysis(df, use_llm=True):
    start, end, prev_start, prev_end = get_periods(df)

    curr_by_channel, _ = summarize_period(df, start, end)
    prev_by_channel, _ = summarize_period(df, prev_start, prev_end)

    merged = curr_by_channel.merge(
        prev_by_channel[["channel", "revenue", "sessions", "conversion_rate"]],
        on="channel",
        suffixes=("_curr", "_prev")
    )
    merged["rev_chg"] = np.where(
        merged["revenue_prev"] != 0,
        (merged["revenue_curr"] - merged["revenue_prev"]) / merged["revenue_prev"],
        np.nan
    )
    merged["sess_chg"] = np.where(
        merged["sessions_prev"] != 0,
        (merged["sessions_curr"] - merged["sessions_prev"]) / merged["sessions_prev"],
        np.nan
    )
    merged["cr_chg"] = merged["conversion_rate_curr"] - merged["conversion_rate_prev"]

    worst = merged.sort_values("rev_chg").iloc[0]

    if worst["sess_chg"] < 0 and worst["cr_chg"] < 0:
        diagnosis = (
            "Both traffic and conversion rate fell — usually points to a broader channel issue. "
            "Check campaign settings, audience targeting, and budget pacing."
        )
    elif worst["sess_chg"] < 0 and worst["cr_chg"] >= 0:
        diagnosis = (
            "Traffic fell but conversion rate held up — the main problem is volume. "
            "Check reach, budget, delivery, and visibility."
        )
    elif worst["sess_chg"] >= 0 and worst["cr_chg"] < 0:
        diagnosis = (
            "Traffic held up but conversion rate fell — the main problem is efficiency. "
            "Check landing pages, offer strength, and audience fit."
        )
    else:
        diagnosis = (
            "Revenue fell even though traffic and conversion rate didn't both decline. "
            "Check value per conversion, channel mix, or measurement."
        )

    context = {
        "question_type": "underperform",
        "main_driver": "channel underperformance",
        "weakest_channel": worst["channel"],
        "revenue_change_pct": float(worst["rev_chg"]),
        "sessions_change_pct": float(worst["sess_chg"]),
        "conversion_rate_change_pct": float(worst["cr_chg"]),
        "next_check": diagnosis,
    }

    fallback = f"""
### Most underperforming channel: **{worst['channel']}**

- Revenue change: **{fmt_pct(worst['rev_chg'], 1)}**
- Sessions change: **{fmt_pct(worst['sess_chg'], 1)}**
- Conversion rate change: **{fmt_pct(worst['cr_chg'], 2)}**

{diagnosis}
"""

    answer, llm_audit, used_llm = finalize_llm_answer(context, fallback, use_llm=use_llm)

    evidence = merged[["channel", "rev_chg", "sess_chg", "cr_chg"]].copy()
    evidence = evidence.sort_values("rev_chg")
    evidence.columns = ["Channel", "Revenue Change", "Sessions Change", "Conv Rate Change"]
    evidence["Revenue Change"] = evidence["Revenue Change"].map(lambda x: fmt_pct(x, 1))
    evidence["Sessions Change"] = evidence["Sessions Change"].map(lambda x: fmt_pct(x, 1))
    evidence["Conv Rate Change"] = evidence["Conv Rate Change"].map(lambda x: fmt_pct(x, 2))

    return answer, evidence, llm_audit, used_llm


def compare_analysis(df, question, use_llm=True):
    start, end, _, _ = get_periods(df)
    curr_by_channel, _ = summarize_period(df, start, end)

    requested = extract_channels(question)

    if len(requested) < 2:
        requested = ["Email", "Paid Search"]
        note = "\n> *(Defaulting to Email vs Paid Search — name two channels to compare others)*\n"
    else:
        note = ""

    channel_a, channel_b = requested[:2]
    compare = curr_by_channel[curr_by_channel["channel"].isin([channel_a, channel_b])].copy()

    row_a = compare[compare["channel"] == channel_a].iloc[0]
    row_b = compare[compare["channel"] == channel_b].iloc[0]

    context = {
        "question_type": "compare",
        "main_driver": f"{channel_a} vs {channel_b}",
        "weakest_channel": "",
        "channel_a": channel_a,
        "sessions_a": float(row_a["sessions"]),
        "cr_a": float(row_a["conversion_rate"]),
        "revenue_a": float(row_a["revenue"]),
        "channel_b": channel_b,
        "sessions_b": float(row_b["sessions"]),
        "cr_b": float(row_b["conversion_rate"]),
        "revenue_b": float(row_b["revenue"]),
    }

    fallback = f"""
### {channel_a} vs {channel_b} (last 7 days)
{note}
Here is a side-by-side comparison across traffic, conversions, conversion rate, revenue, spend, and ROAS.
"""

    answer, audit, used_llm = finalize_llm_answer(context, fallback, use_llm=use_llm)

    has_spend = "spend" in compare.columns and "roas" in compare.columns

    base_cols = ["channel", "sessions", "conversions", "conversion_rate", "revenue"]
    base_labels = ["Channel", "Sessions", "Conversions", "Conv Rate", "Revenue"]

    if has_spend:
        evidence = compare[base_cols + ["spend", "roas"]].copy()
        evidence.columns = base_labels + ["Spend", "ROAS"]
        evidence["Spend"] = evidence["Spend"].map(lambda x: f"${x:,.0f}")
        evidence["ROAS"] = evidence["ROAS"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
    else:
        evidence = compare[base_cols].copy()
        evidence.columns = base_labels

    evidence["Conv Rate"] = evidence["Conv Rate"].map(lambda x: f"{x:.1%}")
    evidence["Revenue"] = evidence["Revenue"].map(lambda x: f"${x:,.0f}")

    return answer, evidence, audit, used_llm


def run_analysis(df, question, use_llm=True):
    intent, error = classify_question(question)

    if error:
        return error, None, None, {"used_llm": False, "passed": False, "notes": ["Question routing failed."]}, intent, False

    if intent == "compare":
        answer, evidence, llm_audit, used_llm = compare_analysis(df, question, use_llm=use_llm)
        return answer, evidence, None, llm_audit, intent, used_llm
    elif intent == "diagnose":
        answer, evidence, mix_evidence, llm_audit, used_llm = diagnose_analysis(df, use_llm=use_llm)
        return answer, evidence, mix_evidence, llm_audit, intent, used_llm
    elif intent == "underperform":
        answer, evidence, llm_audit, used_llm = underperform_analysis(df, use_llm=use_llm)
        return answer, evidence, None, llm_audit, intent, used_llm
    else:
        answer, evidence, llm_audit, used_llm = summary_analysis(df, use_llm=use_llm)
        return answer, evidence, None, llm_audit, intent, used_llm




def get_kpis(df):
    start, end, prev_start, prev_end = get_periods(df)
    _, curr_total = summarize_period(df, start, end)
    _, prev_total = summarize_period(df, prev_start, prev_end)

    return {
        "revenue": curr_total["revenue"],
        "revenue_change": pct_change(curr_total["revenue"], prev_total["revenue"]),
        "sessions_change": pct_change(curr_total["sessions"], prev_total["sessions"]),
        "cr_change": curr_total["conversion_rate"] - prev_total["conversion_rate"],
    }


def get_key_insight(df):
    start, end, prev_start, prev_end = get_periods(df)

    curr_by_channel, _ = summarize_period(df, start, end)
    prev_by_channel, _ = summarize_period(df, prev_start, prev_end)

    merged = curr_by_channel.merge(
        prev_by_channel[["channel", "revenue"]],
        on="channel",
        suffixes=("_curr", "_prev")
    )
    merged["rev_delta"] = merged["revenue_curr"] - merged["revenue_prev"]

    weakest = merged.sort_values("rev_delta").iloc[0]
    strongest = merged.sort_values("rev_delta", ascending=False).iloc[0]

    if weakest["rev_delta"] < 0:
        return f"Key Insight: **{weakest['channel']}** has the largest revenue decline vs the prior 7-day period ({format_signed_money(weakest['rev_delta'])})."
    return f"Key Insight: All channels are up vs the prior 7-day period. Strongest: **{strongest['channel']}** ({format_signed_money(strongest['rev_delta'])})."


st.title("GA4 Channel Diagnostics Assistant")
st.write("Ask plain-English questions about channel performance and get structured, evidence-backed answers.")
st.caption("Note: 'Unknown' reflects sessions with missing attribution in the GA4 event export.")

summary_path = Path("results/benchmark_summary_all_thresholds.csv")

if "question" not in st.session_state:
    st.session_state.question = "Why did revenue drop in the last 7 days?"

def set_question(prompt):
    st.session_state.question = prompt

with st.sidebar:
    st.header("Try these prompts")
    prompt_options = [
        "Why did revenue drop in the last 7 days?",
        "Which channel underperformed most?",
        "Compare email and paid search",
        "Compare direct and referral",
        "Compare organic and paid search",
        "Summarize key changes in the last 7 days",
    ]
    for prompt in prompt_options:
        st.button(
            prompt,
            on_click=set_question,
            args=(prompt,),
            key=f"prompt_{prompt}"
        )

df = load_data()
kpis = get_kpis(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue (Last 7d)", f"${kpis['revenue']:,.0f}")
col2.metric("Revenue Change", fmt_pct(kpis["revenue_change"], 1))
col3.metric("Sessions Change", fmt_pct(kpis["sessions_change"], 1))
col4.metric("Conv Rate Change", fmt_pct(kpis["cr_change"], 2))

st.divider()

use_llm = st.toggle("Use LLM-written explanations", value=True)

status = llm_status()
st.caption(
    f"LLM available: {status['available']} | provider: {status['provider']} | model: {status['model']}"
)

with st.form("question_form", clear_on_submit=False):
    question = st.text_input("Ask a question:", key="question")
    submitted = st.form_submit_button("Analyze")

if submitted:
    answer, evidence, mix_evidence, llm_audit, intent, used_llm = run_analysis(df, question, use_llm=use_llm)

    st.markdown(answer)

    if evidence is not None:
        if intent in {"diagnose", "underperform"}:
            st.info(get_key_insight(df))

        st.subheader("Evidence")
        st.dataframe(evidence, use_container_width=True)

    # Show the mix effect breakdown for diagnose questions
    if mix_evidence is not None:
        with st.expander("Channel mix effect"):
            st.caption(
                "Mix effect shows how much revenue changed just because the share of "
                "sessions shifted across channels — even if per-channel conversion rates "
                "and revenue per conversion stayed the same. "
                "A channel with a large negative mix effect pulled the aggregate down "
                "by attracting a smaller share of high-intent traffic."
            )
            st.dataframe(mix_evidence, use_container_width=True)

    with st.expander("LLM audit"):
        if llm_audit is None:
            st.write("No LLM audit available. Either the toggle was off, the model was unavailable, or the response used fallback text.")
        else:
            score = llm_audit.get("score")
            if score is not None:
                st.progress(score, text=f"Faithfulness score: {score:.2f} / 1.00  ({'passed' if llm_audit.get('passed') else 'fallback'})")
            st.json(llm_audit)

    with st.expander("See raw data sample"):
        st.dataframe(df.tail(16), use_container_width=True)

with st.expander("Show evaluation results"):
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        st.dataframe(summary_df, use_container_width=True)
        st.caption(
            "Evaluation reports benchmark performance from a simulated test harness used to validate the diagnosis logic. "
            "The live app experience shown here runs on GA4 ecommerce data processed into daily channel-level metrics."
        )
    else:
        st.info("Run `python3 benchmark.py` to generate evaluation metrics.")

with st.expander("Show anomalies"):
    anomalies = detect_recent_anomalies(df)

    st.caption(f"Anomaly threshold: z = {ANOMALY_Z_THRESHOLD}")
    st.caption(
        "Weekday-aware anomaly scan across sessions, revenue, conversion rate, and revenue per conversion "
        "for the latest 7 days. Each day is compared against prior values from the same weekday "
        "(for example, Saturday vs previous Saturdays)."
    )

    if anomalies.empty:
        st.success("No unusual channel-level movements detected in the latest 7 days.")
    else:
        st.warning(get_top_anomaly_text(df))

    anomaly_table = anomalies.copy()
    anomaly_table["date"] = anomaly_table["date"].dt.strftime("%Y-%m-%d")
    anomaly_table["metric"] = anomaly_table["metric"].str.replace("_", " ").str.title()
    anomaly_table["value"] = anomaly_table.apply(
        lambda row: format_anomaly_value(
            row["metric"].lower().replace(" ", "_"),
            row["value"]
        ),
        axis=1
    )
    anomaly_table["baseline_mean"] = anomaly_table.apply(
        lambda row: format_anomaly_value(
            row["metric"].lower().replace(" ", "_"),
            row["baseline_mean"]
        ),
        axis=1
    )
    anomaly_table["z_score"] = anomaly_table["z_score"].map(lambda x: f"{x:.2f}")
    anomaly_table["history_points"] = anomaly_table["history_points"].astype(int)

    anomaly_table = anomaly_table.rename(columns={
            "date": "Date",
            "channel": "Channel",
            "metric": "Metric",
            "value": "Observed Value",
            "baseline_mean": "Same-Weekday Baseline",
            "weekday": "Weekday",
            "history_points": "History Points",
            "z_score": "Z-Score",
            "direction": "Direction",
    })

    anomaly_table = anomaly_table[
        ["Date", "Channel", "Metric", "Observed Value", "Same-Weekday Baseline", "Weekday", "History Points", "Z-Score", "Direction"]
    ]

    st.dataframe(anomaly_table, use_container_width=True)