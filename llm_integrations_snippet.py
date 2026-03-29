# HOW TO PLUG IN THE LLM
# Add this import at the top of analytics_assistant.py:
#
#   from llm import generate_summary
#
# Then update each analysis function to build a data dict and call generate_summary.
# If the LLM call fails, it returns None and we fall back to the old f-string.
# The analysis logic and evidence tables don't change at all.


# ---- example: summary_analysis (only the end changes) ----

def summary_analysis(df):
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

    # pack the numbers into a dict and let the LLM write the words
    llm_data = {
        "rev_chg": rev_chg,
        "sess_chg": sess_chg,
        "conv_chg": conv_chg,
        "worst_channel": worst["channel"],
        "worst_chg": worst["rev_chg_pct"],
        "best_channel": best["channel"],
        "best_chg": best["rev_chg_pct"],
    }
    llm_answer = generate_summary("summary", llm_data)

    # fall back to the hardcoded answer if the LLM call failed
    fallback = f"""
### Summary
Revenue changed **{rev_chg:.1%}** vs the prior 7-day period.

- Sessions: **{sess_chg:.1%}**
- Conversions: **{conv_chg:.1%}**
- Biggest decline: **{worst['channel']}** ({worst['rev_chg_pct']:.1%})
- Biggest gain: **{best['channel']}** ({best['rev_chg_pct']:.1%})

### What to look at next
Start with the weakest channel — check whether the issue is traffic volume or conversion rate.
"""

    answer = llm_answer if llm_answer else fallback

    evidence = merged[["channel", "revenue_curr", "revenue_prev", "rev_chg_pct", "sessions_curr", "sessions_prev"]].copy()
    evidence.columns = ["Channel", "Revenue (Last 7d)", "Revenue (Prior 7d)", "Revenue Change", "Sessions (Last 7d)", "Sessions (Prior 7d)"]
    evidence["Revenue (Last 7d)"] = evidence["Revenue (Last 7d)"].map(lambda x: f"${x:,.0f}")
    evidence["Revenue (Prior 7d)"] = evidence["Revenue (Prior 7d)"].map(lambda x: f"${x:,.0f}")
    evidence["Revenue Change"] = evidence["Revenue Change"].map(lambda x: f"{x:.1%}")

    return answer, evidence


# ---- example: diagnose_analysis (only the end changes) ----

def diagnose_analysis(df):
    start, end, prev_start, prev_end = get_periods(df)

    curr_by_channel, curr_total = summarize_period(df, start, end)
    prev_by_channel, prev_total = summarize_period(df, prev_start, prev_end)

    rev_chg = pct_change(curr_total["revenue"], prev_total["revenue"])
    sess_chg = pct_change(curr_total["sessions"], prev_total["sessions"])
    cr_chg = curr_total["conversion_rate"] - prev_total["conversion_rate"]

    merged = curr_by_channel.merge(
        prev_by_channel[["channel", "revenue", "sessions", "conversion_rate"]],
        on="channel",
        suffixes=("_curr", "_prev")
    )
    merged["rev_delta"] = merged["revenue_curr"] - merged["revenue_prev"]
    merged["sess_chg"] = (merged["sessions_curr"] - merged["sessions_prev"]) / merged["sessions_prev"]
    merged["cr_chg"] = merged["conversion_rate_curr"] - merged["conversion_rate_prev"]

    biggest_driver = merged.sort_values("rev_delta").iloc[0]
    main_issue = "traffic volume" if abs(sess_chg) > abs(cr_chg * 100) else "conversion efficiency"

    llm_data = {
        "rev_chg": rev_chg,
        "biggest_driver": biggest_driver["channel"],
        "rev_delta": biggest_driver["rev_delta"],
        "sess_chg": sess_chg,
        "cr_chg": cr_chg,
        "main_issue": main_issue,
    }
    llm_answer = generate_summary("diagnose", llm_data)

    fallback = f"""
### Why did it drop?
Overall revenue changed **{rev_chg:.1%}** vs the prior 7 days.

The biggest single driver was **{biggest_driver['channel']}**, which lost **${abs(biggest_driver['rev_delta']):,.0f}** in revenue.

The main issue looks like **{main_issue}**:
- Sessions changed **{sess_chg:.1%}**
- Conversion rate changed **{cr_chg:.2%}**

### What to check
Look at {biggest_driver['channel']} first — review spend pacing, campaign delivery, and traffic quality.
"""

    answer = llm_answer if llm_answer else fallback

    evidence = merged[["channel", "rev_delta", "sess_chg", "cr_chg"]].copy()
    evidence = evidence.sort_values("rev_delta")
    evidence.columns = ["Channel", "Revenue Delta ($)", "Sessions Change", "Conv Rate Change"]
    evidence["Revenue Delta ($)"] = evidence["Revenue Delta ($)"].map(lambda x: f"${x:,.0f}")
    evidence["Sessions Change"] = evidence["Sessions Change"].map(lambda x: f"{x:.1%}")
    evidence["Conv Rate Change"] = evidence["Conv Rate Change"].map(lambda x: f"{x:.2%}")

    return answer, evidence


# ---- example: compare_analysis (only the end changes) ----

def compare_analysis(df, question):
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

    llm_data = {
        "channel_a": channel_a,
        "sessions_a": row_a["sessions"],
        "cr_a": row_a["conversion_rate"],
        "revenue_a": row_a["revenue"],
        "roas_a": f"{row_a['roas']:.2f}" if not pd.isna(row_a["roas"]) else "N/A",
        "channel_b": channel_b,
        "sessions_b": row_b["sessions"],
        "cr_b": row_b["conversion_rate"],
        "revenue_b": row_b["revenue"],
        "roas_b": f"{row_b['roas']:.2f}" if not pd.isna(row_b["roas"]) else "N/A",
    }
    llm_answer = generate_summary("compare", llm_data)

    fallback = f"""
### {channel_a} vs {channel_b} (last 7 days)
{note}
Here's a side-by-side of **{channel_a}** and **{channel_b}** across traffic, conversions, revenue, spend, and ROAS.
"""

    answer = llm_answer if llm_answer else fallback

    evidence = compare[["channel", "sessions", "conversions", "conversion_rate", "revenue", "spend", "roas"]].copy()
    evidence.columns = ["Channel", "Sessions", "Conversions", "Conv Rate", "Revenue", "Spend", "ROAS"]
    evidence["Conv Rate"] = evidence["Conv Rate"].map(lambda x: f"{x:.1%}")
    evidence["Revenue"] = evidence["Revenue"].map(lambda x: f"${x:,.0f}")
    evidence["Spend"] = evidence["Spend"].map(lambda x: f"${x:,.0f}")
    evidence["ROAS"] = evidence["ROAS"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")

    return answer, evidence


# ---- underperform_analysis follows the same pattern ----
# build llm_data with: worst_channel, rev_chg, sess_chg, cr_chg
# call generate_summary("underperform", llm_data)
# fall back to the f-string if None