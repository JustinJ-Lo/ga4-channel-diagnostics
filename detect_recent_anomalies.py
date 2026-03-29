import pandas as pd

ANOMALY_Z_THRESHOLD = 2.0

def get_periods(df):
    end = df["date"].max()
    start = end - pd.Timedelta(days=6)
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=6)
    return start, end, prev_start, prev_end


def detect_recent_anomalies(
    df,
    metrics=("sessions", "revenue", "conversion_rate", "revenue_per_conversion"),
    same_weekday_lookback=4,
    min_history_points=3,
    z_thresh=ANOMALY_Z_THRESHOLD
):
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

            for _, row in grp.iterrows():
                current_date = row["date"]
                current_weekday = row["weekday"]
                current_value = row["value"]

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
    out = out.sort_values("z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    return out[[
        "date", "channel", "metric", "value", "weekday",
        "baseline_mean", "baseline_std", "history_points",
        "z_score", "direction"
    ]]