from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from simulation import generate_random_scenario


def get_periods(df: pd.DataFrame):
    end = df["date"].max()
    start = end - pd.Timedelta(days=6)
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=6)
    return start, end, prev_start, prev_end


def summarize_period(df: pd.DataFrame, start_date, end_date):
    period = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    by_channel = period.groupby("channel", as_index=False).agg(
        sessions=("sessions", "sum"),
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
        spend=("spend", "sum"),
    )

    by_channel["conversion_rate"] = by_channel["conversions"] / by_channel["sessions"]
    by_channel["revenue_per_conversion"] = np.where(
        by_channel["conversions"] > 0,
        by_channel["revenue"] / by_channel["conversions"],
        np.nan,
    )
    by_channel["cpa"] = np.where(
        by_channel["conversions"] > 0,
        by_channel["spend"] / by_channel["conversions"],
        np.nan,
    )
    by_channel["roas"] = np.where(
        by_channel["spend"] > 0,
        by_channel["revenue"] / by_channel["spend"],
        np.nan,
    )

    total = {
        "channel": "Total",
        "sessions": by_channel["sessions"].sum(),
        "conversions": by_channel["conversions"].sum(),
        "revenue": by_channel["revenue"].sum(),
        "spend": by_channel["spend"].sum(),
    }
    total["conversion_rate"] = total["conversions"] / total["sessions"]
    total["revenue_per_conversion"] = (
        total["revenue"] / total["conversions"] if total["conversions"] > 0 else np.nan
    )
    total["cpa"] = total["spend"] / total["conversions"] if total["conversions"] > 0 else np.nan
    total["roas"] = total["revenue"] / total["spend"] if total["spend"] > 0 else np.nan

    return by_channel, total


def detect_recent_anomalies(
    df: pd.DataFrame,
    metrics=("sessions", "revenue", "conversion_rate", "revenue_per_conversion"),
    same_weekday_lookback: int = 4,
    min_history_points: int = 3,
    z_thresh: float = 2.0,
) -> pd.DataFrame:
    """
    Weekday-aware anomaly detection.

    For each channel, metric, and date in the latest 7 days:
    - find prior observations from the same weekday (e.g. compare Monday to past Mondays)
    - compute a z-score against that weekday-specific baseline

    This replaces the old plain rolling window, which was biased
    because weekends and weekdays have structurally different traffic levels.
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

            for _, row in grp.iterrows():
                current_date = row["date"]
                current_weekday = row["weekday"]
                current_value = row["value"]

                if current_date < start or current_date > end:
                    continue

                # Only compare this date to prior observations from the same weekday
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
            "z_score", "direction",
        ])

    out = pd.concat(all_results, ignore_index=True)
    out = out.sort_values("z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    return out[[
        "date", "channel", "metric", "value",
        "z_score", "direction",
    ]]


def diagnose_root_cause(df: pd.DataFrame) -> dict:
    start, end, prev_start, prev_end = get_periods(df)

    curr_by_channel, _ = summarize_period(df, start, end)
    prev_by_channel, _ = summarize_period(df, prev_start, prev_end)

    merged = curr_by_channel.merge(
        prev_by_channel[
            ["channel", "sessions", "revenue", "conversion_rate", "revenue_per_conversion"]
        ],
        on="channel",
        suffixes=("_curr", "_prev"),
    )

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
        merged["traffic_effect"] +
        merged["conversion_effect"] +
        merged["value_effect"]
    )

    worst = merged.sort_values("rev_delta").iloc[0]

    effect_to_metric = {
        "traffic_effect": "sessions",
        "conversion_effect": "conversion_rate",
        "value_effect": "revenue_per_conversion",
    }

    worst_effect_name = min(
        ["traffic_effect", "conversion_effect", "value_effect"],
        key=lambda col: worst[col]
    )

    return {
        "pred_channel": worst["channel"],
        "pred_metric": effect_to_metric[worst_effect_name],
        "pred_rev_delta": worst["rev_delta"],
    }


def evaluate(n_scenarios: int = 500, z_thresh: float = 2.0):
    rows = []

    for seed in range(n_scenarios):
        df, truth = generate_random_scenario(seed)

        anomalies = detect_recent_anomalies(df, z_thresh=z_thresh)
        diagnosis = diagnose_root_cause(df)

        if anomalies.empty:
            anom_channel = None
            anom_metric = None
            anomaly_flagged = False
        else:
            channel_filtered = anomalies[
                anomalies["channel"] == diagnosis["pred_channel"]
            ].copy()

            if channel_filtered.empty:
                anom_channel = None
                anom_metric = None
                anomaly_flagged = False
            else:
                top = channel_filtered.iloc[0]
                anom_channel = top["channel"]
                anom_metric = top["metric"]
                anomaly_flagged = True

        truth_channel = truth["channel"]
        truth_metric = truth["metric"]

        rows.append({
            "seed": seed,
            "scenario_name": truth["scenario_name"],
            "truth_channel": truth_channel,
            "truth_metric": truth_metric,
            "anomaly_flagged": anomaly_flagged,
            "anom_channel": anom_channel,
            "anom_metric": anom_metric,
            "diag_channel": diagnosis["pred_channel"],
            "diag_metric": diagnosis["pred_metric"],
            "anom_channel_correct": anom_channel == truth_channel,
            "anom_metric_correct": anom_metric == truth_metric,
            "anom_exact_correct": (
                anom_channel == truth_channel
                and anom_metric == truth_metric
            ),
            "diag_channel_correct": diagnosis["pred_channel"] == truth_channel,
            "diag_metric_correct": diagnosis["pred_metric"] == truth_metric,
            "diag_exact_correct": (
                diagnosis["pred_channel"] == truth_channel
                and diagnosis["pred_metric"] == truth_metric
            ),
        })

    results = pd.DataFrame(rows)

    shock = results[results["truth_metric"].notna()].copy()
    control = results[results["truth_metric"].isna()].copy()

    summary = pd.DataFrame([{
        "n_scenarios": n_scenarios,
        "shock_scenarios": len(shock),
        "control_scenarios": len(control),
        "z_threshold": z_thresh,
        "anomaly_channel_accuracy": shock["anom_channel_correct"].mean() if len(shock) > 0 else np.nan,
        "anomaly_metric_accuracy": shock["anom_metric_correct"].mean() if len(shock) > 0 else np.nan,
        "anomaly_exact_accuracy": shock["anom_exact_correct"].mean() if len(shock) > 0 else np.nan,
        "anomaly_false_positive_rate": control["anomaly_flagged"].mean() if len(control) > 0 else np.nan,
        "diagnosis_channel_accuracy": shock["diag_channel_correct"].mean() if len(shock) > 0 else np.nan,
        "diagnosis_metric_accuracy": shock["diag_metric_correct"].mean() if len(shock) > 0 else np.nan,
        "diagnosis_exact_accuracy": shock["diag_exact_correct"].mean() if len(shock) > 0 else np.nan,
    }])

    return results, summary


if __name__ == "__main__":
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    summaries = []

    for z in [2.0, 2.5, 3.0, 3.5]:
        results, summary = evaluate(n_scenarios=500, z_thresh=z)

        print(f"\nz={z} -> rows={len(results)}, shocks={summary.loc[0, 'shock_scenarios']}, controls={summary.loc[0, 'control_scenarios']}")

        results.to_csv(
            out_dir / f"benchmark_detail_z{str(z).replace('.', '_')}.csv",
            index=False
        )
        summaries.append(summary)

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all.to_csv(out_dir / "benchmark_summary_all_thresholds.csv", index=False)

    print("\nSaved:")
    print(f"  {out_dir / 'benchmark_summary_all_thresholds.csv'}")
    print("\nSummary:\n")
    print(summary_all.to_string(index=False))