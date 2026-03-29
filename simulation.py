from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

CHANNELS = ["Paid Search", "Email", "Social", "Organic"]

CHANNEL_CONFIG = {
    "Paid Search": {
        "sessions_low": 900,
        "sessions_high": 1200,
        "cr_low": 0.035,
        "cr_high": 0.050,
        "rpc_low": 95,
        "rpc_high": 115,
        "spend_low": 1800,
        "spend_high": 2400,
    },
    "Email": {
        "sessions_low": 400,
        "sessions_high": 650,
        "cr_low": 0.070,
        "cr_high": 0.110,
        "rpc_low": 85,
        "rpc_high": 105,
        "spend_low": 150,
        "spend_high": 300,
    },
    "Social": {
        "sessions_low": 500,
        "sessions_high": 750,
        "cr_low": 0.020,
        "cr_high": 0.035,
        "rpc_low": 70,
        "rpc_high": 90,
        "spend_low": 700,
        "spend_high": 1000,
    },
    "Organic": {
        "sessions_low": 700,
        "sessions_high": 950,
        "cr_low": 0.045,
        "cr_high": 0.065,
        "rpc_low": 90,
        "rpc_high": 110,
        "spend_low": 0,
        "spend_high": 0,
    },
}


def recompute_observed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sessions"] = out["sessions"].round().astype(int).clip(lower=1)
    out["conversions"] = np.maximum(
        1,
        np.floor(out["sessions"] * out["conv_rate_true"]).astype(int)
    )
    out["revenue"] = (out["conversions"] * out["rpc_true"]).round(2)
    out["spend"] = out["spend"].round(2)

    out["conversion_rate"] = out["conversions"] / out["sessions"]
    out["revenue_per_conversion"] = np.where(
        out["conversions"] > 0,
        out["revenue"] / out["conversions"],
        np.nan
    )
    out["cpa"] = np.where(
        out["conversions"] > 0,
        out["spend"] / out["conversions"],
        np.nan
    )
    out["roas"] = np.where(
        out["spend"] > 0,
        out["revenue"] / out["spend"],
        np.nan
    )

    return out


def generate_base_data(
    seed: int = 42,
    start_date: str = "2026-02-01",
    end_date: str = "2026-03-15"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    rows = []

    for date in dates:
        for channel in CHANNELS:
            cfg = CHANNEL_CONFIG[channel]

            sessions = rng.integers(cfg["sessions_low"], cfg["sessions_high"] + 1)
            conv_rate = rng.uniform(cfg["cr_low"], cfg["cr_high"])
            rpc = rng.uniform(cfg["rpc_low"], cfg["rpc_high"])
            spend = rng.uniform(cfg["spend_low"], cfg["spend_high"]) if cfg["spend_high"] > 0 else 0.0

            rows.append({
                "date": date,
                "channel": channel,
                "sessions": sessions,
                "conv_rate_true": conv_rate,
                "rpc_true": rpc,
                "spend": spend,
            })

    df = pd.DataFrame(rows)
    return recompute_observed(df)


def inject_scenario(
    df: pd.DataFrame,
    channel: Optional[str],
    metric: Optional[str],
    severity: float,
    start_date: str = "2026-03-09",
    end_date: str = "2026-03-15",
) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()

    if metric is None:
        return out, {
            "channel": None,
            "metric": None,
            "severity": None,
            "scenario_name": "no_shock",
        }

    mask = (
        (out["date"] >= pd.Timestamp(start_date)) &
        (out["date"] <= pd.Timestamp(end_date)) &
        (out["channel"] == channel)
    )

    if metric == "sessions":
        out.loc[mask, "sessions"] = (
            out.loc[mask, "sessions"] * severity
        ).round().astype(int).clip(lower=1)

    elif metric == "conversion_rate":
        out.loc[mask, "conv_rate_true"] = (
            out.loc[mask, "conv_rate_true"] * severity
        ).clip(lower=0.001)

    elif metric == "revenue_per_conversion":
        out.loc[mask, "rpc_true"] = (
            out.loc[mask, "rpc_true"] * severity
        ).clip(lower=1.0)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    out = recompute_observed(out)

    return out, {
        "channel": channel,
        "metric": metric,
        "severity": severity,
        "scenario_name": f"{channel}_{metric}_shock",
    }


def sample_scenario(seed: int) -> Dict:
    rng = np.random.default_rng(seed)

    metric = rng.choice(
        ["sessions", "conversion_rate", "revenue_per_conversion", None],
        p=[0.30, 0.30, 0.30, 0.10]
    )

    if metric is None:
        return {
            "channel": None,
            "metric": None,
            "severity": None,
        }

    channel = rng.choice(CHANNELS)

    if metric == "sessions":
        severity = rng.uniform(0.65, 0.90)
    elif metric == "conversion_rate":
        severity = rng.uniform(0.60, 0.90)
    else:
        severity = rng.uniform(0.70, 0.92)

    return {
        "channel": channel,
        "metric": metric,
        "severity": float(severity),
    }


def generate_random_scenario(seed: int) -> Tuple[pd.DataFrame, Dict]:
    base = generate_base_data(seed=seed)
    scenario = sample_scenario(seed + 1000)

    return inject_scenario(
        df=base,
        channel=scenario["channel"],
        metric=scenario["metric"],
        severity=scenario["severity"] if scenario["severity"] is not None else 1.0,
    )