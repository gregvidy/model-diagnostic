import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
    eps: float = 1e-10
) -> float:
    """
    Calculate Information Value (IV) for a single feature
    """
    # Work on a copy to avoid modifying input dataframe
    df_copy = df[[feature, target]].copy()
    
    try:
        df_copy["bin"] = pd.qcut(df_copy[feature], q=bins, duplicates='drop')
    except ValueError:
        df_copy["bin"] = pd.cut(df_copy[feature], bins=bins)
        
    grouped = df_copy.groupby("bin")[target].agg(["count", "sum"])
    grouped["non_event"] = grouped["count"] - grouped["sum"]

    event_total = grouped["sum"].sum()
    non_event_total = grouped["non_event"].sum()

    grouped["event_rate"] = grouped["sum"] / event_total
    grouped["non_event_rate"] = grouped["non_event"] / non_event_total

    grouped["woe"] = np.log(
        (grouped["event_rate"] + eps) / (grouped["non_event_rate"] + eps)
    )
    grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped["woe"]

    return grouped["iv"].sum()


def calculate_iv_for_all_features(df, features, target, bins=10):
    iv_dict = {}
    for feature in tqdm(features, desc="Calculate IV Progress..."):
        try:
            iv = calculate_iv(df, feature, target, bins)
            iv_dict[feature] = iv
        except Exception as e:
            print(f"Failed for feature {feature}: {e}")
            iv_dict[feature] = np.nan

    iv_df = pd.DataFrame(list(iv_dict.items()), columns=["Feature", "IV"])
    iv_df = iv_df.sort_values(by="IV", ascending=False)

    return iv_df


def calculate_psi(expected, actual, buckets=10, eps=1e-10):
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    expected_bins = pd.cut(expected, breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, breakpoints, include_lowest=True)

    expected_dist = expected_bins.value_counts() / len(expected_bins)
    actual_dist = actual_bins.value_counts() / len(actual_bins)

    psi_df = pd.DataFrame({
        "expected": expected_dist,
        "actual": actual_dist
    }).fillna(eps)
    
    # Correct PSI formula: (actual - expected) * ln(actual/expected)
    psi_df["psi"] = (psi_df["actual"] - psi_df["expected"]) * np.log(
        (psi_df["actual"] + eps) / (psi_df["expected"] + eps)
    )

    return psi_df["psi"].sum()


def calculate_psi_for_all_features(train_df, test_df, features, buckets=10):
    psi_dict = {}
    for feature in tqdm(features, desc="Calculate PSI Progress..."):
        try:
            psi = calculate_psi(train_df[feature], test_df[feature], buckets)
            psi_dict[feature] = psi
        except Exception as e:
            print(f"Failed for feature {feature}: {e}")
            psi_dict[feature] = np.nan

    psi_df = pd.DataFrame(list(psi_dict.items()), columns=["Feature", "PSI"])
    psi_df = psi_df.sort_values(by="PSI", ascending=False)

    return psi_df