import polars as pl
from typing import Optional, List, Dict


def calculate_frequency(
    dataset: pl.DataFrame,
    datetime_col: str,
    key: str,
    groupby: str,
    amount_col: str,
    groupby_type: str = "No",
    groupby_col: Optional[str] = None,
    window: str = "30d",
    na_value: Optional[float] = None,
    out_col: str = "frequency",
) -> pl.DataFrame:
    """
    Calculate the monetary value of transactions using a rolling window.

    Parameters:
    - dataset: pl.DataFrame
    - datetime_col: str
    - key: str
    - groupby: str
    - amount_col: str
    - groupby_type: str ('No' or 'Yes')
    - groupby_col: Optional[str] (optional)
    - window: str (rolling window size)
    - na_value: Optional[float] (value to fill NA)
    - out_col: str (output column name)
    - agg_func: str ('mean', 'max', or 'sum')

    Returns:
    - pl.DataFrame with monetary value of transactions
    """
    # Ensure datetime column is in datetime format and sorted
    df = dataset.with_columns(
        pl.col(datetime_col).str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S")
    ).sort(datetime_col)

    # Define grouping keys
    if groupby_type == "No":
        grouping_keys = [groupby]
    else:
        if groupby_col is None:
            raise ValueError("groupby_col must be provided when groupby_type is 'Yes'")
        grouping_keys = [groupby, groupby_col]

    # Apply dynamic grouping with frequency count
    rolled = (
        df.groupby_dynamic(
            index_column=datetime_col,
            every=window,
            period=window,
            by=grouping_keys,
            closed="left"
        )
        .agg(pl.count().alias(out_col))
    )

    # Fill nulls if needed
    if na_value is not None:
        rolled = rolled.with_columns(pl.col(out_col).fill_null(na_value))

    # Join back to original dataset
    result = df.join(rolled, on=grouping_keys + [datetime_col], how="left")

    return result.select([key, groupby, datetime_col, out_col])


def calculate_monetary(
    dataset: pl.DataFrame,
    datetime_col: str,
    key: str,
    groupby: str,
    amount_col: str,
    groupby_type: str = "No",
    groupby_col: Optional[str] = None,
    window: str = "30d",
    na_value: Optional[float] = None,
    out_col: str = "monetary",
    agg_func: str = "mean",  # 'mean', 'max', or 'sum'
) -> pl.DataFrame:
    """
    Calculate the monetary value of transactions using a rolling window.

    Parameters:
    - dataset: pl.DataFrame
    - datetime_col: str
    - key: str
    - groupby: str
    - amount_col: str
    - groupby_type: str ('No' or 'Yes')
    - groupby_col: Optional[str] (optional)
    - window: str (rolling window size)
    - na_value: Optional[float] (value to fill NA)
    - out_col: str (output column name)
    - agg_func: str ('mean', 'max', or 'sum')

    Returns:
    - pl.DataFrame with monetary value of transactions
    """
    # Parse datetime
    df = dataset.with_columns(
        pl.col(datetime_col).str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S")
    ).sort(by=datetime_col)

    # Validate aggregation function
    if agg_func not in ["mean", "max", "sum"]:
        raise ValueError("agg_func must be one of: 'mean', 'max', 'sum'")

    # Convert window string to duration
    duration_kwargs = {}
    if window.endswith("d"):
        duration_kwargs["days"] = int(window[:-1])
    elif window.endswith("h"):
        duration_kwargs["hours"] = int(window[:-1])
    elif window.endswith("m"):
        duration_kwargs["minutes"] = int(window[:-1])
    else:
        raise ValueError("Unsupported window format. Use 'd', 'h', or 'm'.")

    # Define aggregation function
    def aggregate_window(group):
        return group.with_columns([
            pl.col(datetime_col).map_elements(
                lambda current_time, group=group: getattr(
                    group.filter(
                        (pl.col(datetime_col) < current_time) &
                        (pl.col(datetime_col) >= current_time - pl.duration(**duration_kwargs))
                    )[amount_col],
                    agg_func
                )(),
                return_dtype=pl.Float64
            ).alias(out_col)
        ])

    # Apply groupby logic
    if groupby_type == "No":
        result = df.groupby(groupby).apply(aggregate_window)
    else:
        if groupby_col is None:
            raise ValueError("groupby_col must be provided when groupby_type is 'Yes'")
        result = df.groupby([groupby, groupby_col]).apply(aggregate_window)

    # Fill NA if needed
    if na_value is not None:
        result = result.with_columns(
            pl.col(out_col).fill_null(na_value)
        )

    # Select final columns
    return result.select([key, groupby, datetime_col, out_col])


def calculate_unique_count(
    dataset: pl.DataFrame,
    datetime_col: str,
    count_col: str,
    groupby: str,
    window: str = "30d",
    na_value: Optional[float] = None,
    out_col: str = "unique_count",
) -> pl.DataFrame:
    """
    Calculate the unique count of transactions.

    Parameters:
    - dataset: pl.DataFrame
    - datetime_col: str
    - count_col: str
    - groupby: str
    - window: str (rolling window size)
    - na_value: Optional[float] (value to fill NA)
    - out_col: str (output column name)

    Returns:
    - pl.DataFrame with unique count of transactions
    """
    # Ensure datetime column is in datetime format
    df = dataset.with_columns(
        pl.col(datetime_col).str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S")
    )

    # Sort by datetime
    df = df.sort(by=datetime_col)

    # Convert window to duration (e.g., "30d" -> pl.duration(days=30))
    duration_kwargs = {}
    if window.endswith("d"):
        duration_kwargs["days"] = int(window[:-1])
    elif window.endswith("h"):
        duration_kwargs["hours"] = int(window[:-1])
    elif window.endswith("m"):
        duration_kwargs["minutes"] = int(window[:-1])
    else:
        raise ValueError("Unsupported window format. Use 'd', 'h', or 'm'.")

    # Apply rolling unique count manually
    result = (
        df.groupby(groupby)
        .apply(
            lambda group: group.with_columns([
                pl.col(datetime_col).map_elements(
                    lambda current_time, group=group: group.filter(
                        (pl.col(datetime_col) < current_time) &
                        (pl.col(datetime_col) >= current_time - pl.duration(**duration_kwargs))
                    )[count_col].unique().len(),
                    return_dtype=pl.Int32
                ).alias(out_col)
            ])
        )
    )

    # Fill NA if specified
    if na_value is not None:
        result = result.with_columns(
            pl.col(out_col).fill_null(na_value)
        )

    return result



def calculate_time_differences(
    df: pl.DataFrame,
    datetime_col: str,
    groupby_col: str,
    time_window: List[str],
    config: Dict[str, List[str]],
) -> pl.DataFrame:
    """
    Calculate time differences between transactions.

    Parameters:
    - df: pl.DataFrame
    - datetime_col: str
    - groupby_col: str
    - time_window: List[str]
    - config: Dict[str, List[str]]

    Returns:
    - pl.DataFrame with time differences and rolling averages
    """
    df = df.with_columns(pl.col(datetime_col).str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S"))

    for new_col, groupby_cols in config.items():
        df = df.sort(by=groupby_cols + [datetime_col])

        if len(groupby_cols) == 1:
            # Simple time difference
            df = df.with_columns(
                (pl.col(datetime_col).diff().dt.seconds() / 60).over(groupby_cols).alias(new_col)
            )
        else:
            # Conditional time difference
            primary_group = groupby_cols[0]
            change_col = groupby_cols[-1]

            df = df.with_columns([
                pl.col(datetime_col).shift().over(primary_group).alias("prev_time"),
                pl.col(change_col).shift().over(primary_group).alias("prev_val"),
            ])
            df = df.with_columns([
                ((pl.col(datetime_col) - pl.col("prev_time")).dt.seconds() / 60)
                .filter(pl.col(change_col) != pl.col("prev_val"))
                .alias(new_col)
            ])
            df = df.drop(["prev_time", "prev_val"])

        # Rolling averages
        for window in time_window:
            window_int = int(window[:-1])  # assumes format like '3h', '5m'
            df = df.with_columns(
                pl.col(new_col)
                .rolling_mean(window_size=window_int)
                .over(groupby_col)
                .alias(f"avg_{new_col}_L{window}")
            )

    return df


def generate_rolling_features(
    df: pl.DataFrame, datetime_col: str, key_col: str, features_config: List[Dict]
) -> pl.DataFrame:
    """
    Generate rolling window features (frequency, unique count, monetary) based on configuration.

    Parameters:
    - df: Input DataFrame
    - datetime_col: Name of the datetime column
    - key_col: Unique transaction identifier column
    - features_config: List of feature configurations

    Returns:
    - DataFrame with all rolling features merged
    """
    all_feature_dfs = []

    for config in features_config:
        feature_type = config["type"]
        groupby = config["groupby"]
        windows = config["windows"]
        groupby_type = config.get("groupby_type", "No")
        groupby_col = config.get("groupby_col", None)
        na_value = config.get("na_value", 0)

        for window, out_col in windows.items():
            if feature_type == "frequency":
                feature_df = calculate_frequency(
                    dataset=df,
                    datetime_col=datetime_col,
                    key=key_col,
                    groupby=groupby,
                    amount_col=config["amount_col"],
                    groupby_type=groupby_type,
                    groupby_col=groupby_col,
                    window=window,
                    na_value=na_value,
                    out_col=out_col,
                )
            elif feature_type == "unique":
                feature_df = calculate_unique_count(
                    dataset=df,
                    datetime_col=datetime_col,
                    count_col=config["count_col"],
                    groupby=groupby,
                    window=window,
                    na_value=na_value,
                    out_col=out_col,
                )
            elif feature_type == "monetary":
                agg_func = config.get("agg_func", "mean")  # Default to mean
                feature_df = calculate_monetary(
                    dataset=df,
                    datetime_col=datetime_col,
                    key=key_col,
                    groupby=groupby,
                    amount_col=config["amount_col"],
                    groupby_type=groupby_type,
                    groupby_col=groupby_col,
                    window=window,
                    na_value=na_value,
                    out_col=out_col,
                    agg_func=agg_func,
                )
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

            all_feature_dfs.append((feature_df, [key_col, groupby, datetime_col]))

    # Merge all features with original df using appropriate keys
    df_merged = df
    for feat_df, merge_keys in all_feature_dfs:
        df_merged = df_merged.join(
            feat_df,
            left_on=merge_keys,
            right_on=merge_keys,
            how="left"
        )

    return df_merged

