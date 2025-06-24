import pandas as pd
import numpy as np
import dask.dataframe as dd
from functools import reduce
from typing import Optional, List, Dict
from tqdm import tqdm


####################
# Utility Function #
####################

def prepare_dask_dataframe(
    df: dd.DataFrame,
    datetime_col: str,
    partition_size: str = "500MB",
    target_partitions: int = 50
) -> dd.DataFrame:
    """
    Prepare dask dataframe: sort index, repartition, compute divisions
    """
    # make sure datetime column is valid
    df[datetime_col] = dd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[datetime_col])

    # early exit if dataframe becomes empty after cleaning
    if df.shape[0].compute() == 0:
        print("[INFO] Dataframe is empty after datetime cleaning.")
        return df
        
    # compute min/max datetime from whole dataset
    min_dt, max_dt = dd.compute(
        df[datetime_col].min(),
        df[datetime_col].max()
    )

    # handle edge case if min_dt == max_dt
    if min_dt == max_dt:
        divisions = [min_dt, max_dt]
        print("[INFO] only single timestamp found. no partitioning needed")
        df = df.set_index(datetime_col, sorted=False, compute_divisions=True)
        return df
        
    # generate pre-compute divisions
    divisions = pd.date_range(
        start=min_dt, end=max_dt, periods=target_partitions+1
    ).to_list()
    
    # set index directly with divisions
    df = df.set_index(datetime_col, divisions=divisions, sorted=False)

    # verify divisions exist
    assert df.known_divisions, "Divisions are still unknown after preparation!"
    return df


#####################################
# Core Frequency Calculation (Dask) #
#####################################

def calculate_frequency_dask(
    dataset: dd.DataFrame,
    datetime_col: str,
    key: str,
    groupby: str,
    amount_col: str,
    groupby_type: str = "No",
    groupby_col: Optional[str] = None,
    window: str = "30D",
    na_value: Optional[float] = None,
    out_col: str = "frequency",
):
    before_window = pd.Timedelta(window)

    if groupby_type == "No":
        # apply rolling count using map_overlap for time-based rolling
        meta = {amount_col: 'float64'}
        df_num_trnx = dataset.map_overlap(
            lambda df: df[[amount_col]].rolling(window, closed="left").count(),
            before=before_window,
            after=pd.Timedelta('0D'),
            meta=meta
        ).fillna(na_value)

    else:
        # if grouped rolling, we need apply groupby then map_overlap per group
        def group_rolling(df):
            return (
                df.groupby(groupby_col)
                .rolling(window, closed="left")[amount_col]
                .count()
                .fillna(na_value)
                .reset_index(level=0, drop=True)
            )
            
        meta = {amount_col: 'float64'}
        df_num_trnx = dataset.map_overlap(
            group_rolling,
            before=before_window,
            after=pd.Timedelta('0D'),
            meta=meta
        )
    
    # rename output column
    df_num_trnx = df_num_trnx.rename(columns={amount_col: out_col})

    # join back original keys
    result = dd.merge(
        dataset[[key, groupby]],
        df_num_trnx,
        left_index=True,
        right_index=True
    )

    return result


####################################
# Core Monetary Calculation (Dask) #
####################################

def calculate_monetary_dask(
    dataset: dd.DataFrame,
    datetime_col: str,
    key: str,
    groupby: str,
    amount_col: str,
    groupby_type: str = "No",
    groupby_col: Optional[str] = None,
    window: str = "30D",
    na_value: Optional[float] = None,
    out_col: str = "monetary",
    agg_func: str = "mean",  # parameter: 'mean', 'max', or 'sum'
) -> dd.DataFrame:

    # validate aggregation function
    if agg_func not in ["mean", "max", "sum"]:
        raise ValueError("agg_func must be on of: 'mean', 'max', 'sum'")

    before_window = pd.Timedelta(window)

    if groupby_type == "No":
        # case without additional groupby
        meta = {amount_col: 'float64'}
        df_amt_trnx = dataset.map_overlap(
            lambda df: getattr(df[[amount_col]].rolling(window, closed="left"), agg_func)(),
            before=before_window,
            after=pd.Timedelta('0D'),
            meta=meta
        ).fillna(na_value)

    else:
        def group_rolling(df):
            return (
                df.groupby(groupby_col)
                .rolling(window, closed="left")[amount_col]
                .agg(agg_func)
                .fillna(na_value)
                .reset_index(level=0, drop=True)
            )

        meta = {amount_col: 'float64'}
        df_amt_trnx = dataset.map_overlap(
            group_rolling,
            before=before_window,
            after=pd.Timedelta('0D'),
            meta=meta
        )

    # rename column
    df_amt_trnx = df_amt_trnx.rename(columns={amount_col: out_col})

    # merge back with original column
    result = dd.merge(
        dataset[[key, groupby]],
        df_amt_trnx,
        left_index=True,
        right_index=True
    )

    return result


####################################
# Rolling Feature Generator (Dask) #
####################################

def generate_rolling_features_dask(
    df: dd.DataFrame,
    datetime_col: str,
    key_col: str,
    features_config: List[Dict]
) -> dd.DataFrame:

    # Persist base dataframe once to avoid redundant computation
    df = df.persist()

    feature_dfs = []

    for config in features_config:
        feature_type = config["type"]
        groupby = config["groupby"]
        windows = config["windows"]
        groupby_type = config.get("groupby_type", "No")
        groupby_col = config.get("groupby_col", None)
        na_value = config.get("na_value", 0)
        amount_col = config["amount_col"]

        for window, out_col in windows.items():
            print(f"Processing feature: {feature_type} | Window:{window}")

            # process frequency feature
            if feature_type == "frequency":
                feature_df = calculate_frequency_dask(
                    dataset=df,
                    datetime_col=datetime_col,
                    key=key_col,
                    groupby=groupby,
                    amount_col=amount_col,
                    groupby_type=groupby_type,
                    groupby_col=groupby_col,
                    window=window,
                    na_value=na_value,
                    out_col=out_col,
                )
            # process monetary feature
            elif feature_type == "monetary":
                agg_func = config.get("agg_func", "mean") # default mean
                feature_df = calculate_monetary_dask(
                    dataset=df,
                    datetime_col=datetime_col,
                    key=key_col,
                    groupby=groupby,
                    amount_col=amount_col,
                    groupby_type=groupby_type,
                    groupby_col=groupby_col,
                    window=window,
                    na_value=na_value,
                    out_col=out_col,
                    agg_func=agg_func
                )
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

            # persist after each window to prevent graph fusion
            feature_df = feature_df.persist()

            # keep only merge keys + new feature column
            feature_df = feature_df[[key_col, groupby, datetime_col, out_col]]
            feature_dfs.append(feature_df)

    # merge all features sequentially using reduce logic
    df_final = df
    merge_keys = [key_col, groupby, datetime_col]

    for feature_df in feature_dfs:
        df_final = dd.merge(df_final, feature_df, on=merge_keys, how='left')

    return df_final


def calculate_frequency(
    dataset: pd.DataFrame,
    datetime_col: str,
    key: str,
    groupby: str,
    amount_col: str,
    groupby_type: str = "No",
    groupby_col: Optional[str] = None,
    window: str = "30D",
    na_value: Optional[float] = None,
    out_col: str = "frequency",
) -> pd.DataFrame:
    """
    Calculate the frequency of transactions.

    Parameters:
    - dataset: pd.DataFrame
    - datetime_col: str
    - key: str
    - groupby: str
    - amount_col: str
    - groupby_type: str ('No' or 'Yes')
    - groupby_col: Optional[str] (optional)
    - window: str (rolling window size)
    - na_value: Optional[float] (value to fill NA)
    - out_col: str (output column name)

    Returns:
    - pd.DataFrame with frequency of transactions
    """
    dataset = dataset.sort_values(by=datetime_col, ascending=True)
    if groupby_type == "No":
        df_num_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby(groupby)[amount_col]
            .rolling(window, closed="left")
            .count()
            .fillna(na_value)
            .reset_index()
        )
    else:
        df_num_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby([groupby, groupby_col])[amount_col]
            .rolling(window, closed="left")
            .count()
            .fillna(na_value)
            .reset_index()
        )

    df_num_trnx.rename(columns={amount_col: out_col}, inplace=True)
    df_num_trnx = df_num_trnx.drop_duplicates(
        subset=[groupby, datetime_col], keep="last"
    )
    dataset_TJ = dataset[[key, groupby, datetime_col]]
    join_data = dataset_TJ.merge(df_num_trnx, how="left", on=[groupby, datetime_col])
    return join_data[[key, groupby, datetime_col, out_col]]


def calculate_monetary(
    dataset: pd.DataFrame,
    datetime_col: str,
    key: str,
    groupby: str,
    amount_col: str,
    groupby_type: str = "No",
    groupby_col: Optional[str] = None,
    window: str = "30D",
    na_value: Optional[float] = None,
    out_col: str = "monetary",
    agg_func: str = "mean",  # parameter: 'mean', 'max', or 'sum'
) -> pd.DataFrame:
    """
    Calculate the monetary value of transactions using a rolling window.

    Parameters:
    - dataset: pd.DataFrame
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
    - pd.DataFrame with monetary value of transactions
    """
    dataset = dataset.sort_values(by=datetime_col, ascending=True)

    # Validate aggregation function
    if agg_func not in ["mean", "max", "sum"]:
        raise ValueError("agg_func must be one of: 'mean', 'max', 'sum'")

    if groupby_type == "No":
        df_amt_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby(groupby)[amount_col]
            .rolling(window, closed="left")
            .agg(agg_func)
            .fillna(na_value)
            .reset_index()
        )
    else:
        df_amt_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby([groupby, groupby_col])[amount_col]
            .rolling(window, closed="left")
            .agg(agg_func)
            .fillna(na_value)
            .reset_index()
        )

    df_amt_trnx.rename(columns={amount_col: out_col}, inplace=True)
    df_amt_trnx = df_amt_trnx.drop_duplicates(
        subset=[groupby, datetime_col], keep="last"
    )
    dataset_TJ = dataset[[key, groupby, datetime_col]]
    join_data = dataset_TJ.merge(df_amt_trnx, how="left", on=[groupby, datetime_col])
    return join_data[[key, groupby, datetime_col, out_col]]


def calculate_unique_count(
    dataset: pd.DataFrame,
    datetime_col: str,
    count_col: str,
    groupby: str,
    window: str = "30D",
    na_value: Optional[float] = None,
    out_col: str = "unique_count",
) -> pd.DataFrame:
    """
    Calculate the unique count of transactions.
    Parameters:
    - dataset: pd.DataFrame
    - datetime_col: str
    - count_col: str
    - groupby: str
    - window: str (rolling window size)
    - na_value: Optional[float] (value to fill NA)
    - out_col: str (output column name)
    Returns:
    - pd.DataFrame with unique count of transactions
    """
    # Ensure input is sorted by datetime
    dataset[datetime_col] = pd.to_datetime(dataset[datetime_col])
    dataset = dataset.sort_values(by=[groupby, datetime_col], ascending=True)

    # initiate result list
    results = []
    
    # Compute rolling unique counts per group
    for key, group in dataset.groupby(groupby):
        group = group.set_index(datetime_col)
        rolled = (
            group[count_col]
            .rolling(window=window, closed="left", min_periods=1)
            .apply(lambda x: pd.Series(x).nunique(), raw=False)
        )
        group[out_col] = rolled.values
        results.append(group.reset_index())

    # combine back all groups
    result_df = pd.concat(results, axis=0, ignore_index=True)
    
    return result_df


def calculate_time_differences(
    df: pd.DataFrame,
    datetime_col: str,
    groupby_col: str,
    time_window: List[str],
    config: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Calculate time differences between transactions.
    
    Parameters:
    - df: pd.DataFrame
    - datetime_col: str
    - groupby_col: str
    - time_window: List[str]
    - config: Dict[str, List[str]]
    
    Returns:
    - pd.DataFrame with time differences and rolling averages
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=[groupby_col, datetime_col])

    # process each config item
    for new_col, groupby_cols in tqdm(
        config.items(), desc="Processing time diffs"
    ):
        df = df.sort_values(by=groupby_cols + [datetime_col])

        if len(groupby_cols) == 1:
            # Simple time difference
            df[new_col] = (
                df.groupby(groupby_cols)[datetime_col]
                .diff()
                .dt.total_seconds() / 60
            )
        else:
            # Conditional time difference: only when the last column changes
            primary_group = groupby_cols[0]
            change_col = groupby_cols[-1]

            # compute shifted previous time and value
            prev_time = df.groupby(primary_group)[datetime_col].shift(1)
            prev_val = df.groupby(primary_group)[change_col].shift(1)
            changed = df[change_col] != prev_val

            # compute time diff where change occured
            time_diff = (
                (df[datetime_col] - prev_time)
                .dt.total_seconds() / 60
            )
            df[new_col] = time_diff.where(changed, np.nan)

    # Rolling averages
    for window in tqdm(time_window, desc="Calculating rolling averages"):
        temp_df = df[
            [datetime_col, groupby_col] + list(config.keys())
        ].copy()

        # sort again if needed (minimal slice)
        temp_df = temp_df.sort_values(
            by=[groupby_col, datetime_col]
        ).set_index(datetime_col)

        for new_col in config.keys():
            rolled = (
                temp_df.groupby(groupby_col)[new_col]
                .rolling(window=window)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"avg_{new_col}_L{window}"] = rolled.values

    return df


# Generate rolling features Pandas:
def generate_rolling_features(
    df: pd.DataFrame, datetime_col: str, key_col: str, features_config: List[Dict]
) -> pd.DataFrame:
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

    for config in tqdm(features_config, desc="Feature Config Progress"):
        feature_type = config["type"]
        groupby = config["groupby"]
        windows = config["windows"]
        groupby_type = config.get("groupby_type", "No")
        groupby_col = config.get("groupby_col", None)
        na_value = config.get("na_value", 0)

        for window, out_col in tqdm(windows.items(),
                                    desc=f"{feature_type} Windows Progress",
                                    leave=False):
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
        df_merged = pd.merge(
            df_merged,
            feat_df[[*merge_keys, feat_df.columns[-1]]],
            on=merge_keys,
            how="left",
        )

    return df_merged


# example usage
# df = generate_rolling_features(df, datetime_col="transaction_datetime", key_col="transaction_id", features_config=features_config)
