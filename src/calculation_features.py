import pandas as pd
import numpy as np
from typing import Optional, List, Dict


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
) -> pd.DataFrame:
    """
    Calculate the monetary value of transactions.

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
    - pd.DataFrame with monetary value of transactions
    """
    dataset = dataset.sort_values(by=datetime_col, ascending=True)
    if groupby_type == "No":
        df_amt_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby(groupby)[amount_col]
            .rolling(window, closed="left")
            .mean()
            .fillna(na_value)
            .reset_index()
        )
    else:
        df_amt_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby([groupby, groupby_col])[amount_col]
            .rolling(window, closed="left")
            .mean()
            .fillna(na_value)
            .reset_index()
        )

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
    dataset = dataset.sort_values(by=datetime_col, ascending=True)
    df_num = (
        dataset.set_index(datetime_col)
        .sort_index()
        .groupby(groupby)[count_col]
        .rolling(window, closed="left", min_periods=1)
        .apply(lambda x: np.unique(x[~np.isnan(x)]).size, raw=True)
        .fillna(na_value)
        .reset_index()
    )
    df_num = df_num.drop_duplicates(subset=[groupby, datetime_col], keep="last")
    df_output = dataset.merge(df_num, on=[groupby, datetime_col], how="left")
    return df_output[[groupby, datetime_col, out_col]]


def calculate_time_differences(
    df: pd.DataFrame,
    datetime_col: str,
    shift_columns: Dict[str, List[str]],
    time_diff_columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Calculate time differences between transactions.

    Parameters:
    - df: pd.DataFrame
    - datetime_col: str
    - shift_columns: Dict[str, List[str]] (columns to shift and their groupby columns)
    - time_diff_columns: Dict[str, str] (time difference columns and their corresponding shift columns)

    Returns:
    - pd.DataFrame with time differences
    """
    df = df.sort_values(by=[datetime_col], ascending=True)

    for col, groupby_cols in shift_columns.items():
        df[col] = df.groupby(by=groupby_cols)[datetime_col].shift(1)

    for col, shift_col in time_diff_columns.items():
        df[col] = np.where(
            df[shift_col].isnull(),
            -1,
            (df[datetime_col] - df[shift_col]).dt.total_seconds() / 60,
        )

    return df


def calculate_monetary_max(
    dataset: pd.DataFrame,
    datetime_col: str,
    key: str,
    groupby: str,
    amount_col: str,
    groupby_type: str = "No",
    groupby_col: Optional[str] = None,
    window: str = "30D",
    na_value: Optional[float] = None,
    out_col: str = "monetary_max",
) -> pd.DataFrame:
    """
    Calculate the maximum monetary value of transactions.

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
    - pd.DataFrame with maximum monetary value of transactions
    """
    dataset = dataset.sort_values(by=datetime_col, ascending=True)
    if groupby_type == "No":
        df_amt_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby(groupby)[amount_col]
            .rolling(window, closed="left")
            .max()
            .fillna(na_value)
            .reset_index()
        )
    else:
        df_amt_trnx = (
            dataset.set_index(datetime_col)
            .sort_index()
            .groupby([groupby, groupby_col])[amount_col]
            .rolling(window, closed="left")
            .max()
            .fillna(na_value)
            .reset_index()
        )

    df_amt_trnx = df_amt_trnx.drop_duplicates(
        subset=[groupby, datetime_col], keep="last"
    )
    dataset_TJ = dataset[[key, groupby, datetime_col]]
    join_data = dataset_TJ.merge(df_amt_trnx, how="left", on=[groupby, datetime_col])
    return join_data[[key, groupby, datetime_col, out_col]]


# Example usage:
# df = pd.read_csv('your_data.csv')
# frequency_df = calculate_frequency(df, 'date', 'id', 'group', 'amount')
# monetary_df = calculate_monetary(df, 'date', 'id', 'group', 'amount')
# unique_count_df = calculate_unique_count(df, 'date', 'transaction_id', 'group')
# time_diff_df = calculate_time_differences(df)
# monetary_max_df = calculate_monetary_max(df, 'date', 'id', 'group', 'amount')
