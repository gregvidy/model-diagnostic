features_config = [
    {
        "type": "frequency",
        "groupby": "from_account_no",
        "amount_col": "transaction_id",
        "groupby_type": "No",
        "groupby_col": None,
        "windows": {"1H": "TxnCount_L1H", "1D": "TxnCount_L1D"},
    },
    {
        "type": "unique",
        "groupby": "from_account_no",
        "count_col": "to_account_no_num",
        "windows": {"1H": "UniqueToAcc_L1H", "1D": "UniqueToAcc_L1D"},
    },
    {
        "type": "monetary",
        "groupby": "from_account_no",
        "amount_col": "amount",
        "groupby_type": "Yes",
        "groupby_col": "transaction_type",
        "windows": {"1H": "AvgAmt_SameType_L1H", "1D": "AvgAmt_SameType_L1D"},
    },
    {
        "type": "monetary_max",
        "groupby": "from_account_no",
        "amount_col": "amount",
        "groupby_type": "Yes",
        "groupby_col": "transaction_type",
        "windows": {"1H": "MaxAmt_SameType_L1H", "1D": "MaxAmt_SameType_L1D"},
    },
]
