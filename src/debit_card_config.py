##################################
### DYNAMIC HIGH RISK CATEGORY ###
##################################
dynamic_high_risk_config = {
    "groupby": "MCC",
    "fraud_label_col": "Confirmed",
    "agg_func": "sum",
    "months_period": 1,
    "top_n": 10
}


######################################
### DURATION SINCE FIRST TIME TRNX ###
######################################
duration_since_first_trnx_config = [
    {
        # Average duration time between first and current trnx grouped by MCC
        "type": "first_trnx_duration",
        "groupby": "Debit_No",
        "groupby_col": "MCC",
        "agg_func": "mean",
        "windows": {
            "900s": "AvgTimeFirstTxnToCurrentMCCL15minin",
            "30D": "AvgTimeFirstTxnToCurrentMCCL30D",
        },
    },
    {
        # Average duration time between first and current trnx grouped by MCC
        "type": "first_trnx_duration",
        "groupby": "Debit_No",
        "groupby_col": "Country Code",
        "agg_func": "mean",
        "windows": {
            "900s": "AvgTimeFirstTxnToCurrentCountryCodeL15minin",
            "30D": "AvgTimeFirstTxnToCurrentCountryCodeL30D",
        },
    },
]


#######################
### TIME DIFFERENCE ###
#######################
time_shift_config = {
    "TxnTimeDifference": ["Debit_No"],
}

time_windows = [
    "900s",  # 15 mins
    "30D",  # 30 days
]


#################
### FREQUENCY ###
#################
freq_config = [
    {
        # Transaction count grouped by Card_no/Debit_No
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "No",
        "groupby_col": None,
        "windows": {
            "900s": "TxnCountL15min",
            "30D": "TxnCountL30D",
        },
    },
    {
        # Transaction count to each MCC grouped by Card_no/Debit_No
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "MCC",
        "windows": {
            "900s": "TxnCountToMCCL15min",
            "30D": "TxnCountToMCCL30D",
        },
    },
    {
        # Transaction Count Same to Country Code
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "windows": {
            "900s": "TxnCountToCountryCodeL15min",
            "30D": "TxnCountToCountryCodeL30D",
        },
    },
    {
        # Transaction Count grouped by Debit_No and POSMode
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "POSMode",
        "windows": {
            "900s": "TxnCountToPOSModeL15minin",
            "30D": "TxnCountToPOSModeL30D",
        },
    },
]


################
### MONETARY ###
################
monetary_config_1 = [
    {
        # Average Transaction Amount grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "No",
        "groupby_col": None,
        "agg_func": "mean",  # need to be defined, if not the default value is `mean`
        "windows": {
            "900s": "AvgAmtL15min",
            "30D": "AvgAmtL30D",
        },
    },
    {
        # Maximum Transaction Amount grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "No",
        "groupby_col": None,
        "agg_func": "max",
        "windows": {
            "900s": "MaxAmtL15min",
            "30D": "MaxAmtL30D",
        },
    },
    {
        # Sum Transaction Amount grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "No",
        "groupby_col": None,
        "agg_func": "sum",
        "windows": {
            "900s": "SumAmtL15min",
            "30D": "SumAmtL30D",
        },
    },
]

monetary_config_2 = [
    {
        # Average Transaction Amount to MCC grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "MCC",
        "agg_func": "mean",
        "windows": {
            "900s": "AvgAmtToMCCL15min",
            "30D": "AvgAmtToMCCL30D",
        },
    },
    {
        # Maximum Transaction Amount to MCC grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "MCC",
        "agg_func": "max",
        "windows": {
            "900s": "MaxAmtToMCCL15min",
            "30D": "MaxAmtToMCCL30D",
        },
    },
    {
        # Sum Transaction Amount to MCC grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "MCC",
        "agg_func": "sum",
        "windows": {
            "900s": "SumAmtToMCCL15min",
            "30D": "SumAmtToMCCL30D",
        },
    },
]

monetary_config_3 = [
    {
        # Average Transaction Amount to POSMode grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "POSMode",
        "agg_func": "mean",
        "windows": {
            "900s": "AvgAmtToPOSModeL15min",
            "30D": "AvgAmtToPOSModeL30D",
        },
    },
    {
        # Maximum Transaction Amount to POSMode grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "POSMode",
        "agg_func": "max",
        "windows": {
            "900s": "MaxAmtToPOSModeL15min",
            "30D": "MaxAmtToPOSModeL30D",
        },
    },
    {
        # Sum Transaction Amount to POSMode grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "POSMode",
        "agg_func": "sum",
        "windows": {
            "900s": "SumAmtToPOSModeL15min",
            "30D": "SumAmtToPOSModeL30D",
        },
    },
]


####################
### UNIQUE COUNT ###
####################
unique_count_config = [
    {
        # Count unique (distinct) Card_no/Debit_No grouped by MCC
        "type": "unique",
        "groupby": "MCC",
        "count_col": "Debit_No Num",
        "windows": {
            "900s": "CntUniqueCardNoByMCCL15min",
            "30D": "CntUniqueCardNoByMCCL30D",
        },
    },
]