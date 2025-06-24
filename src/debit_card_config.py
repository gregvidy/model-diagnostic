#######################
### TIME DIFFERENCE ###
#######################
time_shift_config = {
    "time_diff": ["Debit_No"],
    # "time_diff_before_mcc": ["Debit_No", "MCC"],
    # "time_diff_before_mcc_cat": ["Debit_No", "MCC Category"],
    # "time_diff_before_country_code": ["Debit_No", "Country Code"],
    # "time_diff_before_currency_code": ["Debit_No", "Transaction Currency Code"],
    # "time_diff_before_card_acceptor_name_cat": ["Debit_No", "Cat Card Acceptor Name"],
    # "time_diff_before_card_acceptor_reg_code": ["Debit_No", "Card Acceptor Region Code"],
    # "time_diff_before_card_acceptor_country_code": [
    #     "Debit_No",
    #     "Card Acceptor Country Code",
    # ],
}

time_windows = [
    "900S",  # 15 mins
    "1H",  # 1 hour
    "1D",  # 1 day
    "7D",  # 7 days
    "14D",  # 14 days
    "30D",  # 30 days
    "90D",  # 90 days
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
            "900S": "TxnCount_L15M",
            "1H": "TxnCount_L1H",
            "1D": "TxnCount_L1D",
            "7D": "TxnCount_L7D",
            "14D": "TxnCount_L14D",
            "30D": "TxnCount_L30D",
            "90D": "TxnCount_L90D",
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
            "900S": "TxnCount_to_MCC_L15M",
            "1H": "TxnCount_to_MCC_L1H",
            "1D": "TxnCount_to_MCC_L1D",
            "7D": "TxnCount_to_MCC_L7D",
            "14D": "TxnCount_to_MCC_L14D",
            "30D": "TxnCount_to_MCC_L30D",
            "90D": "TxnCount_to_MCC_L90D",
        },
    },
    {
        # Transaction count to each Country Code grouped by Card_no/Debit_No
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "windows": {
            "900S": "TxnCount_to_CountryCode_L15M",
            "1H": "TxnCount_to_CountryCode_L1H",
            "1D": "TxnCount_to_CountryCode_L1D",
            "7D": "TxnCount_to_CountryCode_L7D",
            "14D": "TxnCount_to_CountryCode_L14D",
            "30D": "TxnCount_to_CountryCode_L30D",
            "90D": "TxnCount_to_CountryCode_L90D",
        },
    },
    {
        # Transaction count to each MCC Trnx Category Code grouped by Card_no/Debit_No
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "MCC Trnx Category Code",
        "windows": {
            "900S": "TxnCount_to_MCC_catcode_L15M",
            "1H": "TxnCount_to_MCC_catcode_L1H",
            "1D": "TxnCount_to_MCC_catcode_L1D",
            "7D": "TxnCount_to_MCC_catcode_L7D",
            "14D": "TxnCount_to_MCC_catcode_L14D",
            "30D": "TxnCount_to_MCC_catcode_L30D",
            "90D": "TxnCount_to_MCC_catcode_L90D",
        },
    },
    {
        # Transaction count to each MCC Category grouped by Card_no/Debit_No
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "MCC Category",
        "windows": {
            "900S": "TxnCount_to_MCC_cat_L15M",
            "1H": "TxnCount_to_MCC_cat_L1H",
            "1D": "TxnCount_to_MCC_cat_L1D",
            "7D": "TxnCount_to_MCC_cat_L7D",
            "14D": "TxnCount_to_MCC_cat_L14D",
            "30D": "TxnCount_to_MCC_cat_L30D",
            "90D": "TxnCount_to_MCC_cat_L90D",
        },
    },
    {
        # Transaction Count Same to Category Cat Card Acceptor Name
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "Cat Card Acceptor Name",
        "windows": {
            "900S": "TxnCount_to_cardAcceptor_cat_L15M",
            "1H": "TxnCount_to_cardAcceptor_cat_L1H",
            "1D": "TxnCount_to_cardAcceptor_cat_L1D",
            "7D": "TxnCount_to_cardAcceptor_cat_L7D",
            "14D": "TxnCount_to_cardAcceptor_cat_L14D",
            "30D": "TxnCount_to_cardAcceptor_cat_L30D",
            "90D": "TxnCount_to_cardAcceptor_cat_L90D",
        },
    },
    {
        # Transaction Count Same to Card Acceptor Country Code
        "type": "frequency",
        "groupby": "Debit_No",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "Card Acceptor Country Code",
        "windows": {
            "900S": "TxnCount_to_cardAcceptor_country_L15M",
            "1H": "TxnCount_to_cardAcceptor_country_L1H",
            "1D": "TxnCount_to_cardAcceptor_country_L1D",
            "7D": "TxnCount_to_cardAcceptor_country_L7D",
            "14D": "TxnCount_to_cardAcceptor_country_L14D",
            "30D": "TxnCount_to_cardAcceptor_country_L30D",
            "90D": "TxnCount_to_cardAcceptor_country_L90D",
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
            "900S": "Avg_Amt_L15M",
            "1H": "Avg_Amt_L1H",
            "1D": "Avg_Amt_L1D",
            "7D": "Avg_Amt_L7D",
            "14D": "Avg_Amt_L14D",
            "30D": "Avg_Amt_L30D",
            "90D": "Avg_Amt_L90D",
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
            "900S": "Max_Amt_L15M",
            "1H": "Max_Amt_L1H",
            "1D": "Max_Amt_L1D",
            "7D": "Max_Amt_L7D",
            "14D": "Max_Amt_L14D",
            "30D": "Max_Amt_L30D",
            "90D": "Max_Amt_L90D",
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
            "14D": "Sum_Amt_L14D",
            "30D": "Sum_Amt_L30D",
            "90D": "Sum_Amt_L90D",
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
            "900S": "Avg_Amt_to_MCC_L15M",
            "1H": "Avg_Amt_to_MCC_L1H",
            "1D": "Avg_Amt_to_MCC_L1D",
            "7D": "Avg_Amt_to_MCC_L7D",
            "14D": "Avg_Amt_to_MCC_L14D",
            "30D": "Avg_Amt_to_MCC_L30D",
            "90D": "Avg_Amt_to_MCC_L90D",
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
            "900S": "Max_Amt_to_MCC_L15M",
            "1H": "Max_Amt_to_MCC_L1H",
            "1D": "Max_Amt_to_MCC_L1D",
            "7D": "Max_Amt_to_MCC_L7D",
            "14D": "Max_Amt_to_MCC_L14D",
            "30D": "Max_Amt_to_MCC_L30D",
            "90D": "Max_Amt_to_MCC_L90D",
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
            "900S": "Sum_Amt_to_MCC_L15M",
            "1H": "Sum_Amt_to_MCC_L1H",
            "1D": "Sum_Amt_to_MCC_L1D",
            "7D": "Sum_Amt_to_MCC_L7D",
            "14D": "Sum_Amt_to_MCC_L14D",
            "30D": "Sum_Amt_to_MCC_L30D",
            "90D": "Sum_Amt_to_MCC_L90D",
        },
    },
]

monetary_config_3 = [
    {
        # Average Transaction Amount to Country Code grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_Amt_to_CountryCode_L15M",
            "1H": "Avg_Amt_to_CountryCode_L1H",
            "1D": "Avg_Amt_to_CountryCode_L1D",
            "7D": "Avg_Amt_to_CountryCode_L7D",
            "14D": "Avg_Amt_to_CountryCode_L14D",
            "30D": "Avg_Amt_to_CountryCode_L30D",
            "90D": "Avg_Amt_to_CountryCode_L90D",
        },
    },
    {
        # Maximum Transaction Amount to Country Code grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "agg_func": "max",
        "windows": {
            "900S": "Max_Amt_to_CountryCode_L15M",
            "1H": "Max_Amt_to_CountryCode_L1H",
            "1D": "Max_Amt_to_CountryCode_L1D",
            "7D": "Max_Amt_to_CountryCode_L7D",
            "14D": "Max_Amt_to_CountryCode_L14D",
            "30D": "Max_Amt_to_CountryCode_L30D",
            "90D": "Max_Amt_to_CountryCode_L90D",
        },
    },
    {
        # Sum Transaction Amount to Country Code grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "agg_func": "sum",
        "windows": {
            "900S": "Sum_Amt_to_CountryCode_L15M",
            "1H": "Sum_Amt_to_CountryCode_L1H",
            "1D": "Sum_Amt_to_CountryCode_L1D",
            "7D": "Sum_Amt_to_CountryCode_L7D",
            "14D": "Sum_Amt_to_CountryCode_L14D",
            "30D": "Sum_Amt_to_CountryCode_L30D",
            "90D": "Sum_Amt_to_CountryCode_L90D",
        },
    },
]

monetary_config_4 = [
    {
        # Average Transaction Amount to MCC Category grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "MCC Category",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_Amt_to_MCC_cat_L15M",
            "1H": "Avg_Amt_to_MCC_cat_L1H",
            "1D": "Avg_Amt_to_MCC_cat_L1D",
            "7D": "Avg_Amt_to_MCC_cat_L7D",
            "14D": "Avg_Amt_to_MCC_cat_L14D",
            "30D": "Avg_Amt_to_MCC_cat_L30D",
            "90D": "Avg_Amt_to_MCC_cat_L90D",
        },
    },
    {
        # Maximum Transaction Amount to MCC Category grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "MCC Category",
        "agg_func": "max",
        "windows": {
            "900S": "Max_Amt_to_MCC_cat_L15M",
            "1H": "Max_Amt_to_MCC_cat_L1H",
            "1D": "Max_Amt_to_MCC_cat_L1D",
            "7D": "Max_Amt_to_MCC_cat_L7D",
            "14D": "Max_Amt_to_MCC_cat_L14D",
            "30D": "Max_Amt_to_MCC_cat_L30D",
            "90D": "Max_Amt_to_MCC_cat_L90D",
        },
    },
    {
        # Sum Transaction Amount to MCC Category grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "MCC Category",
        "agg_func": "sum",
        "windows": {
            "900S": "Sum_Amt_to_MCC_cat_L15M",
            "1H": "Sum_Amt_to_MCC_cat_L1H",
            "1D": "Sum_Amt_to_MCC_cat_L1D",
            "7D": "Sum_Amt_to_MCC_cat_L7D",
            "14D": "Sum_Amt_to_MCC_cat_L14D",
            "30D": "Sum_Amt_to_MCC_cat_L30D",
            "90D": "Sum_Amt_to_MCC_cat_L90D",
        },
    },
]

monetary_config_5 = [
    {
        # Average Transaction Amount to Cat Card Acceptor Name grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Cat Card Acceptor Name",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_Amt_to_cardAcceptor_cat_L15M",
            "1H": "Avg_Amt_to_cardAcceptor_cat_L1H",
            "1D": "Avg_Amt_to_cardAcceptor_cat_L1D",
            "7D": "Avg_Amt_to_cardAcceptor_cat_L7D",
            "14D": "Avg_Amt_to_cardAcceptor_cat_L14D",
            "30D": "Avg_Amt_to_cardAcceptor_cat_L30D",
            "90D": "Avg_Amt_to_cardAcceptor_cat_L90D",
        },
    },
    {
        # Maximum Transaction Amount to Cat Card Acceptor Name grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Cat Card Acceptor Name",
        "agg_func": "max",
        "windows": {
            "900S": "Max_Amt_to_cardAcceptor_cat_L15M",
            "1H": "Max_Amt_to_cardAcceptor_cat_L1H",
            "1D": "Max_Amt_to_cardAcceptor_cat_L1D",
            "7D": "Max_Amt_to_cardAcceptor_cat_L7D",
            "14D": "Max_Amt_to_cardAcceptor_cat_L14D",
            "30D": "Max_Amt_to_cardAcceptor_cat_L30D",
            "90D": "Max_Amt_to_cardAcceptor_cat_L90D",
        },
    },
    {
        # Sum Transaction Amount to MCC Category grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Cat Card Acceptor Name",
        "agg_func": "sum",
        "windows": {
            "900S": "Sum_Amt_to_cardAcceptor_cat_L15M",
            "1H": "Sum_Amt_to_cardAcceptor_cat_L1H",
            "1D": "Sum_Amt_to_cardAcceptor_cat_L1D",
            "7D": "Sum_Amt_to_cardAcceptor_cat_L7D",
            "14D": "Sum_Amt_to_cardAcceptor_cat_L14D",
            "30D": "Sum_Amt_to_cardAcceptor_cat_L30D",
            "90D": "Sum_Amt_to_cardAcceptor_cat_L90D",
        },
    },
]

monetary_config_6 = [
    {
        # Average Transaction Amount to Card Acceptor Country Code grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Card Acceptor Country Code",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_Amt_to_cardAcceptor_country_L15M",
            "1H": "Avg_Amt_to_cardAcceptor_country_L1H",
            "1D": "Avg_Amt_to_cardAcceptor_country_L1D",
            "7D": "Avg_Amt_to_cardAcceptor_country_L7D",
            "14D": "Avg_Amt_to_cardAcceptor_country_L14D",
            "30D": "Avg_Amt_to_cardAcceptor_country_L30D",
            "90D": "Avg_Amt_to_cardAcceptor_country_L90D",
        },
    },
    {
        # Maximum Transaction Amount to Card Acceptor Country Code grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Card Acceptor Country Code",
        "agg_func": "max",
        "windows": {
            "900S": "Max_Amt_to_cardAcceptor_country_L15M",
            "1H": "Max_Amt_to_cardAcceptor_country_L1H",
            "1D": "Max_Amt_to_cardAcceptor_country_L1D",
            "7D": "Max_Amt_to_cardAcceptor_country_L7D",
            "14D": "Max_Amt_to_cardAcceptor_country_L14D",
            "30D": "Max_Amt_to_cardAcceptor_country_L30D",
            "90D": "Max_Amt_to_cardAcceptor_country_L90D",
        },
    },
    {
        # Sum Transaction Amount to Card Acceptor Country Code grouped by Debit_No
        "type": "monetary",
        "groupby": "Debit_No",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Card Acceptor Country Code",
        "agg_func": "sum",
        "windows": {
            "900S": "Sum_Amt_to_cardAcceptor_country_L15M",
            "1H": "Sum_Amt_to_cardAcceptor_country_L1H",
            "1D": "Sum_Amt_to_cardAcceptor_country_L1D",
            "7D": "Sum_Amt_to_cardAcceptor_country_L7D",
            "14D": "Sum_Amt_to_cardAcceptor_country_L14D",
            "30D": "Sum_Amt_to_cardAcceptor_country_L30D",
            "90D": "Sum_Amt_to_cardAcceptor_country_L90D",
        },
    },
]


####################
### UNIQUE COUNT ###
####################
unique_count_config = [
    {
        # Count unique (distinct) MCC grouped by Debit_No
        "type": "unique",
        "groupby": "Debit_No",
        "count_col": "MCC Num",
        "windows": {
            "900S": "CntUnique_MCC_by_CardNo_L15M",
            "1H": "CntUnique_MCC_by_CardNo_L1H",
            "1D": "CntUnique_MCC_by_CardNo_L1D",
            "7D": "CntUnique_MCC_by_CardNo_L7D",
            "14D": "CntUnique_MCC_by_CardNo_L14D",
            "30D": "CntUnique_MCC_by_CardNo_L30D",
            "90D": "CntUnique_MCC_by_CardNo_L90D",
        },
    },
    {
        # Count unique (distinct) Card_no/Debit_No grouped by MCC
        "type": "unique",
        "groupby": "MCC",
        "count_col": "Debit_No Num",
        "windows": {
            "900S": "CntUnique_CardNo_by_MCC_L15M",
            "1H": "CntUnique_CardNo_by_MCC_L1H",
            "1D": "CntUnique_CardNo_by_MCC_L1D",
            "7D": "CntUnique_CardNo_by_MCC_L7D",
            "14D": "CntUnique_CardNo_by_MCC_L14D",
            "30D": "CntUnique_CardNo_by_MCC_L30D",
            "90D": "CntUnique_CardNo_by_MCC_L90D",
        },
    },
    {
        # Count unique (distinct) Card_no/Debit_No grouped by Card Acceptor Name Cat
        "type": "unique",
        "groupby": "Cat Card Acceptor Name",
        "count_col": "Debit_No Num",
        "windows": {
            "900S": "CntUnique_CardNo_by_cardAcceptor_cat_L15M",
            "1H": "CntUnique_CardNo_by_cardAcceptor_cat_L1H",
            "1D": "CntUnique_CardNo_by_cardAcceptor_cat_L1D",
            "7D": "CntUnique_CardNo_by_cardAcceptor_cat_L7D",
            "14D": "CntUnique_CardNo_by_cardAcceptor_cat_L14D",
            "30D": "CntUnique_CardNo_by_cardAcceptor_cat_L30D",
            "90D": "CntUnique_CardNo_by_cardAcceptor_cat_L90D",
        },
    },
    {
        # Count unique (distinct) Card_no/Debit_No grouped by Card Acceptor Country Code
        "type": "unique",
        "groupby": "Card Acceptor Country Code",
        "count_col": "Debit_No Num",
        "windows": {
            "900S": "CntUnique_CardNo_by_cardAcceptor_country_L15M",
            "1H": "CntUnique_CardNo_by_cardAcceptor_country_L1H",
            "1D": "CntUnique_CardNo_by_cardAcceptor_country_L1D",
            "7D": "CntUnique_CardNo_by_cardAcceptor_country_L7D",
            "14D": "CntUnique_CardNo_by_cardAcceptor_country_L14D",
            "30D": "CntUnique_CardNo_by_cardAcceptor_country_L30D",
            "90D": "CntUnique_CardNo_by_cardAcceptor_country_L90D",
        },
    },
]
