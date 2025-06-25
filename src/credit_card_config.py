#######################
### TIME DIFFERENCE ###
#######################
time_shift_config = {
    "time_diff": ["PANNumber"],
    # "time_diff_before_mcc": ["PANNumber", "MCC"],
    # "time_diff_before_mcc_cat": ["PANNumber", "MCC Category"],
    # "time_diff_before_country_code": ["PANNumber", "Country Code"],
    # "time_diff_before_currency_code": ["PANNumber", "Currency Code"],
    # "time_diff_before_card_acceptor_name_cat": ["PANNumber", "Cat Card Acceptor Name"],
    # "time_diff_before_card_acceptor_reg_code": ["PANNumber", "Country Code"],
    # "time_diff_before_card_acceptor_country_code": [
    #     "PANNumber",
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
        # Transaction count grouped by Card_no/PANNumber
        "type": "frequency",
        "groupby": "PANNumber",
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
        # Transaction count to each MCC grouped by Card_no/PANNumber
        "type": "frequency",
        "groupby": "PANNumber",
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
        # Transaction count to each MCC Details grouped by Card_no/PANNumber
        "type": "frequency",
        "groupby": "PANNumber",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "MCC Details",
        "windows": {
            "900S": "TxnCount_to_MCC_details_L15M",
            "1H": "TxnCount_to_MCC_details_L1H",
            "1D": "TxnCount_to_MCC_details_L1D",
            "7D": "TxnCount_to_MCC_details_L7D",
            "14D": "TxnCount_to_MCC_details_L14D",
            "30D": "TxnCount_to_MCC_details_L30D",
            "90D": "TxnCount_to_MCC_details_L90D",
        },
    },
    {
        # Transaction count to each MCC Trnx Category Code grouped by Card_no/PANNumber
        "type": "frequency",
        "groupby": "PANNumber",
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
        # Transaction count to each MCC Category grouped by Card_no/PANNumber
        "type": "frequency",
        "groupby": "PANNumber",
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
        "groupby": "PANNumber",
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
        # Transaction Count Same to Country Code
        "type": "frequency",
        "groupby": "PANNumber",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "windows": {
            "900S": "TxnCount_to_countryCode_L15M",
            "1H": "TxnCount_to_countryCode_L1H",
            "1D": "TxnCount_to_countryCode_L1D",
            "7D": "TxnCount_to_countryCode_L7D",
            "14D": "TxnCount_to_countryCode_L14D",
            "30D": "TxnCount_to_countryCode_L30D",
            "90D": "TxnCount_to_countryCode_L90D",
        },
    },
    {
        # Transaction Count Same to Card Acceptor Country Code
        "type": "frequency",
        "groupby": "PANNumber",
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

freq_config_pos_mode = [
    {
        # Transaction Count grouped by PANNumber and POSMode
        "type": "frequency",
        "groupby": "PANNumber",
        "amount_col": "Transaction Serial No",
        "groupby_type": "Yes",
        "groupby_col": "POS Entry Mode",
        "windows": {
            "900S": "TxnCount_to_POSMode_L15M",
            "1H": "TxnCount_to_POSMode_L1H",
            "1D": "TxnCount_to_POSMode_L1D",
            "7D": "TxnCount_to_POSMode_L7D",
            "14D": "TxnCount_to_POSMode_L14D",
            "30D": "TxnCount_to_POSMode_L30D",
            "90D": "TxnCount_to_POSMode_L90D",
        },
    },
]


################
### MONETARY ###
################
monetary_config_1 = [
    {
        # Average Transaction Amount grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Maximum Transaction Amount grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Sum Transaction Amount grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
        "amount_col": "Transaction Amount",
        "groupby_type": "No",
        "groupby_col": None,
        "agg_func": "sum",
        "windows": {
            "900S": "Sum_Amt_L15M",
            "1H": "Sum_Amt_L1H",
            "1D": "Sum_Amt_L1D",
            "7D": "Sum_Amt_L7D",
            "14D": "Sum_Amt_L14D",
            "30D": "Sum_Amt_L30D",
            "90D": "Sum_Amt_L90D",
        },
    },
]

monetary_config_2 = [
    {
        # Average Transaction Amount to MCC grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Maximum Transaction Amount to MCC grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Sum Transaction Amount to MCC grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Average Transaction Amount to MCC Category grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Maximum Transaction Amount to MCC Category grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Sum Transaction Amount to MCC Category grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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

monetary_config_4 = [
    {
        # Average Transaction Amount to MCC Category grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Maximum Transaction Amount to MCC Category grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Sum Transaction Amount to MCC Category grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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

monetary_config_5 = [
    {
        # Average Transaction Amount to Country Code grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_Amt_to_countryCode_L15M",
            "1H": "Avg_Amt_to_countryCode_L1H",
            "1D": "Avg_Amt_to_countryCode_L1D",
            "7D": "Avg_Amt_to_countryCode_L7D",
            "14D": "Avg_Amt_to_countryCode_L14D",
            "30D": "Avg_Amt_to_countryCode_L30D",
            "90D": "Avg_Amt_to_countryCode_L90D",
        },
    },
    {
        # Maximum Transaction Amount to Country Code grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "agg_func": "max",
        "windows": {
            "900S": "Max_Amt_to_countryCode_L15M",
            "1H": "Max_Amt_to_countryCode_L1H",
            "1D": "Max_Amt_to_countryCode_L1D",
            "7D": "Max_Amt_to_countryCode_L7D",
            "14D": "Max_Amt_to_countryCode_L14D",
            "30D": "Max_Amt_to_countryCode_L30D",
            "90D": "Max_Amt_to_countryCode_L90D",
        },
    },
    {
        # Sum Transaction Amount to Country Code grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "Country Code",
        "agg_func": "sum",
        "windows": {
            "900S": "Sum_Amt_to_countryCode_L15M",
            "1H": "Sum_Amt_to_countryCode_L1H",
            "1D": "Sum_Amt_to_countryCode_L1D",
            "7D": "Sum_Amt_to_countryCode_L7D",
            "14D": "Sum_Amt_to_countryCode_L14D",
            "30D": "Sum_Amt_to_countryCode_L30D",
            "90D": "Sum_Amt_to_countryCode_L90D",
        },
    },
]

monetary_config_6 = [
    {
        # Average Transaction Amount to Card Acceptor Country Code grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Maximum Transaction Amount to Card Acceptor Country Code grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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
        # Sum Transaction Amount to Card Acceptor Country Code grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
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

monetary_config_pos_mode = [
    {
        # Average Transaction Amount to POSMode grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "POS Entry Mode",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_Amt_to_POSMode_L15M",
            "1H": "Avg_Amt_to_POSMode_L1H",
            "1D": "Avg_Amt_to_POSMode_L1D",
            "7D": "Avg_Amt_to_POSMode_L7D",
            "14D": "Avg_Amt_to_POSMode_L14D",
            "30D": "Avg_Amt_to_POSMode_L30D",
            "90D": "Avg_Amt_to_POSMode_L90D",
        },
    },
    {
        # Maximum Transaction Amount to POSMode grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "POS Entry Mode",
        "agg_func": "max",
        "windows": {
            "900S": "Max_Amt_to_POSMode_L15M",
            "1H": "Max_Amt_to_POSMode_L1H",
            "1D": "Max_Amt_to_POSMode_L1D",
            "7D": "Max_Amt_to_POSMode_L7D",
            "14D": "Max_Amt_to_POSMode_L14D",
            "30D": "Max_Amt_to_POSMode_L30D",
            "90D": "Max_Amt_to_POSMode_L90D",
        },
    },
    {
        # Sum Transaction Amount to POSMode grouped by PANNumber
        "type": "monetary",
        "groupby": "PANNumber",
        "amount_col": "Transaction Amount",
        "groupby_type": "Yes",
        "groupby_col": "POS Entry Mode",
        "agg_func": "sum",
        "windows": {
            "900S": "Sum_Amt_to_POSMode_L15M",
            "1H": "Sum_Amt_to_POSMode_L1H",
            "1D": "Sum_Amt_to_POSMode_L1D",
            "7D": "Sum_Amt_to_POSMode_L7D",
            "14D": "Sum_Amt_to_POSMode_L14D",
            "30D": "Sum_Amt_to_POSMode_L30D",
            "90D": "Sum_Amt_to_POSMode_L90D",
        },
    },
]


####################
### UNIQUE COUNT ###
####################
unique_count_config = [
    {
        # Count unique (distinct) MCC grouped by PANNumber
        "type": "unique",
        "groupby": "PANNumber",
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
        # Count unique (distinct) Card_no/PANNumber grouped by MCC
        "type": "unique",
        "groupby": "MCC",
        "count_col": "PANNumber Num",
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
        # Count unique (distinct) Card_no/PANNumber grouped by Card Acceptor Name Cat
        "type": "unique",
        "groupby": "Cat Card Acceptor Name",
        "count_col": "PANNumber Num",
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
        # Count unique (distinct) Card_no/PANNumber grouped by Country Code
        "type": "unique",
        "groupby": "Currency Code",
        "count_col": "PANNumber Num",
        "windows": {
            "900S": "CntUnique_CardNo_by_currencyCode_L15M",
            "1H": "CntUnique_CardNo_by_currencyCode_L1H",
            "1D": "CntUnique_CardNo_by_currencyCode_L1D",
            "7D": "CntUnique_CardNo_by_currencyCode_L7D",
            "14D": "CntUnique_CardNo_by_currencyCode_L14D",
            "30D": "CntUnique_CardNo_by_currencyCode_L30D",
            "90D": "CntUnique_CardNo_by_currencyCode_L90D",
        },
    },
    {
        # Count unique (distinct) Card_no/PANNumber grouped by Card Acceptor Country Code
        "type": "unique",
        "groupby": "Card Acceptor Country Code",
        "count_col": "PANNumber Num",
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


################################
### FIRST TIME TRNX DURATION ###
################################
first_time_trnx_duration_config = [
    {
        # Average unique (distinct) MCC grouped by PANNumber
        "type": "first_trnx_duration",
        "groupby": "PANNumber",
        "groupby_col": "MCC",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_DurationSince_FirstTrnx_to_Current_MCC_L15M",
            "1H": "Avg_DurationSince_FirstTrnx_to_Current_MCC_L1H",
            "1D": "Avg_DurationSince_FirstTrnx_to_Current_MCC_L1D",
            "7D": "Avg_DurationSince_FirstTrnx_to_Current_MCC_L7D",
            "14D": "Avg_DurationSince_FirstTrnx_to_Current_MCC_L14D",
            "30D": "Avg_DurationSince_FirstTrnx_to_Current_MCC_L30D",
            "90D": "Avg_DurationSince_FirstTrnx_to_Current_MCC_L90D",
        },
    },
    {
        # Count unique (distinct) Card_no/PANNumber grouped by MCC
        "type": "first_trnx_duration",
        "groupby": "PANNumber",
        "groupby_col": "CountryCode",
        "agg_func": "mean",
        "windows": {
            "900S": "Avg_DurationSince_FirstTrnx_to_Current_CountryCode_L15M",
            "1H": "Avg_DurationSince_FirstTrnx_to_Current_CountryCode_L1H",
            "1D": "Avg_DurationSince_FirstTrnx_to_Current_CountryCode_L1D",
            "7D": "Avg_DurationSince_FirstTrnx_to_Current_CountryCode_L7D",
            "14D": "Avg_DurationSince_FirstTrnx_to_Current_CountryCode_L14D",
            "30D": "Avg_DurationSince_FirstTrnx_to_Current_CountryCode_L30D",
            "90D": "Avg_DurationSince_FirstTrnx_to_Current_CountryCode_L90D",
        },
    },
]