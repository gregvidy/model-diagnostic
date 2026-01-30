"""
Configuration file for Application Fraud Detection Feature Engineering
========================================================================

This configuration file defines all feature engineering parameters for
Credit Card and Personal Loan application fraud detection models.

Usage:
------
from calculation_features import apply_feature_engineering
from application_fraud_config import FEATURE_CONFIG

# Apply feature engineering
df_with_features = apply_feature_engineering(df, FEATURE_CONFIG)

Configuration Structure:
------------------------
- demographic: Demographic features configuration
- economic: Economic/financial features configuration
- application_behavior: Application behavior patterns configuration
- sales_agent: Sales agent performance metrics configuration
- collateral: Collateral-related features configuration
"""

###############################
### TIME PERIOD DEFINITIONS ###
###############################

# Standard time periods for rolling features (in months)
TIME_PERIODS = [1, 3, 6, 12, 24]


#############################
### DEMOGRAPHIC FEATURES ###
#############################

demographic_config = [
    {
        # Age (direct mapping)
        "feature_name": "Age",
        "source_col": "USIA",
        "agg_type": "direct",
    },
    {
        # Age group categorization
        "feature_name": "Age_Group",
        "source_col": "USIA",
        "agg_type": "age_group",
        "bins": [0, 25, 35, 45, 55, 65, 100],
        "labels": ["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
    },
    {
        # Education level
        "feature_name": "Education_Level",
        "source_col": "PENDIDIKAN TERAKHIR",
        "agg_type": "direct",
    },
    {
        # Occupation
        "feature_name": "Occupation",
        "source_col": "PEKERJAAN",
        "agg_type": "direct",
    },
]


##########################
### ECONOMIC FEATURES ###
##########################

economic_config = [
    {
        # Annual income (direct)
        "feature_name": "Annual_Income",
        "calc_type": "direct",
        "source_col": "GAJI/TAHUN",
    },
    {
        # Loan to Value Ratio
        "feature_name": "Loan_to_Value_Ratio",
        "calc_type": "loan_to_value",
        "loan_col": "Amount_Limit",
        "value_col": "APPRISAL PRICE",
    },
    {
        # Loan to Income Ratio
        "feature_name": "Loan_to_Income_Ratio",
        "calc_type": "ratio",
        "numerator_col": "Amount_Limit",
        "denominator_col": "GAJI/TAHUN",
    },
    {
        # Purchase to Appraisal Ratio
        "feature_name": "Purchase_to_Appraisal_Ratio",
        "calc_type": "ratio",
        "numerator_col": "PURCHASE PRICE",
        "denominator_col": "APPRISAL PRICE",
    },
]


######################################
### APPLICATION BEHAVIOR FEATURES ###
######################################

application_behavior_config = [
    {
        # Application frequency by customer
        "type": "frequency",
        "groupby": "ID/KTP/PASPOR/KITAS",
        "value_col": "Application_Number",
        "agg_func": "count",
        "periods": TIME_PERIODS,
        "feature_prefix": "Application_Count",
        "calculate_ratios": True,
    },
    {
        # Average loan amount by customer over time
        "type": "frequency",
        "groupby": "ID/KTP/PASPOR/KITAS",
        "value_col": "Amount_Limit",
        "agg_func": "mean",
        "periods": TIME_PERIODS,
        "feature_prefix": "Avg_Loan_Amount",
        "calculate_ratios": True,
    },
    {
        # Maximum loan amount by customer over time
        "type": "frequency",
        "groupby": "ID/KTP/PASPOR/KITAS",
        "value_col": "Amount_Limit",
        "agg_func": "max",
        "periods": TIME_PERIODS,
        "feature_prefix": "Max_Loan_Amount",
        "calculate_ratios": False,
    },
    {
        # Sum of loan amounts by customer over time
        "type": "frequency",
        "groupby": "ID/KTP/PASPOR/KITAS",
        "value_col": "Amount_Limit",
        "agg_func": "sum",
        "periods": TIME_PERIODS,
        "feature_prefix": "Total_Loan_Amount",
        "calculate_ratios": True,
    },
    {
        # Application rejection rate
        "type": "rejection_rate",
        "groupby": "ID/KTP/PASPOR/KITAS",
        "rejection_col": "REJECTION CODE",  # Assuming 1 if rejected, 0 otherwise
        "periods": TIME_PERIODS,
        "feature_prefix": "Rejection_Rate",
    },
    {
        # Time since last application (days)
        "type": "time_since_last",
        "groupby": "ID/KTP/PASPOR/KITAS",
        "feature_name": "Days_Since_Last_Application",
    },
    {
        # Application count by product type (RCC/RPL)
        "type": "frequency",
        "groupby": ["ID/KTP/PASPOR/KITAS", "Application_Type"],
        "value_col": "Application_Number",
        "agg_func": "count",
        "periods": TIME_PERIODS,
        "feature_prefix": "Application_Count_By_Type",
        "calculate_ratios": False,
    },
    {
        # Application count by branch
        "type": "frequency",
        "groupby": ["ID/KTP/PASPOR/KITAS", "Branch"],
        "value_col": "Application_Number",
        "agg_func": "count",
        "periods": TIME_PERIODS,
        "feature_prefix": "Application_Count_By_Branch",
        "calculate_ratios": False,
    },
]


#####################################
### SALES AGENT BEHAVIOR FEATURES ###
#####################################

sales_agent_config = [
    {
        # Sales agent success rate
        "type": "agent_success_rate",
        "agent_col": "SALES CODE",
        "success_col": "REJECTION CODE",  # 0 if approved, 1 if rejected
        "periods": TIME_PERIODS,
        "feature_prefix": "Agent_Success_Rate",
    },
    {
        # Number of applications processed by agent
        "type": "agent_volume",
        "groupby": "SALES CODE",
        "value_col": "Application_Number",
        "agg_func": "count",
        "periods": TIME_PERIODS,
        "feature_prefix": "Agent_Application_Count",
        "calculate_ratios": True,
    },
    {
        # Average loan amount processed by agent
        "type": "agent_avg_amount",
        "groupby": "SALES CODE",
        "value_col": "Amount_Limit",
        "agg_func": "mean",
        "periods": TIME_PERIODS,
        "feature_prefix": "Agent_Avg_Loan_Amount",
        "calculate_ratios": True,
    },
    {
        # Total loan amount processed by agent
        "type": "agent_avg_amount",
        "groupby": "SALES CODE",
        "value_col": "Amount_Limit",
        "agg_func": "sum",
        "periods": TIME_PERIODS,
        "feature_prefix": "Agent_Total_Loan_Amount",
        "calculate_ratios": True,
    },
    {
        # Number of unique customers per agent
        "type": "agent_volume",
        "groupby": "SALES CODE",
        "value_col": "ID/KTP/PASPOR/KITAS",
        "agg_func": "count",
        "periods": TIME_PERIODS,
        "feature_prefix": "Agent_Unique_Customers",
        "calculate_ratios": False,
    },
    {
        # Agent tenure (days since join date)
        "type": "agent_volume",
        "groupby": "SALES CODE",
        "value_col": "JOIN DATE",
        "agg_func": "min",
        "periods": [24],
        "feature_prefix": "Agent_Tenure_Days",
        "calculate_ratios": False,
    },
]


############################
### COLLATERAL FEATURES ###
############################

collateral_config = [
    {
        # Certificate number (direct)
        "feature_name": "Certificate_Number",
        "calc_type": "direct",
        "source_col": "NO SERTIFIKAT",
    },
    {
        # Purchase price (direct)
        "feature_name": "Collateral_Purchase_Price",
        "calc_type": "direct",
        "source_col": "PURCHASE PRICE",
    },
    {
        # Appraisal price (direct)
        "feature_name": "Collateral_Appraisal_Price",
        "calc_type": "direct",
        "source_col": "APPRISAL PRICE",
    },
    {
        # Purchase to appraisal ratio
        "feature_name": "Purchase_to_Appraisal_Ratio",
        "calc_type": "ratio",
        "numerator_col": "PURCHASE PRICE",
        "denominator_col": "APPRISAL PRICE",
    },
    {
        # Ownership match (applicant name vs certificate name)
        "feature_name": "Ownership_Match",
        "calc_type": "ownership_match",
        "applicant_col": "NAMA",
        "owner_col": "SERTIFIKAT ATAS NAMA",
    },
    {
        # Collateral postcode
        "feature_name": "Collateral_Postcode",
        "calc_type": "direct",
        "source_col": "KODE POS SERTIFIKAT",
    },
]


#################################
### LOCATION-BASED FEATURES ###
#################################

location_config = [
    {
        # Home postcode match with collateral postcode
        "feature_name": "Home_Collateral_Postcode_Match",
        "calc_type": "ownership_match",
        "applicant_col": "KODE POS RUMAH",
        "owner_col": "KODE POS SERTIFIKAT",
    },
    {
        # Home postcode match with company postcode
        "feature_name": "Home_Company_Postcode_Match",
        "calc_type": "ownership_match",
        "applicant_col": "KODE POS RUMAH",
        "owner_col": "KODE POS PERUSAHAAN",
    },
]


##############################
### ADDITIONAL FEATURES ###
##############################

additional_config = [
    {
        # Application type
        "feature_name": "Application_Type",
        "calc_type": "direct",
        "source_col": "Application_Type",
    },
    {
        # Loan purpose
        "feature_name": "Loan_Purpose",
        "calc_type": "direct",
        "source_col": "TUJUAN PINJAMAN",
    },
    {
        # Bank disbursement
        "feature_name": "Bank_Disbursement",
        "calc_type": "direct",
        "source_col": "BANK PENCAIRAN",
    },
    {
        # APO (Account Processing Officer)
        "feature_name": "APO",
        "calc_type": "direct",
        "source_col": "APO",
    },
    {
        # Primary/Secondary indicator
        "feature_name": "Primary_Secondary",
        "calc_type": "direct",
        "source_col": "PRIMARY/SECONDARY",
    },
    {
        # Marketing program
        "feature_name": "Marketing_Program",
        "calc_type": "direct",
        "source_col": "MARKETING PROGRAM",
    },
]


###############################
### COMPLETE CONFIGURATION ###
###############################

# Main configuration dictionary that combines all feature groups
FEATURE_CONFIG = {
    "demographic": demographic_config,
    "economic": economic_config,
    "application_behavior": application_behavior_config,
    "sales_agent": sales_agent_config,
    "collateral": collateral_config + location_config,
}

# Configuration for specific product types
FEATURE_CONFIG_RCC = FEATURE_CONFIG.copy()
FEATURE_CONFIG_RPL = FEATURE_CONFIG.copy()

# You can customize configurations per product if needed
# Example: RPL requires more collateral features
FEATURE_CONFIG_RPL["collateral"].extend(additional_config)


#####################################
### CONFIGURATION USAGE EXAMPLES ###
#####################################

"""
Example 1: Basic usage with all features
-----------------------------------------
from calculation_features import apply_feature_engineering
from application_fraud_config import FEATURE_CONFIG
import pandas as pd

df = pd.read_csv('application_data.csv')
df_with_features = apply_feature_engineering(df, FEATURE_CONFIG)


Example 2: Custom time periods
-------------------------------
from application_fraud_config import application_behavior_config

# Modify time periods for specific use case
custom_config = application_behavior_config.copy()
custom_config[0]['periods'] = [1, 6, 12]  # Only 1, 6, 12 months

df_with_features = apply_feature_engineering(df, {"application_behavior": custom_config})


Example 3: Product-specific configuration
------------------------------------------
from application_fraud_config import FEATURE_CONFIG_RCC, FEATURE_CONFIG_RPL

# For RCC applications
df_rcc = df[df['Application_Type'] == 'RCC']
df_rcc_features = apply_feature_engineering(df_rcc, FEATURE_CONFIG_RCC)

# For RPL applications
df_rpl = df[df['Application_Type'] == 'RPL']
df_rpl_features = apply_feature_engineering(df_rpl, FEATURE_CONFIG_RPL)


Example 4: Selective feature groups
------------------------------------
# Only calculate demographic and economic features
selective_config = {
    "demographic": demographic_config,
    "economic": economic_config,
}

df_with_features = apply_feature_engineering(df, selective_config)


Example 5: Adding custom features
----------------------------------
custom_economic_config = economic_config.copy()
custom_economic_config.append({
    "feature_name": "Custom_Ratio",
    "calc_type": "ratio",
    "numerator_col": "Amount_Limit",
    "denominator_col": "Custom_Column",
})

custom_config = {
    "economic": custom_economic_config,
}

df_with_features = apply_feature_engineering(df, custom_config)
"""
