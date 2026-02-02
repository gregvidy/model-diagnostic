# Application Fraud Detection - Feature Engineering Guide

This guide provides comprehensive documentation for using the modular feature engineering system for application fraud detection.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Feature Categories](#feature-categories)
6. [Usage Examples](#usage-examples)
7. [Configuration Guide](#configuration-guide)
8. [Customization](#customization)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This feature engineering system provides a **modular**, **configurable**, and **scalable** approach to building fraud detection features for credit card (RCC) and personal loan (RPL) applications.

### Key Features

- ✅ **Modular Architecture**: Calculate features independently or in groups
- ✅ **Configurable Time Periods**: Customizable rolling windows (1, 3, 6, 12, 24 months)
- ✅ **Product-Specific Configs**: Separate configurations for RCC and RPL
- ✅ **Automatic Ratio Calculations**: Period-over-period ratios
- ✅ **Missing Data Handling**: Robust handling of edge cases
- ✅ **Easy Extension**: Add custom features via configuration

### Feature Categories

1. **Demographic Features** - Age, education, occupation
2. **Economic Features** - Income, loan-to-value, debt-to-income ratios
3. **Application Behavior** - Frequency, rejection patterns, temporal features
4. **Sales Agent Features** - Agent performance metrics
5. **Collateral Features** - Property valuation, ownership matching

---

## 🚀 Quick Start

### Basic Usage

```python
from calculation_features import apply_feature_engineering
from application_fraud_config import FEATURE_CONFIG
import pandas as pd

# Load your data
df = pd.read_csv('application_data.csv')
df['Application_Date'] = pd.to_datetime(df['Application_Date'])

# Apply feature engineering
df_with_features = apply_feature_engineering(
    df=df,
    config=FEATURE_CONFIG,
    date_col='Application_Date',
    entity_col='ID/KTP/PASPOR/KITAS'
)

print(f"Features created: {df_with_features.shape[1] - df.shape[1]}")
```

### Running the Example Notebook

1. Open `application_fraud_feature_engineering_example.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. Review the 7 comprehensive examples provided

---

## 💾 Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib
```

### Directory Setup

```
instinct-ml/
├── src/
│   ├── calculation_features.py          # Core feature calculation engine
│   ├── application_fraud_config.py      # Feature configurations
│   └── example_usage.py                 # Python script examples
├── data/
│   ├── generate_dummy_data.py           # Data generator
│   └── application_data.csv             # Sample data
└── notebook/
    ├── application_fraud_feature_engineering_example.ipynb
    └── README.md                         # This file
```

---

## 📂 Project Structure

### Core Modules

#### 1. `calculation_features.py`

The main engine for feature calculation. Contains:

- **`FeatureCalculator`** class: Main calculation engine
  - `calculate_demographic_features()`
  - `calculate_economic_features()`
  - `calculate_application_behavior_features()`
  - `calculate_sales_agent_features()`
  - `calculate_collateral_features()`
  - `calculate_all_features()`

- **`apply_feature_engineering()`** function: High-level wrapper for easy usage

#### 2. `application_fraud_config.py`

Configuration definitions for all features:

- **`FEATURE_CONFIG`**: Complete configuration for all features
- **`FEATURE_CONFIG_RCC`**: Credit card specific configuration
- **`FEATURE_CONFIG_RPL`**: Personal loan specific configuration
- Individual configs: `demographic_config`, `economic_config`, etc.
- **`TIME_PERIODS`**: Default time windows `[1, 3, 6, 12, 24]` months

---

## 🏷️ Feature Categories

### 1. Demographic Features

Features related to customer demographics:

```python
demographic_config = [
    {
        "feature_name": "Age",
        "source_col": "USIA",
        "agg_type": "direct",
    },
    {
        "feature_name": "Age_Group",
        "source_col": "USIA",
        "agg_type": "age_group",
        "bins": [0, 25, 35, 45, 55, 65, 100],
        "labels": ["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
    },
    # ... more features
]
```

**Generated Features:**
- `Age`: Customer age
- `Age_Group`: Categorical age groups
- `Education_Level`: Education level
- `Occupation`: Occupation type

### 2. Economic Features

Financial and economic indicators:

```python
economic_config = [
    {
        "feature_name": "Loan_to_Income_Ratio",
        "calc_type": "ratio",
        "numerator_col": "Amount_Limit",
        "denominator_col": "GAJI/TAHUN",
    },
    # ... more features
]
```

**Generated Features:**
- `Annual_Income`: Yearly income
- `Loan_to_Value_Ratio`: LTV for secured loans
- `Loan_to_Income_Ratio`: Loan amount / annual income
- `Purchase_to_Appraisal_Ratio`: Property valuation ratio

### 3. Application Behavior Features

Temporal patterns and application history:

```python
application_behavior_config = [
    {
        "type": "frequency",
        "groupby": "ID/KTP/PASPOR/KITAS",
        "value_col": "Application_Number",
        "agg_func": "count",
        "periods": [1, 3, 6, 12, 24],  # months
        "feature_prefix": "Application_Count",
        "calculate_ratios": True,
    },
    # ... more features
]
```

**Generated Features (per time period):**
- `Application_Count_{period}M`: Number of applications
- `Avg_Loan_Amount_{period}M`: Average loan amount
- `Max_Loan_Amount_{period}M`: Maximum loan amount
- `Total_Loan_Amount_{period}M`: Total loan amount
- `Rejection_Rate_{period}M`: Rejection rate
- `Days_Since_Last_Application`: Days since last application
- Ratio features (e.g., `Application_Count_Ratio_1M_to_3M`)

### 4. Sales Agent Features

Sales agent performance metrics:

```python
sales_agent_config = [
    {
        "type": "agent_success_rate",
        "agent_col": "SALES CODE",
        "success_col": "REJECTION CODE",
        "periods": [1, 3, 6, 12, 24],
        "feature_prefix": "Agent_Success_Rate",
    },
    # ... more features
]
```

**Generated Features:**
- `Agent_Success_Rate_{period}M`: Agent approval rate
- `Agent_Application_Count_{period}M`: Applications processed
- `Agent_Avg_Loan_Amount_{period}M`: Average loan amount
- `Agent_Unique_Customers_{period}M`: Unique customer count

### 5. Collateral Features

Property and collateral-related features (mainly for RPL):

```python
collateral_config = [
    {
        "feature_name": "Loan_to_Value_Ratio",
        "calc_type": "loan_to_value",
        "loan_col": "Amount_Limit",
        "value_col": "APPRISAL PRICE",
    },
    {
        "feature_name": "Ownership_Match",
        "calc_type": "ownership_match",
        "applicant_col": "NAMA",
        "owner_col": "SERTIFIKAT ATAS NAMA",
    },
    # ... more features
]
```

**Generated Features:**
- `Collateral_Purchase_Price`: Purchase price
- `Collateral_Appraisal_Price`: Appraisal price
- `Purchase_to_Appraisal_Ratio`: Price ratio
- `Ownership_Match`: Name matching indicator
- `Home_Collateral_Postcode_Match`: Location matching

---

## 📚 Usage Examples

### Example 1: All Features

```python
from calculation_features import apply_feature_engineering
from application_fraud_config import FEATURE_CONFIG

df_features = apply_feature_engineering(
    df=df,
    config=FEATURE_CONFIG,
    date_col='Application_Date',
    entity_col='ID/KTP/PASPOR/KITAS'
)
```

### Example 2: Product-Specific Features

```python
from application_fraud_config import FEATURE_CONFIG_RCC, FEATURE_CONFIG_RPL

# For RCC (Credit Card)
df_rcc = df[df['Application_Type'] == 'RCC']
df_rcc_features = apply_feature_engineering(df_rcc, FEATURE_CONFIG_RCC)

# For RPL (Personal Loan)
df_rpl = df[df['Application_Type'] == 'RPL']
df_rpl_features = apply_feature_engineering(df_rpl, FEATURE_CONFIG_RPL)
```

### Example 3: Selective Feature Groups

```python
from application_fraud_config import demographic_config, economic_config

# Only demographic and economic features
selective_config = {
    "demographic": demographic_config,
    "economic": economic_config,
}

df_selective = apply_feature_engineering(df, selective_config)
```

### Example 4: Custom Time Periods

```python
from application_fraud_config import application_behavior_config

# Modify time periods
custom_config = application_behavior_config.copy()
for feature_dict in custom_config:
    if 'periods' in feature_dict:
        feature_dict['periods'] = [3, 6, 12]  # Only 3, 6, 12 months

df_custom = apply_feature_engineering(
    df, 
    {"application_behavior": custom_config}
)
```

### Example 5: Step-by-Step Calculation

```python
from calculation_features import FeatureCalculator

calculator = FeatureCalculator(df, 'Application_Date', 'ID/KTP/PASPOR/KITAS')

# Calculate each group separately
df = calculator.calculate_demographic_features(demographic_config)
df = calculator.calculate_economic_features(economic_config)
df = calculator.calculate_application_behavior_features(application_behavior_config)
```

---

## ⚙️ Configuration Guide

### Configuration Structure

Each feature configuration is a dictionary or list of dictionaries:

```python
{
    "feature_name": "Feature_Name",      # Output column name
    "calc_type": "ratio",                 # Calculation type
    "numerator_col": "Column1",           # Input column 1
    "denominator_col": "Column2",         # Input column 2
}
```

### Calculation Types

1. **`direct`**: Direct column mapping
   ```python
   {"feature_name": "Age", "source_col": "USIA", "agg_type": "direct"}
   ```

2. **`ratio`**: Numerator / Denominator
   ```python
   {
       "feature_name": "Loan_to_Income",
       "calc_type": "ratio",
       "numerator_col": "Amount_Limit",
       "denominator_col": "GAJI/TAHUN"
   }
   ```

3. **`frequency`**: Rolling aggregations over time
   ```python
   {
       "type": "frequency",
       "groupby": "ID/KTP/PASPOR/KITAS",
       "value_col": "Application_Number",
       "agg_func": "count",
       "periods": [1, 3, 6, 12, 24],
       "feature_prefix": "App_Count",
       "calculate_ratios": True
   }
   ```

4. **`rejection_rate`**: Rejection rate over time
   ```python
   {
       "type": "rejection_rate",
       "groupby": "ID/KTP/PASPOR/KITAS",
       "rejection_col": "REJECTION CODE",
       "periods": [1, 3, 6, 12, 24],
       "feature_prefix": "Rejection_Rate"
   }
   ```

5. **`time_since_last`**: Days since last event
   ```python
   {
       "type": "time_since_last",
       "groupby": "ID/KTP/PASPOR/KITAS",
       "feature_name": "Days_Since_Last_App"
   }
   ```

### Aggregation Functions

Supported `agg_func` values:
- `count`: Count of records
- `sum`: Sum of values
- `mean`: Average of values
- `median`: Median of values
- `min`: Minimum value
- `max`: Maximum value
- `std`: Standard deviation

---

## 🛠️ Customization

### Adding Custom Features

#### Method 1: Extend Existing Config

```python
from application_fraud_config import economic_config

# Add to existing config
custom_economic = economic_config.copy()
custom_economic.append({
    "feature_name": "Custom_Ratio",
    "calc_type": "ratio",
    "numerator_col": "Amount_Limit",
    "denominator_col": "Custom_Column",
})

df_custom = apply_feature_engineering(
    df, 
    {"economic": custom_economic}
)
```

#### Method 2: Create New Config

```python
# Define new feature group
custom_features = [
    {
        "feature_name": "Income_Per_Age",
        "calc_type": "ratio",
        "numerator_col": "GAJI/TAHUN",
        "denominator_col": "USIA",
    },
    {
        "feature_name": "Loan_Per_Age",
        "calc_type": "ratio",
        "numerator_col": "Amount_Limit",
        "denominator_col": "USIA",
    },
]

df_custom = apply_feature_engineering(
    df,
    {"custom": custom_features}
)
```

### Modifying Time Periods

```python
from application_fraud_config import FEATURE_CONFIG

# Deep copy to avoid modifying original
import copy
custom_config = copy.deepcopy(FEATURE_CONFIG)

# Modify all time periods
for group in custom_config.values():
    if isinstance(group, list):
        for feature in group:
            if 'periods' in feature:
                feature['periods'] = [1, 6, 12]  # Custom periods
```

---

## 💡 Best Practices

### 1. Data Preparation

```python
# Always convert dates to datetime
df['Application_Date'] = pd.to_datetime(df['Application_Date'])

# Sort by entity and date
df = df.sort_values(['ID/KTP/PASPOR/KITAS', 'Application_Date'])

# Handle missing values before feature engineering
df = df.fillna({'Amount_Limit': 0, 'GAJI/TAHUN': df['GAJI/TAHUN'].median()})
```

### 2. Memory Management

For large datasets:

```python
# Calculate features in batches
batch_size = 10000
results = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    batch_features = apply_feature_engineering(batch, FEATURE_CONFIG)
    results.append(batch_features)

df_final = pd.concat(results, ignore_index=True)
```

### 3. Feature Selection

```python
# Calculate all features first
df_all = apply_feature_engineering(df, FEATURE_CONFIG)

# Select important features
important_features = [
    'Application_Count_3M', 'Application_Count_6M',
    'Rejection_Rate_6M', 'Loan_to_Income_Ratio',
    'Age_Group', 'Agent_Success_Rate_12M'
]

df_selected = df_all[['Application_Number'] + important_features]
```

### 4. Validation

```python
# Check for missing values
print(df_features.isnull().sum())

# Check feature distributions
print(df_features.describe())

# Verify temporal features
print(df_features['Application_Count_3M'].value_counts())
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. KeyError: Column not found

**Problem**: Input column doesn't exist in dataframe

**Solution**:
```python
# Check required columns
required_cols = ['Application_Date', 'ID/KTP/PASPOR/KITAS', 'Amount_Limit']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
```

#### 2. All values are NaN for rolling features

**Problem**: Not enough historical data

**Solution**:
```python
# Check data distribution over time
print(df.groupby(df['Application_Date'].dt.to_period('M')).size())

# Use shorter time periods
feature_dict['periods'] = [1, 3, 6]  # Instead of [1, 3, 6, 12, 24]
```

#### 3. Memory error with large datasets

**Problem**: Dataset too large to process at once

**Solution**:
```python
# Process in smaller batches (see Best Practices section)
# Or use only essential features
minimal_config = {
    "demographic": demographic_config,
    "economic": economic_config[:3],  # First 3 features only
}
```

#### 4. Slow performance

**Problem**: Complex calculations on large dataset

**Solution**:
```python
# Reduce time periods
# Reduce number of feature groups
# Process product types separately (RCC vs RPL)
# Use sampling for development

df_sample = df.sample(frac=0.1, random_state=42)
df_sample_features = apply_feature_engineering(df_sample, FEATURE_CONFIG)
```

---

## 📖 Additional Resources

### Files in this Repository

1. **`application_fraud_feature_engineering_example.ipynb`**
   - Interactive examples of all functionality
   - Includes visualizations and explanations
   - Best for learning and experimentation

2. **`../src/example_usage.py`**
   - Python script with 7 comprehensive examples
   - Good for batch processing
   - Can be run from command line

3. **`../data/generate_dummy_data.py`**
   - Generate synthetic data for testing
   - Customize data size and distributions

### Running Examples

```bash
# Generate fresh dummy data
cd ../data
python generate_dummy_data.py

# Run example usage script
cd ../src
python example_usage.py
```

### Getting Help

For questions or issues:
1. Review this README thoroughly
2. Check the example notebook for similar use cases
3. Review the configuration file comments
4. Test with small sample data first

---

## 📝 Data Schema

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `Application_Number` | string | Unique application ID |
| `Application_Date` | datetime | Application submission date |
| `Application_Type` | string | RCC or RPL |
| `Amount_Limit` | numeric | Loan amount |
| `ID/KTP/PASPOR/KITAS` | string | Customer identifier (hashed) |
| `USIA` | numeric | Age |
| `GAJI/TAHUN` | numeric | Annual income |

### Optional Columns (for specific features)

| Column | Required For | Type |
|--------|--------------|------|
| `REJECTION CODE` | Rejection features | string |
| `SALES CODE` | Agent features | string |
| `PURCHASE PRICE` | Collateral features | numeric |
| `APPRISAL PRICE` | Collateral features | numeric |
| `KODE POS RUMAH` | Location features | string |
| `PENDIDIKAN TERAKHIR` | Demographic features | string |
| `PEKERJAAN` | Demographic features | string |

---

## 🎓 Tutorial: From Scratch

### Step 1: Prepare Your Data

```python
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Convert date columns
df['Application_Date'] = pd.to_datetime(df['Application_Date'])

# Verify data
print(df.info())
print(df.head())
```

### Step 2: Choose Configuration

```python
from application_fraud_config import FEATURE_CONFIG

# Start with complete config
config = FEATURE_CONFIG

# Or choose specific groups
from application_fraud_config import (
    demographic_config,
    economic_config,
    application_behavior_config
)

config = {
    "demographic": demographic_config,
    "economic": economic_config,
    "application_behavior": application_behavior_config,
}
```

### Step 3: Apply Feature Engineering

```python
from calculation_features import apply_feature_engineering

df_features = apply_feature_engineering(
    df=df,
    config=config,
    date_col='Application_Date',
    entity_col='ID/KTP/PASPOR/KITAS'
)

print(f"Original columns: {df.shape[1]}")
print(f"With features: {df_features.shape[1]}")
print(f"New features: {df_features.shape[1] - df.shape[1]}")
```

### Step 4: Validate and Export

```python
# Check for issues
print("\nMissing values:")
print(df_features.isnull().sum().sort_values(ascending=False).head(10))

# Export
df_features.to_csv('data_with_features.csv', index=False)
print("\n✓ Features exported successfully")
```

---

## 📊 Performance Tips

1. **Start Small**: Test with sample data first
2. **Selective Features**: Only calculate what you need
3. **Batch Processing**: Process large datasets in chunks
4. **Product Separation**: Process RCC and RPL separately
5. **Time Period Optimization**: Use fewer time periods in development

---

## 🔄 Version History

- **v1.0** (Feb 2026): Initial release with complete feature engineering system

---

## 📧 Support

For issues, questions, or contributions, please refer to the example notebook or contact the data science team.

---

**Happy Feature Engineering! 🚀**
