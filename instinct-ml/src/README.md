# Application Fraud Detection - Feature Engineering System

A modular and configurable feature engineering system for building fraud detection models for Credit Card and Personal Loan applications.

## 📋 Overview

This system provides a reusable framework for creating fraud detection features without redefining logic. Data scientists can configure features through simple configuration files, making it easy to:

- Create demographic, economic, and behavioral features
- Generate rolling time-based aggregations (1, 3, 6, 12, 24 months)
- Calculate ratio features between time periods
- Apply product-specific feature engineering (RCC vs RPL)
- Reuse feature engineering logic across different projects

## 🏗️ Architecture

```
src/
├── calculation_features.py       # Core feature calculation module
├── application_fraud_config.py   # Configuration file for features
└── example_usage.py              # Usage examples
```

### Core Components

1. **`calculation_features.py`**: Contains the `FeatureCalculator` class with modular methods for different feature types
2. **`application_fraud_config.py`**: Configuration dictionaries that define what features to create
3. **`example_usage.py`**: Demonstrates various usage patterns

## 🚀 Quick Start

### Basic Usage

```python
from calculation_features import apply_feature_engineering
from application_fraud_config import FEATURE_CONFIG
import pandas as pd

# Load your data
df = pd.read_csv('application_data.csv')

# Apply feature engineering
df_with_features = apply_feature_engineering(
    df=df,
    config=FEATURE_CONFIG,
    date_col='Application_Date',
    entity_col='ID/KTP/PASPOR/KITAS'
)
```

## 📊 Feature Categories

### 1. Demographic Features
- Age, age groups
- Education level
- Occupation
- Gender (if available)

### 2. Economic Features
- Annual income
- Loan to Value (LTV) ratio
- Loan to Income ratio
- Debt to Income ratio
- Purchase to Appraisal ratio

### 3. Application Behavior Features
- Application frequency (1M, 3M, 6M, 12M, 24M)
- Average/Maximum/Total loan amounts over time
- Rejection rate history
- Days since last application
- Application patterns by product type
- Application patterns by branch

### 4. Sales Agent Features
- Agent success rate
- Number of applications processed
- Average loan amount per agent
- Total loan volume per agent
- Number of unique customers per agent
- Agent tenure

### 5. Collateral Features
- Purchase price vs Appraisal price
- Ownership match (applicant vs certificate owner)
- Location match (home vs collateral)
- Certificate details

## ⚙️ Configuration System

### Time Periods

Define rolling time windows in `application_fraud_config.py`:

```python
TIME_PERIODS = [1, 3, 6, 12, 24]  # months
```

### Feature Configuration Structure

Each feature group has a specific configuration structure:

#### Demographic Features

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
]
```

#### Rolling Features (Application Behavior)

```python
{
    "type": "frequency",
    "groupby": "ID/KTP/PASPOR/KITAS",
    "value_col": "Application_Number",
    "agg_func": "count",
    "periods": [1, 3, 6, 12, 24],
    "feature_prefix": "Application_Count",
    "calculate_ratios": True,
}
```

This creates:
- `Application_Count_1M`
- `Application_Count_3M`
- `Application_Count_6M`
- `Application_Count_12M`
- `Application_Count_24M`
- `Application_Count_Ratio_1M_to_3M`
- `Application_Count_Ratio_3M_to_6M`
- etc.

#### Economic Features

```python
{
    "feature_name": "Loan_to_Value_Ratio",
    "calc_type": "loan_to_value",
    "loan_col": "Amount_Limit",
    "value_col": "APPRISAL PRICE",
}
```

## 💡 Usage Examples

### Example 1: Product-Specific Features

```python
from application_fraud_config import FEATURE_CONFIG_RCC, FEATURE_CONFIG_RPL

# For RCC applications
df_rcc = df[df['Application_Type'] == 'RCC']
df_rcc_features = apply_feature_engineering(df_rcc, FEATURE_CONFIG_RCC)

# For RPL applications
df_rpl = df[df['Application_Type'] == 'RPL']
df_rpl_features = apply_feature_engineering(df_rpl, FEATURE_CONFIG_RPL)
```

### Example 2: Custom Time Periods

```python
from application_fraud_config import application_behavior_config

# Modify to use only 3, 6, 12 months
custom_config = application_behavior_config.copy()
for feature in custom_config:
    if 'periods' in feature:
        feature['periods'] = [3, 6, 12]

df_features = apply_feature_engineering(
    df, 
    {"application_behavior": custom_config}
)
```

### Example 3: Selective Feature Groups

```python
from application_fraud_config import demographic_config, economic_config

# Only calculate demographic and economic features
selective_config = {
    "demographic": demographic_config,
    "economic": economic_config,
}

df_features = apply_feature_engineering(df, selective_config)
```

### Example 4: Add Custom Features

```python
from application_fraud_config import economic_config

# Add a custom ratio feature
custom_config = economic_config.copy()
custom_config.append({
    "feature_name": "Custom_Ratio",
    "calc_type": "ratio",
    "numerator_col": "Amount_Limit",
    "denominator_col": "Custom_Column",
})

df_features = apply_feature_engineering(
    df, 
    {"economic": custom_config}
)
```

### Example 5: Step-by-Step Calculation

```python
from calculation_features import FeatureCalculator

# Initialize calculator
calculator = FeatureCalculator(df, 'Application_Date', 'ID/KTP/PASPOR/KITAS')

# Calculate features step by step
df_demo = calculator.calculate_demographic_features(demographic_config)
df_econ = calculator.calculate_economic_features(economic_config)
df_behavior = calculator.calculate_application_behavior_features(application_behavior_config)
```

## 🔧 Extending the System

### Adding a New Feature Type

1. Add a new calculation method to `FeatureCalculator` class in `calculation_features.py`:

```python
def calculate_custom_features(self, config: Dict) -> pd.DataFrame:
    result_df = self.df.copy()
    # Your custom logic here
    return result_df
```

2. Update the `calculate_all_features` method:

```python
def calculate_all_features(self, full_config: Dict) -> pd.DataFrame:
    # ... existing code ...
    
    if 'custom' in full_config:
        result_df = self.calculate_custom_features(full_config['custom'])
    
    return result_df
```

3. Add configuration in `application_fraud_config.py`:

```python
custom_config = [
    {
        "feature_name": "Custom_Feature",
        # ... your config ...
    }
]

FEATURE_CONFIG['custom'] = custom_config
```

## 📝 Data Requirements

### Required Columns

The system expects the following columns (based on SQL queries):

- `Application_Number`: Unique application identifier
- `Application_Date`: Date of application
- `Application_Type`: Type of application (RCC, RPL)
- `Amount_Limit`: Loan amount
- `Branch`: Branch code
- `ID/KTP/PASPOR/KITAS`: Customer identifier (hashed)
- `NAMA`: Customer name (hashed)
- `USIA`: Age
- `GAJI/TAHUN`: Annual income
- `PEKERJAAN`: Occupation
- `PENDIDIKAN TERAKHIR`: Education level
- `SALES CODE`: Sales agent identifier
- `REJECTION CODE`: Rejection code (if rejected)
- `PURCHASE PRICE`: Collateral purchase price
- `APPRISAL PRICE`: Collateral appraisal price
- And other fields as defined in SQL scripts

### Data Preparation

```python
import pandas as pd

# Load data
df = pd.read_csv('application_data.csv')

# Convert date columns
df['Application_Date'] = pd.to_datetime(df['Application_Date'])

# Create rejection flag
df['Is_Rejected'] = df['REJECTION CODE'].notna().astype(int)

# Handle missing values as needed
df.fillna(0, inplace=True)
```

## 🎯 Best Practices

1. **Always sort data by entity and date** before feature calculation (done automatically by `FeatureCalculator`)
2. **Use consistent naming conventions** for features
3. **Document custom features** in the configuration file
4. **Test with small datasets** before applying to full data
5. **Monitor memory usage** for large datasets with many time periods
6. **Version control your configurations** to track feature changes
7. **Create separate configs per product** when features differ significantly

## 🔍 Troubleshooting

### Memory Issues

For large datasets:
```python
# Process in chunks
chunk_size = 10000
results = []

for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
    chunk_features = apply_feature_engineering(chunk, FEATURE_CONFIG)
    results.append(chunk_features)

df_final = pd.concat(results, ignore_index=True)
```

### Missing Values

Handle missing values before or after feature engineering:
```python
# Before
df.fillna({'GAJI/TAHUN': df['GAJI/TAHUN'].median()}, inplace=True)

# After (for ratio features)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
```

## 📚 References

- Based on credit card feature engineering patterns from `credit_card_config.py`
- SQL queries from CIMB Niaga ID Instinct project
- Rolling aggregation concepts from time series analysis

## 👥 Contributing

To contribute new feature types:
1. Add the calculation method to `FeatureCalculator`
2. Create configuration templates
3. Add examples to `example_usage.py`
4. Update this README

## 📄 License

Internal use for GBG Analytics team.

---

**Contact**: Data Science Team - GBG Analytics
**Last Updated**: January 2026
