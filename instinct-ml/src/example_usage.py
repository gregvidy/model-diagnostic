"""
Example Usage Script for Application Fraud Detection Feature Engineering
==========================================================================

This script demonstrates how to use the modular feature engineering system
for application fraud detection.
"""

import pandas as pd
import numpy as np
from calculation_features import apply_feature_engineering, FeatureCalculator
from application_fraud_config import (
    FEATURE_CONFIG,
    FEATURE_CONFIG_RCC,
    FEATURE_CONFIG_RPL,
    TIME_PERIODS
)


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load and prepare application data.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
        
    Returns:
    --------
    pd.DataFrame with prepared data
    """
    df = pd.read_csv(filepath)
    
    # Convert date columns to datetime
    df['Application_Date'] = pd.to_datetime(df['Application_Date'])
    
    # Handle rejection code (convert to binary: 1 if rejected, 0 if approved)
    df['Is_Rejected'] = df['REJECTION CODE'].notna().astype(int)
    
    return df


def example_1_basic_usage():
    """
    Example 1: Basic usage with all feature groups
    """
    print("=" * 80)
    print("Example 1: Basic Usage with All Features")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data('application_data.csv')
    
    # Apply feature engineering
    df_with_features = apply_feature_engineering(
        df=df,
        config=FEATURE_CONFIG,
        date_col='Application_Date',
        entity_col='ID/KTP/PASPOR/KITAS'
    )
    
    print(f"Original shape: {df.shape}")
    print(f"Shape with features: {df_with_features.shape}")
    print(f"New features created: {df_with_features.shape[1] - df.shape[1]}")
    print("\nSample of new features:")
    print(df_with_features.head())
    
    return df_with_features


def example_2_product_specific():
    """
    Example 2: Product-specific feature engineering (RCC vs RPL)
    """
    print("\n" + "=" * 80)
    print("Example 2: Product-Specific Feature Engineering")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data('application_data.csv')
    
    # Separate by application type
    df_rcc = df[df['Application_Type'] == 'RCC'].copy()
    df_rpl = df[df['Application_Type'] == 'RPL'].copy()
    
    print(f"RCC applications: {len(df_rcc)}")
    print(f"RPL applications: {len(df_rpl)}")
    
    # Apply RCC-specific features
    if len(df_rcc) > 0:
        df_rcc_features = apply_feature_engineering(
            df=df_rcc,
            config=FEATURE_CONFIG_RCC,
            date_col='Application_Date',
            entity_col='ID/KTP/PASPOR/KITAS'
        )
        print(f"\nRCC features shape: {df_rcc_features.shape}")
    
    # Apply RPL-specific features
    if len(df_rpl) > 0:
        df_rpl_features = apply_feature_engineering(
            df=df_rpl,
            config=FEATURE_CONFIG_RPL,
            date_col='Application_Date',
            entity_col='ID/KTP/PASPOR/KITAS'
        )
        print(f"RPL features shape: {df_rpl_features.shape}")
    
    return df_rcc_features, df_rpl_features


def example_3_custom_time_periods():
    """
    Example 3: Custom time periods for rolling features
    """
    print("\n" + "=" * 80)
    print("Example 3: Custom Time Periods")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data('application_data.csv')
    
    # Create custom configuration with different time periods
    from application_fraud_config import application_behavior_config
    
    custom_config = application_behavior_config.copy()
    
    # Change time periods to only 3, 6, and 12 months
    for feature_dict in custom_config:
        if 'periods' in feature_dict:
            feature_dict['periods'] = [3, 6, 12]
    
    # Apply feature engineering
    df_custom = apply_feature_engineering(
        df=df,
        config={"application_behavior": custom_config},
        date_col='Application_Date',
        entity_col='ID/KTP/PASPOR/KITAS'
    )
    
    print(f"Shape with custom periods: {df_custom.shape}")
    print("\nFeatures with custom periods:")
    feature_cols = [col for col in df_custom.columns if any(p in col for p in ['3M', '6M', '12M'])]
    print(feature_cols)
    
    return df_custom


def example_4_selective_features():
    """
    Example 4: Calculate only specific feature groups
    """
    print("\n" + "=" * 80)
    print("Example 4: Selective Feature Groups")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data('application_data.csv')
    
    from application_fraud_config import demographic_config, economic_config
    
    # Only calculate demographic and economic features
    selective_config = {
        "demographic": demographic_config,
        "economic": economic_config,
    }
    
    df_selective = apply_feature_engineering(
        df=df,
        config=selective_config,
        date_col='Application_Date',
        entity_col='ID/KTP/PASPOR/KITAS'
    )
    
    print(f"Shape with selective features: {df_selective.shape}")
    print("\nDemographic and economic features only:")
    new_features = [col for col in df_selective.columns if col not in df.columns]
    print(new_features)
    
    return df_selective


def example_5_add_custom_features():
    """
    Example 5: Add custom features to the configuration
    """
    print("\n" + "=" * 80)
    print("Example 5: Adding Custom Features")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data('application_data.csv')
    
    from application_fraud_config import economic_config
    
    # Create a custom economic configuration
    custom_economic = economic_config.copy()
    
    # Add a new custom ratio feature
    custom_economic.append({
        "feature_name": "Amount_to_Income_Per_Age",
        "calc_type": "ratio",
        "numerator_col": "Amount_Limit",
        "denominator_col": "GAJI/TAHUN",
    })
    
    custom_config = {
        "economic": custom_economic,
    }
    
    df_custom = apply_feature_engineering(
        df=df,
        config=custom_config,
        date_col='Application_Date',
        entity_col='ID/KTP/PASPOR/KITAS'
    )
    
    print(f"Shape with custom features: {df_custom.shape}")
    print("\nCustom feature added:")
    if 'Amount_to_Income_Per_Age' in df_custom.columns:
        print("Amount_to_Income_Per_Age successfully created")
        print(df_custom['Amount_to_Income_Per_Age'].describe())
    
    return df_custom


def example_6_step_by_step():
    """
    Example 6: Step-by-step feature calculation using FeatureCalculator
    """
    print("\n" + "=" * 80)
    print("Example 6: Step-by-Step Feature Calculation")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data('application_data.csv')
    
    # Initialize calculator
    calculator = FeatureCalculator(
        df=df,
        date_col='Application_Date',
        entity_col='ID/KTP/PASPOR/KITAS'
    )
    
    # Step 1: Calculate demographic features
    from application_fraud_config import demographic_config
    df_step1 = calculator.calculate_demographic_features(demographic_config)
    print(f"After demographic features: {df_step1.shape}")
    
    # Step 2: Calculate economic features
    from application_fraud_config import economic_config
    df_step2 = calculator.calculate_economic_features(economic_config)
    print(f"After economic features: {df_step2.shape}")
    
    # Step 3: Calculate application behavior features
    from application_fraud_config import application_behavior_config
    df_step3 = calculator.calculate_application_behavior_features(application_behavior_config)
    print(f"After application behavior features: {df_step3.shape}")
    
    print("\nStep-by-step calculation completed!")
    
    return df_step3


def example_7_export_features():
    """
    Example 7: Export features to file
    """
    print("\n" + "=" * 80)
    print("Example 7: Export Features")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data('application_data.csv')
    
    # Apply feature engineering
    df_with_features = apply_feature_engineering(
        df=df,
        config=FEATURE_CONFIG,
        date_col='Application_Date',
        entity_col='ID/KTP/PASPOR/KITAS'
    )
    
    # Export to CSV
    output_file = 'application_data_with_features.csv'
    df_with_features.to_csv(output_file, index=False)
    print(f"Features exported to: {output_file}")
    
    # Export feature list
    feature_list = [col for col in df_with_features.columns if col not in df.columns]
    feature_list_df = pd.DataFrame({'Feature_Name': feature_list})
    feature_list_file = 'feature_list.csv'
    feature_list_df.to_csv(feature_list_file, index=False)
    print(f"Feature list exported to: {feature_list_file}")
    print(f"Total features created: {len(feature_list)}")
    
    return df_with_features


def main():
    """
    Main function to run all examples
    """
    print("Application Fraud Detection - Feature Engineering Examples")
    print("=" * 80)
    print()
    
    try:
        # Run all examples
        example_1_basic_usage()
        example_2_product_specific()
        example_3_custom_time_periods()
        example_4_selective_features()
        example_5_add_custom_features()
        example_6_step_by_step()
        example_7_export_features()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
