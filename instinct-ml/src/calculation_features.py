"""
Calculation Features Module for Application Fraud Detection
=============================================================

This module provides modular functionalities for building fraud detection features
for Credit Card and Personal Loan applications.

Features are categorized into:
- Demographic Features (age, gender, education)
- Economic Features (income, employment status, loan to value ratio)
- Application Behavior Features (frequency, rejection history)
- Sales Agent Acquisition Behavior Features (agent performance metrics)
- Collateral Features (collateral type, value, ownership)

All features support configurable rolling time periods: 1, 3, 6, 12, 24 months
Ratio features between time periods are automatically generated.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
from datetime import datetime, timedelta


class FeatureCalculator:
    """
    Main class for calculating application fraud detection features.
    """
    
    def __init__(self, df: pd.DataFrame, date_col: str = 'Application_Date', 
                 entity_col: str = 'ID/KTP/PASPOR/KITAS'):
        """
        Initialize the feature calculator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with application data
        date_col : str
            Name of the date column
        entity_col : str
            Name of the entity identifier column (e.g., customer ID)
        """
        self.df = df.copy()
        self.date_col = date_col
        self.entity_col = entity_col
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # Sort by entity and date
        self.df = self.df.sort_values([entity_col, date_col])
        self.df.reset_index(drop=True, inplace=True)
        
    def calculate_demographic_features(self, config: Dict) -> pd.DataFrame:
        """
        Calculate demographic features based on configuration.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary for demographic features
            
        Returns:
        --------
        pd.DataFrame with demographic features
        """
        result_df = self.df.copy()
        
        for feature_config in config:
            feature_name = feature_config['feature_name']
            source_col = feature_config['source_col']
            agg_type = feature_config.get('agg_type', 'direct')
            
            if agg_type == 'direct':
                # Direct mapping (e.g., age, gender)
                result_df[feature_name] = result_df[source_col]
                
            elif agg_type == 'age_group':
                # Create age groups
                bins = feature_config.get('bins', [0, 25, 35, 45, 55, 65, 100])
                labels = feature_config.get('labels', ['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
                result_df[feature_name] = pd.cut(result_df[source_col], bins=bins, labels=labels)
                
            elif agg_type == 'encode':
                # Encode categorical variables
                encoding_map = feature_config.get('encoding_map', {})
                result_df[feature_name] = result_df[source_col].map(encoding_map)
        
        return result_df
    
    def calculate_economic_features(self, config: Dict) -> pd.DataFrame:
        """
        Calculate economic features based on configuration.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary for economic features
            
        Returns:
        --------
        pd.DataFrame with economic features
        """
        result_df = self.df.copy()
        
        for feature_config in config:
            feature_name = feature_config['feature_name']
            calc_type = feature_config['calc_type']
            
            if calc_type == 'direct':
                source_col = feature_config['source_col']
                result_df[feature_name] = result_df[source_col]
                
            elif calc_type == 'ratio':
                numerator_col = feature_config['numerator_col']
                denominator_col = feature_config['denominator_col']
                result_df[feature_name] = result_df[numerator_col] / result_df[denominator_col].replace(0, np.nan)
                
            elif calc_type == 'loan_to_value':
                loan_col = feature_config['loan_col']
                value_col = feature_config['value_col']
                result_df[feature_name] = result_df[loan_col] / result_df[value_col].replace(0, np.nan)
                
            elif calc_type == 'debt_to_income':
                debt_col = feature_config['debt_col']
                income_col = feature_config['income_col']
                result_df[feature_name] = result_df[debt_col] / result_df[income_col].replace(0, np.nan)
        
        return result_df
    
    def calculate_rolling_features(self, config: Dict) -> pd.DataFrame:
        """
        Calculate rolling aggregation features with multiple time windows.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary for rolling features
            
        Returns:
        --------
        pd.DataFrame with rolling features
        """
        result_df = self.df.copy()
        
        groupby_col = config['groupby']
        value_col = config['value_col']
        agg_func = config['agg_func']
        periods = config['periods']  # e.g., [1, 3, 6, 12, 24] months
        feature_prefix = config['feature_prefix']
        
        for period in periods:
            feature_name = f"{feature_prefix}_{period}M"
            
            # Calculate rolling features
            result_df[feature_name] = result_df.groupby(groupby_col).apply(
                lambda x: self._calculate_period_agg(
                    x, value_col, agg_func, period, self.date_col
                )
            ).reset_index(level=0, drop=True)
        
        # Calculate ratio features between periods if requested
        if config.get('calculate_ratios', False):
            result_df = self._calculate_period_ratios(result_df, feature_prefix, periods)
        
        return result_df
    
    def _calculate_period_agg(self, group_df: pd.DataFrame, value_col: str, 
                              agg_func: str, months: int, date_col: str) -> pd.Series:
        """
        Helper function to calculate aggregation over a specific period.
        """
        result = []
        
        for idx, row in group_df.iterrows():
            current_date = row[date_col]
            start_date = current_date - pd.DateOffset(months=months)
            
            # Get historical data within the period (excluding current record)
            mask = (group_df[date_col] >= start_date) & (group_df[date_col] < current_date)
            period_data = group_df.loc[mask, value_col]
            
            if len(period_data) > 0:
                if agg_func == 'count':
                    agg_value = len(period_data)
                elif agg_func == 'sum':
                    agg_value = period_data.sum()
                elif agg_func == 'mean':
                    agg_value = period_data.mean()
                elif agg_func == 'max':
                    agg_value = period_data.max()
                elif agg_func == 'min':
                    agg_value = period_data.min()
                elif agg_func == 'std':
                    agg_value = period_data.std()
                else:
                    agg_value = np.nan
            else:
                agg_value = 0 if agg_func == 'count' else np.nan
            
            result.append(agg_value)
        
        return pd.Series(result, index=group_df.index)
    
    def _calculate_period_ratios(self, df: pd.DataFrame, feature_prefix: str, 
                                 periods: List[int]) -> pd.DataFrame:
        """
        Calculate ratio features between different time periods.
        """
        result_df = df.copy()
        
        for i in range(len(periods) - 1):
            short_period = periods[i]
            long_period = periods[i + 1]
            
            short_col = f"{feature_prefix}_{short_period}M"
            long_col = f"{feature_prefix}_{long_period}M"
            ratio_col = f"{feature_prefix}_Ratio_{short_period}M_to_{long_period}M"
            
            result_df[ratio_col] = result_df[short_col] / result_df[long_col].replace(0, np.nan)
        
        return result_df
    
    def calculate_application_behavior_features(self, config: List[Dict]) -> pd.DataFrame:
        """
        Calculate application behavior features (frequency, rejection patterns, etc.).
        
        Parameters:
        -----------
        config : list of dict
            Configuration list for application behavior features
            
        Returns:
        --------
        pd.DataFrame with application behavior features
        """
        result_df = self.df.copy()
        
        for feature_config in config:
            feature_type = feature_config['type']
            
            if feature_type == 'frequency':
                result_df = self.calculate_rolling_features(feature_config)
                
            elif feature_type == 'rejection_rate':
                groupby_col = feature_config['groupby']
                rejection_col = feature_config['rejection_col']
                periods = feature_config['periods']
                feature_prefix = feature_config['feature_prefix']
                
                for period in periods:
                    # Calculate number of rejections
                    rejection_count_col = f"Rejection_Count_{period}M"
                    result_df[rejection_count_col] = result_df.groupby(groupby_col).apply(
                        lambda x: self._calculate_period_agg(
                            x, rejection_col, 'sum', period, self.date_col
                        )
                    ).reset_index(level=0, drop=True)
                    
                    # Calculate total applications
                    total_count_col = f"Total_Applications_{period}M"
                    result_df[total_count_col] = result_df.groupby(groupby_col).apply(
                        lambda x: self._calculate_period_agg(
                            x, 'Application_Number', 'count', period, self.date_col
                        )
                    ).reset_index(level=0, drop=True)
                    
                    # Calculate rejection rate
                    rate_col = f"{feature_prefix}_{period}M"
                    result_df[rate_col] = result_df[rejection_count_col] / result_df[total_count_col].replace(0, np.nan)
                
            elif feature_type == 'time_since_last':
                groupby_col = feature_config['groupby']
                feature_name = feature_config['feature_name']
                
                result_df[feature_name] = result_df.groupby(groupby_col)[self.date_col].diff().dt.days
        
        return result_df
    
    def calculate_sales_agent_features(self, config: List[Dict]) -> pd.DataFrame:
        """
        Calculate sales agent acquisition behavior features.
        
        Parameters:
        -----------
        config : list of dict
            Configuration list for sales agent features
            
        Returns:
        --------
        pd.DataFrame with sales agent features
        """
        result_df = self.df.copy()
        
        for feature_config in config:
            feature_type = feature_config['type']
            
            if feature_type == 'agent_success_rate':
                agent_col = feature_config['agent_col']
                success_col = feature_config['success_col']
                periods = feature_config['periods']
                feature_prefix = feature_config['feature_prefix']
                
                for period in periods:
                    # Calculate success count
                    success_count_col = f"Agent_Success_Count_{period}M"
                    result_df[success_count_col] = result_df.groupby(agent_col).apply(
                        lambda x: self._calculate_period_agg(
                            x, success_col, 'sum', period, self.date_col
                        )
                    ).reset_index(level=0, drop=True)
                    
                    # Calculate total applications by agent
                    total_count_col = f"Agent_Total_Applications_{period}M"
                    result_df[total_count_col] = result_df.groupby(agent_col).apply(
                        lambda x: self._calculate_period_agg(
                            x, 'Application_Number', 'count', period, self.date_col
                        )
                    ).reset_index(level=0, drop=True)
                    
                    # Calculate success rate
                    rate_col = f"{feature_prefix}_{period}M"
                    result_df[rate_col] = result_df[success_count_col] / result_df[total_count_col].replace(0, np.nan)
            
            elif feature_type == 'agent_volume':
                result_df = self.calculate_rolling_features(feature_config)
            
            elif feature_type == 'agent_avg_amount':
                result_df = self.calculate_rolling_features(feature_config)
        
        return result_df
    
    def calculate_collateral_features(self, config: List[Dict]) -> pd.DataFrame:
        """
        Calculate collateral-related features.
        
        Parameters:
        -----------
        config : list of dict
            Configuration list for collateral features
            
        Returns:
        --------
        pd.DataFrame with collateral features
        """
        result_df = self.df.copy()
        
        for feature_config in config:
            feature_name = feature_config['feature_name']
            calc_type = feature_config['calc_type']
            
            if calc_type == 'direct':
                source_col = feature_config['source_col']
                result_df[feature_name] = result_df[source_col]
            
            elif calc_type == 'ratio':
                numerator_col = feature_config['numerator_col']
                denominator_col = feature_config['denominator_col']
                result_df[feature_name] = result_df[numerator_col] / result_df[denominator_col].replace(0, np.nan)
            
            elif calc_type == 'ownership_match':
                applicant_col = feature_config['applicant_col']
                owner_col = feature_config['owner_col']
                result_df[feature_name] = (result_df[applicant_col] == result_df[owner_col]).astype(int)
        
        return result_df
    
    def calculate_all_features(self, full_config: Dict) -> pd.DataFrame:
        """
        Calculate all features based on a complete configuration.
        
        Parameters:
        -----------
        full_config : dict
            Complete configuration dictionary containing all feature configs
            
        Returns:
        --------
        pd.DataFrame with all calculated features
        """
        result_df = self.df.copy()
        
        # Calculate demographic features
        if 'demographic' in full_config:
            result_df = self.calculate_demographic_features(full_config['demographic'])
        
        # Calculate economic features
        if 'economic' in full_config:
            result_df = self.calculate_economic_features(full_config['economic'])
        
        # Calculate application behavior features
        if 'application_behavior' in full_config:
            result_df = self.calculate_application_behavior_features(full_config['application_behavior'])
        
        # Calculate sales agent features
        if 'sales_agent' in full_config:
            result_df = self.calculate_sales_agent_features(full_config['sales_agent'])
        
        # Calculate collateral features
        if 'collateral' in full_config:
            result_df = self.calculate_collateral_features(full_config['collateral'])
        
        return result_df


def apply_feature_engineering(df: pd.DataFrame, config: Dict, 
                              date_col: str = 'Application_Date',
                              entity_col: str = 'ID/KTP/PASPOR/KITAS') -> pd.DataFrame:
    """
    Main function to apply feature engineering based on configuration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : dict
        Feature engineering configuration
    date_col : str
        Date column name
    entity_col : str
        Entity identifier column name
        
    Returns:
    --------
    pd.DataFrame with engineered features
    """
    calculator = FeatureCalculator(df, date_col, entity_col)
    result_df = calculator.calculate_all_features(config)
    
    return result_df