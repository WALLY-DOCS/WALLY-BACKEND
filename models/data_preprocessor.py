import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

class AdvancedDataPreprocessor:
    """Enhanced data preprocessing with feature engineering and validation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_encoders = {}
        self.feature_importance = {}
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._validate_input(data)
        data = self._create_temporal_features(data)
        data = self._create_categorical_features(data)
        data = self._handle_missing_values(data)
        data = self._create_advanced_features(data)
        return data
    
    def _validate_input(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['Order Date', 'Sales', 'Category']
        assert all(col in data.columns for col in required_cols)
        
        data = data.drop_duplicates().reset_index(drop=True)
        data = data.dropna(subset=['Order Date', 'Sales'])
        return data
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Order Date'] = pd.to_datetime(data['Order Date'])
        
        # Basic temporal features
        data['Year'] = data['Order Date'].dt.year
        data['Month'] = data['Order Date'].dt.month
        data['DayOfWeek'] = data['Order Date'].dt.dayofweek
        data['Quarter'] = data['Order Date'].dt.quarter
        data['WeekOfYear'] = data['Order Date'].dt.isocalendar().week
        
        # Cyclic encoding
        data['Month_Sin'] = np.sin(2 * np.pi * data['Month']/12)
        data['Month_Cos'] = np.cos(2 * np.pi * data['Month']/12)
        data['DayOfWeek_Sin'] = np.sin(2 * np.pi * data['DayOfWeek']/7)
        data['DayOfWeek_Cos'] = np.cos(2 * np.pi * data['DayOfWeek']/7)
        
        return data
    
    def _create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # Revenue features
        data['MonthlyRevenue'] = data.groupby(['Year', 'Month'])['Sales'].transform('sum')
        data['CategoryRevenue'] = data.groupby('Category')['Sales'].transform('sum')
        
        # Time series features
        for lag in [1, 3, 6, 12]:
            data[f'Sales_Lag_{lag}'] = data.groupby('Category')['Sales'].shift(lag)
            data[f'Sales_Rolling_Mean_{lag}'] = (
                data.groupby('Category')['Sales']
                .rolling(lag, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            data[f'Sales_Rolling_Std_{lag}'] = (
                data.groupby('Category')['Sales']
                .rolling(lag, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
        
        return data
