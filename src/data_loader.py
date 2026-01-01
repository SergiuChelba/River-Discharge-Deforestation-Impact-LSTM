import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

class FloodDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.scaler = MinMaxScaler()
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.file_path)
        
        if 'Date_Time' in self.df.columns:
            self.df['Date_Time'] = pd.to_datetime(self.df['Date_Time'])
            self.df = self.df.set_index('Date_Time').sort_index()
            
        return self.df

    def get_basin_data(self, basin_name: str) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
            
        basin_df = self.df[self.df['Nume_Bazin'] == basin_name].copy()
        
        return basin_df.drop(columns=['Nume_Bazin'])

    def preprocess(self, df: pd.DataFrame, target_col: str = 'Runoff_mm') -> Tuple[np.ndarray, MinMaxScaler]:
        scaled_data = self.scaler.fit_transform(df.values)
        return scaled_data, self.scaler

    def create_sequences(self, data: np.ndarray, seq_length: int, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, target_idx])
            
        return np.array(X), np.array(y)