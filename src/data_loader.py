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
        if self.df is None:
            # Optimizare memorie: specificăm tipurile de date pentru a consuma mai puțin RAM
            dtypes = {
                'Ploaie_mm': 'float32',
                'Temp_C': 'float32',
                'Runoff_mm': 'float32',
                'Nume_Bazin': 'category'
            }
            self.df = pd.read_csv(self.file_path, dtype=dtypes)
            
            if 'Date_Time' in self.df.columns:
                self.df['Date_Time'] = pd.to_datetime(self.df['Date_Time'])
                self.df = self.df.set_index('Date_Time').sort_index()
            
        return self.df

    def get_basin_data(self, basin_name: str) -> pd.DataFrame:
        """Filters data for a specific river basin."""
        if self.df is None:
            self.load_data()
            
        # Returnăm o copie ca să nu afectăm originalul
        return self.df[self.df['Nume_Bazin'] == basin_name].copy()

    def get_resampled_data(self, basin_name: str, rule='D') -> pd.DataFrame:
        """
        OPTIMIZARE CRITICĂ: Agregare date pentru vizualizare rapidă.
        rule='D' înseamnă Daily (Zilnic).
        Folosim 'max' pentru Runoff/Ploaie ca să nu pierdem vârfurile de viitură.
        """
        df_basin = self.get_basin_data(basin_name)
        
        # Resampling inteligent:
        # - Ploaia și Runoff: luăm MAXIMUL (să vedem inundația) sau SUMA (volumul total)
        # - Temperatura: luăm MEDIA
        agg_dict = {
            'Ploaie_mm': 'sum',   # Vrem volumul total de ploaie pe zi
            'Runoff_mm': 'max',   # Vrem cel mai mare debit din acea zi (Vârful viiturii)
            'Temp_C': 'mean'      # Temperatura medie a zilei
        }
        
        # Facem resampling doar pe coloanele numerice
        resampled = df_basin.resample(rule).agg(agg_dict)
        return resampled

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
        """Normalizes data between 0 and 1."""
        scaled_data = self.scaler.fit_transform(df.values)
        return scaled_data, self.scaler

    def create_sequences(self, data: np.ndarray, seq_length: int, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, target_idx])
            
        return np.array(X), np.array(y)