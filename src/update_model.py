import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys

# --- CONFIGURARE ---
FULL_DATASET = 'data/raw/date_finale.csv'
MODEL_PATH = 'models/lstm_flood_model.h5'
SCALER_X_PATH = 'models/scaler_x.pkl'
SCALER_Y_PATH = 'models/scaler_y.pkl'

# Definim ariile reale (Modelul trebuie să le învețe!)
BAZIN_AREAS = {
    'Suceava_Itcani': 2200, 
    'Moldova_Tupilati': 4000, 
    'Bistrita_Frumosu': 2800, 
    'Neagra_Brosteni': 350
}

def train_smart_model():
    print("🧠 Începem antrenarea modelului REAL (care știe suprafața bazinelor)...")
    
    # 1. Încărcare Date
    if not os.path.exists(FULL_DATASET):
        print("❌ Nu găsesc date_finale.csv")
        return

    df = pd.read_csv(FULL_DATASET, low_memory=False)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Date_Time'])
    
    # Conversii numerice
    df['Runoff_mm'] = pd.to_numeric(df['Runoff_mm'], errors='coerce').fillna(0)
    rain_col = 'Ploaie_mm' if 'Ploaie_mm' in df.columns else 'Ploiaie_mm'
    df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0)
    
    # --- CRITIC: ADĂUGĂM ARIA ÎN DATE ---
    df['Nume_Bazin'] = df['Nume_Bazin'].astype(str)
    df['Area_km2'] = df['Nume_Bazin'].map(BAZIN_AREAS).fillna(1000) # Fallback 1000
    
    # Calculăm Debitul REAL (Istoric) pe care să îl învețe modelul
    # Q = (Runoff * Area * 1000) / 86400
    df['Debit_Reale_m3s'] = (df['Runoff_mm'] * df['Area_km2'] * 1000) / 86400

    # Defrișări (Hansen)
    if os.path.exists('data/raw/hansen_real.csv'):
        try:
            hansen = pd.read_csv('data/raw/hansen_real.csv')
            hansen['Cumulative_Loss_Ha'] = hansen.groupby('Nume')['Forest_Loss_Ha'].cumsum()
            df['Year'] = df['Date_Time'].dt.year
            df = pd.merge(df, hansen[['Year', 'Nume', 'Cumulative_Loss_Ha']], 
                         left_on=['Year', 'Nume_Bazin'], right_on=['Year', 'Nume'], how='left')
            df['Cumulative_Loss_Ha'] = df['Cumulative_Loss_Ha'].ffill().fillna(0)
        except:
            df['Cumulative_Loss_Ha'] = 0
    else:
        df['Cumulative_Loss_Ha'] = 0

    df['Month'] = df['Date_Time'].dt.month

    # 2. PREGĂTIRE INPUT (FEATURES)
    # Acum includem 'Area_km2' ca să știe diferența dintre Neagra și Moldova!
    features = [rain_col, 'Runoff_mm', 'Cumulative_Loss_Ha', 'Month', 'Area_km2']
    target = ['Debit_Reale_m3s'] # Modelul va învăța să prezică direct debitul corect
    
    print(f"Features folosite: {features}")
    
    # Scalare
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X = df[features].values
    y = df[target].values
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Salvare Scalere
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    
    # Creare Secvențe (LSTM are nevoie de istoric)
    SEQ_LENGTH = 30
    X_seq, y_seq = [], []
    
    # Facem secvențe separat pentru fiecare bazin (ca să nu amestecăm datele)
    for bazin in df['Nume_Bazin'].unique():
        bazin_df = df[df['Nume_Bazin'] == bazin]
        # Trebuie să refacem X_scaled doar pentru acest bazin
        # (Pentru simplitate în acest script rapid, folosim indexarea globală, dar cu grijă)
        indices = df.index[df['Nume_Bazin'] == bazin].tolist()
        
        # Luăm datele scalate corespunzătoare acestor indici
        bazin_data = X_scaled[indices]
        bazin_target = y_scaled[indices]
        
        if len(bazin_data) > SEQ_LENGTH:
            for i in range(len(bazin_data) - SEQ_LENGTH):
                X_seq.append(bazin_data[i:(i + SEQ_LENGTH)])
                y_seq.append(bazin_target[i + SEQ_LENGTH])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"📦 Date pregătite: {X_seq.shape}")

    # 3. CREARE MODEL LSTM
    model = Sequential([
        # Input shape: (30 zile, 5 variabile)
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 5)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1) # Prezice Debitul
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print("🚀 Începe antrenarea (poate dura 2-3 minute)...")
    # Epochs puține pentru test rapid, poți crește la 10-20
    model.fit(X_seq, y_seq, batch_size=64, epochs=5, validation_split=0.1)
    
    model.save(MODEL_PATH)
    print("✅ MODEL NOU SALVAT! Acum știe diferența între bazine.")

if __name__ == "__main__":
    train_smart_model()