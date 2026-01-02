import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from data_loader import FloodDataLoader

# --- CONFIGURARE ---
SEQ_LENGTH = 30  # Ne uităm la ultimele 30 de zile pentru a prezice ziua următoare
EPOCHS = 20      # Cât timp antrenăm (epoci)
BATCH_SIZE = 32

# Configurare Bazine (Aceleași ca în Dashboard pentru consistență)
BAZIN_AREAS = {
    'Suceava_Itcani': 2200, 'Moldova_Tupilati': 4000, 
    'Bistrita_Frumosu': 2800, 'Neagra_Brosteni': 350
}

def prepare_data():
    print("🔄 Loading and preparing data...")
    
    # 1. Încărcare Date (Aceeași logică ca la Dashboard)
    loader = FloodDataLoader('data/raw/final_dataset.csv')
    df = loader.load_data().reset_index()
    
    # Conversii numerice
    df['Runoff_mm'] = pd.to_numeric(df['Runoff_mm'], errors='coerce')
    rain_col = 'Ploaie_mm' if 'Ploaie_mm' in df.columns else 'Ploiaie_mm'
    df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce')
    
    # Suprafața
    df['Nume_Bazin'] = df['Nume_Bazin'].astype(str)
    df['Area_km2'] = df['Nume_Bazin'].map(BAZIN_AREAS).fillna(1000)

    # 2. Integrare Hansen (Pentru ca AI-ul să învețe efectul defrișării)
    try:
        hansen = pd.read_csv('data/raw/hansen_real.csv')
        hansen = hansen.sort_values('Year')
        hansen['Cumulative_Loss_Ha'] = hansen.groupby('Nume')['Forest_Loss_Ha'].cumsum()
        
        df['Year'] = df['Date_Time'].dt.year
        df = pd.merge(df, hansen[['Year', 'Nume', 'Cumulative_Loss_Ha']], 
                     left_on=['Year', 'Nume_Bazin'], right_on=['Year', 'Nume'], how='left')
        df['Cumulative_Loss_Ha'] = df['Cumulative_Loss_Ha'].fillna(0)
    except:
        print("⚠️ Warning: Hansen data not found. Training without deforestation features.")
        df['Cumulative_Loss_Ha'] = 0

    # 3. Calculăm Debitul Total (Target-ul nostru)
    # Folosim logica calibrată din Dashboard
    df['Groundwater'] = df.groupby('Nume_Bazin')[rain_col].transform(lambda x: x.rolling(2160, min_periods=1).mean())
    df['Baseflow'] = df['Groundwater'] * df['Area_km2'] * 0.004
    
    # Impact Defrișare
    max_loss = 15000.0
    impact = (df['Cumulative_Loss_Ha'] / max_loss).clip(upper=1.0)
    
    # Debit Final (Target)
    factor_seceta = 1.0 - (impact * 0.4)
    factor_viitura = 1.0 + (impact * 0.6)
    
    baseflow_final = df['Baseflow'] * factor_seceta
    viitura_final = df['Runoff_mm'] * df['Area_km2'] * 0.15 * factor_viitura
    
    df['Debit_Target'] = baseflow_final + viitura_final
    
    # Selectăm doar coloanele relevante pentru antrenare
    # Input: Ploaie, Runoff, Defrișare, Luna (sezonalitate)
    # Output: Debit_Target
    df['Month'] = df['Date_Time'].dt.month
    feature_cols = [rain_col, 'Runoff_mm', 'Cumulative_Loss_Ha', 'Month']
    target_col = 'Debit_Target'
    
    data = df[feature_cols + [target_col]].dropna()
    
    return data, feature_cols, target_col

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train():
    # 1. Pregătire Date
    df, feature_cols, target_col = prepare_data()
    
    # Scalare (LSTM are nevoie de date între 0 și 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Salvează scaler-ul pentru a-l folosi la predicții viitoare
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Separăm Input (X) și Output (Y)
    X = scaled_data[:, :-1] # Toate coloanele mai puțin ultima
    y = scaled_data[:, -1]  # Ultima coloană (Debit)
    
    # Creăm secvențe temporale
    print(f"⏳ Creating sequences ({SEQ_LENGTH} days history)...")
    X_seq, y_seq = create_sequences(X, SEQ_LENGTH) # Aici e o mică eroare logică în matrice, corectez mai jos
    
    # FIX: Trebuie să scalăm separat features și target pentru a putea inversa scalarea corect
    # Refacem scalarea corect
    scaler_x = MinMaxScaler()
    X_raw = df[feature_cols].values
    X_scaled = scaler_x.fit_transform(X_raw)
    
    scaler_y = MinMaxScaler()
    y_raw = df[[target_col]].values
    y_scaled = scaler_y.fit_transform(y_raw)
    
    joblib.dump(scaler_x, 'models/scaler_x.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    
    # Creare secvente corecte
    X_final, y_final = [], []
    for i in range(len(X_scaled) - SEQ_LENGTH):
        X_final.append(X_scaled[i:(i + SEQ_LENGTH)])
        y_final.append(y_scaled[i + SEQ_LENGTH])
        
    X_final, y_final = np.array(X_final), np.array(y_final)
    
    # Split Train/Test (80% antrenare, 20% testare)
    train_size = int(len(X_final) * 0.8)
    X_train, X_test = X_final[:train_size], X_final[train_size:]
    y_train, y_test = y_final[:train_size], y_final[train_size:]
    
    print(f"🚀 Training LSTM on {len(X_train)} samples...")

    # 2. Construire Model LSTM
    model = Sequential([
        # Strat LSTM 1: Învață secvențe complexe
        LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, len(feature_cols))),
        Dropout(0.2), # Previne overfitting-ul
        
        # Strat LSTM 2: Rafinează memoria
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        
        # Strat Dens: Predicția finală
        Dense(25),
        Dense(1) # O singură valoare: Debitul prezis
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 3. Antrenare efectivă
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
    
    # 4. Salvare
    model.save('models/lstm_flood_model.h5')
    print("✅ Model trained and saved to 'models/lstm_flood_model.h5'")

if __name__ == "__main__":
    train()