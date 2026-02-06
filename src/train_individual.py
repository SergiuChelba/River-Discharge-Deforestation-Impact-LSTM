import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys

# --- CONFIGURARE ---
FULL_DATASET = 'data/raw/date_finale.csv'
MODELS_DIR = 'models'

# Ariile sunt necesare DOAR pentru a calcula debitul istoric pe care îl învață modelul
BAZIN_AREAS = {
    'Suceava_Itcani': 2200,
    'Moldova_Tupilati': 4000,
    'Bistrita_Frumosu': 2800,
    'Neagra_Brosteni': 350
}

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def train_separate_brains():
    print("🚀 Începem antrenarea a 4 MODELE DISTINCTE...")

    # 1. Citire Date
    if not os.path.exists(FULL_DATASET):
        print(f"❌ Nu găsesc {FULL_DATASET}")
        return

    print("📖 Citesc CSV-ul mare...")
    df = pd.read_csv(FULL_DATASET, low_memory=False)
    df['Date_Time'] = pd.to_datetime(
        df['Date_Time'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Date_Time'])

    # Conversii numerice
    df['Runoff_mm'] = pd.to_numeric(df['Runoff_mm'], errors='coerce').fillna(0)
    rain_col = 'Ploaie_mm' if 'Ploaie_mm' in df.columns else 'Ploiaie_mm'
    df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0)

    # Adăugare Hansen (Defrișări)
    if os.path.exists('data/raw/hansen_real.csv'):
        hansen = pd.read_csv('data/raw/hansen_real.csv')
        hansen['Cumulative_Loss_Ha'] = hansen.groupby(
            'Nume')['Forest_Loss_Ha'].cumsum()
        df['Year'] = df['Date_Time'].dt.year
        df = pd.merge(df, hansen[['Year', 'Nume', 'Cumulative_Loss_Ha']],
                      left_on=['Year', 'Nume_Bazin'], right_on=['Year', 'Nume'], how='left')
        df['Cumulative_Loss_Ha'] = df['Cumulative_Loss_Ha'].ffill().fillna(0)
    else:
        df['Cumulative_Loss_Ha'] = 0

    df['Month'] = df['Date_Time'].dt.month

    # 2. Iterăm prin fiecare bazin
    bazine = df['Nume_Bazin'].unique()

    for bazin in bazine:
        safe_name = str(bazin).replace(" ", "_")
        print(f"\n🌊 --- ANTRENARE: {bazin} ---")

        # A. Filtrăm datele DOAR pentru acest bazin
        bazin_df = df[df['Nume_Bazin'] == bazin].copy()
        bazin_df = bazin_df.sort_values('Date_Time')

        # B. Calculăm TARGET-ul (Debitul Real m3/s)
        # Modelul va învăța direct să scoată m3/s specifici acestui râu
        aria = BAZIN_AREAS.get(safe_name, 1000)
        bazin_df['Debit_Target'] = (
            bazin_df['Runoff_mm'] * aria * 1000) / 86400

        # C. Features (Inputuri)
        # [Ploaie, Debit_Anterior, Defrișare, Luna]
        features = [rain_col, 'Debit_Target', 'Cumulative_Loss_Ha', 'Month']
        target = ['Debit_Target']

        # D. Scalare (CRUCIAL: Fiecare bazin are scara lui!)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X = bazin_df[features].values
        y = bazin_df[target].values

        # Antrenăm scalerele doar pe acest bazin
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Salvare Scalere
        joblib.dump(scaler_x, f'{MODELS_DIR}/scaler_x_{safe_name}.pkl')
        joblib.dump(scaler_y, f'{MODELS_DIR}/scaler_y_{safe_name}.pkl')

        # E. Creare Secvențe (LSTM)
        SEQ_LENGTH = 30
        X_seq, y_seq = [], []
        if len(X_scaled) > SEQ_LENGTH:
            for i in range(len(X_scaled) - SEQ_LENGTH):
                X_seq.append(X_scaled[i:(i + SEQ_LENGTH)])
                y_seq.append(y_scaled[i + SEQ_LENGTH])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            # F. Definire Model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 4)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(1)  # Ieșire: Debit (scaled)
            ])

            model.compile(optimizer='adam', loss='mse')

            # Early Stopping (Se oprește dacă nu mai învață -> Economisește timp)
            early_stop = EarlyStopping(
                monitor='val_loss', patience=2, restore_best_weights=True)

            print(f"⏳ Se antrenează... (Max 10 epoci)")
            model.fit(X_seq, y_seq, batch_size=32, epochs=10,
                      validation_split=0.1, callbacks=[early_stop], verbose=1)

            # Salvare
            model.save(f'{MODELS_DIR}/lstm_{safe_name}.h5')
            print(f"✅ Gata {bazin}! Model salvat.")
        else:
            print(f"⚠️ Prea puține date pentru {bazin}. Sar peste.")

    print("\n✨ TOATE CELE 4 MODELE AU FOST ANTRENATE CU SUCCES!")


if __name__ == "__main__":
    train_separate_brains()
