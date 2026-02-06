import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

st.set_page_config(page_title="Sistem Hidrologic Complet", layout="wide")
st.title("🌲 Sistem Hidrologic: Analiză Completă (Baseflow + Runoff)")

BAZIN_AREAS = {
    'Suceava_Itcani': 2200,
    'Moldova_Tupilati': 4000,
    'Bistrita_Frumosu': 2800,
    'Neagra_Brosteni': 350
}

# --- 2. COMPONENTA HIDROGEOLOGICĂ (DEBIT DE BAZĂ) ---
# Acesta este debitul care curge PERMANENT, din izvoare, chiar dacă nu plouă.
# Valori medii multianuale minime (estimpri hidrologice).
BASEFLOW = {
    'Suceava_Itcani': 12.0,   # Râu mare, nu scade sub 12 mc/s
    'Moldova_Tupilati': 18.0,  # Cel mai mare
    'Bistrita_Frumosu': 15.0,  # Regulat
    'Neagra_Brosteni': 3.5    # Pârâu de munte, debit mic dar constant
}

# --- 3. CALIBRARE DINAMICĂ (PENTRU VIITURI) ---


def get_dynamic_calibration(rain_mm, basin_name):
    """
    Transformă scurgerea medie zilnică în debit de VÂRF (Viitură).
    La ploi mici -> Factor mic (scurgere lentă).
    La ploi torențiale -> Factor uriaș (Flash Flood), mai ales pe râuri mici.
    """
    is_flashy = (
        basin_name == 'Neagra_Brosteni')  # Neagra e "nervoasă" (torențială)

    if rain_mm < 15:
        return 1.2  # Ploaie normală, debitul crește puțin peste medie

    elif rain_mm < 40:
        # Ploaie serioasă
        return 5.0 if is_flashy else 3.0

    elif rain_mm < 60:
        # Cod Portocaliu
        return 15.0 if is_flashy else 8.0

    else:
        # COD ROȘU (70mm+)
        # Aici e secretul: Pe Neagra, apa vine toată odată.
        # Satelitul vede o medie, noi trebuie să reconstruim VÂRFUL.
        return 35.0 if is_flashy else 15.0

# --- 4. MODEL LOADING ---


def load_specific_brain(basin_name):
    try:
        safe_name = str(basin_name).replace(" ", "_")
        model_path = os.path.join(root_dir, f'models/lstm_{safe_name}.h5')
        scaler_x_path = os.path.join(
            root_dir, f'models/scaler_x_{safe_name}.pkl')
        scaler_y_path = os.path.join(
            root_dir, f'models/scaler_y_{safe_name}.pkl')

        if not os.path.exists(model_path):
            return None, None, None, f"⚠️ Modelul pentru {basin_name} nu e gata."

        model = load_model(model_path, compile=False)
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_x, scaler_y, None
    except Exception as e:
        return None, None, None, str(e)

# --- 5. DATE ---


@st.cache_data
def load_data_complete():
    try:
        csv_path = os.path.join(root_dir, 'data/raw/date_finale.csv')
        df = pd.read_csv(csv_path, low_memory=False)
        df['Date_Time'] = pd.to_datetime(
            df['Date_Time'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Date_Time'])

        df['Nume_Bazin'] = df['Nume_Bazin'].astype(str)
        df['Runoff_mm'] = pd.to_numeric(
            df['Runoff_mm'], errors='coerce').fillna(0)
        rain_col = 'Ploaie_mm' if 'Ploaie_mm' in df.columns else 'Ploiaie_mm'
        df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0)
        df['Area_km2'] = df['Nume_Bazin'].map(BAZIN_AREAS).fillna(1000)

        # --- CALCUL DEBIT ISTORIC (CU BASEFLOW) ---
        # Q_total = Q_scurgere + Q_baza
        def calc_hist_q(row):
            base = BASEFLOW.get(row['Nume_Bazin'], 5.0)
            runoff_q = (row['Runoff_mm'] * row['Area_km2'] * 1000) / 86400
            # Pentru istoric, folosim un factor mediu de calibrare (ex: 5)
            return (runoff_q * 5.0) + base

        df['Debit_Istoric_Afisare'] = df.apply(calc_hist_q, axis=1)

        # Hansen
        if os.path.exists(os.path.join(root_dir, 'data/raw/hansen_real.csv')):
            hansen = pd.read_csv(os.path.join(
                root_dir, 'data/raw/hansen_real.csv'))
            hansen['Cumulative_Loss_Ha'] = hansen.groupby(
                'Nume')['Forest_Loss_Ha'].cumsum()
            df['Year'] = df['Date_Time'].dt.year
            df = pd.merge(df, hansen[['Year', 'Nume', 'Cumulative_Loss_Ha']],
                          left_on=['Year', 'Nume_Bazin'], right_on=['Year', 'Nume'], how='left')
            df['Cumulative_Loss_Ha'] = df['Cumulative_Loss_Ha'].ffill().fillna(0)
        else:
            df['Cumulative_Loss_Ha'] = 0.0

        df = df.set_index('Date_Time').sort_index()
        return df, None
    except Exception as e:
        return None, str(e)


data, error_msg = load_data_complete()
if data is None:
    st.error(f"Eroare date: {error_msg}")
    st.stop()

# --- UI ---
selected_basin = st.sidebar.selectbox(
    "📍 Alege Bazinul", data['Nume_Bazin'].unique())
basin_data = data[data['Nume_Bazin'] == selected_basin]
model, scaler_x, scaler_y, err = load_specific_brain(selected_basin)

# Statistici
max_debit_historic = basin_data['Debit_Istoric_Afisare'].max()
mean_debit_historic = basin_data['Debit_Istoric_Afisare'].mean()
base_flow_current = BASEFLOW.get(selected_basin, 5.0)

tab1, tab2 = st.tabs(["📊 Regim Hidrologic", "🧠 Simulator Viituri"])

with tab1:
    st.subheader(f"Istoric: {selected_basin}")
    st.write(f"ℹ️ **Debit de Bază (Fără ploaie):** {base_flow_current} m³/s")

    basin_data['DayOfYear'] = basin_data.index.dayofyear
    periods = {"2000-2010": (2000, 2010, 'green'),
               "2021-2025": (2021, 2025, 'red')}

    fig = go.Figure()
    for label, (start, end, color) in periods.items():
        mask = (basin_data.index.year >= start) & (
            basin_data.index.year <= end)
        daily_avg = basin_data[mask].groupby(
            'DayOfYear')['Debit_Istoric_Afisare'].mean()
        smoothed = daily_avg.rolling(
            window=30, center=True, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=smoothed.index, y=smoothed,
                      name=label, line=dict(color=color, width=2)))
    fig.update_layout(title="Debit Mediu Zilnic (m³/s)",
                      height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"🧠 Simulator Viituri ({selected_basin})")

    c1, c2 = st.columns([1, 2])
    with c1:
        rain_input = st.slider("🌧️ Ploaie (mm/24h)", 0, 100, 10)

        # Feedback Calibrare
        factor = get_dynamic_calibration(rain_input, selected_basin)
        if rain_input == 0:
            st.info(
                f"Fără ploaie. Rămâne doar debitul de bază: {base_flow_current} m³/s")
        elif rain_input > 60:
            st.error(f"🔴 VIITURĂ ISTORICĂ! Factor multiplicare: x{factor}")

        deforest_input = st.slider("🪓 Defrișare Extra (Ha)", 0, 5000, 0)

        if st.button("PREZICE DEBIT", type="primary"):
            run_ai = True
        else:
            run_ai = False

    with c2:
        if run_ai and model:
            # 1. Date Intrare
            last_30 = basin_data.tail(30).copy()
            rain_col = 'Ploaie_mm' if 'Ploaie_mm' in last_30.columns else 'Ploiaie_mm'

            # Scenariu
            idx = last_30.index[-1]
            last_30.at[idx, rain_col] = rain_input
            last_30.at[idx, 'Cumulative_Loss_Ha'] += deforest_input
            last_30['Month'] = last_30.index.month

            # Reconstruim target-ul brut (ca la antrenare)
            last_30['Debit_Target'] = (
                last_30['Runoff_mm'] * last_30['Area_km2'] * 1000) / 86400
            input_df = last_30[[rain_col, 'Debit_Target',
                                'Cumulative_Loss_Ha', 'Month']].copy()

            try:
                # 2. Predicție AI (Componenta Meteo)
                input_scaled = scaler_x.transform(input_df.values)
                input_tensor = input_scaled.reshape(1, 30, 4)

                pred_scaled = model.predict(input_tensor, verbose=0)
                pred_raw = scaler_y.inverse_transform(
                    pred_scaled)[0][0]  # Debitul generat DOAR de ploaie

                if pred_raw < 0:
                    pred_raw = 0.0

                # 3. COMPUNERE FINALĂ (FIZICĂ)
                # Debit Total = (Debit Ploaie * Factor Viitură) + Debit Bază
                # Factorul transformă scurgerea medie în vârf de viitură

                debit_viitura = pred_raw * factor
                debit_final = debit_viitura + base_flow_current

                # Afișare
                st.metric("Debit Vârf Estimat", f"{debit_final:.0f} m³/s",
                          delta=f"Din care Bază: {base_flow_current} m³/s")

                # Gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=debit_final,
                    title={
                        'text': f"Nivel Risc<br><span style='font-size:0.8em;color:gray'>Max Istoric: {max_debit_historic:.0f} m³/s</span>"},
                    gauge={'axis': {'range': [0, max_debit_historic * 1.2]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, mean_debit_historic *
                                          1.5], 'color': "lightgreen"},
                               {'range': [
                                   mean_debit_historic * 1.5, max_debit_historic * 0.9], 'color': "orange"},
                               {'range': [max_debit_historic * 0.9,
                                          max_debit_historic * 1.5], 'color': "red"}
                    ]}
                ))
                st.plotly_chart(fig_g, use_container_width=True)

            except Exception as e:
                st.error(f"Eroare: {e}")
