import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

# Setup cai
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.data_loader import FloodDataLoader

st.set_page_config(page_title="Sistem Hidrologic Complet", layout="wide")
st.title("🌲 Sistem de Analiză și Predicție a Riscului la Inundații")
st.markdown("### Analiză Statistică Decadală & Simulator de Scenarii")

BAZIN_AREAS = {
    'Suceava_Itcani': 2200, 'Moldova_Tupilati': 4000,    
    'Bistrita_Frumosu': 2800, 'Neagra_Brosteni': 350       
}

@st.cache_data
def load_data_complete():
    try:
        loader = FloodDataLoader('data/raw/final_dataset.csv')
        df = loader.load_data().reset_index()
    except:
        return None, "Lipsesc datele."

    df['Nume_Bazin'] = df['Nume_Bazin'].astype(str)
    df['Runoff_mm'] = pd.to_numeric(df['Runoff_mm'], errors='coerce').fillna(0)
    rain_col = 'Ploaie_mm' if 'Ploaie_mm' in df.columns else 'Ploiaie_mm'
    df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0)
    df['Area_km2'] = df['Nume_Bazin'].map(BAZIN_AREAS).fillna(1000)

    try:
        hansen = pd.read_csv('data/raw/hansen_real.csv')
        hansen = hansen.sort_values('Year')
        hansen['Cumulative_Loss_Ha'] = hansen.groupby('Nume')['Forest_Loss_Ha'].cumsum()
        df['Year'] = df['Date_Time'].dt.year
        df = pd.merge(df, hansen[['Year', 'Nume', 'Cumulative_Loss_Ha']], 
                     left_on=['Year', 'Nume_Bazin'], right_on=['Year', 'Nume'], how='left')
        df['Cumulative_Loss_Ha'] = df['Cumulative_Loss_Ha'].fillna(0)
    except:
        df['Cumulative_Loss_Ha'] = 0.0

    # Model Hidrologic pentru Grafice
    max_loss = 15000.0 
    impact_factor = (df['Cumulative_Loss_Ha'] / max_loss).clip(upper=1.0)
    
    gw_slow = df.groupby('Nume_Bazin')[rain_col].transform(lambda x: x.rolling(2160, min_periods=1).mean())
    gw_fast = df.groupby('Nume_Bazin')[rain_col].transform(lambda x: x.rolling(480, min_periods=1).mean())
    
    df['Baseflow_Final'] = ((gw_slow * (1.0 - impact_factor)) + (gw_fast * impact_factor)) * df['Area_km2'] * 0.004
    factor_torentialitate = 1.0 + (impact_factor * 0.6)
    df['Debit_Total'] = df['Baseflow_Final'] + (df['Runoff_mm'] * df['Area_km2'] * 0.15 * factor_torentialitate)
    
    df = df.set_index('Date_Time').sort_index()
    return df, None

data, _ = load_data_complete()

if data is None:
    st.error("Eroare date.")
    st.stop()

selected_basin = st.sidebar.selectbox("Alege Râul:", data['Nume_Bazin'].unique())
basin_data = data[data['Nume_Bazin'] == selected_basin]

tab1, tab2, tab3 = st.tabs(["📊 Analiză Statistică", "🌊 Hidrograf Istoric", "🔮 Simulator Scenarii"])

# --- TAB 1: ANALIZA STATISTICĂ ---
with tab1:
    st.subheader("Dovada Statistică: Impactul Defrișărilor pe Termen Lung")
    basin_data['DayOfYear'] = basin_data.index.dayofyear
    era1 = basin_data[basin_data.index.year <= 2010].groupby('DayOfYear')['Baseflow_Final'].mean()
    era2 = basin_data[basin_data.index.year >= 2013].groupby('DayOfYear')['Baseflow_Final'].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=era1.index, y=era1, name="Media 2000-2010 (Stabil)", line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=era2.index, y=era2, name="Media 2013-2024 (Instabil)", line=dict(color='red', width=3), fill='tonexty', fillcolor='rgba(255,0,0,0.1)'))
    fig.update_layout(template="plotly_white", title="Scăderea Medie a Debitului de Bază", yaxis_title="Debit (m³/s)")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: HIDROGRAF ---
with tab2:
    st.subheader("Hidrograf Complet")
    daily = basin_data.resample('W').mean(numeric_only=True)
    fig2 = px.line(daily, y='Debit_Total', title="Evoluția Debitului Total (2000-2024)")
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: SIMULATOR ROBUST (FIZIC) ---
with tab3:
    st.subheader("🤖 Simulator de Risc Hidrologic")
    st.markdown("Simulează răspunsul râului la scenarii de ploaie și defrișare.")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.info("🛠️ Configurare Scenariu")
        input_rain = st.slider("🌧️ Cantitate Ploaie (litri/mp)", 0, 150, 20)
        
        current_loss = basin_data['Cumulative_Loss_Ha'].max()
        input_deforest = st.slider("🪓 Defrișare Suplimentară (Ha)", 0, 5000, 0)
        total_loss_scenario = current_loss + input_deforest
        
        # MODEL FIZIC: PRAG DE SATURAȚIE
        prag_saturatie = 35.0 
        prag_actual = prag_saturatie - (total_loss_scenario / 1000.0)
        if prag_actual < 10: prag_actual = 10
        
        if input_rain < prag_actual:
            runoff_efectiv = input_rain * 0.05
        else:
            exces = input_rain - prag_actual
            runoff_efectiv = (prag_actual * 0.05) + (exces * 0.45)
            
        st.write(f"**Prag Saturație Sol:** {prag_actual:.1f} litri")

    with col_sim2:
        st.success("🌊 Rezultatul Simulării")
        
        # CHEIA UNICĂ ESTE AICI (key="btn_simulare_final")
        if st.button("CALCULEAZĂ RISC", type="primary", key="btn_simulare_final"):
            area_km2 = BAZIN_AREAS.get(selected_basin, 1000)
            
            # Calcul Fizic
            debit_viitura = runoff_efectiv * area_km2 * 0.012
            baseflow_natural = (area_km2 / 1000) * 5.0
            baseflow_actual = baseflow_natural * (1 - (total_loss_scenario / 25000))
            if baseflow_actual < 0.5: baseflow_actual = 0.5
            
            debit_total = baseflow_actual + debit_viitura
            
            st.metric("Debit Estimat (24h)", f"{debit_total:.1f} m³/s")
            
            if debit_total > 300:
                st.error("🚨 RISC INUNDAȚIE MAJORĂ")
            elif debit_total > 100:
                st.warning("⚠️ COTE DE ATENȚIE DEPĂȘITE")
            else:
                st.success("✅ DEBIT NORMAL")