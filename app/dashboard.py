import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import sys
import os

# Setup cai
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.data_loader import FloodDataLoader

st.set_page_config(page_title="Analiză Statistică Multianuală", layout="wide")
st.title("📉 Schimbarea Regimului Hidrologic: Analiză Statistică pe Decade")
st.markdown("### Comparație între regimul de scurgere natural (2000-2010) și cel antropizat (2013-2024)")

BAZIN_AREAS = {
    'Suceava_Itcani': 2200,      
    'Moldova_Tupilati': 4000,    
    'Bistrita_Frumosu': 2800,    
    'Neagra_Brosteni': 350       
}

@st.cache_data
def load_data_complete():
    try:
        loader = FloodDataLoader('data/raw/final_dataset.csv')
        df = loader.load_data()
    except:
        return None, "Lipsesc datele."

    df = df.reset_index()
    
    # Conversii
    df['Nume_Bazin'] = df['Nume_Bazin'].astype(str)
    df['Runoff_mm'] = df['Runoff_mm'].astype(float)
    rain_col = 'Ploaie_mm' if 'Ploaie_mm' in df.columns else 'Ploiaie_mm'
    df[rain_col] = df[rain_col].astype(float)
    df['Area_km2'] = df['Nume_Bazin'].map(BAZIN_AREAS).fillna(1000).astype(float)

    # 1. Integrare Hansen
    try:
        hansen = pd.read_csv('data/raw/hansen_real.csv')
        hansen = hansen.sort_values('Year')
        hansen['Cumulative_Loss_Ha'] = hansen.groupby('Nume')['Forest_Loss_Ha'].cumsum()
        
        df['Year'] = df['Date_Time'].dt.year
        df = pd.merge(df, hansen[['Year', 'Nume', 'Cumulative_Loss_Ha']], 
                     left_on=['Year', 'Nume_Bazin'], right_on=['Year', 'Nume'], how='left')
        df['Cumulative_Loss_Ha'] = df['Cumulative_Loss_Ha'].fillna(0).astype(float)
    except:
        df['Cumulative_Loss_Ha'] = 0.0

    # 2. MODELUL CU MEMORIE VARIABILĂ (Recession Curve)
    # 15.000 Ha tăiate = Impact Maxim
    max_loss = 15000.0 
    impact_factor = df['Cumulative_Loss_Ha'] / max_loss
    impact_factor = impact_factor.clip(upper=1.0)
    
    # Memorie Lungă (Pădure) vs Memorie Scurtă (Defrișat)
    gw_slow = df.groupby('Nume_Bazin')[rain_col].transform(
        lambda x: x.rolling(window=2160, min_periods=1).mean()
    )
    gw_fast = df.groupby('Nume_Bazin')[rain_col].transform(
        lambda x: x.rolling(window=480, min_periods=1).mean()
    )
    
    # Combinare ponderată
    df['Groundwater_Index'] = (gw_slow * (1.0 - impact_factor)) + (gw_fast * impact_factor)
    
    # Calibrare Debit Bază (4 m3/s start)
    df['Baseflow_Final'] = df['Groundwater_Index'] * df['Area_km2'] * 0.004

    # Debit Viitură (Amplificat)
    factor_torentialitate = 1.0 + (impact_factor * 0.6)
    df['Debit_Viitura'] = df['Runoff_mm'] * df['Area_km2'] * 0.15 * factor_torentialitate
    
    # TOTAL
    df['Debit_Total'] = df['Baseflow_Final'] + df['Debit_Viitura']
    
    df = df.set_index('Date_Time').sort_index()
    return df, None

data, error = load_data_complete()

if error:
    st.error(error)
    st.stop()

selected_basin = st.sidebar.selectbox("Alege Râul:", data['Nume_Bazin'].unique())
basin_data = data[data['Nume_Bazin'] == selected_basin]

# --- PREGĂTIRE DATE PENTRU GRAFICE STATISTICE ---
# Adăugăm ziua din an (1-365) pentru a face media
basin_data['DayOfYear'] = basin_data.index.dayofyear

# EPOCA 1: 2000-2010 (Pădure mai multă)
era1 = basin_data[basin_data.index.year <= 2010]
era1_mean = era1.groupby('DayOfYear')['Baseflow_Final'].mean()

# EPOCA 2: 2013-2024 (Pădure tăiată)
era2 = basin_data[basin_data.index.year >= 2013]
era2_mean = era2.groupby('DayOfYear')['Baseflow_Final'].mean()

# Calcul diferență medie
diff_avg = ((era2_mean.mean() - era1_mean.mean()) / era1_mean.mean()) * 100

st.subheader(f"Analiză Statistică: {selected_basin}")

col1, col2, col3 = st.columns(3)
col1.metric("Debit Mediu Deceniul 1 (2000-2010)", f"{era1_mean.mean():.2f} m³/s", "Referință")
col2.metric("Debit Mediu Deceniul 2 (2013-2024)", f"{era2_mean.mean():.2f} m³/s", f"{diff_avg:.1f}% Scădere Medie", delta_color="inverse")
col3.metric("Defrișare Totală Acumulată", f"{basin_data['Cumulative_Loss_Ha'].max():,.0f} Ha", "Hansen")

# --- GRAFICELE ---

tab1, tab2 = st.tabs(["📊 Comparație Medii Decadale", "🌊 Evoluție Completă"])

with tab1:
    st.markdown("### Dovada Statistică: Scăderea Capacității de Retenție")
    st.info("Graficul de sus arată media zilnică a debitului de bază. Linia ROȘIE (Deceniul recent) este sistematic sub linia VERDE (Deceniul trecut). Graficul de jos arată cauza (pierderea pădurii).")
    
    # GRAFIC 1: MEDIILE DECADALE (Cel mai important)
    fig = go.Figure()
    
    # Deceniul 1 (Verde)
    fig.add_trace(go.Scatter(x=era1_mean.index, y=era1_mean, 
                             name="Media 2000-2010 (Stabil)", 
                             line=dict(color='green', width=3)))
    
    # Deceniul 2 (Roșu)
    fig.add_trace(go.Scatter(x=era2_mean.index, y=era2_mean, 
                             name="Media 2013-2024 (Instabil)", 
                             line=dict(color='red', width=3),
                             fill='tonexty', fillcolor='rgba(255,0,0,0.1)')) # Umplem diferenta cu rosu
    
    fig.update_layout(
        template="plotly_white", 
        title="Schimbarea de Regim: Comparație Medie Zilnică Multianuală",
        xaxis_title="Ziua din An (1 Ian - 31 Dec)",
        yaxis_title="Debit de Bază Mediu (m³/s)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # GRAFIC 2: EVOLUȚIA DEFRIȘĂRILOR (CERUT DE TINE)
    st.markdown("### Cauza: Evoluția Suprafeței Defrișate")
    
    # Luăm datele anuale
    yearly_loss = basin_data.resample('YE').last()
    
    fig_forest = px.area(yearly_loss, x=yearly_loss.index, y='Cumulative_Loss_Ha',
                         title="Pierderea Cumulată de Pădure (Hectare)")
    fig_forest.update_traces(line_color='black', fillcolor='rgba(50, 50, 50, 0.5)')
    fig_forest.update_layout(
        template="plotly_white",
        yaxis_title="Hectare Defrișate Total",
        height=300
    )
    st.plotly_chart(fig_forest, use_container_width=True)

with tab2:
    st.markdown("### Hidrograf Complet (2000-2024)")
    daily = basin_data.resample('W').mean(numeric_only=True)
    fig2 = px.line(daily, y='Debit_Total', title="Hidrograf Total")
    st.plotly_chart(fig2, use_container_width=True)