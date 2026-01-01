# Sistem Integrat de Predicție a Viiturilor Rapide (Flash-Floods) utilizând LSTM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Acest repository conține codul sursă și documentația tehnică pentru Lucrarea de Licență 2026.

## 📌 Descriere
Proiectul propune o arhitectură Deep Learning (Long Short-Term Memory) pentru modelarea relației ploaie-scurgere în bazinele hidrografice montane din România, utilizând date satelitare ERA5 Land.

## 🛠️ Arhitectura Sistemului
Proiectul este structurat modular:
- `src/`: Pipeline-ul de date și logica modelului (Backend).
- `app/`: Interfața grafică dezvoltată în Streamlit (Frontend).
- `notebooks/`: Experimente exploratorii și validarea ipotezelor.

## 🚀 Instalare și Rulare
```bash
# 1. Clonare repository
git clone [https://github.com/USERUL_TAU/flood-prediction-licenta.git](https://github.com/USERUL_TAU/flood-prediction-licenta.git)

# 2. Instalare dependențe
pip install -r requirements.txt

# 3. Rulare Dashboard
streamlit run app/dashboard.py