import pandas as pd

# Pune numele exact al fișierului nou pe care l-ai descărcat
df = pd.read_csv('data/raw/dataset_2024_2026.csv') # Sau cum l-ai numit

print("=== RAPORT FISIER NOU ===")
print(f"Data de START: {df['Date_Time'].min()}")
print(f"Data de FINAL: {df['Date_Time'].max()}")
print(f"Număr total de linii: {len(df)}")