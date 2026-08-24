import pandas as pd

# Lê o CSV ou o ZIP
df = pd.read_csv('archive/creditcard.csv')

# Otimiza tipos para reduzir drasticamente o tamanho
float_cols = [c for c in df.columns if df[c].dtype == 'float64']
df[float_cols] = df[float_cols].astype('float32')
df['Class'] = df['Class'].astype('int8')

# Salva em Parquet (~25 MB)
df.to_parquet('archive/creditcard.parquet', index=False)