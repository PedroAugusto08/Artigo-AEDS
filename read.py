import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load movie dataset from CSV.""" 
    df = pd.read_csv(file_path, delimiter=';')
    return df

def preprocess_data(df, rating_threshold, revenue_threshold):
    """Filter dataset and preprocess.""" 
    scaler = MinMaxScaler()

    # Substituir vírgulas por pontos e converter colunas numéricas para float
    df['Revenue (Millions)'] = df['Revenue (Millions)'].str.replace(',', '.').astype(float)
    df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)
    if 'Budget (Million)' in df.columns:
        df['Budget (Million)'] = df['Budget (Million)'].astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)').astype(float)
    else:
        df['Budget (Million)'] = np.nan  # Adiciona a coluna como NaN caso não exista

    # Converter colunas numéricas para float
    df['Runtime (Minutes)'] = df['Runtime (Minutes)'].replace({'N/A': None, 'Unknown': None}).astype(float)

    # Aplicar filtros baseados em nota e receita
    df = df[(df['Rating'] >= rating_threshold) & (df['Revenue (Millions)'] >= revenue_threshold)]

    # Tratar valores faltantes
    df['Runtime (Minutes)'] = df['Runtime (Minutes)'].fillna(df['Runtime (Minutes)'].median())
    df['Revenue (Millions)'] = df['Revenue (Millions)'].fillna(df['Revenue (Millions)'].median())
    df['Budget (Million)'] = df['Budget (Million)'].fillna(df['Budget (Million)'].median())

    # Normalizar colunas numéricas
    df[['Runtime (Minutes)', 'Revenue (Millions)', 'Budget (Million)']] = scaler.fit_transform(df[['Runtime (Minutes)', 'Revenue (Millions)', 'Budget (Million)']])

    # Codificar gêneros (apenas o Primary Genre)
    genre_encoded = pd.get_dummies(df['Primary Genre'], prefix='genre')
    df = pd.concat([df, genre_encoded], axis=1)

    return df, scaler

def denormalize_data(df, scaler, feature_cols):
    """Denormalize the data using the original scaler.""" 
    df_denormalized = df.copy()
    df_denormalized[feature_cols] = scaler.inverse_transform(df[feature_cols])
    return df_denormalized