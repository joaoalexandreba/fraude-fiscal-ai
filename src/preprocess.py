
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

def load_and_normalize_data(df, feature_cols):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler

def build_autoencoder(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def avaliar_anomalias(model, X, df, prefixo):
    reconstructions = model.predict(X)
    mse = np.mean(np.square(X - reconstructions), axis=1)
    threshold = np.percentile(mse, 95)
    df[f"{prefixo}_Erro_Reconstrucao"] = mse
    df[f"{prefixo}_Anomalia"] = (mse > threshold).astype(int)
    print(f"Threshold para {prefixo}: {threshold:.6f}")
    return df

def gerar_justificativas(model, X, df, prefixo, feature_names):
    reconstructions = model.predict(X)
    diffs = np.abs(X - reconstructions)
    justificativas = []
    for i in range(len(X)):
        if df[f"{prefixo}_Anomalia"].iloc[i] == 1:
            idx = np.argmax(diffs[i])
            var = feature_names[idx]
            real = X[i][idx]
            recon = reconstructions[i][idx]
            perc = (real - recon) / (real + 1e-5)
            direcao = "acima" if perc > 0 else "abaixo"
            justificativa = f"{var} reconstruído {abs(perc*100):.1f}% {direcao} do valor original"
        else:
            justificativa = ""
        justificativas.append(justificativa)
    df[f"{prefixo}_Justificativa_Anomalia"] = justificativas
    return df

def gerar_ranking_anomalias(df, prefixo, valor_col, agrupadores=['CNPJ Emitente', 'Ano']):
    df['Ano'] = pd.to_datetime(df['DHEMI']).dt.year
    df_anomalias = df[df[f'{prefixo}_Anomalia'] == 1]

    ranking = (
        df_anomalias
        .groupby(agrupadores)
        .agg({
            valor_col: 'sum',
            f'{prefixo}_Anomalia': 'count'
        })
        .rename(columns={
            valor_col: 'Valor Total Anômalo',
            f'{prefixo}_Anomalia': 'Qtd Notas Anômalas'
        })
        .reset_index()
        .sort_values(by='Valor Total Anômalo', ascending=False)
    )
    return ranking
