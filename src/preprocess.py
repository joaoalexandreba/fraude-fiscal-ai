import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

def load_and_normalize_data(df, feature_cols):
    '''
    Carrega e normaliza os dados de entrada.
    Retorna X normalizado e o scaler treinado.
    '''
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler

def build_autoencoder(input_dim):
    '''
    Cria um modelo de autoencoder simples para detecção de anomalias.
    '''
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def avaliar_anomalias(model, X, df, prefixo):
    '''
    Calcula erro de reconstrução e marca anomalias com base em um limiar.
    '''
    reconstructions = model.predict(X)
    mse = np.mean(np.square(X - reconstructions), axis=1)
    threshold = np.percentile(mse, 95)
    df[f"{prefixo}_Erro_Reconstrucao"] = mse
    df[f"{prefixo}_Anomalia"] = (mse > threshold).astype(int)
    return df

def gerar_justificativas(model, X, df, prefixo, feature_names):
    '''
    Gera justificativas textuais para explicar por que cada anomalia foi marcada.
    '''
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
