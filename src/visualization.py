
import matplotlib.pyplot as plt

def plot_history(history, title="Treinamento"):
    '''
    Plota a curva de perda (loss) de treino e validação.
    '''
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Erro de reconstrução - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_top_anomalias(df, valor_col, titulo, top_n=10):
    '''
    Plota um gráfico de barras com os CNPJs com maiores valores anômalos.
    '''
    import seaborn as sns

    df_ordenado = df.sort_values(by=valor_col, ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_ordenado, x=valor_col, y='CNPJ Emitente', palette='Reds_r')
    plt.title(titulo)
    plt.xlabel("Valor Total Anômalo")
    plt.ylabel("CNPJ Emitente")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
