
# Detecção de Fraudes Fiscais com IA

Este projeto aplica técnicas de Machine Learning e Deep Learning para detectar possíveis fraudes fiscais em documentos eletrônicos (NFe e NFCe), utilizando dados simulados de postos de combustíveis.

## 🔍 Objetivo

Criar modelos inteligentes capazes de identificar notas fiscais com comportamento anômalo (potencialmente fraudulento), a partir de padrões históricos e características fiscais.

## 📁 Estrutura do Projeto

```
fraude-fiscal-ai/
├── data/                   # Bases de dados simuladas
│   ├── base_simulada_nfe.csv
│   └── base_simulada_nfce.csv
├── notebooks/              # Jupyter Notebooks
│   └── autoencoder_fraudes.ipynb
├── models/                 # Modelos treinados
├── src/                    # Scripts e funções auxiliares (futuramente)
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

## 🧠 Técnicas utilizadas

- Autoencoder (Keras/TensorFlow)
- Detecção de anomalias via erro de reconstrução
- Análise por CNPJ e ano
- Visualização de justificativas explicando o motivo das anomalias

## 🚀 Como usar

1. Clone este repositório ou baixe os arquivos
2. Instale as dependências com `pip install -r requirements.txt`
3. Abra o notebook `autoencoder_fraudes.ipynb`
4. Execute as células para:
   - Carregar os dados simulados
   - Treinar os modelos
   - Detectar anomalias
   - Ver justificativas e rankings por CNPJ

## 📌 Observações

- Este projeto está em fase inicial, usando dados simulados
- Quando os dados reais forem disponibilizados, será possível treinar modelos mais robustos e refinados

## 📄 Licença

MIT
