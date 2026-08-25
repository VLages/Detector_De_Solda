import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# TRUQUE DE CAMINHO
# ==========================================
CAMINHO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CAMINHO_BP = os.path.join(CAMINHO_RAIZ, 'backpropagation')
CAMINHO_RF = os.path.join(CAMINHO_RAIZ, 'random_forest')
CAMINHO_YL = os.path.join(CAMINHO_RAIZ, 'yolov8')
for caminho in [CAMINHO_RAIZ, CAMINHO_BP, CAMINHO_RF, CAMINHO_YL]:
    if caminho not in sys.path:
        sys.path.append(caminho)


from random_forest.quality_classification import QualityClassification

MODELO_RF_PATH = "modelos_treinados/modelo.joblib"

# Dicionário para traduzir os nomes técnicos para o gráfico do artigo
TRADUCAO_FEATURES = {
    "pct_azul":   "Azul", 
    "pct_cinza":  "Cinza", 
    "pct_verde":  "Verde",
    "pct_marrom": "Marrom", 
    "pct_roxo":   "Roxo",
    "valor":      "Brilho (Luz)", 
    "saturacao":  "Saturação",
    "h_mean":     "Média do Matiz", 
    "h_std":      "Desvio Padrão do Matiz"
}

def main():
    print("Carregando o modelo Random Forest...")
    try:
        modelo_rf = QualityClassification.load(MODELO_RF_PATH)
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o modelo: {e}")
        return

    # Extrai os nomes das colunas e os pesos matemáticos que a IA deu para elas
    features_originais = modelo_rf.feature_names
    importancias = modelo_rf.model.feature_importances_

    # Traduz os nomes e cria uma tabela (DataFrame)
    features_traduzidas = [TRADUCAO_FEATURES.get(f, f) for f in features_originais]
    df_importancia = pd.DataFrame({
        'Parâmetro Visual (HSV)': features_traduzidas,
        'Importância (Gini)': importancias
    })

    # Organiza do mais importante para o menos importante
    df_importancia = df_importancia.sort_values(by='Importância (Gini)', ascending=False)

    # ==========================================
    # GERAÇÃO DO GRÁFICO CIENTÍFICO
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Usa uma paleta de cores acadêmica (cinza para o menos importante, azul/vermelho para os destaques)
    ax = sns.barplot(x='Importância (Gini)', y='Parâmetro Visual (HSV)', data=df_importancia, palette='Greys_r')

    plt.title('Importância das Variáveis Extraídas (Random Forest)', fontsize=16, pad=15)
    plt.xlabel('Peso de Decisão (Gini Importance)', fontsize=14)
    plt.ylabel('Característica Visual', fontsize=14)
    
    # Adiciona os valores numéricos no final de cada barra para facilitar a leitura
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', 
                    xytext=(5, 0), textcoords='offset points', fontsize=11)

    plt.xlim(0, max(importancias) * 1.15) # Dá um respiro visual no final da maior barra
    plt.tight_layout()

    caminho_grafico = "dados_estatisticos/comparativo_importancia_variaveis.png"
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    print(f"\n[SUCESSO] Gráfico acadêmico salvo em: {caminho_grafico}")
    
    plt.show()

if __name__ == "__main__":
    main()