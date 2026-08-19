import os
import sys
import glob
import re
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# TRUQUE DE CAMINHO (Para achar o RF)
# ==========================================

CAMINHO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CAMINHO_BP = os.path.join(CAMINHO_RAIZ, 'backpropagation')
CAMINHO_RF = os.path.join(CAMINHO_RAIZ, 'random_forest')
CAMINHO_YL = os.path.join(CAMINHO_RAIZ, 'yolov8')
for caminho in [CAMINHO_RAIZ, CAMINHO_BP, CAMINHO_RF, CAMINHO_YL]:
    if caminho not in sys.path:
        sys.path.append(caminho)

from backpropagation.enhancement_LE import Enhancement
from backpropagation.features_LA import FeaturesLA as Features
from random_forest.quality_classification import QualityClassification
from yolov8.detector_yolo import ExtratorCordaoYOLO

# ==========================================
# ARQUITETURA DA MLP (Cópia Exata)
# ==========================================
class MLPSolda(nn.Module):
    def __init__(self, input_size=6, num_classes=2):
        super(MLPSolda, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.rede(x)

# ==========================================
# CONFIGURAÇÕES DE CAMINHO
# ==========================================
PASTA_TESTES = r"D:\Lages\Detector_De_Solda\banco_de_dados\base_GasPurga_balanceada" # <-- COLOQUE AQUI A SUA PASTA DE TESTES
MODELO_MLP_PATH = "modelos_treinados/modelo_mlp_features_solda.pth"
MODELO_RF_PATH = "modelos_treinados/modelo.joblib"
MODELO_YOLO_PATH = "modelos_treinados/yolov8n_solda.pt" # <-- Caminho do YOLO

LABELS_MACRO = ["Nível 1 a 4\n(Aceitável)", "Nível 5 a 10\n(Inaceitável)"]

def ppm_para_macro(ppm):
    """Traduz o PPM exato para as 3 Macro-Classes do projeto"""
    if ppm <= 100: return 0
    else: return 1

def main():
    print("="*50)
    print("INICIANDO VALIDAÇÃO CIENTÍFICA (MLP vs RANDOM FOREST)")
    print("="*50)

    # 1. Instanciando as Ferramentas (Agora com YOLO)
    yolo = ExtratorCordaoYOLO(model_path=MODELO_YOLO_PATH)
    enhancer = Enhancement()
    extrator = Features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Carregando os Modelos
    modelo_mlp = MLPSolda(input_size=6, num_classes=2).to(device)
    modelo_mlp.load_state_dict(torch.load(MODELO_MLP_PATH, map_location=device, weights_only=True))
    modelo_mlp.eval()
    
    modelo_rf = QualityClassification.load(MODELO_RF_PATH)

    # 3. Lendo Imagens de Teste
    caminhos = glob.glob(os.path.join(PASTA_TESTES, "*.png")) + glob.glob(os.path.join(PASTA_TESTES, "*.jpg"))
    if not caminhos:
        print(f"[ERRO] Nenhuma imagem encontrada na pasta: {PASTA_TESTES}")
        return

    y_verdadeiro = []
    y_pred_mlp = []
    y_pred_rf = []

    print(f"Analisando {len(caminhos)} imagens inéditas com YOLO...\n")

    # 4. Loop de Inferência
    for caminho in caminhos:
        nome_arquivo = os.path.basename(caminho)
        
        busca = re.search(r'g(\d+)-', nome_arquivo, re.IGNORECASE)
        if not busca:
            print(f"[Aviso] Ignorando {nome_arquivo}")
            continue
            
        ppm_real = int(busca.group(1))
        if ppm_real == 0:
            continue
        macro_real = ppm_para_macro(ppm_real)
        
        # Leitura da imagem bruta
        img_bruta = cv2.imread(caminho)
        
        # --- O PULO DO GATO: CORTE DO YOLO ---
        _, recortes = yolo.detectar_e_recortar(img_bruta, padding_ratio=0.05)
        
        if not recortes:
            img_para_analise = img_bruta.copy()
        else:
            # Como é uma foto de validação unitária, pegamos o primeiro cordão detectado
            img_para_analise = recortes[0]['imagem']

        # Processamento Ótico na imagem RECORTADA
        img_melhorada = enhancer.enhance(img_para_analise)
        features_dict = extrator.extract(img_melhorada)
        lista_features = [features_dict[c] for c in extrator.FEATURE_COLUMNS]

        # --- AVALIAÇÃO MLP ---
        tensor_feat = torch.tensor(lista_features, dtype=torch.float32).unsqueeze(0).to(device) / 100.0
        with torch.no_grad():
            saida_mlp = modelo_mlp(tensor_feat)
            _, pred_mlp = torch.max(saida_mlp, 1)
            y_pred_mlp.append(pred_mlp.item())

        # --- AVALIAÇÃO RANDOM FOREST (SOMA DE PROBABILIDADES) ---
        probs_rf_originais = modelo_rf.predict_proba(features_dict)[0]
        probs_rf_macro = [0.0, 0.0] 
        
        # O modelo .joblib tem 11 classes, precisamos agrupar nos 2 baldes!
        for idx, cls_label in enumerate(modelo_rf.classes_):
            try:
                cls_val = int(cls_label)
                if cls_val <= 100:
                    probs_rf_macro[0] += probs_rf_originais[idx]
                else:
                    probs_rf_macro[1] += probs_rf_originais[idx]
            except ValueError:
                pass
                
        # Agora sim a decisão será apenas 0 ou 1
        macro_pred_rf = int(np.argmax(probs_rf_macro))
        y_pred_rf.append(macro_pred_rf)

        y_verdadeiro.append(macro_real)

    # CÁLCULO DAS MÉTRICAS CIENTÍFICAS
    # ==========================================
    acc_mlp = accuracy_score(y_verdadeiro, y_pred_mlp)
    acc_rf = accuracy_score(y_verdadeiro, y_pred_rf)

    print("="*50)
    print("MÉTRICAS DA REDE NEURAL (MLP)")
    print("="*50)
    print(f"Acurácia Global: {acc_mlp * 100:.2f}%")
    print(classification_report(y_verdadeiro, y_pred_mlp, labels=[0, 1], target_names=[l.replace('\n', ' ') for l in LABELS_MACRO], zero_division=0))

    print("\n" + "="*50)
    print("MÉTRICAS DO RANDOM FOREST")
    print("="*50)
    print(f"Acurácia Global: {acc_rf * 100:.2f}%")
    print(classification_report(y_verdadeiro, y_pred_rf, labels=[0, 1], target_names=[l.replace('\n', ' ') for l in LABELS_MACRO], zero_division=0))

    # ==========================================
    # GERAÇÃO DAS MATRIZES DE CONFUSÃO (GRÁFICOS)
    # ==========================================
    # As matrizes agora são 2x2 (Binárias)
    matriz_mlp = confusion_matrix(y_verdadeiro, y_pred_mlp, labels=[0, 1])
    matriz_rf = confusion_matrix(y_verdadeiro, y_pred_rf, labels=[0, 1])

    fig, eixos = plt.subplots(1, 2, figsize=(14, 6))

    # Plot MLP
    sns.heatmap(matriz_mlp, annot=True, fmt='d', cmap='Blues', ax=eixos[0], cbar=False,
                xticklabels=LABELS_MACRO, yticklabels=LABELS_MACRO, annot_kws={"size": 14})
    eixos[0].set_title('Rede Neural (MLP) - Matriz de Confusão', fontsize=14, pad=15)
    eixos[0].set_xlabel('Previsão da IA', fontsize=12)
    eixos[0].set_ylabel('Gabarito Real (PPM)', fontsize=12)

    # Plot RF
    sns.heatmap(matriz_rf, annot=True, fmt='d', cmap='Greens', ax=eixos[1], cbar=False,
                xticklabels=LABELS_MACRO, yticklabels=LABELS_MACRO, annot_kws={"size": 14})
    eixos[1].set_title('Random Forest - Matriz de Confusão', fontsize=14, pad=15)
    eixos[1].set_xlabel('Previsão da IA', fontsize=12)
    eixos[1].set_ylabel('Gabarito Real (PPM)', fontsize=12)

    plt.tight_layout()
    caminho_grafico = "dados_estatisticos/comparativo_matrizes_confusao.png"
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    print(f"\n[SUCESSO] Gráfico de alta resolução salvo em: '{caminho_grafico}'")
    plt.show()

if __name__ == "__main__":
    main()