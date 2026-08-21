import os
import sys
import glob
import re
import cv2
import torch
import pickle
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
            nn.Linear(input_size, 16), 
            nn.BatchNorm1d(16),
            nn.ReLU(),                 
            nn.Dropout(0.3),           
            
            nn.Linear(16, 8),         
            nn.ReLU(),
            
            nn.Linear(8, num_classes) 
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

LABELS_YOLO = ["Metal Limpo\n(Sem Solda)", "Cordão Presente\n(Com Solda)"]
LABELS_IA = ["Nível 1 a 4\n(Aceitável)", "Nível 5 a 10\n(Inaceitável)"]

def ppm_para_macro(ppm):
    """Traduz o PPM exato apenas para as 2 Macro-Classes da IA (Ignora o nível 0)"""
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

    caminho_scaler = os.path.join(CAMINHO_RAIZ, "modelos_treinados", "scaler.pkl")
    with open(caminho_scaler, "rb") as f:
        scaler = pickle.load(f)
    
    modelo_rf = QualityClassification.load(MODELO_RF_PATH)

    # 3. Lendo Imagens de Teste
    caminhos = glob.glob(os.path.join(PASTA_TESTES, "*.png")) + glob.glob(os.path.join(PASTA_TESTES, "*.jpg"))
    if not caminhos:
        print(f"[ERRO] Nenhuma imagem encontrada na pasta: {PASTA_TESTES}")
        return

    # Listas do YOLO (Avalia TODAS as imagens)
    y_verdadeiro_yolo = []
    y_pred_yolo = []

    # Listas da IA (Avalia APENAS imagens com solda)
    y_verdadeiro_ia = []
    y_pred_mlp = []
    y_pred_rf = []

    print(f"Analisando {len(caminhos)} imagens inéditas...\n")

    # 4. Loop de Inferência
    for caminho in caminhos:
        nome_arquivo = os.path.basename(caminho)
        
        busca = re.search(r'g(\d+)-', nome_arquivo, re.IGNORECASE)
        if not busca:
            continue
            
        ppm_real = int(busca.group(1))
        img_bruta = cv2.imread(caminho)
        
        # ==========================================
        # FASE 1: AVALIAÇÃO DO YOLO (DETECÇÃO)
        # ==========================================
        _, recortes = yolo.detectar_e_recortar(img_bruta, padding_ratio=0.05)
        
        # Gabarito do YOLO: 0 se for g0, 1 se tiver PPM
        y_verdadeiro_yolo.append(0 if ppm_real == 0 else 1)
        # Previsão do YOLO: 0 se não gerou recorte, 1 se gerou
        y_pred_yolo.append(0 if not recortes else 1)

        # ==========================================
        # FASE 2: AVALIAÇÃO DA IA (QUALIDADE)
        # ==========================================
        # Se for metal limpo (g0), a IA não precisa avaliar. Pulamos para a próxima foto.
        if ppm_real == 0:
            continue
            
        macro_real = ppm_para_macro(ppm_real)
        y_verdadeiro_ia.append(macro_real)
        
        # Fallback: Se o YOLO falhou em achar a solda, mandamos a foto inteira para a IA não quebrar
        img_para_analise = img_bruta.copy() if not recortes else recortes[0]['imagem']

        # Processamento Ótico 
        img_melhorada = enhancer.enhance(img_para_analise)
        features_dict = extrator.extract(img_melhorada)
        
        # 1. Garante que vai extrair apenas as 6 primeiras features (como no treino)
        lista_features = [features_dict[c] for c in extrator.FEATURE_COLUMNS[:6]]

        # 2. Aplica a padronização do treino usando o Scaler
        features_normalizadas = scaler.transform([lista_features])
        
        # 3. Transforma em Tensor (Sem dividir por 100.0)
        tensor_feat = torch.tensor(features_normalizadas, dtype=torch.float32).to(device)
        # -------------------------

        # --- AVALIAÇÃO MLP ---
        with torch.no_grad():
            saida_mlp = modelo_mlp(tensor_feat)
            probs_mlp = torch.nn.functional.softmax(saida_mlp, dim=1)[0]
            probabilidade_ruim = probs_mlp[1].item()
            LIMITE_CORTE = 0.30  # 30%
            if probabilidade_ruim >= LIMITE_CORTE:
                y_pred_mlp.append(1) # Reprova a solda
            else:
                y_pred_mlp.append(0) # Aceita a solda

        # --- AVALIAÇÃO RANDOM FOREST ---
        probs_rf_originais = modelo_rf.predict_proba(features_dict)[0]
        probs_rf_macro = [0.0, 0.0] 
        
        for idx, cls_label in enumerate(modelo_rf.classes_):
            try:
                cls_val = int(cls_label)
                if cls_val <= 100: probs_rf_macro[0] += probs_rf_originais[idx]
                else: probs_rf_macro[1] += probs_rf_originais[idx]
            except ValueError:
                pass
                
        macro_pred_rf = int(np.argmax(probs_rf_macro))
        y_pred_rf.append(macro_pred_rf)

    # ==========================================
    # CÁLCULO DAS MÉTRICAS CIENTÍFICAS
    # ==========================================
    print("="*50)
    print("MÉTRICAS DO YOLO (DETECÇÃO DE OBJETO)")
    print("="*50)
    print(f"Acurácia Global: {accuracy_score(y_verdadeiro_yolo, y_pred_yolo) * 100:.2f}%")
    print(classification_report(y_verdadeiro_yolo, y_pred_yolo, target_names=[l.replace('\n', ' ') for l in LABELS_YOLO], zero_division=0))

    print("\n" + "="*50)
    print("MÉTRICAS DA REDE NEURAL (QUALIDADE)")
    print("="*50)
    print(f"Acurácia Global: {accuracy_score(y_verdadeiro_ia, y_pred_mlp) * 100:.2f}%")
    print(classification_report(y_verdadeiro_ia, y_pred_mlp, target_names=[l.replace('\n', ' ') for l in LABELS_IA], zero_division=0))

    print("\n" + "="*50)
    print("MÉTRICAS DO RANDOM FOREST (QUALIDADE)")
    print("="*50)
    print(f"Acurácia Global: {accuracy_score(y_verdadeiro_ia, y_pred_rf) * 100:.2f}%")
    print(classification_report(y_verdadeiro_ia, y_pred_rf, target_names=[l.replace('\n', ' ') for l in LABELS_IA], zero_division=0))

    # ==========================================
    # GERAÇÃO DO PAINEL DE MATRIZES (1x3)
    # ==========================================
    matriz_yolo = confusion_matrix(y_verdadeiro_yolo, y_pred_yolo, labels=[0, 1])
    matriz_mlp = confusion_matrix(y_verdadeiro_ia, y_pred_mlp, labels=[0, 1])
    matriz_rf = confusion_matrix(y_verdadeiro_ia, y_pred_rf, labels=[0, 1])

    fig, eixos = plt.subplots(1, 3, figsize=(20, 6))

    # Plot YOLO (Matriz de Detecção)
    sns.heatmap(matriz_yolo, annot=True, fmt='d', cmap='Oranges', ax=eixos[0], cbar=False,
                xticklabels=LABELS_YOLO, yticklabels=LABELS_YOLO, annot_kws={"size": 15})
    eixos[0].set_title('YOLOv8 - Matriz de Detecção', fontsize=15, pad=15)
    eixos[0].set_xlabel('Previsão do YOLO', fontsize=12)
    eixos[0].set_ylabel('Gabarito Real', fontsize=12)

    # Plot MLP (Matriz de Classificação)
    sns.heatmap(matriz_mlp, annot=True, fmt='d', cmap='Blues', ax=eixos[1], cbar=False,
                xticklabels=LABELS_IA, yticklabels=LABELS_IA, annot_kws={"size": 15})
    eixos[1].set_title('Rede Neural (MLP) - Matriz de Qualidade', fontsize=15, pad=15)
    eixos[1].set_xlabel('Previsão da IA', fontsize=12)
    eixos[1].set_ylabel('Gabarito Real (PPM)', fontsize=12)

    # Plot RF (Matriz de Classificação)
    sns.heatmap(matriz_rf, annot=True, fmt='d', cmap='Greens', ax=eixos[2], cbar=False,
                xticklabels=LABELS_IA, yticklabels=LABELS_IA, annot_kws={"size": 15})
    eixos[2].set_title('Random Forest - Matriz de Qualidade', fontsize=15, pad=15)
    eixos[2].set_xlabel('Previsão da IA', fontsize=12)
    eixos[2].set_ylabel('Gabarito Real (PPM)', fontsize=12)

    plt.tight_layout()
    caminho_grafico = "dados_estatisticos/comparativo_matrizes_confusao.png"
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    print(f"\n[SUCESSO] Painel estatístico salvo em: '{caminho_grafico}'")
    plt.show()

if __name__ == "__main__":
    main()