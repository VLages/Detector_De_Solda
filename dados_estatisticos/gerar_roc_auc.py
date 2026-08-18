import os
import sys
import glob
import re
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ==========================================
# TRUQUE DE CAMINHO PARA O JOBLIB
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
# ARQUITETURA DA MLP
# ==========================================
class MLPSolda(nn.Module):
    def __init__(self, input_size=6, num_classes=3):
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
# CONFIGURAÇÕES
# ==========================================
PASTA_TESTES = r"D:\Lages\Detector_De_Solda\banco_de_dados\base_GasPurga"
MODELO_MLP_PATH = "modelos_treinados/modelo_mlp_features_solda.pth"
MODELO_RF_PATH = "modelos_treinados/modelo.joblib"
MODELO_YOLO_PATH = "modelos_treinados/yolov8n_solda.pt"

LABELS_MACRO = ["Classe 0 (Limpa)", "Classe 1 (Aceitável)", "Classe 2 (Inaceitável)"]
N_CLASSES = 3

def ppm_para_macro(ppm):
    if ppm == 0: return 0
    elif ppm <= 100: return 1
    else: return 2

def main():
    print("Extraindo probabilidades para as Curvas ROC...")

    yolo = ExtratorCordaoYOLO(model_path=MODELO_YOLO_PATH)
    enhancer = Enhancement()
    extrator = Features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo_mlp = MLPSolda().to(device)
    modelo_mlp.load_state_dict(torch.load(MODELO_MLP_PATH, map_location=device, weights_only=True))
    modelo_mlp.eval()
    
    modelo_rf = QualityClassification.load(MODELO_RF_PATH)

    caminhos = glob.glob(os.path.join(PASTA_TESTES, "*.png")) + glob.glob(os.path.join(PASTA_TESTES, "*.jpg"))
    
    y_verdadeiro = []
    probabilidades_mlp = []
    probabilidades_rf = []

    for caminho in caminhos:
        nome_arquivo = os.path.basename(caminho)
        busca = re.search(r'g(\d+)-', nome_arquivo, re.IGNORECASE)
        if not busca: continue
            
        ppm_real = int(busca.group(1))
        y_verdadeiro.append(ppm_para_macro(ppm_real))

        img_bruta = cv2.imread(caminho)
        _, recortes = yolo.detectar_e_recortar(img_bruta, padding_ratio=0.05)
        img_para_analise = img_bruta.copy() if not recortes else recortes[0]['imagem']

        img_melhorada = enhancer.enhance(img_para_analise)
        features_dict = extrator.extract(img_melhorada)
        lista_features = [features_dict[c] for c in extrator.FEATURE_COLUMNS]

        # Probabilidades MLP
        tensor_feat = torch.tensor(lista_features, dtype=torch.float32).unsqueeze(0).to(device) / 100.0
        with torch.no_grad():
            saida_mlp = modelo_mlp(tensor_feat)
            probs_mlp = F.softmax(saida_mlp, dim=1)[0].cpu().numpy()
            probabilidades_mlp.append(probs_mlp)

        # Probabilidades RF (Traduzindo 11 classes para 3)
        probs_rf_originais = modelo_rf.predict_proba(features_dict)[0]
        probs_rf_macro = [0.0, 0.0, 0.0]
        for idx, cls_label in enumerate(modelo_rf.classes_):
            try:
                cls_val = int(cls_label)
                if cls_val == 0: probs_rf_macro[0] += probs_rf_originais[idx]
                elif cls_val <= 100: probs_rf_macro[1] += probs_rf_originais[idx]
                else: probs_rf_macro[2] += probs_rf_originais[idx]
            except ValueError:
                pass
        probabilidades_rf.append(probs_rf_macro)

    # Binarizando os labels reais para plotagem multiclasse (One-vs-Rest)
    Y_bin = label_binarize(y_verdadeiro, classes=[0, 1, 2])
    Y_score_mlp = np.array(probabilidades_mlp)
    Y_score_rf = np.array(probabilidades_rf)

    # ==========================================
    # CÁLCULO ROC E AUC (Por Classe e Macro-Média)
    # ==========================================
    def calcular_roc_auc(Y_verdadeiro, Y_score):
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(N_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(Y_verdadeiro[:, i], Y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Cálculo do Macro-Average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_CLASSES)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(N_CLASSES):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= N_CLASSES
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        return fpr, tpr, roc_auc

    fpr_mlp, tpr_mlp, auc_mlp = calcular_roc_auc(Y_bin, Y_score_mlp)
    fpr_rf, tpr_rf, auc_rf = calcular_roc_auc(Y_bin, Y_score_rf)

    # ==========================================
    # PLOTAGEM DOS GRÁFICOS (Lado a Lado)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cores = cycle(['#22C55E', '#F5C542', '#EF4444']) # Verde, Amarelo, Vermelho

    def plotar_curvas(ax, fpr, tpr, roc_auc, titulo):
        for i, cor in zip(range(N_CLASSES), cores):
            ax.plot(fpr[i], tpr[i], color=cor, lw=2,
                    label=f'{LABELS_MACRO[i]} (AUC = {roc_auc[i]:.2f})')
        
        ax.plot(fpr["macro"], tpr["macro"],
                label=f'Média Macro (AUC = {roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', lw=3)
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5) # Linha de aleatoriedade
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
        ax.set_title(titulo, fontsize=15, pad=15)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

    plotar_curvas(ax1, fpr_mlp, tpr_mlp, auc_mlp, 'Curvas ROC - Rede Neural (MLP)')
    plotar_curvas(ax2, fpr_rf, tpr_rf, auc_rf, 'Curvas ROC - Random Forest')

    plt.tight_layout()
    caminho_grafico = "dados_estatisticos/comparativo_curvas_roc.png"
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    print(f"[SUCESSO] Gráfico ROC/AUC salvo em: {caminho_grafico}")
    plt.show()

if __name__ == "__main__":
    main()