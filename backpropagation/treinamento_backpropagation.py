import os
import glob
import cv2
import torch
import re
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

from features_LA import FeaturesLA as Features
from enhancement_LE import Enhancement
from torch.utils.data import random_split

# ==========================================
# CONFIGURAÇÕES GERAIS 
# ==========================================
NUM_CLASSES = 2          
EPOCHS = 100              
BATCH_SIZE = 16
LEARNING_RATE = 0.001    # Taxa reduzida para um aprendizado mais estável

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de Treinamento: {device.type.upper()}")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, path='modelo_mlp_features_solda.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss, model):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(model)
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.save_checkpoint(model)
            self.counter = 0 

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class FeatureDataset(Dataset):
    def __init__(self, features_array, labels_list):
        # Recebe o array já normalizado pelo StandardScaler
        self.X = torch.tensor(features_array, dtype=torch.float32)
        self.y = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preparar_dados_features():
    pasta_imagens = "D:/Lages/Detector_De_Solda/backpropagation/imagens_geradas"
    caminhos = sorted(glob.glob(os.path.join(pasta_imagens, '*.png')))

    if not caminhos:
        print(f"[Erro] Nenhuma imagem encontrada em {pasta_imagens}")
        exit()

    realcador = Enhancement(denoise=True, clahe=True, sharpen=True)
    extrator_hsv = Features()
    
    features_list = []
    labels_list = []
    
    print(f"Tratando e extraindo features de {len(caminhos)} imagens...")
    
    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is None:
            continue
            
        busca_classe = re.search(r'classe_(\d+)', os.path.basename(caminho))
        if not busca_classe:
            continue
            
        label = 0 if int(busca_classe.group(1)) <= 1 else 1

        # 1. Aplica o realce antes da extração
        img_tratada = realcador.enhance(img)
        
        # 2. Extrai as características
        feat_dict = extrator_hsv.extract(img_tratada)
        
        # Garante que apenas as 6 features definidas na classe sejam usadas
        linha_features = [feat_dict[col] for col in extrator_hsv.FEATURE_COLUMNS[:6]]
        
        features_list.append(linha_features)
        labels_list.append(label)

    # 3. Normalização Profissional
    scaler = StandardScaler()
    features_normalizadas = scaler.fit_transform(features_list)
    
    # Salva o normalizador para ser usado no script de inferência/teste
    os.makedirs("D:/Lages/Detector_De_Solda/modelos_treinados", exist_ok=True)
    with open("D:/Lages/Detector_De_Solda/modelos_treinados/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    dataset = FeatureDataset(features_normalizadas, labels_list)
    return dataset, len(dataset)

# ==========================================
# ARQUITETURA REDUZIDA (Prevenção de Overfitting)
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

from torch.utils.data import random_split # Adicione no topo do arquivo

if __name__ == "__main__":
    # 1. Carrega o dataset completo
    dataset_completo, total_dados = preparar_dados_features()
    
    # 2. Separa 20% dos dados para Validação Cega
    tamanho_val = int(0.2 * len(dataset_completo))
    tamanho_treino = len(dataset_completo) - tamanho_val
    
    dataset_treino, dataset_val = random_split(dataset_completo, [tamanho_treino, tamanho_val])
    
    dataloader_treino = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset: {tamanho_treino} amostras de Treino | {tamanho_val} amostras de Validação.")

    modelo = MLPSolda(input_size=6, num_classes=NUM_CLASSES).to(device)
    criterio = nn.CrossEntropyLoss(label_smoothing=0.01)
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) # Aumentado o peso da regularização

    pasta_modelos = "D:/Lages/Detector_De_Solda/modelos_treinados"
    os.makedirs(pasta_modelos, exist_ok=True)
    caminho_modelo_final = os.path.join(pasta_modelos, 'modelo_mlp_features_solda.pth')

    # O vigilante agora será alimentado com o erro da validação
    vigilante = EarlyStopping(patience=15, min_delta=0.001, path=caminho_modelo_final)

    for epoch in range(EPOCHS):
        # --- FASE DE TREINO ---
        modelo.train() 
        erro_treino_acumulado, acertos_treino = 0.0, 0

        for entradas_features, labels in dataloader_treino:
            entradas_features, labels = entradas_features.to(device), labels.to(device)

            otimizador.zero_grad()
            previsoes = modelo(entradas_features)
            loss = criterio(previsoes, labels)
            loss.backward()
            otimizador.step()

            erro_treino_acumulado += loss.item() * entradas_features.size(0)
            acertos_treino += torch.sum(torch.max(previsoes, 1)[1] == labels.data).item()

        # --- FASE DE VALIDAÇÃO (O Segredo contra Overfitting) ---
        modelo.eval()
        erro_val_acumulado, acertos_val = 0.0, 0
        
        with torch.no_grad():
            for entradas_val, labels_val in dataloader_val:
                entradas_val, labels_val = entradas_val.to(device), labels_val.to(device)
                
                previsoes_val = modelo(entradas_val)
                loss_val = criterio(previsoes_val, labels_val)
                
                erro_val_acumulado += loss_val.item() * entradas_val.size(0)
                acertos_val += torch.sum(torch.max(previsoes_val, 1)[1] == labels_val.data).item()

        erro_treino = erro_treino_acumulado / tamanho_treino
        acc_treino = (acertos_treino / tamanho_treino) * 100
        
        erro_val = erro_val_acumulado / tamanho_val
        acc_val = (acertos_val / tamanho_val) * 100

        print(f"Época [{epoch+1:02d}/{EPOCHS}] | Treino: {erro_treino:.4f} ({acc_treino:.1f}%) | Validação: {erro_val:.4f} ({acc_val:.1f}%)")
        
        # O vigilante decide parar com base no erro de imagens que a rede não usou para atualizar pesos
        vigilante(erro_val, modelo)
        
        if vigilante.early_stop:
            print(f"\n[!] Treinamento interrompido na época {epoch+1} pois a validação parou de melhorar.")
            break