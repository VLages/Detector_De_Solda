import os
import glob
import cv2
import torch
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from features_LA import FeaturesLA as Features

# ==========================================
# CONFIGURAÇÕES GERAIS (HIPERPARÂMETROS)
# ==========================================
NUM_CLASSES = 2          # Usando as 3 Macro-Classes (Aceitável vs Inaceitável)
EPOCHS = 100              # Redes tabulares precisam de mais épocas, mas treinam em milissegundos
BATCH_SIZE = 16
LEARNING_RATE = 0.005    # Taxa de aprendizado um pouco maior para MLPs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de Treinamento: {device.type.upper()}")

# ==========================================
# CLASSE DE VIGILÂNCIA (EARLY STOPPING)
# ==========================================
class EarlyStopping:
    """
    Vigia a taxa de erro da rede MLP. Se o erro parar de cair, ele interrompe
    o treinamento para evitar o vício (overfitting) e salva o melhor modelo.
    """
    def __init__(self, patience=8, min_delta=0.001, path='modelo_mlp_features_solda.pth'):
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
            print(f"   [Checkpoint] Melhor modelo salvo! (Loss: {current_loss:.4f})")
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"   [Aviso] Sem melhoras significativas. Paciência em {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.save_checkpoint(model)
            print(f"   [Checkpoint] Novo melhor modelo salvo! (Loss: {current_loss:.4f})")
            self.counter = 0 

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# ==========================================
# CLASSE DO DATALOADER (DADOS TABULARES)
# ==========================================
class FeatureDataset(Dataset):
    def __init__(self, features_list, labels_list):
        # Converte a lista de números em Tensores Float
        self.X = torch.tensor(features_list, dtype=torch.float32)
        
        # Normalização: Como as features são porcentagens (0-100), 
        # dividimos por 100 para a rede neural processar os dados entre 0.0 e 1.0
        self.X = self.X / 100.0 
        
        self.y = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# PREPARAÇÃO DOS DADOS (IMAGEM -> AUGMENTATION -> FEATURES)
# ==========================================
def preparar_dados_features():
    pasta_imagens = "D:/Lages/Detector_De_Solda/backpropagation/imagens_geradas"
    
    print(f"Lendo imagens já aumentadas de: {pasta_imagens}...")
    busca = os.path.join(pasta_imagens, '*.png')
    caminhos = sorted(glob.glob(busca))

    if not caminhos:
        print(f"[Erro] Nenhuma imagem encontrada em {pasta_imagens}")
        exit()

    extrator_hsv = Features()
    features_list = []
    labels_list = []
    
    print(f"Extraindo Features HSV de {len(caminhos)} imagens...")
    
    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is None:
            continue
            
        # Extrai a classe diretamente do nome do arquivo gerado pelo builder
        # Exemplo: 'aug_0000_classe_2.png' -> pega o número '2'
        nome_arquivo = os.path.basename(caminho)
        busca_classe = re.search(r'classe_(\d+)', nome_arquivo)
        
        if not busca_classe:
            continue
            
        label = int(busca_classe.group(1))

        if label <= 1: label = 0
        else: label = 1

        # Extrai o dicionário de cores original
        feat_dict = extrator_hsv.extract(img)
        
        # Converte o dicionário para uma lista de 7 números na ordem correta
        linha_features = [feat_dict[col] for col in extrator_hsv.FEATURE_COLUMNS]
        
        features_list.append(linha_features)
        labels_list.append(label)

    # Cria o Dataloader tabular
    dataset = FeatureDataset(features_list, labels_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return dataloader, len(dataset)
# ==========================================
# CRIAÇÃO DA ARQUITETURA (MULTILAYER PERCEPTRON - MLP)
# ==========================================
class MLPSolda(nn.Module):
    def __init__(self, input_size=6, num_classes=2):
        super(MLPSolda, self).__init__()
        
        self.rede = nn.Sequential(
            nn.Linear(input_size, 32), # Camada Oculta 1
            nn.ReLU(),                 
            nn.Dropout(0.2),           
            
            nn.Linear(32, 16),         # Camada Oculta 2
            nn.ReLU(),
            
            nn.Linear(16, num_classes) # Camada de Saída
        )

    def forward(self, x):
        return self.rede(x)

# ==========================================
# O LOOP DE TREINAMENTO (BACKPROPAGATION)
# ==========================================
if __name__ == "__main__":
    dataloader_treino, total_dados = preparar_dados_features()
    print(f"Dataset pronto: {total_dados} amostras tabulares para treinamento.")

    # Instancia o Cérebro Tabular com 7 Features
    modelo = MLPSolda(input_size=6, num_classes=NUM_CLASSES).to(device)
    
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    # Garante que a pasta de destino dos modelos exista
    pasta_modelos = "D:/Lages/Detector_De_Solda/modelos_treinados"
    os.makedirs(pasta_modelos, exist_ok=True)
    
    caminho_modelo_final = os.path.join(pasta_modelos, 'modelo_mlp_features_solda.pth')

    # Instancia o vigilante apontando para a nova pasta de destino
    vigilante_overfitting = EarlyStopping(
        patience=8, 
        min_delta=0.005, 
        path=caminho_modelo_final
    )

    print("\n" + "="*40)
    print("INICIANDO TREINAMENTO POR FEATURES")
    print("="*40)

    for epoch in range(EPOCHS):
        modelo.train() 
        erro_acumulado = 0.0
        acertos = 0

        for entradas_features, labels in dataloader_treino:
            entradas_features = entradas_features.to(device)
            labels = labels.to(device)

            otimizador.zero_grad()

            previsoes = modelo(entradas_features)

            loss = criterio(previsoes, labels)
            loss.backward()
            otimizador.step()

            erro_acumulado += loss.item() * entradas_features.size(0)
            _, classe_prevista = torch.max(previsoes, 1)
            acertos += torch.sum(classe_prevista == labels.data).item()

        erro_medio = erro_acumulado / total_dados
        acuracia = (acertos / total_dados) * 100

        print(f"\nÉpoca [{epoch+1:02d}/{EPOCHS}] | Erro: {erro_medio:.4f} | Acurácia: {acuracia:.2f}%")
        
        # O vigilante analisa o erro médio desta época
        vigilante_overfitting(erro_medio, modelo)
        
        # Se a paciência acabar, interrompemos o loop
        if vigilante_overfitting.early_stop:
            print(f"\n[!] Treinamento interrompido na época {epoch+1} para evitar Overfitting.")
            break

    print("\n" + "="*40)
    print("TREINAMENTO CONCLUÍDO!")
    print(f"Pesos do melhor MLP salvos em: '{vigilante_overfitting.path}'")
    print("="*40)