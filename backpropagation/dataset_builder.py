import os
import glob
import cv2
import torch
import random
import re             # <--- ADICIONE ESTA LINHA AQUI
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ==========================================
# 1. CLASSE DE DATA AUGMENTATION (FUSÃO DE TÉCNICAS)
# ==========================================
class FusionWeldAugmentation:
    def __init__(self, crop_size=(224, 224), n_crops_per_image=10, seed=42):
        self.crop_size = crop_size
        self.n_crops = n_crops_per_image
        self.seed = seed
        random.seed(self.seed)

    # ------------------------------------------
    # MÉTODOS BASE E ÓTICOS (Nossa Abordagem)
    # ------------------------------------------
    def _random_crop(self, img):
        """Recorte focado no centro horizontal e aleatório na vertical (ROI Natural)."""
        h, w = img.shape[:2]
        ch, cw = self.crop_size
        if h <= ch or w <= cw:
            return cv2.resize(img, self.crop_size, interpolation=cv2.INTER_AREA)
        y = random.randint(0, h - ch)
        x = (w - cw) // 2
        return img[y:y + ch, x:x + cw]

    @staticmethod
    def _motion_blur(img, size=5):
        kernel = np.zeros((size, size))
        np.fill_diagonal(kernel, 1)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def _vignette(img):
        h, w = img.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(w, w/2)
        Y_resultant_kernel = cv2.getGaussianKernel(h, h/2)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        img_vignette = np.copy(img).astype(np.float32)
        for i in range(3):
            img_vignette[:, :, i] = img_vignette[:, :, i] * mask
        return np.clip(img_vignette, 0, 255).astype(np.uint8)

    @staticmethod
    def _clahe_luminance(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # ------------------------------------------
    # MÉTODOS ESTRUTURAIS (Abordagem da Chefe)
    # ------------------------------------------
    @staticmethod
    def _rotate(img, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        # BORDER_REFLECT é vital para não gerar bordas pretas que confundiriam a rede
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _gamma(img, g):
        inv = 1.0 / g
        table = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
        return cv2.LUT(img, table)

    @staticmethod
    def _noise(img, rng, sigma=8):
        noise = rng.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _zoom(img, factor=1.1):
        h, w = img.shape[:2]
        ch, cw = int(h / factor), int(w / factor)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = img[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------
    # MOTOR DE FUSÃO
    # ------------------------------------------
    def augment(self, img: np.ndarray, base_seed: int = 0) -> list:
        rng = np.random.default_rng(self.seed + base_seed)
        variants = []
        
        # Para cada recorte na solda, geramos 8 variações unindo as duas lógicas
        for _ in range(self.n_crops):
            # 1. Base geométrica (Recorte focado, igual a versão 1)
            base_crop = self._random_crop(img)
            
            # Variante 0: O recorte puro com cores originais
            variants.append(base_crop)
            
            # Variante 1: Espelhamento Horizontal
            variants.append(cv2.flip(base_crop, 1))
            
            # Variantes 2 e 3: Rotação suave da chefe
            variants.append(self._rotate(base_crop, 6))
            variants.append(self._rotate(base_crop, -6))
            
            # Variante 4: A Vinheta do boroscópio combinada com o Gamma Escuro da chefe
            variants.append(self._gamma(self._vignette(base_crop), 0.80))
            
            # Variante 5: Micro-contraste térmico combinado com Gamma Claro da chefe
            variants.append(self._gamma(self._clahe_luminance(base_crop), 1.25))
            
            # Variante 6: A câmara tremeu (Motion Blur) e o sensor gerou Ruído
            variants.append(self._noise(self._motion_blur(base_crop, size=5), rng))
            
            # Variante 7: Zoom da chefe
            variants.append(self._zoom(base_crop, factor=1.10))
            
        return variants

    def augment_dataset(self, images: list, labels: list):
        aug_imgs, aug_labels = [], []
        for idx, (img, label) in enumerate(zip(images, labels)):
            for v in self.augment(img, base_seed=idx):
                aug_imgs.append(v)
                aug_labels.append(label)
        return aug_imgs, aug_labels

# ==========================================
# 2. CLASSE DO DATALOADER (PYTORCH)
# ==========================================
class SoldaDataset(Dataset):
    def __init__(self, imagens, labels):
        self.imagens = imagens
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        img_bgr = self.imagens[idx]
        label = self.labels[idx]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

# ==========================================
# 3. EXECUÇÃO PARA VALIDAÇÃO VISUAL
# ==========================================
if __name__ == "__main__":
    
    # Novo mapeamento para 3 Macro-Classes:
    # 0: Nível Zero (Sem oxidação)
    # 1: Solda Aceitável (Níveis 1, 2, 3 e 4)
    # 2: Solda Inaceitável (Níveis 5 a 10)
    dicionario_classes = {
        0: 0, 
        10: 1, 25: 1, 50: 1, 100: 1, 
        200: 2, 500: 2, 1000: 2, 5000: 2, 12500: 2, 25000: 2
    }

    print("Carregando imagens originais dos bancos de dados...")
    
    # Lista com todas as pastas fontes de imagens originais
    pastas_entrada = [
        "D:/Lages/Detector_De_Solda/banco_de_dados/base_AWSC5.5",
    ]

    caminhos_imagens = []
    
    # Percorre cada pasta e junta todas as imagens numa lista só
    for pasta in pastas_entrada:
        busca = os.path.join(pasta, '*.png')
        arquivos_encontrados = sorted(glob.glob(busca))
        caminhos_imagens.extend(arquivos_encontrados)
        print(f" -> Encontradas {len(arquivos_encontrados):03d} imagens em: {pasta}")
    
    if not caminhos_imagens:
        print("\n[Erro] Nenhuma imagem encontrada. Verifique se os caminhos das pastas estão corretos.")
        exit()

    imagens_base, labels_base = [], []

    for caminho in caminhos_imagens:
        img = cv2.imread(caminho)
        if img is not None:
            nome_arquivo = os.path.basename(caminho).replace('.png', '')
            
            # Encontra TODOS os números isolados no nome do arquivo
            busca_numeros = re.findall(r'\d+', nome_arquivo)
            
            classe = -1
            # Testa cada número encontrado para ver se ele faz parte do nosso dicionário de PPM
            for num_str in busca_numeros:
                ppm_val = int(num_str)
                if ppm_val in dicionario_classes:
                    classe = dicionario_classes[ppm_val]
                    break  # Achou a classe correta, sai do loop de busca
                    
            # Só adiciona a imagem se a classe for encontrada
            if classe != -1:
                imagens_base.append(img)
                labels_base.append(classe)
            else:
                print(f" -> [Aviso] Arquivo ignorado: '{nome_arquivo}' (Nenhum PPM reconhecido no nome).")

    print(f"\nTotal de imagens originais unificadas: {len(imagens_base)}")
    print("Executando a FUSÃO de Data Augmentation (Isso pode demorar um pouco)...")
    
    # 10 recortes x 8 variantes = 80 imagens por original
    aug_fusion = FusionWeldAugmentation(n_crops_per_image=10)
    imgs_aumentadas, labels_aumentados = aug_fusion.augment_dataset(imagens_base, labels_base)
    
    # Define o novo diretório de destino na nova subpasta solicitada
    pasta_saida = "D:/Lages/Detector_De_Solda/backpropagation/imagens_geradas"
    os.makedirs(pasta_saida, exist_ok=True)
    
    print(f"Salvando fisicamente {len(imgs_aumentadas)} imagens robustas...")
    for i, (img, label) in enumerate(zip(imgs_aumentadas, labels_aumentados)):
        nome_arquivo = os.path.join(pasta_saida, f"aug_{i:04d}_classe_{label}.png")
        cv2.imwrite(nome_arquivo, img)

    print(f"\nSucesso! O dataset definitivo está pronto na pasta:\n -> '{pasta_saida}'")