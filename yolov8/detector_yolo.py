import cv2
import numpy as np
from ultralytics import YOLO

class ExtratorCordaoYOLO:
    
    """
    Motor de detecção visual usando YOLOv8.
    Responsável por encontrar o cordão de solda em qualquer lugar da imagem
    e recortá-lo com as coordenadas exatas da detecção.
    """

    def __init__(self, model_path="yolov8n_solda.pt"):
        # Carrega o modelo YOLO treinado (Nano)
        try:
            self.model = YOLO(model_path)
            self.modelo_carregado = True
        except Exception as e:
            print(f"[Aviso] Modelo YOLO não encontrado em {model_path}. Operando em modo de falha.")
            self.modelo_carregado = False

    def detectar_e_recortar(self, img: np.ndarray, padding_ratio: float = 0.05) -> tuple:
        """
        Analisa a imagem, encontra TODAS as soldas e retorna:
        1. img_anotada: Imagem original com todas as caixas desenhadas.
        2. lista_recortes: Uma lista de dicionários contendo os recortes das soldas.
        """
        if not self.modelo_carregado:
            return img, []

        # 1. Inferência do YOLO
        resultados = self.model(img, verbose=False, conf=0.8)
        caixas = resultados[0].boxes
        
        if len(caixas) == 0:
            print("[Aviso YOLO] Nenhuma solda detectada na imagem.")
            return img, []

        img_anotada = img.copy()
        lista_recortes = []

        # 2. Ordena as caixas da Esquerda para a Direita (Baseado na coordenada X)
        # Isso garante que a leitura faça sentido para o operador (Solda 1, 2, 3...)
        caixas_ordenadas = sorted(caixas, key=lambda c: c.xyxy[0][0].item())

        h_img, w_img = img.shape[:2]

        # 3. Processa cada solda encontrada
        for i, caixa in enumerate(caixas_ordenadas):
            x1, y1, x2, y2 = caixa.xyxy[0].cpu().numpy().astype(int)
            confianca = float(caixa.conf[0].cpu().numpy())

            # Calcula a margem (padding) da ZTA
            largura_caixa = x2 - x1
            altura_caixa = y2 - y1
            pad_x = int(largura_caixa * padding_ratio)
            pad_y = int(altura_caixa * padding_ratio)

            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w_img, x2 + pad_x)
            y2_pad = min(h_img, y2 + pad_y)

            # Efetua o recorte
            img_recortada = img[y1_pad:y2_pad, x1_pad:x2_pad]

            # Desenha a caixa na imagem original com o Número da Solda (Ex: #1, #2)
            cv2.rectangle(img_anotada, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 0), 2)
            texto = f"#{i+1} ({confianca*100:.1f}%)"
            cv2.putText(img_anotada, texto, (x1_pad, max(15, y1_pad - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Adiciona à lista de resultados
            lista_recortes.append({
                'id': i + 1,
                'imagem': img_recortada,
                'confianca_yolo': confianca
            })

        return img_anotada, lista_recortes