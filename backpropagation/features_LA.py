"""
features_LA.py
===========
Classe aprimorada para EXTRAÇÃO DE PARÂMETROS HSV (Versão Estudo de Ablação).
Além das proporções de cores tradicionais, esta versão fatia o espectro
quente para detectar "Amarelo Palha" vs "Marrom/Vermelho" e extrai os
momentos estatísticos do Matiz (H_mean e H_std) da área oxidada.

OpenCV usa HSV com H em [0,179], S e V em [0,255].
"""

import cv2
import numpy as np
import pandas as pd


class FeaturesLA:
    # Novas colunas atualizadas para o classificador
    FEATURE_COLUMNS = [
        "pct_azul", 
        "pct_roxo", 
        "pct_cinza", 
        "pct_marrom", 
        "h_mean", 
        "h_std",
        #"pct_amarelo_palha",
        #"saturacao",
        #"valor"
    ]

    def __init__(
        self,
        s_min_color: int = 50,   # Abaixo disso = cinza/acromático
        v_min: int = 25,         # Abaixo disso = preto/sombra (ignorado)
        brown_v_max: int = 170,  # Limiar de brilho para separar amarelo palha de marrom escuro
    ):
        self.s_min_color = s_min_color
        self.v_min = v_min
        self.brown_v_max = brown_v_max

    # ---------- NÚCLEO MATEMÁTICO ----------
    def extract(self, img: np.ndarray) -> dict:
        """Recebe imagem BGR (uint8) e devolve dict com as 9 características aprimoradas."""
        if img is None:
            raise ValueError("Imagem invalida (None).")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        total = H.size

        # Filtros Base
        colored = (S >= self.s_min_color) & (V >= self.v_min)
        achrom = (S < self.s_min_color) & (V >= self.v_min)
        
        # Filtros de Cor Fria
        blue = colored & (H >= 95) & (H <= 130)
        purple = colored & (H >= 131) & (H <= 160)

        # Amarelo Palha: H entre 15 e 35, com brilho alto (V >= brown_v_max)
        straw_yellow = colored & (H >= 15) & (H <= 35) & (V >= self.brown_v_max)
        
        # Marrom/Vermelho: H nas extremidades (perto de 0 ou 165-179), com brilho menor
        warm_hue = ((H <= 30) | (H >= 170))
        brown = colored & warm_hue & (V < self.brown_v_max)

        # ----------------------------------------------------
        # MOMENTOS ESTATÍSTICOS (Apenas onde há oxidação/cor)
        # ----------------------------------------------------
        H_colored = H[colored]
        h_mean = float(np.mean(H_colored)) if H_colored.size > 0 else 0.0
        h_std = float(np.std(H_colored)) if H_colored.size > 0 else 0.0

        def pct(mask):
            return 100.0 * float(np.count_nonzero(mask)) / total

        return {
            "pct_azul": pct(blue),
            "pct_roxo": pct(purple),
            "pct_cinza": pct(achrom),
            "pct_amarelo_palha": pct(straw_yellow),
            "pct_marrom": pct(brown),
            "valor": 100.0 * float(V.mean()) / 255.0,
            "saturacao": 100.0 * float(S.mean()) / 255.0,
            "h_mean": h_mean,
            "h_std": h_std
        }

    # ---------- TABELAS ----------
    def extract_row(self, img, label=None, name=None) -> dict:
        row = {}
        if name is not None:
            row["imagem"] = name
        row.update(self.extract(img))
        if label is not None:
            row["nivel_solda"] = label
        return row

    def extract_table(self, images, labels=None, names=None) -> pd.DataFrame:
        """Monta um DataFrame (uma linha por imagem) com as características."""
        n = len(images)
        labels = labels if labels is not None else [None] * n
        names = names if names is not None else [None] * n
        rows = [
            self.extract_row(img, lab, nm)
            for img, lab, nm in zip(images, labels, names)
        ]
        df = pd.DataFrame(rows)
        # ordena colunas: imagem | features | nivel_solda
        ordered = (
            (["imagem"] if "imagem" in df else [])
            + self.FEATURE_COLUMNS
            + (["nivel_solda"] if "nivel_solda" in df else [])
        )
        return df[ordered]