"""
features.py
===========
Classe que EXTRAI OS PARAMETROS HSV de cada imagem e monta a TABELA com:
    - % de pixels AZUL
    - % de pixels CINZA
    - % de pixels VERDE
    - % de pixels MARROM
    - % de pixels ROXO
    - VALOR medio    (brilho medio, 0-100 %)
    - SATURACAO media (0-100 %)

OpenCV usa HSV com H em [0,179], S e V em [0,255].
Os intervalos abaixo sao constantes da classe e podem ser ajustados.
Observacao: as porcentagens de cor NAO somam 100 %, pois pixels brancos,
pretos, vermelhos, laranjas e amarelos ficam fora das 5 categorias pedidas.
"""

import cv2
import numpy as np
import pandas as pd


class Features:
    # nomes das colunas (na ordem usada pelo classificador)
    FEATURE_COLUMNS = [
        "pct_azul", "pct_cinza", "pct_marrom", "pct_roxo", "valor"
    ]

    def __init__(
        self,
        s_min_color: int = 50,   # abaixo disso o pixel e considerado acromatico (cinza)
        v_min: int = 25,         # abaixo disso o pixel e quase preto (ignorado nas cores)
        brown_v_max: int = 170,  # marrom = tom quente porem escuro
    ):
        self.s_min_color = s_min_color
        self.v_min = v_min
        self.brown_v_max = brown_v_max

    # ---------- nucleo ----------
    def extract(self, img: np.ndarray) -> dict:
        """Recebe imagem BGR (uint8) e devolve dict com as 7 caracteristicas."""
        if img is None:
            raise ValueError("Imagem invalida (None).")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        total = H.size

        colored = (S >= self.s_min_color) & (V >= self.v_min)
        achrom = (S < self.s_min_color) & (V >= self.v_min)

        blue = colored & (H >= 95) & (H <= 130)
        green = colored & (H >= 40) & (H <= 85)
        purple = colored & (H >= 131) & (H <= 160)
        # marrom = laranja/vermelho escuro (hue baixo OU proximo de 180) e V baixo
        warm_hue = ((H <= 30) | (H >= 170))
        brown = colored & warm_hue & (V < self.brown_v_max)

        def pct(mask):
            return 100.0 * float(np.count_nonzero(mask)) / total

        return {
            "pct_azul": pct(blue),
            "pct_cinza": pct(achrom),
            "pct_verde": pct(green),
            "pct_marrom": pct(brown),
            "pct_roxo": pct(purple),
            "valor": 100.0 * float(V.mean()) / 255.0,
            "saturacao": 100.0 * float(S.mean()) / 255.0,
        }

    # ---------- tabelas ----------
    def extract_row(self, img, label=None, name=None) -> dict:
        row = {}
        if name is not None:
            row["imagem"] = name
        row.update(self.extract(img))
        if label is not None:
            row["nivel_solda"] = label
        return row

    def extract_table(self, images, labels=None, names=None) -> pd.DataFrame:
        """Monta um DataFrame (uma linha por imagem) com as caracteristicas HSV."""
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
