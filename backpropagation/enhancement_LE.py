"""
enhancement.py
==============
Classe responsavel por MELHORAR A QUALIDADE da imagem de solda antes de
extrair caracteristicas. As mesmas configuracoes devem ser usadas no
treino e na inferencia (GUI), caso contrario as cores extraidas mudam.

Pipeline de realce:
    1. Reducao de ruido (filtro bilateral - preserva bordas)
    2. Aumento de contraste local (CLAHE no canal L do espaco LAB)
    3. Realce de nitidez (unsharp mask)
    4. (Opcional) Balanco de branco gray-world -> desligado por padrao,
       pois ele altera a distribuicao de cores, que aqui e o "sinal".
"""

import cv2
import numpy as np


class Enhancement:
    def __init__(
        self,
        denoise: bool = True,
        bilateral_d: int = 7,
        bilateral_sigma_color: int = 50,
        bilateral_sigma_space: int = 50,
        clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_grid: tuple = (8, 8),
        sharpen: bool = True,
        sharpen_amount: float = 1.5,
        white_balance: bool = False,
    ):
        self.denoise = denoise
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe = clahe
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.sharpen = sharpen
        self.sharpen_amount = sharpen_amount
        self.white_balance = white_balance

    # ---------- etapas individuais ----------
    def _denoise(self, img):
        return cv2.bilateralFilter(
            img,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

    def _clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _sharpen(self, img):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        return cv2.addWeighted(
            img, self.sharpen_amount, blurred, 1.0 - self.sharpen_amount, 0
        )

    def _white_balance(self, img):
        # gray-world: assume que a media de cada canal deve ser igual
        result = img.astype(np.float32)
        avg = result.reshape(-1, 3).mean(axis=0)
        gray = avg.mean()
        for c in range(3):
            if avg[c] > 1e-6:
                result[:, :, c] *= gray / avg[c]
        return np.clip(result, 0, 255).astype(np.uint8)

    # ---------- API publica ----------
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """Recebe imagem BGR (uint8) e devolve a versao realcada."""
        if img is None:
            raise ValueError("Imagem invalida (None).")
        out = img.copy()
        if self.denoise:
            out = self._denoise(out)
        if self.clahe:
            out = self._clahe(out)
        if self.sharpen:
            out = self._sharpen(out)
        if self.white_balance:
            out = self._white_balance(out)
        return out

    def enhance_path(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Nao foi possivel ler a imagem: {path}")
        return self.enhance(img)
