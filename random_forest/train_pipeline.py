import os
import glob
import sys
import re
import cv2
import pandas as pd

CAMINHO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if CAMINHO_RAIZ not in sys.path:
    sys.path.append(CAMINHO_RAIZ)

from backpropagation.enhancement_LE import Enhancement
from backpropagation.dataset_builder import FusionWeldAugmentation
from backpropagation.features_LA import FeaturesLA as Features
from quality_classification import QualityClassification

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = "D:\\Lages\\Detector_De_Solda\\banco_de_dados\\base_AWSC5.5"
MODEL_PATH = "D:\\Lages\\Detector_De_Solda\\modelos_treinados\\modelo.joblib"


def label_from_filename(path: str) -> int:
    """Extrai o numero do nome do arquivo como nivel da solda (ex.: 3.png -> 3)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"\d+", stem)
    if not m:
        raise ValueError(f"Sem numero no nome do arquivo: {path}")
    return int(m.group())

def load_dataset(images_dir: str):
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg")),
        key=label_from_filename,
    )
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")
    images, labels, names = [], [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [aviso] ignorando ilegivel: {p}")
            continue
        images.append(img)
        labels.append(label_from_filename(p))
        names.append(os.path.basename(p))
    return images, labels, names

def ppm_para_macro(ppm):
    """Traduz o PPM exato para as 2 Macro-Classes (Binário)"""
    if ppm <= 100:
        return 0  # Índice 0: Aceitável
    else:
        return 1  # Índice 1: Inaceitável

def main():
    print("1) Carregando imagens...")
    images, labels, names = load_dataset(IMAGES_DIR)
    print(f"   {len(images)} imagens | niveis: {labels}")

    print("2) Realcando (Enhancement)...")
    enhancer = Enhancement()
    enhanced = [enhancer.enhance(im) for im in images]

    print("3) Data augmentation (Balanceamento de Classes)...")
    aug_imgs = []
    aug_labels = []

    # Criamos dois motores de augmentation com multiplicadores diferentes
    aug_aceitavel = FusionWeldAugmentation(crop_size=(224, 224), n_crops_per_image=15, seed=42)
    aug_inaceitavel = FusionWeldAugmentation(crop_size=(224, 224), n_crops_per_image=10, seed=42)

    for img, lbl_ppm in zip(enhanced, labels):
        if lbl_ppm <= 100:
            novas_imgs, novas_labels = aug_aceitavel.augment_dataset([img], [lbl_ppm])
        else:
            novas_imgs, novas_labels = aug_inaceitavel.augment_dataset([img], [lbl_ppm])

        aug_imgs.extend(novas_imgs)
        aug_labels.extend(novas_labels)

    qtd_aceitaveis = sum(1 for lbl in aug_labels if lbl <= 100)
    qtd_inaceitaveis = sum(1 for lbl in aug_labels if lbl > 100)

    print(f"   Total apos aumento: {len(aug_imgs)} imagens balanceadas ({qtd_aceitaveis} Aceitáveis vs {qtd_inaceitaveis} Inaceitáveis)")    

    print("4) Extraindo parametros HSV (Features)...")
    feat = Features()
    # tabela das 10 imagens originais (a "tabela HSV" pedida)
    base_table = feat.extract_table(enhanced, labels, names)
    base_table.to_csv(os.path.join(HERE, "features_base.csv"), index=False)
    # tabela completa do conjunto aumentado (treino)
    full_table = feat.extract_table(aug_imgs, aug_labels)

    print("\n=== TABELA HSV DAS 10 IMAGENS ORIGINAIS ===")
    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:6.2f}"):
        print(base_table.to_string(index=False))
    print("===========================================\n")

    print("5) Treinando classificador (Random Forest)...")
    X = full_table[Features.FEATURE_COLUMNS]
    y = full_table["nivel_solda"]
    clf = QualityClassification(n_estimators=300)
    clf.fit(X, y)
    print("   ", clf.evaluate(X, y))

    print("6) Salvando modelo e tabelas...")
    clf.save(MODEL_PATH)
    full_table.to_csv(os.path.join(HERE, "features_aumentadas.csv"), index=False)
    print(f"   modelo  -> {MODEL_PATH}")
    print("   tabelas -> features_base.csv | features_aumentadas.csv")
    print("\nPronto. Agora rode a interface:  python gui.py")


if __name__ == "__main__":
    main()