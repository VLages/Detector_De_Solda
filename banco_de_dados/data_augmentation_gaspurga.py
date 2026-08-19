import os
import glob
import re
import cv2
import random
import shutil

# ==========================================
# CONFIGURAÇÕES DE PASTAS
# ==========================================
PASTA_ORIGEM = r"D:\Lages\Detector_De_Solda\banco_de_dados\base_GasPurga"
PASTA_DESTINO = r"D:\Lages\Detector_De_Solda\banco_de_dados\base_GasPurga_balanceada"

def aplicar_variacao_simples(img):
    """Aplica transformações leves na imagem inteira para não enganar o YOLO."""
    img_aug = img.copy()
    
    # 1. Espelhamento aleatório (Horizontal, Vertical ou Ambos)
    flip_code = random.choice([-1, 0, 1, None])
    if flip_code is not None:
        img_aug = cv2.flip(img_aug, flip_code)
        
    # 2. Alteração leve de Brilho e Contraste
    alfa = random.uniform(0.85, 1.15)
    beta = random.randint(-15, 15)
    img_aug = cv2.convertScaleAbs(img_aug, alpha=alfa, beta=beta)
    
    # 3. Borrão super leve ocasional
    if random.random() > 0.5:
        img_aug = cv2.GaussianBlur(img_aug, (3, 3), 0)
        
    return img_aug

def main():
    os.makedirs(PASTA_DESTINO, exist_ok=True)
    caminhos = glob.glob(os.path.join(PASTA_ORIGEM, "*.png")) + glob.glob(os.path.join(PASTA_ORIGEM, "*.jpg"))
    
    cont_0 = 0
    cont_1 = 0
    cont_2 = 0
    
    # Contador global para gerar o ID final da imagem de forma sequencial (ex: g0-101)
    id_gerador = 100 
    
    print("Iniciando balanceamento do banco de testes...\n")
    
    for caminho in caminhos:
        nome_arquivo = os.path.basename(caminho)
        img = cv2.imread(caminho)
        if img is None: continue
            
        # Regex flexível: aceita tanto traço (g0-1) quanto underline (g0_1)
        busca = re.search(r'g(\d+)[-_]', nome_arquivo, re.IGNORECASE)
            
        if not busca:
            shutil.copy(caminho, os.path.join(PASTA_DESTINO, nome_arquivo))
            continue
            
        ppm_real = int(busca.group(1))
        ext = os.path.splitext(nome_arquivo)[1]
        
        # Salva a imagem original na nova pasta
        cv2.imwrite(os.path.join(PASTA_DESTINO, nome_arquivo), img)
        
        # ==========================================
        # REGRAS MATEMÁTICAS CORRIGIDAS
        # ==========================================
        if ppm_real == 0:
            novas_para_gerar = 10 # 1 original + 10 novas = 11 por imagem
            cont_0 += (1 + novas_para_gerar)
        elif ppm_real <= 100:
            novas_para_gerar = 6  # 1 original + 6 novas = 7 por imagem
            cont_1 += (1 + novas_para_gerar)
        else:
            novas_para_gerar = 0  # Apenas a original
            cont_2 += 1
            
        # Gerando as cópias com nomenclatura padrão
        for _ in range(novas_para_gerar):
            img_nova = aplicar_variacao_simples(img)
            
            # Formato rigoroso para enganar o .gitignore: g[PPM]-[ID_GERADOR]
            novo_nome = f"g{ppm_real}-{id_gerador}{ext}"
            cv2.imwrite(os.path.join(PASTA_DESTINO, novo_nome), img_nova)
            id_gerador += 1
            
    print("="*50)
    print("BANCO DE TESTES BALANCEADO COM SUCESSO!")
    print("="*50)
    print(f"Total de Imagens 0 PPM (Metal Limpo): {cont_0}")
    print(f"Total de Imagens 10-100 PPM (Aceitáveis): {cont_1}")
    print(f"Total de Imagens 200+ PPM (Inaceitáveis): {cont_2}")
    print(f"\nAs imagens prontas para validação estão em:\n'{PASTA_DESTINO}'")

if __name__ == "__main__":
    main()