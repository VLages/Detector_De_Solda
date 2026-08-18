from ultralytics import YOLO
import os
import glob
import shutil

def treinar_yolo_solda():
    """
    Inicia o treinamento do modelo YOLOv8 Nano para detecção do cordão de solda.
    """
    # ==========================================
    # ANCORAGEM DE DIRETÓRIO
    # ==========================================
    # Descobre o caminho absoluto da pasta exata onde ESTE script está salvo (a pasta yolov8)
    DIRETORIO_YOLO = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Carrega o modelo base (ancorado na pasta yolov8)
    caminho_modelo = os.path.join(DIRETORIO_YOLO, 'yolov8n.pt')
    modelo = YOLO(caminho_modelo)
    
    # 2. Caminho para o arquivo data.yaml (ancorado na pasta yolov8/dataset)
    caminho_yaml = os.path.join(DIRETORIO_YOLO, "dataset", "data.yaml")
    
    if not os.path.exists(caminho_yaml):
        print(f"ERRO: Arquivo '{caminho_yaml}' não encontrado.")
        print("Verifique se você extraiu o ZIP do Roboflow na mesma pasta deste script.")
        return

    print("Iniciando o treinamento do YOLOv8...")
    print(f"Lendo dados de: {caminho_yaml}")
    
    # 3. Inicia o Treinamento
    pasta_runs = os.path.join(DIRETORIO_YOLO, "runs")
    
    resultados = modelo.train(
        data=caminho_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        project=pasta_runs,             # <--- O SEGREDO ESTÁ AQUI: Força a criar a pasta runs aqui
        name='detect/detector_solda',   # Mantém a organização runs/detect/detector_solda
        plots=True
    )

    print("\n" + "="*50)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    
    # ==========================================
    # ORGANIZAÇÃO AUTOMÁTICA DOS ARQUIVOS
    # ==========================================
    # A pasta destino precisa ficar na RAIZ do projeto para a Interface usar.
    # O ".." significa "voltar uma pasta" (Sair de yolov8 e ir para Detector_De_Solda)
    pasta_destino = os.path.abspath(os.path.join(DIRETORIO_YOLO, "..", "modelos_treinados"))
    os.makedirs(pasta_destino, exist_ok=True)

    # Procura o best.pt ancorado na nova pasta runs
    busca_runs = os.path.join(pasta_runs, 'detect', '*', 'weights', 'best.pt')
    arquivos_yolo = glob.glob(busca_runs)
    
    if arquivos_yolo:
        # Pega o arquivo mais recente criado
        caminho_yolo_origem = max(arquivos_yolo, key=os.path.getctime)
        caminho_yolo_destino = os.path.join(pasta_destino, "yolov8n_solda.pt")
        
        # Copia e renomeia o arquivo
        shutil.copy2(caminho_yolo_origem, caminho_yolo_destino)
        
        print(f"O seu modelo pronto foi copiado da pasta:")
        print(f" -> '{caminho_yolo_origem}'")
        print(f"E salvo e renomeado com sucesso em:")
        print(f" -> '{caminho_yolo_destino}'")
        print("\nVocê já pode usar este modelo na sua Interface Gráfica!")
    else:
        print("\n[Aviso] Arquivo 'best.pt' não encontrado na pasta 'runs/'.")

    print("="*50)

if __name__ == '__main__':
    # Necessário no Windows para não bugar o multiprocessamento do PyTorch
    from multiprocessing import freeze_support
    freeze_support()
    
    treinar_yolo_solda()