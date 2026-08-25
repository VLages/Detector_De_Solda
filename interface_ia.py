import os
import sys
import cv2
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.font as tkfont

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

CAMINHO_RAIZ = os.path.dirname(os.path.abspath(__file__))
CAMINHO_RF = os.path.join(CAMINHO_RAIZ, 'random_forest')
if CAMINHO_RF not in sys.path:
    sys.path.append(CAMINHO_RF)

# Importando o pipeline completo
from backpropagation.enhancement_LE import Enhancement
from backpropagation.features_LA import FeaturesLA as Features
from yolov8.detector_yolo import ExtratorCordaoYOLO

# Importando a classe do Random Forest
from random_forest.quality_classification import QualityClassification

# ==========================================
# ARQUITETURA DA REDE MLP
# ==========================================
# ==========================================
# ARQUITETURA DA REDE MLP (Sincronizada)
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
# ==========================================
# CAMINHOS E CONFIGURAÇÕES
# ==========================================
MODEL_PATH = "modelos_treinados/modelo_mlp_features_solda.pth" 
RF_PATH = "modelos_treinados/modelo.joblib"  # <--- Caminho do Random Forest
YOLO_PATH = "modelos_treinados/yolov8n_solda.pt"
LOGO_PATH = "modelos_treinados/logo.png"

# ==========================================
# PALETA DE CORES
# ==========================================
BG_MAIN   = "#081326"
BG_HEADER = "#0C1E3D"
BG_CARD   = "#11254A"
BG_ROW    = "#15294F"
BG_ROW_ALT= "#0F2042"
BG_HEAD   = "#1C3A6E"
TRACK     = "#203E6B"
ACCENT    = "#4F9DF7"
TEAL      = "#2DD4BF"
TEXT      = "#E8F0FB"
TEXT_MUT  = "#8DA6CC"
BORDER    = "#1E3C68"

SWATCH = {
    "pct_azul":   "#4F9DF7", 
    "pct_cinza":  "#9AA7B8",
    "pct_verde":  "#34D399", 
    "pct_marrom": "#B5743A",
    "pct_roxo":   "#A78BFA", 
    "valor":      "#F5D67B",
    "saturacao":  "#2DD4BF",
    "pct_amarelo_palha":   "#FDE047", # Amarelo vibrante/palha
    "pct_marrom_vermelho": "#9A3412", # Marrom avermelhado escuro
    "h_mean":              "#F472B6", # Rosa/Magenta (Apenas para a barra)
    "h_std":               "#C084FC"  # Roxo claro (Apenas para a barra)
}

LABEL_PT = {
    "pct_azul": "Azul", 
    "pct_cinza": "Cinza/Prata", 
    "pct_verde": "Verde",
    "pct_marrom": "Marrom Clássico", 
    "pct_roxo": "Roxo",
    "valor": "Brilho (Luz)", 
    "saturacao": "Saturação",
    "pct_amarelo_palha": "Amarelo Palha",
    "pct_marrom_vermelho": "Marrom/Vermelho",
    "h_mean": "Matiz Médio (H)",
    "h_std": "Desvio do Matiz"

}
LABELS_MLP = [
    "Aceitável (Níveis 1 a 4)", 
    "Inaceitável (Níveis 5 a 10)"
]

def lerp(c1, c2, t):
    a = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i+2], 16) for i in (1, 3, 5))
    r = tuple(round(a[i] + (b[i]-a[i])*t) for i in range(3))
    return f"#{r[0]:02x}{r[1]:02x}{r[2]:02x}"

def severity_color(level):
    if level == 0: return "#22C55E"
    return "#EF4444"

def round_rect(canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
           x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)

class GUI:
    BAR_W = 132
    BAR_H = 9

    def __init__(self, model_path: str = MODEL_PATH):
        # Motores de IA e Visão
        self.enhancer = Enhancement()
        self.features = Features()
        self.yolo = ExtratorCordaoYOLO(model_path=YOLO_PATH)
        self.model_path = model_path
        
        self.modelo_mlp = None
        self.modelo_rf = None # <--- Instância do RF
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._photo = None
        
        # Variáveis de Estado para Múltiplas Soldas
        self.resultados_atuais = []
        self.solda_atual_idx = 0

        # ==========================================
        # CORREÇÃO: CRIA A JANELA PRIMEIRO
        # ==========================================
        self.root = tk.Tk()
        
        # SÓ ENTÃO CRIA A VARIÁVEL DO TKINTER
        self.model_choice = tk.StringVar(value="MLP") 

        self.root.title("Inspeção de Solda por Inteligência Artificial (MLP + Random Forest + YOLO)")
        self.root.geometry("1160x800")
        self.root.minsize(1040, 750)
        self.root.configure(bg=BG_MAIN)

        fams = set(tkfont.families(self.root))
        fam = next((f for f in ["Segoe UI", "Helvetica Neue", "Arial"] if f in fams), "Helvetica")
        self.f_title = tkfont.Font(family=fam, size=17, weight="bold")
        self.f_sub   = tkfont.Font(family=fam, size=10)
        self.f_card  = tkfont.Font(family=fam, size=10, weight="bold")
        self.f_text  = tkfont.Font(family=fam, size=10)
        self.f_big   = tkfont.Font(family=fam, size=32, weight="bold")
        self.f_med   = tkfont.Font(family=fam, size=13, weight="bold")
        self.f_small = tkfont.Font(family=fam, size=9)

        self._build()
        self._load_model()

    def _load_model(self):
        # Carrega MLP
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Modelo não encontrado",
                                   f"O arquivo {self.model_path} não existe.\nRode o treinamento primeiro.")
        else:
            try:
                self.modelo_mlp = MLPSolda(input_size=6, num_classes=2)
                self.modelo_mlp.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                self.modelo_mlp.eval()
                self.modelo_mlp.to(self.device)
                print("Modelo MLP carregado com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro ao carregar modelo MLP", str(e))

        # Carrega Random Forest
        if not os.path.exists(RF_PATH):
            messagebox.showwarning("Modelo Random Forest Ausente", 
                                   f"O arquivo '{RF_PATH}' não foi encontrado.\nO sistema funcionará, mas o Random Forest ficará zerado.")
        else:
            try:
                self.modelo_rf = QualityClassification.load(RF_PATH)
                print("Modelo Random Forest carregado com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro no Random Forest", f"Erro ao carregar o modelo:\n{e}")

        # Carrega o Normalizador Matemático (Scaler)
        caminho_scaler = "modelos_treinados/scaler.pkl"
        if os.path.exists(caminho_scaler):
            with open(caminho_scaler, "rb") as f:
                self.scaler = pickle.load(f)
            print("Scaler matemático carregado com sucesso!")
        else:
            self.scaler = None
            messagebox.showwarning("Aviso", "Arquivo scaler.pkl não encontrado. A MLP pode falhar.")        

    def _card(self, parent, title, accent_top=False):
        outer = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        if accent_top:
            tk.Frame(outer, bg=ACCENT, height=3).pack(fill=tk.X)
        head = tk.Frame(outer, bg=BG_CARD)
        head.pack(fill=tk.X, padx=14, pady=(12, 0))
        tk.Label(head, text=title, bg=BG_CARD, fg=ACCENT if accent_top else TEXT_MUT,
                 font=self.f_card).pack(side=tk.LEFT)
        body = tk.Frame(outer, bg=BG_CARD)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)
        return outer, body

    def _build(self):
        header = tk.Frame(self.root, bg=BG_HEADER)
        header.pack(fill=tk.X)
        inner = tk.Frame(header, bg=BG_HEADER)
        inner.pack(fill=tk.X, padx=22, pady=14)

        self.img_logo_tk = None
        if os.path.exists(LOGO_PATH):
            try:
                pil_logo = Image.open(LOGO_PATH).convert("RGBA")
                # Redimensiona para uma altura de 50 pixels mantendo a proporção
                h_alvo = 50
                w_alvo = int((h_alvo / pil_logo.height) * pil_logo.width)
                pil_logo = pil_logo.resize((w_alvo, h_alvo), Image.Resampling.LANCZOS)
                self.img_logo_tk = ImageTk.PhotoImage(pil_logo)
                
                lbl_logo = tk.Label(inner, image=self.img_logo_tk, bg=BG_HEADER)
                lbl_logo.pack(side=tk.LEFT, padx=(0, 20))
            except Exception as e:
                print(f"[Aviso] Não foi possível processar a logo: {e}")

        tk.Frame(inner, bg=ACCENT, width=5, height=42).pack(side=tk.LEFT, padx=(0, 14))
        
        tcol = tk.Frame(inner, bg=BG_HEADER)
        tcol.pack(side=tk.LEFT)
        tk.Label(tcol, text="Qualidade da Solda", bg=BG_HEADER, fg=TEXT, font=self.f_title).pack(anchor="w")
        tk.Label(tcol, text="Avaliação por Inteligência Artificial (MLP vs Random Forest)", bg=BG_HEADER, fg=TEXT_MUT, font=self.f_sub).pack(anchor="w")

        grid = tk.Frame(self.root, bg=BG_MAIN)
        grid.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        grid.columnconfigure(0, weight=1, uniform="c")
        grid.columnconfigure(1, weight=1, uniform="c")
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        img_card, self.img_body = self._card(grid, "IMAGEM ORIGINAL (DETECÇÃO YOLO)")
        img_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        
        res_card, self.res_body = self._card(grid, "RESULTADO DA INTELIGÊNCIA ARTIFICIAL", accent_top=True)
        res_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        
        tab_card, self.tab_body = self._card(grid, "REFERÊNCIA HSV (TABELA DE CORES)")
        tab_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        
        exp_card, self.exp_body = self._card(grid, "CONFIANÇA DO MODELO")
        exp_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))

        self._build_image(self.img_body)
        self._build_result(self.res_body)
        self._build_table(self.tab_body, None)
        self._build_charts(self.exp_body)

    def _build_image(self, body):
        btn = tk.Button(body, text="  Carregar imagem  ", command=self.on_load,
                        bg=ACCENT, fg="white", activebackground=lerp(ACCENT, "#000000", 0.15),
                        activeforeground="white", relief="flat", font=self.f_card,
                        cursor="hand2", pady=8, bd=0)
        btn.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))

        # Painel de Navegação de Múltiplas Soldas
        self.nav_frame = tk.Frame(body, bg=BG_CARD)
        
        self.btn_prev = tk.Button(self.nav_frame, text="< Anterior", command=self.prev_weld, 
                                  bg=BG_ROW, fg=TEXT, activebackground=ACCENT, relief="flat", cursor="hand2")
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 10))
        
        self.lbl_nav = tk.Label(self.nav_frame, text="Solda 1 de 1", bg=BG_CARD, fg=TEXT_MUT, font=self.f_card)
        self.lbl_nav.pack(side=tk.LEFT, expand=True)
        
        self.btn_next = tk.Button(self.nav_frame, text="Próxima >", command=self.next_weld, 
                                  bg=BG_ROW, fg=TEXT, activebackground=ACCENT, relief="flat", cursor="hand2")
        self.btn_next.pack(side=tk.RIGHT, padx=(10, 0))

        self.canvas_img = tk.Label(body, text="(nenhuma imagem)", bg=BG_ROW_ALT, fg=TEXT_MUT, font=self.f_text)
        self.canvas_img.pack(fill=tk.BOTH, expand=True)

    def _build_result(self, body):
        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill=tk.BOTH, expand=True)

        self.badge = tk.Canvas(row, width=168, height=168, bg=BG_CARD, highlightthickness=0)
        self.badge.pack(side=tk.LEFT, padx=(4, 18))

        info = tk.Frame(row, bg=BG_CARD)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor="n")
        
        # ---------------------------------------------
        # SELETOR DE MODELOS (MLP vs RF)
        # ---------------------------------------------
        tk.Label(info, text="SELECIONAR MODELO", bg=BG_CARD, fg=TEXT_MUT, font=self.f_small).pack(anchor="w", pady=(0, 2))
        frame_radios = tk.Frame(info, bg=BG_CARD)
        frame_radios.pack(anchor="w", pady=(0, 10))
        
        tk.Radiobutton(frame_radios, text="Rede Neural (MLP)", variable=self.model_choice, value="MLP", 
                       command=self.atualizar_tela_solda, bg=BG_CARD, fg=TEXT, selectcolor=BG_ROW_ALT, 
                       activebackground=BG_CARD, activeforeground=ACCENT, cursor="hand2").pack(side=tk.LEFT, padx=(0,10))
                       
        tk.Radiobutton(frame_radios, text="Random Forest", variable=self.model_choice, value="RF", 
                       command=self.atualizar_tela_solda, bg=BG_CARD, fg=TEXT, selectcolor=BG_ROW_ALT, 
                       activebackground=BG_CARD, activeforeground=ACCENT, cursor="hand2").pack(side=tk.LEFT)
        # ---------------------------------------------
        
        tk.Label(info, text="VEREDITO FINAL", bg=BG_CARD, fg=TEXT_MUT, font=self.f_small).pack(anchor="w", pady=(6, 0))
        self.level_lbl = tk.Label(info, text="-", bg=BG_CARD, fg=TEXT, font=self.f_med)
        self.level_lbl.pack(anchor="w")

        tk.Label(info, text="CERTEZA DO MODELO", bg=BG_CARD, fg=TEXT_MUT, font=self.f_small).pack(anchor="w", pady=(14, 2))
        self.conf_canvas = tk.Canvas(info, width=240, height=16, bg=BG_CARD, highlightthickness=0)
        self.conf_canvas.pack(anchor="w")
        self.conf_lbl = tk.Label(info, text="-", bg=BG_CARD, fg=ACCENT, font=self.f_card)
        self.conf_lbl.pack(anchor="w", pady=(4, 0))

        self._draw_badge(None)
        self._draw_conf(0.0)

    def _draw_badge(self, level_num, custom_color=None, custom_text=None):
        c = self.badge
        c.delete("all")
        cx, cy, r = 84, 84, 70
        
        # Se receber uma cor customizada (caso do Metal Limpo)
        if custom_color is not None:
            col = custom_color
            texto = custom_text
        # Caso padrão da IA
        else:
            col = "#33425E" if level_num is None else severity_color(level_num)
            texto = "?"
            if level_num == 0: texto = "OK\nAceita"
            elif level_num == 1: texto = "X\nFalhou"
            
        ring = lerp(col, "#FFFFFF", 0.18)
        
        c.create_oval(cx-r-5, cy-r-5, cx+r+5, cy+r+5, fill=ring, outline="")
        c.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline="")
        
        c.create_text(cx, cy, text=texto, fill="white", font=self.f_med, justify="center")

    def _draw_conf(self, frac):
        c = self.conf_canvas
        c.delete("all")
        w, h = 240, 16
        round_rect(c, 0, 2, w, h-2, (h-4)/2, fill=TRACK, outline="")
        fw = max(6, int(w * max(0.0, min(1.0, frac))))
        round_rect(c, 0, 2, fw, h-2, (h-4)/2, fill=TEAL, outline="")

    def _build_table(self, body, row_data):
        for w in body.winfo_children(): w.destroy()
        
        hd = tk.Frame(body, bg=BG_HEAD)
        hd.pack(fill=tk.X)
        for c, mn, w in [(0, 22, 0), (1, 150, 1), (2, 70, 0), (3, self.BAR_W+6, 0)]:
            hd.grid_columnconfigure(c, minsize=mn, weight=w)
            
        tk.Label(hd, text="PARÂMETROS", bg=BG_HEAD, fg=TEXT, font=self.f_small).grid(row=0, column=1, sticky="w", padx=6, pady=7)
        tk.Label(hd, text="VALOR", bg=BG_HEAD, fg=TEXT, font=self.f_small).grid(row=0, column=2, sticky="e", padx=6)
        tk.Label(hd, text="PROPORÇÃO", bg=BG_HEAD, fg=TEXT, font=self.f_small).grid(row=0, column=3, sticky="e", padx=(6, 8))

        cols = self.features.FEATURE_COLUMNS
        for i, key in enumerate(cols):
            bg = BG_ROW if i % 2 == 0 else BG_ROW_ALT
            rf = tk.Frame(body, bg=bg)
            rf.pack(fill=tk.X)
            for c, mn, w in [(0, 22, 0), (1, 150, 1), (2, 70, 0), (3, self.BAR_W+6, 0)]:
                rf.grid_columnconfigure(c, minsize=mn, weight=w)
            sw = SWATCH[key]
            cv = tk.Canvas(rf, width=12, height=12, bg=bg, highlightthickness=0)
            cv.grid(row=0, column=0, padx=(8, 0), pady=6)
            round_rect(cv, 0, 0, 12, 12, 3, fill=sw, outline="")
            tk.Label(rf, text=LABEL_PT[key], bg=bg, fg=TEXT, font=self.f_text).grid(row=0, column=1, sticky="w", padx=6)
            
            val = 0.0 if row_data is None else row_data[key]
            
            # ==========================================
            # CORREÇÃO: TRATAMENTO DO MATIZ (H) VS PORCENTAGENS
            # ==========================================
            if key in ["h_mean", "h_std"]:
                texto_valor = f"{val:5.1f}"    # Remove o símbolo de %
                razao_barra = val / 179.0      # O limite máximo do Hue no OpenCV é 179
            else:
                texto_valor = f"{val:5.1f} %"  # Mantém a porcentagem normal
                razao_barra = val / 100.0      # Limite normal é 100%

            tk.Label(rf, text=texto_valor, bg=bg, fg=TEXT, font=self.f_text).grid(row=0, column=2, sticky="e", padx=6)
            
            bar = tk.Canvas(rf, width=self.BAR_W, height=self.BAR_H, bg=bg, highlightthickness=0)
            bar.grid(row=0, column=3, sticky="e", padx=(6, 8))
            round_rect(bar, 0, 1, self.BAR_W, self.BAR_H-1, (self.BAR_H-2)/2, fill=TRACK, outline="")
            
            # Calcula a largura da barra usando a razão definida acima (travada no máximo de 1.0)
            fw = max(3, int(self.BAR_W * min(razao_barra, 1.0)))
            round_rect(bar, 0, 1, fw, self.BAR_H-1, (self.BAR_H-2)/2, fill=sw, outline="")

    def _build_charts(self, body):
        self.fig = Figure(figsize=(5.6, 3.0), dpi=100, facecolor=BG_CARD)
        self.ax_prob = self.fig.add_subplot(1, 1, 1) 
        self._style_axes()
        self.fig.tight_layout(pad=1.4)
        
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_axes(self):
        self.ax_prob.set_facecolor(BG_CARD)
        self.ax_prob.tick_params(colors=TEXT_MUT, labelsize=10)
        for s in self.ax_prob.spines.values():
            s.set_color(BORDER)
        self.ax_prob.set_title("Probabilidade por Macro-Classe (%)", color=TEXT, fontsize=11)

    # ---------------- NAVEGAÇÃO DE SOLDAS ----------------
    def prev_weld(self):
        if self.solda_atual_idx > 0:
            self.solda_atual_idx -= 1
            self.atualizar_tela_solda()

    def next_weld(self):
        if self.solda_atual_idx < len(self.resultados_atuais) - 1:
            self.solda_atual_idx += 1
            self.atualizar_tela_solda()

    def atualizar_tela_solda(self):
        """Atualiza a Tabela, o Gráfico e o Selo de acordo com a solda selecionada e o modelo (MLP/RF)."""
        if not self.resultados_atuais: return
        
        res = self.resultados_atuais[self.solda_atual_idx]
        
        # Identifica qual modelo o usuário selecionou no RadioButton
        escolha = self.model_choice.get()
        dados_ia = res['mlp'] if escolha == "MLP" else res['rf']
        
        # 1. Atualiza a Avaliação de acordo com o modelo selecionado
        self._draw_badge(dados_ia['classe_num'])
        self.level_lbl.configure(text=LABELS_MLP[dados_ia['classe_num']])
        self._draw_conf(dados_ia['conf_val'])
        self.conf_lbl.configure(text=f"{dados_ia['conf_val']*100:.1f}%")
        self._update_charts(dados_ia['probabilidades'])
        
        # 2. Atualiza a Tabela de Cores
        self._build_table(self.tab_body, res['row'])

        # 3. Atualiza os Botões de Navegação
        total = len(self.resultados_atuais)
        if total > 1:
            self.nav_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10), before=self.canvas_img)
            self.lbl_nav.configure(text=f"Avaliando Solda #{res['id']} de {total}")
            
            # Desabilita o botão 'Anterior' se estiver na primeira
            self.btn_prev.configure(state=tk.NORMAL if self.solda_atual_idx > 0 else tk.DISABLED)
            # Desabilita o botão 'Próxima' se estiver na última
            self.btn_next.configure(state=tk.NORMAL if self.solda_atual_idx < total - 1 else tk.DISABLED)
        else:
            self.nav_frame.pack_forget()

    # ---------------- AÇÕES E INFERÊNCIA ----------------
    def on_load(self):
        path = filedialog.askopenfilename(
            title="Escolha uma imagem de solda",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp"), ("Todos", "*.*")])
        if not path: return
        
        img_original = cv2.imread(path)
        if img_original is None:
            messagebox.showerror("Erro", "Não foi possivel ler a imagem.")
            return

        # 1. YOLO tenta detectar as soldas
        img_anotada, recortes = self.yolo.detectar_e_recortar(img_original, padding_ratio=0.05)
        
        # ==========================================
        # FALLBACK: METAL BASE LIMPO (SEM SOLDA)
        # ==========================================
        if not recortes:
            # Avisos visuais na foto
            cv2.putText(img_anotada, "Nao foi detectado", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_anotada, "cordoes de solda", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            self._show_image(img_anotada)
            
            # Desativa navegação de múltiplas soldas
            self.resultados_atuais = []
            self.nav_frame.pack_forget()
            
            # Interface: Bolinha Azul e Textos
            self._draw_badge(level_num=None, custom_color="#39A0F5", custom_text="Metal\nLimpo")
            self.level_lbl.configure(text="Sem Cordão (Metal Base)")
            self.conf_lbl.configure(text="-")
            self._draw_conf(0.0)
            
            # Interface: Zera a Tabela de Features
            self._build_table(self.tab_body, None)
            
            # Interface: Limpa o gráfico do matplotlib
            self.ax_prob.clear()
            self._style_axes()
            self.fig_canvas.draw()
            
            # Interrompe o processamento (A MLP e o Random Forest não serão acionados)
            return
            
        self._show_image(img_anotada)

        # Limpa o estado anterior
        self.resultados_atuais = []
        
        # 2. Processa cada recorte (ou a imagem inteira) individualmente no pipeline
        for recorte in recortes:
            img_crop = recorte['imagem']
            enhanced = self.enhancer.enhance(img_crop)
            row = self.features.extract(enhanced)
            
            # --- 2A. AVALIAÇÃO DA REDE NEURAL (MLP) ---
            classe_mlp, conf_mlp = 0, 0.0
            probs_mlp = [0.0, 0.0]
            
            if self.modelo_mlp is not None and self.scaler is not None:
                # 1. Garante que vai extrair apenas as 6 primeiras features (como no treino)
                lista_valores = [row[col] for col in self.features.FEATURE_COLUMNS[:6]]
                # 2. Aplica a padronização matemática
                features_normalizadas = self.scaler.transform([lista_valores])
                # 3. Transforma em Tensor
                tensor_features = torch.tensor(features_normalizadas, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    saida = self.modelo_mlp(tensor_features)
                    prob_mlp_tensor = torch.nn.functional.softmax(saida, dim=1)[0]
                    # Probabilidade bruta da solda ser Inaceitável
                    probabilidade_ruim = prob_mlp_tensor[1].item()
                    LIMITE_CORTE = 0.0001
                    if probabilidade_ruim >= LIMITE_CORTE:
                        classe_mlp = 1
                        conf_mlp = probabilidade_ruim # A interface mostra a certeza do erro
                    else:
                        classe_mlp = 0
                        conf_mlp = prob_mlp_tensor[0].item() # A interface mostra a certeza do acerto
                        
                probs_mlp = prob_mlp_tensor.tolist()

            # --- 2B. AVALIAÇÃO DO RANDOM FOREST ---
            classe_rf, conf_rf = 0, 0.0
            probs_rf = [0.0, 0.0]
            if self.modelo_rf is not None:
                probs_rf_arr = self.modelo_rf.predict_proba(row)[0]
                
                # TRADUTOR DE CLASSES BINÁRIO
                if len(probs_rf_arr) > 2:
                    macro_probs = [0.0, 0.0]
                    for idx, cls_label in enumerate(self.modelo_rf.classes_):
                        try:
                            cls_val = int(cls_label)
                            # Menor ou igual a 100 é Aceitável, o resto é Inaceitável
                            if cls_val <= 100:
                                macro_probs[0] += probs_rf_arr[idx]
                            else:
                                macro_probs[1] += probs_rf_arr[idx]
                        except ValueError:
                            pass
                    probs_rf = macro_probs
                else:
                    probs_rf = probs_rf_arr.tolist()
                    
                classe_rf = int(np.argmax(probs_rf))
                conf_rf = float(probs_rf[classe_rf])
                
            # Salva o resultado de ambos os modelos na memória da Interface
            self.resultados_atuais.append({
                'id': recorte['id'],
                'row': row,
                'mlp': {
                    'classe_num': classe_mlp,
                    'conf_val': conf_mlp,
                    'probabilidades': probs_mlp
                },
                'rf': {
                    'classe_num': classe_rf,
                    'conf_val': conf_rf,
                    'probabilidades': probs_rf
                }
            })

        # 3. Renderiza os dados da primeira solda da lista
        self.solda_atual_idx = 0
        self.atualizar_tela_solda()

    def _show_image(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((360, 300))
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas_img.configure(image=self._photo, text="")

    def _update_charts(self, probabilidades):
        self.ax_prob.clear()
        
        # Converte para porcentagem de forma segura (funciona para Tensor e Float do RF)
        probs = [float(p) * 100 for p in probabilidades] 
        colors = [ "#22C55E", "#EF4444"]
        
        self.ax_prob.barh(LABELS_MLP, probs, color=colors)
        
        for y, v in enumerate(probs):
            if v > 1:
                self.ax_prob.text(min(v+2, 99), y, f"{v:.1f}%", va="center", color=TEXT_MUT, fontsize=9)
                
        self.ax_prob.set_xlim(0, 110)
        self._style_axes()
        self.fig.tight_layout(pad=1.4)
        self.fig_canvas.draw()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    GUI().run()