import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from backpropagation.features_LE import Features


class QualityClassification:
    def __init__(self, n_estimators: int = 300, max_depth=None, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
        self.feature_names = list(Features.FEATURE_COLUMNS)
        self.classes_ = None
        self.class_profiles_ = None  

    # ---------- utilitarios internos ----------
    def _as_matrix(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].to_numpy(dtype=float)
        if isinstance(X, dict):
            return np.array([[float(X[f]) for f in self.feature_names]])
        arr = np.asarray(X, dtype=float)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    # ---------- treino ----------
    def fit(self, X, y):
        Xm = self._as_matrix(X)
        y = np.asarray(y)
        self.model.fit(Xm, y)
        self.classes_ = self.model.classes_
        # perfil medio de cada classe (para texto explicativo)
        df = pd.DataFrame(Xm, columns=self.feature_names)
        df["_y"] = y
        self.class_profiles_ = df.groupby("_y").mean()
        return self

    # ---------- previsao ----------
    def predict(self, X):
        return self.model.predict(self._as_matrix(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._as_matrix(X))

    # ---------- contribuicoes locais (tree interpreter) ----------
    def _tree_contributions(self, tree, x):
        """Contribuicao de cada feature p/ a proba de cada classe em UMA arvore."""
        t = tree.tree_
        value = t.value[:, 0, :]
        proba = value / value.sum(axis=1, keepdims=True)  # [n_nos, n_classes]
        n_classes = proba.shape[1]
        contribs = np.zeros((len(self.feature_names), n_classes))

        node = 0
        while t.children_left[node] != -1:  # enquanto nao for folha
            f = t.feature[node]
            if x[f] <= t.threshold[node]:
                child = t.children_left[node]
            else:
                child = t.children_right[node]
            contribs[f] += proba[child] - proba[node]
            node = child
        return proba[0], contribs  # vies (raiz), contribuicoes

    def _contributions(self, x):
        """Media das contribuicoes sobre todas as arvores do Random Forest."""
        n_classes = len(self.classes_)
        bias = np.zeros(n_classes)
        contribs = np.zeros((len(self.feature_names), n_classes))
        for est in self.model.estimators_:
            b, c = self._tree_contributions(est, x)
            bias += b
            contribs += c
        n = len(self.model.estimators_)
        return bias / n, contribs / n

    # ---------- explicacao ----------
    def explain(self, x_row) -> dict:
        """Devolve um dicionario estruturado com a explicacao da previsao."""
        x = self._as_matrix(x_row)[0]
        proba = self.model.predict_proba(x.reshape(1, -1))[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.classes_[pred_idx]

        _, contribs = self._contributions(x)
        contrib_pred = contribs[:, pred_idx]  # contribuicao p/ a classe prevista

        feat_contrib = sorted(
            [
                {"feature": f, "valor": float(x[i]),
                 "contribuicao": float(contrib_pred[i])}
                for i, f in enumerate(self.feature_names)
            ],
            key=lambda d: abs(d["contribuicao"]),
            reverse=True,
        )

        ranking = sorted(
            [{"nivel": int(self.classes_[i]) if str(self.classes_[i]).isdigit()
              else self.classes_[i], "prob": float(p)}
             for i, p in enumerate(proba)],
            key=lambda d: d["prob"], reverse=True,
        )

        global_imp = sorted(
            [{"feature": f, "importancia": float(imp)}
             for f, imp in zip(self.feature_names, self.model.feature_importances_)],
            key=lambda d: d["importancia"], reverse=True,
        )

        return {
            "nivel_previsto": pred_label,
            "confianca": float(proba[pred_idx]),
            "ranking_classes": ranking,
            "contribuicoes_locais": feat_contrib,
            "importancia_global": global_imp,
        }

    def explain_text(self, x_row) -> str:
        """Versao em texto (PT-BR) da explicacao."""
        e = self.explain(x_row)
        linhas = []
        linhas.append(f"NIVEL DE SOLDA PREVISTO: {e['nivel_previsto']}")
        linhas.append(f"Confianca: {e['confianca']*100:.1f}%")
        linhas.append("")
        linhas.append("Probabilidade por nivel (top 3):")
        for r in e["ranking_classes"][:3]:
            linhas.append(f"   nivel {r['nivel']}: {r['prob']*100:.1f}%")
        linhas.append("")
        linhas.append("Por que este resultado (caracteristicas mais influentes):")
        for c in e["contribuicoes_locais"][:4]:
            sinal = "favorece" if c["contribuicao"] >= 0 else "reduz"
            linhas.append(
                f"   {c['feature']} = {c['valor']:.1f}  "
                f"({sinal}, contrib {c['contribuicao']:+.3f})"
            )
        return "\n".join(linhas)

    # ---------- avaliacao ----------
    def evaluate(self, X, y):
        """
        Acuracia no proprio conjunto (com aviso). Com apenas 1 imagem real por
        nivel, nao ha como validar de forma realista: as variantes da mesma
        imagem aparecem no treino. Use mais imagens distintas por nivel para
        uma validacao confiavel.
        """
        Xm = self._as_matrix(X)
        acc = self.model.score(Xm, np.asarray(y))
        return {"acuracia_treino": float(acc),
                "aviso": "otimista: 1 imagem-base por nivel; valide com mais dados reais"}

    # ---------- persistencia ----------
    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "QualityClassification":
        return joblib.load(path)
