"""
Acierto del modelo lineal en el test canonico segun la longitud del asunto y el tipo.

Responde una pregunta que el equipo se hizo el 03/11/2026, despues del cierre de la
tesis: ¿hay un numero minimo de caracteres para acertar, o depende del tipo de
solicitud? Es un analisis **descriptivo, posterior y no prerregistrado**: no entrena
nada, usa las probabilidades ya guardadas del test canonico y los textos del marco
consolidado. El "acierto" es coincidencia con la etiqueta historica, que es ruidosa.
La longitud se confunde con el tipo y con la dependencia, asi que no es causal.

La confianza calibrada usa la temperatura ajustada en validacion
(artifacts/calibracion/calibracion_metadata.json). El script verifica que el top-1, el
top-3 y la cobertura globales coincidan con las cifras canonicas.

Salidas:
    artifacts/analisis_longitud_tipo/acierto_por_longitud.csv
    artifacts/analisis_longitud_tipo/acierto_por_tipo.csv
    artifacts/analisis_longitud_tipo/analisis_longitud_tipo_metadata.json

Uso (basta el Python global, con pandas y numpy):
    python Proyecto/Scripts/analisis_longitud_y_tipo.py
"""
from __future__ import annotations

import json
import math
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "Proyecto"
sys.path.insert(0, str(PROJECT_DIR))

try:
    import sklearn.model_selection  # noqa: F401
except ImportError:
    # canonical_split importa train_test_split en el nivel superior, pero aqui solo
    # se usa _consolidar, que no lo necesita. Asi corre con el Python global.
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    _falso = types.ModuleType("sklearn.model_selection")
    _falso.train_test_split = None
    sys.modules["sklearn.model_selection"] = _falso

import canonical_split as cs  # noqa: E402

NPZ_TEST = PROJECT_DIR / "artifacts" / "ensamble" / "probabilidades_test_4329.npz"
CALIBRACION = PROJECT_DIR / "artifacts" / "calibracion" / "calibracion_metadata.json"
OUT_DIR = PROJECT_DIR / "artifacts" / "analisis_longitud_tipo"

UMBRAL = 0.70
CORTES = [10, 15, 20, 30, 50, 100, 200, 100_000]
TRAMOS = ["10-14", "15-19", "20-29", "30-49", "50-99", "100-199", "200 o más"]
MIN_TIPO = 20

# Cifras canonicas que el analisis debe reproducir (ESTADO, seccion 4).
CANONICAS = {"top1": 0.7059, "top3": 0.8508, "cobertura": 0.5553}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - m, c + m


def marco_test() -> pd.DataFrame:
    z = np.load(NPZ_TEST, allow_pickle=True)
    y_true, clases = z["y_true"].astype(str), z["clases"].astype(str)
    work = cs._consolidar(cs.read_csv_with_fallback(cs.DATA_PATH))
    test = work.loc[z["original_index"], ["TIPO", "ASUNTO", cs.LABEL_COLUMN]].copy()
    if not (test[cs.LABEL_COLUMN].to_numpy() == y_true).all():
        raise RuntimeError("Las etiquetas del marco no coinciden con las del test canonico")

    proba = z["proba_baseline"].astype(float)
    orden = np.argsort(-proba, axis=1)
    posicion = (clases[orden] == y_true[:, None]).argmax(axis=1)
    test["top1"] = posicion == 0
    test["top3"] = posicion < 3
    test["top1_mrbert"] = clases[z["proba_mrbert"].argmax(axis=1)] == y_true

    temperatura = json.loads(CALIBRACION.read_text(encoding="utf-8"))["temperatura"]["T"]
    logits = np.log(np.clip(proba, 1e-12, 1)) / temperatura
    calibrada = np.exp(logits - logits.max(axis=1, keepdims=True))
    calibrada /= calibrada.sum(axis=1, keepdims=True)
    test["confianza"] = calibrada.max(axis=1)
    test["destacada"] = test["confianza"] >= UMBRAL

    test["caracteres"] = test["ASUNTO"].str.len()
    test["palabras"] = test["ASUNTO"].str.split().str.len()
    test["tramo"] = pd.cut(test["caracteres"], bins=CORTES, right=False, labels=TRAMOS)
    test.attrs["temperatura"] = temperatura
    return test


def resumen(grupo: pd.DataFrame) -> dict:
    n, k = len(grupo), int(grupo["top1"].sum())
    lo, hi = wilson(k, n)
    destacados = grupo[grupo["destacada"]]
    return {
        "n": n,
        "top1": round(k / n, 4),
        "top1_ic95": [round(lo, 4), round(hi, 4)],
        "top3": round(float(grupo["top3"].mean()), 4),
        "top1_mrbert": round(float(grupo["top1_mrbert"].mean()), 4),
        "destacada": round(float(grupo["destacada"].mean()), 4),
        "top1_en_destacadas": round(float(destacados["top1"].mean()), 4) if len(destacados) else None,
        "mediana_caracteres": float(grupo["caracteres"].median()),
    }


def main() -> None:
    test = marco_test()
    globales = {"top1": test["top1"].mean(), "top3": test["top3"].mean(),
                "cobertura": test["destacada"].mean()}
    for clave, valor in CANONICAS.items():
        if abs(globales[clave] - valor) > 5e-5:
            raise RuntimeError(f"{clave} = {globales[clave]:.4f} no reproduce la cifra canonica {valor}")

    por_longitud = [{"tramo": str(t), **resumen(g)} for t, g in test.groupby("tramo", observed=True)]
    por_tipo = [{"tipo": t, **resumen(g)} for t, g in test.groupby("TIPO")]
    por_tipo.sort(key=lambda f: (-f["n"], f["tipo"]))

    def agrupado(mascara: pd.Series) -> dict:
        return resumen(test[mascara])

    oficio = test[test["TIPO"] == "Oficio"]
    oficio_tramos = pd.cut(oficio["caracteres"], bins=[10, 20, 50, 100, 100_000], right=False,
                           labels=["10-19", "20-49", "50-99", "100 o más"])
    tipos_altos = [f for f in por_tipo if f["n"] >= MIN_TIPO and f["top1"] >= 0.95]

    metadata = {
        "fecha": date.today().isoformat(),
        "naturaleza": "Analisis descriptivo, posterior y no prerregistrado; no esta en la tesis.",
        "acierto": "Coincidencia con la etiqueta historica (ruidosa), no exactitud real.",
        "advertencia": "La longitud se confunde con el tipo y con la dependencia: no es un efecto causal.",
        "fuentes": [str(NPZ_TEST.relative_to(REPO_ROOT)), str(CALIBRACION.relative_to(REPO_ROOT)),
                    str(cs.DATA_PATH.relative_to(REPO_ROOT))],
        "temperatura": test.attrs["temperatura"],
        "umbral_confianza": UMBRAL,
        "control_cifras_canonicas": {k: round(float(v), 4) for k, v in globales.items()},
        "longitud_global": {
            "minima": int(test["caracteres"].min()),
            "p10": float(test["caracteres"].quantile(0.1)),
            "mediana": float(test["caracteres"].median()),
            "media": round(float(test["caracteres"].mean()), 1),
            "p90": float(test["caracteres"].quantile(0.9)),
            "filtro_de_datos": "El marco consolidado exige al menos 10 caracteres en el asunto.",
        },
        "aciertos_vs_fallos": {
            estado: {"n": int(len(g)), "mediana_caracteres": float(g["caracteres"].median()),
                     "media_caracteres": round(float(g["caracteres"].mean()), 1),
                     "mediana_palabras": float(g["palabras"].median())}
            for estado, g in (("aciertos_top1", test[test["top1"]]), ("fallos_top1", test[~test["top1"]]))
        },
        "por_longitud": por_longitud,
        "oficio_por_longitud": [{"tramo": str(t), **resumen(g)}
                                for t, g in oficio.groupby(oficio_tramos, observed=True)],
        "confianza": {
            "menos_de_20_caracteres": agrupado(test["caracteres"] < 20),
            "de_50_a_199_caracteres": agrupado(test["caracteres"].between(50, 199)),
        },
        # Agrupaciones que citan la tesis (anexo B y recomendaciones) y el articulo.
        "agrupados": {
            "menos_de_30_caracteres": agrupado(test["caracteres"] < 30),
            "50_o_mas_caracteres": agrupado(test["caracteres"] >= 50),
        },
        "tipos_con_top1_95_o_mas": [f["tipo"] for f in tipos_altos],
        "tipos_minimo_n": MIN_TIPO,
        "casos_en_tipos_con_menos_de_20": int(sum(f["n"] for f in por_tipo if f["n"] < MIN_TIPO)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(por_longitud).to_csv(OUT_DIR / "acierto_por_longitud.csv", index=False)
    pd.DataFrame(por_tipo).to_csv(OUT_DIR / "acierto_por_tipo.csv", index=False)
    with open(OUT_DIR / "analisis_longitud_tipo_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Cifras canonicas reproducidas: {metadata['control_cifras_canonicas']}")
    for fila in por_longitud:
        print(f"  {fila['tramo']:>10}  n={fila['n']:>5}  top-1 {fila['top1']:.1%}  top-3 {fila['top3']:.1%}")
    print(f"Salidas en {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
