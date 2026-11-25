"""Auditoría independiente y agregada de la revisión general de la tesis.

No entrena modelos ni modifica datos o resultados históricos. Usa solo numpy y
pandas para contrastar las métricas y documentar límites del diseño ejecutado.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DESTINO = Path(__file__).resolve().parent
RAIZ = DESTINO.parents[2]
ARTIFACTOS = RAIZ / "Proyecto" / "artifacts"


def fisher_bilateral(a: int, b: int, c: int, d: int) -> float:
    """Suma hipergeométrica bilateral con márgenes fijos para una tabla 2x2."""
    r1, r2, col1 = a + b, c + d, a + c
    den = math.comb(r1 + r2, col1)

    def prob(x: int) -> float:
        return math.comb(r1, x) * math.comb(r2, col1 - x) / den

    observado = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, col1 - r2), min(r1, col1) + 1)
                        if prob(x) <= observado * (1.0 + 1e-12)))


def normalizar(valor: object) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(valor).strip())
                   if unicodedata.category(c) != "Mn")


def auditar() -> dict:
    archivo = ARTIFACTOS / "ensamble" / "probabilidades_test_4329.npz"
    with np.load(archivo, allow_pickle=True) as z:
        y, clases = z["y_true"], z["clases"]
        p, q = z["proba_baseline"].astype(float), z["proba_mrbert"].astype(float)
        indices = z["original_index"]
    assert len(y) == len(set(indices)) == 4329 and len(clases) == 42

    def metricas(proba: np.ndarray) -> dict:
        orden = np.argsort(-proba, axis=1)
        pred = clases[orden[:, 0]]
        f1, recalls = [], []
        for clase in clases:
            tp = int(((pred == clase) & (y == clase)).sum())
            reales, estimados = int((y == clase).sum()), int((pred == clase).sum())
            f1.append(2 * tp / (reales + estimados) if reales + estimados else 0)
            recalls.append(tp / reales if reales else 0)
        return {"f1_macro": float(np.mean(f1)), "exactitud": float((pred == y).mean()),
                "exactitud_balanceada": float(np.mean(recalls)),
                "top3": float((clases[orden[:, :3]] == y[:, None]).any(axis=1).mean())}

    lineal, mrbert = metricas(p), metricas(q)
    assert abs(lineal["f1_macro"] - 0.5455370598338695) < 1e-12
    assert abs(mrbert["f1_macro"] - 0.5267705143766618) < 1e-12
    a, b = clases[p.argmax(1)] == y, clases[q.argmax(1)] == y
    discordancias = {"solo_lineal": int((a & ~b).sum()), "solo_mrbert": int((~a & b).sum()),
                     "fallos_conjuntos": int((~a & ~b).sum()), "oraculo_top1": float((a | b).mean())}
    assert [discordancias[k] for k in ["solo_lineal", "solo_mrbert", "fallos_conjuntos"]] == [320, 342, 931]

    temperatura = json.loads((ARTIFACTOS / "calibracion" / "calibracion_metadata.json").read_text(encoding="utf-8"))["temperatura"]["T"]
    cal = p ** (1 / temperatura)
    cal /= cal.sum(axis=1, keepdims=True)
    cobertura = {}
    for nombre, proba in [("sin_calibrar", p), ("calibrada", cal)]:
        orden = np.argsort(-proba, axis=1)
        top = proba[np.arange(len(y)), orden[:, 0]]
        margen = top - proba[np.arange(len(y)), orden[:, 1]]
        acierto = clases[orden[:, 0]] == y
        grupos = {"umbral_0_70": top >= 0.70, "umbrales_0_35_y_0_08": (top >= .35) & (margen >= .08)}
        cobertura[nombre] = {}
        for regla, mascara in grupos.items():
            cobertura[nombre][regla] = {
                "n": int(mascara.sum()), "cobertura": float(mascara.mean()),
                "top1": float(acierto[mascara].mean()),
                "top3": float((clases[orden[:, :3]] == y[:, None]).any(axis=1)[mascara].mean()),
                "top1_excluidos": float(acierto[~mascara].mean()),
            }
    assert cobertura["calibrada"]["umbral_0_70"]["n"] == 2404
    assert np.array_equal(p.argmax(1), cal.argmax(1))

    feedback = ARTIFACTOS / "hitl" / "feedback.csv"
    f = pd.read_csv(feedback)
    A = f[f.sampling_mode.str.startswith("Auditoria")]
    B = f[f.condicion.str.startswith("Asistida") & f.sampling_mode.str.startswith("Muestra")]
    C = f[f.condicion.str.startswith("Ciega")]
    solapamientos = {"A_B": len(set(A.row_id) & set(B.row_id)),
                     "A_C": len(set(A.row_id) & set(C.row_id)),
                     "B_C": len(set(B.row_id) & set(C.row_id))}
    C_nueva = C[~C.row_id.isin(set(A.row_id) | set(B.row_id))]

    def resumen(grupo: pd.DataFrame) -> dict:
        dec = grupo[grupo.decision != "No aplica (informacion insuficiente)"]
        desacuerdos = (dec.etiqueta_validada.astype(str).str.strip()
                       != dec.dependencia_historica_norm.astype(str).str.strip())
        return {"n": len(grupo), "decidibles": len(dec), "desacuerdos_historico": int(desacuerdos.sum()),
                "top1_decidibles": int(dec.accepted_top1.sum()), "top3_decidibles": int(dec.accepted_top3.sum()),
                "indecidibles": len(grupo) - len(dec)}

    b_res, c_res, c_nueva = resumen(B), resumen(C), resumen(C_nueva)
    p_original = fisher_bilateral(0, b_res["decidibles"], c_res["desacuerdos_historico"],
                                 c_res["decidibles"] - c_res["desacuerdos_historico"])
    p_sensible = fisher_bilateral(0, b_res["decidibles"], c_nueva["desacuerdos_historico"],
                                 c_nueva["decidibles"] - c_nueva["desacuerdos_historico"])
    assert len(f) == 250 and f.row_id.nunique() == 247 and solapamientos == {"A_B": 0, "A_C": 2, "B_C": 1}
    assert abs(p_original - 9.44713e-11) < 1e-15

    fuente = RAIZ / "data" / "Registro_PQRS_Mindeportes_20260328.csv"
    w = pd.read_csv(fuente, low_memory=False).dropna(subset=["TIPO", "ASUNTO", "DEPENDENCIA DESTINO"]).copy()
    claves, etiqueta = ["TIPO", "ASUNTO"], "DEPENDENCIA DESTINO"
    for col in claves + [etiqueta]:
        w[col] = w[col].apply(normalizar)
    w["anio"] = pd.to_numeric(w.RADICADO.astype(str).str.extract(r"^(\d{4})")[0], errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fecha = pd.to_datetime(w["FECHA RADICADO"], errors="coerce").dt.year
    w["anio"] = w.anio.where(w.anio.between(2000, 2026)).fillna(fecha.where(fecha.between(2000, 2026)))

    def mayorias(datos: pd.DataFrame) -> pd.DataFrame:
        votos = datos.groupby(claves + [etiqueta]).size().reset_index(name="votos")
        votos["prioridad"] = votos[etiqueta].map(datos[etiqueta].value_counts())
        return (votos.sort_values(["votos", "prioridad", etiqueta], ascending=[False, False, True])
                .drop_duplicates(claves).set_index(claves)[[etiqueta]])

    g = w.groupby(claves)
    marco = mayorias(w).join(g.anio.first()).sort_index().reset_index()
    marco = marco[(marco.ASUNTO.str.len() >= 10) & (marco.TIPO.str.len() >= 3)]
    frecuencias = marco[etiqueta].value_counts()
    marco = marco[marco[etiqueta].isin(frecuencias[frecuencias >= 20].index)].set_index(claves)
    assert len(marco) == 43285
    mixto = g.anio.agg(lambda valores: {2020, 2021}.issubset(set(valores.dropna())))
    contradictorio = g[etiqueta].nunique() > 1
    mayorias_2020 = mayorias(w[w.anio == 2020]).rename(columns={etiqueta: "etiqueta_2020"})
    contraste = marco[marco.anio == 2020].join(mayorias_2020)
    cambio = contraste[etiqueta] != contraste.etiqueta_2020
    temporal = {
        "alcance": "Agrupación TIPO+ASUNTO; etiquetas normalizadas como canonical_split.py; solo cifras agregadas.",
        "grupos_antes_filtros": len(mixto), "grupos_ambos_anios_antes_filtros": int(mixto.sum()),
        "grupos_ambos_anios_etiquetas_distintas_antes_filtros": int((mixto & contradictorio).sum()),
        "registros_consolidados": len(marco),
        "asignados_a_2020": int((marco.anio == 2020).sum()),
        "asignados_a_2021": int((marco.anio == 2021).sum()),
        "grupos_ambos_anios_retenidos": int(mixto.reindex(marco.index).sum()),
        "etiquetas_2020_distintas_al_consolidar_solo_2020": int(cambio.sum()),
        "interpretacion": "Consolidar todo el corpus antes de separar años permite que registros de 2021 influyan en la etiqueta de grupos asignados a 2020. No se reentrenaron modelos ni se estimó el efecto sobre sus métricas.",
    }
    rutas = [archivo, feedback, fuente, RAIZ / "Proyecto" / "canonical_split.py", RAIZ / "Proyecto" / "app_gradio.py"]
    return {
        "fecha": "2026-11-24", "modelos_reentrenados": False,
        "metricas_recalculadas": {"lineal": lineal, "mrbert_reinferido": mrbert},
        "discordancias": discordancias, "cobertura": cobertura,
        "piloto": {"validaciones": len(f), "casos_unicos": int(f.row_id.nunique()), "solapamientos": solapamientos,
                   "B": b_res, "C": c_res, "C_sin_casos_compartidos": c_nueva,
                   "fisher_original": p_original, "fisher_sin_casos_compartidos": p_sensible,
                   "limite": "Análisis posterior; excluir casos compartidos no controla el orden de exposición ni la experiencia del único revisor."},
        "consolidacion_temporal": temporal,
        "correcciones_documentales": [
            "Se acota el oráculo a elegir entre predicciones top-1, sin presentarlo como límite de cualquier ensamble.",
            "Se distinguen discrepancias entre duplicados, error empírico restringido y error irreducible poblacional.",
            "Se separa la calibración retrospectiva de las probabilidades sin calibrar usadas por el prototipo.",
            "Se declara la exploración de umbrales sobre prueba y la necesidad de una nueva evaluación con la regla fijada.",
            "Se documentan el solapamiento del piloto y la consolidación anterior al corte temporal.",
            "Se retiran inferencias de equivalencia, causalidad y ausencia de aporte que los contrastes no establecen.",
            "Se ajustan las declaraciones de derechos y control humano al reglamento institucional y a la evidencia obtenida.",
            "Se corrigen un comando LaTeX dañado y una frase duplicada; se acortan títulos del índice y se revisa la colocación de figuras.",
        ],
        "fuente_normativa": {
            "titulo": "Universidad de La Salle, Acuerdo 025 de 2021, artículos 18 a 20",
            "url": "https://lasalle.edu.co/es/normatividad/propiedad-intelectual",
            "pdf": "https://cmspreprod.lasalle.edu.co/sites/default/files/2024-08/7_1_ACS-No-025-de-2021.pdf",
            "verificacion": "Texto y páginas 8 y 9 inspeccionados; no se presume una cesión general de derechos.",
        },
        "fuentes_sha256": {str(p.relative_to(RAIZ)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in rutas},
    }


if __name__ == "__main__":
    resultado = auditar()
    destino = DESTINO / "REVISION_GENERAL_20261124.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: resultado[k] for k in ["metricas_recalculadas", "piloto", "consolidacion_temporal"]}, ensure_ascii=False, indent=2))
    print(f"Verificación guardada: {destino}")
