"""
Calibracion del baseline y curva de prediccion selectiva (cobertura vs precision).

Cierra dos huecos que el propio proyecto se habia marcado y no ejecuto:

1. **Calibracion.** `BITACORA_CAMBIOS_MODELOS_20260328.md` (secciones 946 y 965)
   registra como tarea de Fase 7: "las probabilidades estaran mal calibradas;
   conviene aplicar temperature scaling antes de fijar el umbral de confianza".
   Nunca se hizo. Sin esto, la "confianza" que muestra el prototipo es un numero
   sin garantia de significar lo que parece significar.

2. **Prediccion selectiva.** El prototipo trae dos umbrales fijos
   (`low_confidence = 0.35`, `max_margin = 0.08`) sin ninguna justificacion
   empirica en la documentacion. Este script produce la curva
   cobertura-precision que los justifica o los refuta, y convierte una metrica
   agregada ("accuracy 0.7059") en evidencia retrospectiva: que fraccion del
   conjunto recibiria una sugerencia por encima de cada umbral y con que
   exactitud historica, siempre sujeta a validacion humana.

La temperatura se ajusta en **validacion** y se evalua en **test**, una sola vez.
Ajustarla en test seria el mismo sesgo de seleccion que se corrigio en el
analisis del ensamble.

Salidas:
    artifacts/calibracion/calibracion_metadata.json
    artifacts/calibracion/cobertura_precision.csv
    artifacts/calibracion/fiabilidad_bins.csv

Uso:
    python3 Proyecto/Scripts/calibracion_y_cobertura.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "Proyecto"
sys.path.insert(0, str(PROJECT_DIR))

import canonical_split as cs  # noqa: E402

NPZ_TEST = PROJECT_DIR / "artifacts" / "ensamble" / "probabilidades_test_4329.npz"
JOBLIB = PROJECT_DIR / "artifacts" / "baseline_b_tfidf_lr_final.joblib"
OUT_DIR = PROJECT_DIR / "artifacts" / "calibracion"

# Umbrales que el prototipo trae fijos, sin justificacion documentada.
APP_LOW_CONFIDENCE = 0.35
APP_MAX_MARGIN = 0.08

N_BINS = 10


# --------------------------------------------------------------------------- #
# Calibracion
# --------------------------------------------------------------------------- #
def ece_y_bins(confianza, acierto, n_bins=N_BINS):
    """Expected Calibration Error con bins de ancho igual, y el detalle por bin.

    ECE = suma ponderada de |confianza media - precision observada| por bin.
    Es 0 si el modelo esta perfectamente calibrado: cuando dice 0.8, acierta el
    80 % de las veces.
    """
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    filas = []
    ece = 0.0
    n = len(confianza)
    for i in range(n_bins):
        lo, hi = bordes[i], bordes[i + 1]
        # El ultimo bin incluye el borde superior.
        m = (confianza > lo) & (confianza <= hi) if i > 0 else (confianza >= lo) & (confianza <= hi)
        if not m.any():
            filas.append(
                {"bin": f"({lo:.1f}, {hi:.1f}]", "n": 0, "confianza_media": None,
                 "precision_observada": None, "brecha": None}
            )
            continue
        conf_m = float(confianza[m].mean())
        acc_m = float(acierto[m].mean())
        peso = m.sum() / n
        ece += peso * abs(conf_m - acc_m)
        filas.append(
            {
                "bin": f"({lo:.1f}, {hi:.1f}]",
                "n": int(m.sum()),
                "confianza_media": round(conf_m, 4),
                "precision_observada": round(acc_m, 4),
                "brecha": round(acc_m - conf_m, 4),
            }
        )
    return float(ece), pd.DataFrame(filas)


def brier_multiclase(proba, y_true_idx):
    """Brier score multiclase (menor es mejor). Mide calibracion y acierto juntos."""
    n, k = proba.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y_true_idx] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def ajustar_temperatura(logits_val, y_val_idx):
    """Temperature scaling: un solo parametro T que divide los logits.

    Se ajusta minimizando la log-verosimilitud negativa en **validacion**. T > 1
    suaviza (el modelo estaba sobreconfiado); T < 1 agudiza.
    """

    def nll(log_t):
        t = np.exp(log_t)  # parametrizacion en log para forzar T > 0
        p = softmax(logits_val / t, axis=1)
        p = np.clip(p, 1e-12, 1.0)
        return -np.log(p[np.arange(len(y_val_idx)), y_val_idx]).mean()

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x)), float(res.fun)


# --------------------------------------------------------------------------- #
# Prediccion selectiva
# --------------------------------------------------------------------------- #
def curva_cobertura(proba, y_true_idx, criterio="confianza"):
    """Cobertura vs precision al abstenerse por debajo de un umbral.

    `criterio`:
      - 'confianza': umbral sobre la probabilidad maxima.
      - 'margen'   : umbral sobre (p1 - p2), que mide cuan disputada esta la
                     decision. Un caso puede tener confianza alta y margen
                     minimo si dos clases empatan arriba.
    """
    orden = np.sort(proba, axis=1)[:, ::-1]
    p1 = orden[:, 0]
    p2 = orden[:, 1]
    score = p1 if criterio == "confianza" else (p1 - p2)

    pred = proba.argmax(axis=1)
    acierto1 = (pred == y_true_idx).astype(float)
    top3 = np.argsort(proba, axis=1)[:, ::-1][:, :3]
    acierto3 = (top3 == y_true_idx[:, None]).any(axis=1).astype(float)

    filas = []
    for u in np.round(np.arange(0.0, 1.0, 0.05), 2):
        m = score >= u
        cob = m.mean()
        filas.append(
            {
                "criterio": criterio,
                "umbral": float(u),
                "cobertura": round(float(cob), 4),
                "n_cubiertos": int(m.sum()),
                "precision_top1": round(float(acierto1[m].mean()), 4) if m.any() else None,
                "precision_top3": round(float(acierto3[m].mean()), 4) if m.any() else None,
                # Precision sobre los que quedan FUERA: si es alta, el umbral
                # esta descartando casos que el modelo resolvia bien.
                "precision_top1_excluidos": round(float(acierto1[~m].mean()), 4) if (~m).any() else None,
            }
        )
    return pd.DataFrame(filas)


def umbral_para_precision(curva, objetivo):
    """Umbral minimo que alcanza una precision top-1 objetivo, y su cobertura."""
    ok = curva[curva["precision_top1"].notna() & (curva["precision_top1"] >= objetivo)]
    if ok.empty:
        return None
    fila = ok.iloc[0]  # el umbral mas bajo que ya cumple
    return {
        "objetivo_precision": objetivo,
        "umbral": float(fila["umbral"]),
        "cobertura": float(fila["cobertura"]),
        "n_cubiertos": int(fila["n_cubiertos"]),
        "precision_top1": float(fila["precision_top1"]),
    }


# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- test: probabilidades ya calculadas y verificadas -------
    z = np.load(NPZ_TEST, allow_pickle=True)
    clases = np.array([str(c) for c in z["clases"]])
    proba_test = z["proba_baseline"]
    y_true_test = np.array([str(v) for v in z["y_true"]])
    idx_de_clase = {c: i for i, c in enumerate(clases)}
    y_test_idx = np.array([idx_de_clase[c] for c in y_true_test])

    assert proba_test.shape == (4329, 42), f"forma inesperada: {proba_test.shape}"
    print(f"Test: {proba_test.shape[0]} filas, {proba_test.shape[1]} clases.")

    # ---------------- validacion: hay que inferir ---------------------------
    # La temperatura se ajusta aqui y NUNCA en test.
    import joblib

    modelo = joblib.load(JOBLIB)
    frame = cs.build_canonical_frame(verify=True)
    val = frame[frame["split"] == "validation"]
    print(f"Validacion: {len(val)} filas (para ajustar la temperatura).")

    # softmax(decision_function) en vez de predict_proba: el .joblib se serializo
    # con una version de scikit-learn cuyo predict_proba no carga en este entorno.
    dec_val = modelo.decision_function(val["text_baseline"].tolist())
    clases_modelo = np.array([str(c) for c in modelo.classes_])
    assert set(clases_modelo) == set(clases), "las clases del modelo no coinciden con el npz"

    # Alinear columnas del modelo al orden de `clases` del npz, por NOMBRE.
    reorden = [list(clases_modelo).index(c) for c in clases]
    dec_val = dec_val[:, reorden]

    y_val_idx = np.array([idx_de_clase[c] for c in val[cs.LABEL_COLUMN].astype(str)])

    # Verificacion: sin temperatura, softmax(decision_function) debe reproducir
    # el accuracy documentado en test.
    dec_test_check = None
    proba_val_sin_T = softmax(dec_val, axis=1)
    acc_val = (proba_val_sin_T.argmax(axis=1) == y_val_idx).mean()
    print(f"Accuracy en validacion (sin calibrar): {acc_val:.4f}")

    # ---------------- calibracion antes / despues ---------------------------
    conf_test = proba_test.max(axis=1)
    acierto_test = (proba_test.argmax(axis=1) == y_test_idx).astype(float)

    ece_antes, bins_antes = ece_y_bins(conf_test, acierto_test)
    brier_antes = brier_multiclase(proba_test, y_test_idx)
    conf_media = float(conf_test.mean())
    acc_test = float(acierto_test.mean())

    print()
    print("=== CALIBRACION EN TEST, SIN CALIBRAR ===")
    print(f"  confianza media    : {conf_media:.4f}")
    print(f"  accuracy observado : {acc_test:.4f}")
    print(f"  brecha             : {acc_test - conf_media:+.4f}  "
          f"({'SUBCONFIADO' if acc_test > conf_media else 'SOBRECONFIADO'})")
    print(f"  ECE                : {ece_antes:.4f}")
    print(f"  Brier              : {brier_antes:.4f}")

    T, nll_val = ajustar_temperatura(dec_val, y_val_idx)
    print()
    print(f"Temperatura ajustada en VALIDACION: T = {T:.4f}  (NLL {nll_val:.4f})")
    print(f"  T > 1 => el modelo estaba sobreconfiado y se suaviza.")
    print(f"  T < 1 => estaba subconfiado y se agudiza.")

    # Aplicar T a test. Recuperar los logits de test desde el npz no es posible
    # (guarda probabilidades), asi que se reinfiere el decision_function.
    test = frame[frame["split"] == "test"]
    dec_test = modelo.decision_function(test["text_baseline"].tolist())[:, reorden]
    y_test_idx_re = np.array([idx_de_clase[c] for c in test[cs.LABEL_COLUMN].astype(str)])

    # El npz y la reconstruccion deben coincidir en accuracy: control de que se
    # esta hablando de las mismas filas.
    acc_reinf = float((softmax(dec_test, axis=1).argmax(axis=1) == y_test_idx_re).mean())
    print()
    print(f"Control de identidad de filas: accuracy npz {acc_test:.4f} vs "
          f"reinferido {acc_reinf:.4f}  (delta {acc_reinf - acc_test:+.4f})")
    if abs(acc_reinf - acc_test) > 0.002:
        print("  AVISO: los dos caminos no coinciden; revisar el orden de filas.")

    proba_test_cal = softmax(dec_test / T, axis=1)
    conf_cal = proba_test_cal.max(axis=1)
    acierto_cal = (proba_test_cal.argmax(axis=1) == y_test_idx_re).astype(float)
    ece_despues, bins_despues = ece_y_bins(conf_cal, acierto_cal)
    brier_despues = brier_multiclase(proba_test_cal, y_test_idx_re)

    print()
    print("=== CALIBRACION EN TEST, DESPUES DE TEMPERATURE SCALING ===")
    print(f"  confianza media    : {float(conf_cal.mean()):.4f}")
    print(f"  accuracy observado : {float(acierto_cal.mean()):.4f}")
    print(f"  ECE                : {ece_despues:.4f}  "
          f"(antes {ece_antes:.4f}, {'MEJORA' if ece_despues < ece_antes else 'EMPEORA'})")
    print(f"  Brier              : {brier_despues:.4f}  (antes {brier_antes:.4f})")
    print()
    print("  Nota: temperature scaling no cambia el argmax, de modo que el")
    print("  accuracy y el top-3 son identicos. Solo cambia lo que significa el")
    print("  numero de confianza, que es exactamente lo que el prototipo usa")
    print("  para decidir a que casos pedir revision humana.")

    # ---------------- diagrama de fiabilidad -------------------------------
    print()
    print("=== DIAGRAMA DE FIABILIDAD (calibrado) ===")
    print(f"  {'bin':14s} {'n':>5s} {'conf':>7s} {'precision':>10s} {'brecha':>8s}")
    for _, r in bins_despues.iterrows():
        if r["n"] == 0:
            continue
        print(f"  {r['bin']:14s} {int(r['n']):5d} {r['confianza_media']:7.4f} "
              f"{r['precision_observada']:10.4f} {r['brecha']:+8.4f}")

    # ---------------- prediccion selectiva ---------------------------------
    # Se usa la probabilidad CALIBRADA: fijar umbrales sobre una confianza mal
    # calibrada es precisamente el error que la bitacora anticipaba.
    curva_conf = curva_cobertura(proba_test_cal, y_test_idx_re, "confianza")
    curva_marg = curva_cobertura(proba_test_cal, y_test_idx_re, "margen")
    curva = pd.concat([curva_conf, curva_marg], ignore_index=True)

    print()
    print("=== COBERTURA vs PRECISION (umbral sobre confianza calibrada) ===")
    print(f"  {'umbral':>7s} {'cobertura':>10s} {'n':>6s} {'prec top-1':>11s} "
          f"{'prec top-3':>11s} {'prec excluidos':>15s}")
    for _, r in curva_conf.iterrows():
        if r["precision_top1"] is None or pd.isna(r["precision_top1"]):
            continue
        pe = r["precision_top1_excluidos"]
        print(f"  {r['umbral']:7.2f} {r['cobertura']:10.4f} {int(r['n_cubiertos']):6d} "
              f"{r['precision_top1']:11.4f} {r['precision_top3']:11.4f} "
              f"{(pe if pe is not None and not pd.isna(pe) else float('nan')):15.4f}")

    objetivos = {}
    for obj in [0.80, 0.85, 0.90, 0.95]:
        r = umbral_para_precision(curva_conf, obj)
        objetivos[f"precision_{int(obj * 100)}"] = r
        if r:
            print()
            print(f"  Para precision top-1 >= {obj:.0%}: umbral {r['umbral']:.2f} -> "
                  f"cobertura {r['cobertura']:.1%} ({r['n_cubiertos']} de "
                  f"{len(y_test_idx_re)} casos), precision real {r['precision_top1']:.1%}")
        else:
            print()
            print(f"  Precision top-1 >= {obj:.0%}: NO alcanzable con ningun umbral.")

    # ---------------- veredicto sobre los umbrales del prototipo -----------
    print()
    print("=== LOS UMBRALES DEL PROTOTIPO ===")
    orden_cal = np.sort(proba_test_cal, axis=1)[:, ::-1]
    m_conf = orden_cal[:, 0] >= APP_LOW_CONFIDENCE
    m_marg = (orden_cal[:, 0] - orden_cal[:, 1]) >= APP_MAX_MARGIN
    m_app = m_conf & m_marg  # el prototipo marca como "confuso" si falla alguno
    veredicto_app = {
        "low_confidence": APP_LOW_CONFIDENCE,
        "max_margin": APP_MAX_MARGIN,
        "cobertura_no_marcada_confusa": round(float(m_app.mean()), 4),
        "precision_top1_no_marcada": round(float(acierto_cal[m_app].mean()), 4) if m_app.any() else None,
        "precision_top1_marcada_confusa": round(float(acierto_cal[~m_app].mean()), 4) if (~m_app).any() else None,
    }
    print(f"  Con confianza >= {APP_LOW_CONFIDENCE} Y margen >= {APP_MAX_MARGIN}:")
    print(f"    cobertura                       {veredicto_app['cobertura_no_marcada_confusa']:.1%}")
    print(f"    precision top-1 de esos casos   {veredicto_app['precision_top1_no_marcada']:.1%}")
    print(f"    precision de los marcados       {veredicto_app['precision_top1_marcada_confusa']:.1%}")
    separa = (
        veredicto_app["precision_top1_no_marcada"] is not None
        and veredicto_app["precision_top1_marcada_confusa"] is not None
        and veredicto_app["precision_top1_no_marcada"]
        - veredicto_app["precision_top1_marcada_confusa"] > 0.15
    )
    print(f"    -> {'Los umbrales SI separan casos faciles de dificiles.' if separa else 'Los umbrales separan poco; conviene reajustarlos con esta curva.'}")

    # ---------------- persistir --------------------------------------------
    bins_antes.to_csv(OUT_DIR / "fiabilidad_bins_sin_calibrar.csv", index=False)
    bins_despues.to_csv(OUT_DIR / "fiabilidad_bins.csv", index=False)
    curva.to_csv(OUT_DIR / "cobertura_precision.csv", index=False)

    meta = {
        "n_test": int(len(y_test_idx_re)),
        "n_validacion": int(len(val)),
        "temperatura": {
            "T": round(T, 4),
            "ajustada_en": "validation (4328 filas)",
            "nll_validacion": round(nll_val, 4),
            "interpretacion": (
                "T > 1 significa que el modelo estaba sobreconfiado; T < 1, "
                "subconfiado. Temperature scaling NO cambia el argmax: accuracy, "
                "F1 y top-3 quedan identicos. Solo corrige el significado del "
                "numero de confianza que el prototipo usa para pedir revision."
            ),
        },
        "calibracion_test": {
            "sin_calibrar": {
                "confianza_media": round(conf_media, 4),
                "accuracy": round(acc_test, 4),
                "brecha_accuracy_menos_confianza": round(acc_test - conf_media, 4),
                "ece": round(ece_antes, 4),
                "brier": round(brier_antes, 4),
            },
            "calibrado": {
                "confianza_media": round(float(conf_cal.mean()), 4),
                "accuracy": round(float(acierto_cal.mean()), 4),
                "ece": round(ece_despues, 4),
                "brier": round(brier_despues, 4),
            },
            "mejora_ece": round(ece_antes - ece_despues, 4),
        },
        "prediccion_selectiva": {
            "criterio": "confianza calibrada",
            "umbrales_por_objetivo_de_precision": objetivos,
            "nota": (
                "La curva completa esta en cobertura_precision.csv, con los dos "
                "criterios (confianza y margen). 'precision_top1_excluidos' dice "
                "cuanto acierta el modelo en los casos que el umbral manda a "
                "revision: si es alta, el umbral esta gastando revision humana en "
                "casos que el modelo resolvia."
            ),
        },
        "umbrales_del_prototipo": veredicto_app,
        "control_identidad_filas": {
            "accuracy_npz": round(acc_test, 4),
            "accuracy_reinferido": round(acc_reinf, 4),
            "delta": round(acc_reinf - acc_test, 4),
        },
        "motivacion": (
            "Cierra dos tareas que el proyecto se habia marcado para Fase 7 y no "
            "ejecuto: temperature scaling antes de fijar umbrales de confianza "
            "(BITACORA seccion 965), y la justificacion empirica de los umbrales "
            "0.35/0.08 que el prototipo trae fijos."
        ),
    }
    with open(OUT_DIR / "calibracion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print(f"Guardado en {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
