"""Barrido clasico, extension: TF-IDF + Random Forest (bosque aleatorio).

ESTUDIO DE EXHAUSTIVIDAD -- NO ES UNA RESELECCION DE MODELO.

Motivacion. La propuesta aprobada (28/03/2026, seccion 4.5) nombra como familia
clasica de referencia "TF-IDF + SVM / regresion logistica / bosque aleatorio".
El barrido del 05/09/2026 (barrido_clasico_tfidf.py) evaluo SVM lineal, Naive
Bayes, SGD y LSA, pero no el bosque aleatorio. Este script cierra ese unico
hueco bajo EXACTAMENTE el mismo protocolo:

    - misma particion canonica (huellas SHA-256 verificadas en ejecucion);
    - misma representacion TF-IDF ajustada una sola vez sobre train;
    - seleccion de hiperparametros en VALIDACION (4.328 filas);
    - evaluacion unica en el TEST canonico (4.329 filas);
    - misma regla de adopcion preregistrada (ver abajo).

REGLA DE ADOPCION PREREGISTRADA (identica a la del 05/09, fijada ANTES de correr):
    El modelo de produccion (TF-IDF + LogisticRegression, C=8.0) SOLO se
    reemplaza si una alternativa cumple LAS DOS condiciones:
      (1) mejora en F1 macro > 0.031 sobre el test canonico (dos veces la
          desviacion multi-semilla del baseline, 0.5241 +/- 0.0153); y
      (2) la mejora sobrevive una prueba pareada con IC 95 % que no cruce cero.

Rejilla del bosque. Cinco configuraciones, todas con 300 arboles, bootstrap y
semilla 42. Se varia el numero de rasgos por division (sqrt, log2, 1.500), el
tope de hojas por arbol (2.000 / 6.000) y la ponderacion de clases. El tope de
hojas no es una eleccion estetica: con 42 clases, cada nodo almacena un vector
de 42 valores y un bosque de 300 arboles sin tope superaria varios GB de
memoria. Se declara como restriccion de recursos del entorno de escritorio.

Prueba pareada. Aunque la condicion (2) solo se exige si se supera el umbral,
el script calcula SIEMPRE el bootstrap pareado (2.000 replicas, semilla 42) y
McNemar frente a las predicciones certificadas del control, para poder
declarar si la diferencia observada esta establecida en cualquiera de las dos
direcciones.

CONTENCION: aditivo. No reescribe barrido_clasico_metadata.json (artefacto del
05/09 corrido en Linux). Escribe unicamente:
    - Proyecto/artifacts/barrido_clasico/barrido_rf_metadata.json
    - Proyecto/artifacts/barrido_clasico/predicciones_rf.npz
    - Proyecto/artifacts/barrido_clasico/tabla_barrido_completa.md  (7 filas)

Uso:
    python Proyecto/Scripts/barrido_clasico_rf.py
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrido_clasico_tfidf as base  # noqa: E402

# El script base fija CACHE_DIR = "/tmp/barrido_cache" (Linux). Aqui se resuelve
# de forma portable; la verificacion de huellas se ejecuta igual.
base.CACHE_DIR = str(Path(tempfile.gettempdir()) / "barrido_cache")

SEED = base.SEED
N_ESTIMATORS = 300
N_JOBS = max(1, (os.cpu_count() or 2) - 2)  # deja margen al sistema
N_BOOT = 2000


def rejilla_rf():
    """Configuraciones evaluadas en validacion. Orden: de la mas barata a la mas cara."""
    return [
        {"max_features": "sqrt", "max_leaf_nodes": 2000, "class_weight": "balanced_subsample"},
        {"max_features": "sqrt", "max_leaf_nodes": 6000, "class_weight": "balanced_subsample"},
        {"max_features": "sqrt", "max_leaf_nodes": 6000, "class_weight": None},
        {"max_features": "log2", "max_leaf_nodes": 6000, "class_weight": "balanced_subsample"},
        {"max_features": 1500, "max_leaf_nodes": 6000, "class_weight": "balanced_subsample"},
    ]


def construir_rf(params):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=params["max_features"],
        max_leaf_nodes=params["max_leaf_nodes"],
        class_weight=params["class_weight"],
        bootstrap=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


# --------------------------------------------------------------------------
# Prueba pareada (misma logica que analisis_pareado_pendiente_p1.py)
# --------------------------------------------------------------------------

def bootstrap_pareado_f1(y_true, pred_a, pred_b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = f1_score(y_true[idx], pred_a[idx], average="macro")
        fb = f1_score(y_true[idx], pred_b[idx], average="macro")
        deltas[i] = fa - fb
    return {
        "delta_puntual": float(
            f1_score(y_true, pred_a, average="macro") - f1_score(y_true, pred_b, average="macro")
        ),
        "ic95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "sd_bootstrap": float(deltas.std(ddof=1)),
        "replicas": n_boot,
        "semilla": seed,
    }


def mcnemar(y_true, pred_a, pred_b):
    ok_a = pred_a == y_true
    ok_b = pred_b == y_true
    b = int(np.sum(ok_a & ~ok_b))  # acierta solo A
    c = int(np.sum(~ok_a & ok_b))  # acierta solo B
    if b + c == 0:
        return {"solo_a": b, "solo_b": c, "chi2": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # con correccion de continuidad
    return {"solo_a": b, "solo_b": c, "chi2": float(stat), "p": float(chi2.sf(stat, 1))}


# --------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "Proyecto" / "artifacts" / "barrido_clasico"
    out_dir.mkdir(parents=True, exist_ok=True)
    inicio = time.time()

    (train_df, valid_df, test_df, label_list, id2label, data_sha256,
     fingerprints) = base.construir_splits(repo_root)
    x_train, y_train = train_df["text_baseline"], train_df["label"].to_numpy()
    x_valid, y_valid = valid_df["text_baseline"], valid_df["label"].to_numpy()
    x_test, y_test = test_df["text_baseline"], test_df["label"].to_numpy()
    print(f"[INFO] train={len(x_train)} valid={len(x_valid)} test={len(x_test)} "
          f"clases={len(label_list)} | n_jobs={N_JOBS}")

    cache = Path(base.CACHE_DIR) / "tfidf_canonico.joblib"
    if cache.exists():
        Xtr, Xva, Xte = joblib.load(cache)
        print(f"[INFO] Reusando TF-IDF cacheado: {cache}")
    else:
        print("[INFO] Ajustando TF-IDF canonico sobre train...")
        vec = TfidfVectorizer(**base.TFIDF_KWARGS)
        Xtr = vec.fit_transform(x_train)
        Xva = vec.transform(x_valid)
        Xte = vec.transform(x_test)
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((Xtr, Xva, Xte), cache, compress=0)
    print(f"[INFO] Vocabulario: {Xtr.shape[1]} rasgos")
    if Xtr.shape[1] != 227980:
        print(f"[AVISO] El vocabulario difiere del barrido del 05/09 (227.980 rasgos).")

    # ---------------- Control: predicciones certificadas ----------------
    meta_ctrl = json.loads(
        (repo_root / "Proyecto" / "artifacts" / "baseline_b_tfidf_lr_final_metadata.json")
        .read_text(encoding="utf-8")
    )
    if meta_ctrl["split_fingerprints_sha256"]["test"] != fingerprints["test"]:
        raise RuntimeError("El control no comparte el test canonico.")
    # Las predicciones del control se obtienen reinfiriendo el artefacto exportado
    # (baseline_b_tfidf_lr_final.joblib, que incluye su propio TF-IDF) sobre el
    # texto del test canonico. Es la via certificada en PENDIENTES_MLPM.md
    # ("las cinco metricas coinciden con la metadata a cuatro decimales") y no
    # depende de pyarrow. Se verifica que reproduzca el F1 macro publicado.
    artefacto = joblib.load(
        repo_root / "Proyecto" / "artifacts" / "baseline_b_tfidf_lr_final.joblib"
    )
    modelo_ctrl = artefacto["pipeline"] if isinstance(artefacto, dict) and "pipeline" in artefacto else artefacto
    salida_ctrl = modelo_ctrl.predict(x_test)
    if salida_ctrl.dtype.kind in "iu":
        y_pred_ctrl = np.asarray(salida_ctrl)
    else:
        label2id = {v: k for k, v in id2label.items()}
        y_pred_ctrl = np.array([label2id[v] for v in salida_ctrl])
    f1_control = float(f1_score(y_test, y_pred_ctrl, average="macro"))
    if abs(f1_control - base.F1_MACRO_BASELINE_REFERENCIA) > 5e-4:
        raise RuntimeError(
            f"La reinferencia del control ({f1_control:.4f}) no reproduce el F1 macro "
            f"publicado ({base.F1_MACRO_BASELINE_REFERENCIA:.4f}); se aborta la prueba pareada."
        )
    print(f"[OK] Control reinferido. F1 macro = {f1_control:.4f} "
          f"(publicado {base.F1_MACRO_BASELINE_REFERENCIA:.4f})")

    # Verificacion lateral: F1 macro restringido a clases con soporte >= 20 en test.
    # La tesis reporta 0,6201 y el documento de Investigacion II reporta 0,6270;
    # se calculan las dos variantes para dejar constancia de cual es cual.
    soporte = pd.Series(y_test).value_counts()
    clases_grandes = sorted(int(c) for c in soporte[soporte >= 20].index)
    f1_labels_grandes = float(f1_score(y_test, y_pred_ctrl, labels=clases_grandes, average="macro"))
    mask = np.isin(y_test, clases_grandes)
    f1_filas_grandes = float(f1_score(y_test[mask], y_pred_ctrl[mask], average="macro"))
    verificacion_soporte = {
        "n_clases_soporte_ge20": len(clases_grandes),
        "f1_macro_labels_restringidos_todas_las_filas": round(f1_labels_grandes, 5),
        "f1_macro_filas_restringidas_labels_presentes": round(f1_filas_grandes, 5),
        "nota": (
            "Variante 1: f1_score(labels=clases con soporte>=20) sobre las 4.329 filas. "
            "Variante 2: filas cuya etiqueta verdadera esta en esas clases, promediando "
            "sobre las etiquetas presentes (verdaderas o predichas)."
        ),
    }
    print(f"[INFO] F1 macro clases soporte>=20: variante labels={f1_labels_grandes:.4f} | "
          f"variante filas={f1_filas_grandes:.4f} ({len(clases_grandes)} clases)")

    # ---------------- Barrido RF: seleccion en validacion ----------------
    detalle, mejor = [], None
    for params in rejilla_rf():
        t0 = time.time()
        modelo = construir_rf(params)
        modelo.fit(Xtr, y_train)
        f1_val = float(f1_score(y_valid, modelo.predict(Xva), average="macro"))
        dt = time.time() - t0
        detalle.append({"params": params, "f1_macro_validacion": f1_val, "segundos": round(dt, 1)})
        print(f"    {params} -> F1 macro (val) = {f1_val:.4f}  [{dt:.0f}s]", flush=True)
        if mejor is None or f1_val > mejor[0]:
            if mejor is not None:
                del mejor
            mejor = (f1_val, params, modelo)
        else:
            del modelo
        gc.collect()

    f1_val, params, modelo = mejor
    print(f"[INFO] Configuracion elegida en validacion: {params} (F1 val {f1_val:.4f})")

    # ---------------- Evaluacion unica en test ----------------
    y_pred = modelo.predict(Xte)
    scores = base.scores_de(modelo, Xte)
    fila = base.evaluar("TF-IDF + Random Forest", y_test, y_pred, scores)
    fila["familia"] = "rf"
    fila["mejores_params"] = params
    fila["n_estimators"] = N_ESTIMATORS
    fila["f1_macro_validacion"] = f1_val
    fila["seleccion"] = "F1 macro en validacion (4328 filas)"
    fila["entorno"] = {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "sklearn": __import__("sklearn").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_jobs": N_JOBS,
    }
    np.savez_compressed(out_dir / "predicciones_rf.npz",
                        y_pred=y_pred, y_true=y_test, scores=scores.astype(np.float32))
    print(f"  -> TEST: F1 macro {fila['f1_macro']:.4f} | bal.acc {fila['balanced_accuracy']:.4f} "
          f"| acc {fila['accuracy']:.4f} | top-3 {fila['top3_accuracy']:.4f} "
          f"| F1 pond {fila['f1_weighted']:.4f}")

    # ---------------- Regla de adopcion + prueba pareada ----------------
    delta = fila["f1_macro"] - f1_control
    supera = bool(delta > base.UMBRAL_ADOPCION)
    print("[INFO] Bootstrap pareado (2.000 replicas) y McNemar frente al control...", flush=True)
    boot = bootstrap_pareado_f1(y_test, y_pred, y_pred_ctrl)
    mcn = mcnemar(y_test, y_pred, y_pred_ctrl)
    ic = boot["ic95"]
    if ic[0] > 0:
        veredicto = "RF supera al control (IC excluye cero)"
    elif ic[1] < 0:
        veredicto = "RF queda por debajo del control (IC excluye cero): diferencia establecida"
    else:
        veredicto = "diferencia no distinguible del ruido de muestreo (IC cruza cero)"

    # ---------------- Tabla completa (6 filas del 05/09 + RF) ----------------
    meta_prev = json.loads((out_dir / "barrido_clasico_metadata.json").read_text(encoding="utf-8"))
    if meta_prev["split_fingerprints_sha256"] != fingerprints:
        raise RuntimeError("El barrido del 05/09 no comparte la particion canonica.")
    filas = list(meta_prev["resultados_test"]) + [fila]
    lineas = [
        "| Modelo | F1 macro | Bal. acc. | Exactitud | Top-3 | F1 pond. |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(filas, key=lambda x: -x["f1_macro"]):
        lineas.append(
            f"| {r['modelo']} | {r['f1_macro']:.4f} | {r['balanced_accuracy']:.4f} "
            f"| {r['accuracy']:.4f} | {r['top3_accuracy']:.4f} | {r['f1_weighted']:.4f} |"
        )
    (out_dir / "tabla_barrido_completa.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    metadata = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "encuadre": (
            "ESTUDIO DE EXHAUSTIVIDAD, extension del barrido del 05/09/2026 con el bosque "
            "aleatorio nombrado en la propuesta aprobada (seccion 4.5). NO es una "
            "reseleccion del modelo de produccion."
        ),
        "regla_de_adopcion_preregistrada": meta_prev["regla_de_adopcion_preregistrada"],
        "entorno": fila["entorno"],
        "nota_entorno": (
            "Las cinco familias del 05/09 se ejecutaron en Linux (sklearn 1.7.2); esta "
            "extension en el equipo de escritorio Windows del prototipo. Las huellas de "
            "particion coinciden exactamente, que es lo que garantiza la comparabilidad."
        ),
        "data_sha256": data_sha256,
        "split_fingerprints_sha256": fingerprints,
        "split": meta_prev["split"],
        "representacion": {"tfidf": base.TFIDF_KWARGS, "n_rasgos": int(Xtr.shape[1])},
        "rejilla": rejilla_rf(),
        "n_estimators": N_ESTIMATORS,
        "detalle_validacion": detalle,
        "resultado_test": fila,
        "verificacion_f1_macro_clases_soporte_ge20": verificacion_soporte,
        "prueba_pareada_vs_control": {
            "f1_macro_control": f1_control,
            "delta_f1_macro_rf_menos_control": round(delta, 4),
            "bootstrap_pareado": boot,
            "mcnemar": mcn,
            "veredicto_pareado": veredicto,
        },
        "veredicto": {
            "umbral_preregistrado": base.UMBRAL_ADOPCION,
            "supera_umbral_preregistrado": supera,
            "conclusion": (
                "El bosque aleatorio NO supera el umbral preregistrado; el modelo de "
                "produccion NO cambia."
                if not supera
                else "ATENCION: RF supera el umbral. Verificar condicion (2) antes de decidir."
            ),
        },
        "minutos_totales": round((time.time() - inicio) / 60, 1),
    }
    (out_dir / "barrido_rf_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "\n".join(lineas))
    print(f"\n[PAREADO] delta={delta:+.4f} IC95=[{ic[0]:+.4f}; {ic[1]:+.4f}] "
          f"McNemar chi2={mcn['chi2']:.3f} p={mcn['p']:.2e} (solo RF {mcn['solo_a']}, "
          f"solo control {mcn['solo_b']}) -> {veredicto}")
    print("[VEREDICTO] " + metadata["veredicto"]["conclusion"])
    print(f"[OK] Escrito en {out_dir} ({metadata['minutos_totales']} min)")


if __name__ == "__main__":
    main()
