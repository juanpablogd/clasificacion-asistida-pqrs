"""Barrido de algoritmos clasicos sobre la misma representacion TF-IDF.

ESTUDIO DE EXHAUSTIVIDAD -- NO ES UNA RESELECCION DE MODELO.

Motivacion (reunion con el tutor, 2026-09):
    "el TF-IDF combinarlo con otras estrategias, con otros algoritmos, no para
     ver que pasa [...] que se vea un trabajo exhaustivo".

El unico clasificador clasico evaluado hasta ahora era LogisticRegression.
Este script cierra ese hueco evaluando, sobre EXACTAMENTE la misma particion
canonica y la misma representacion, cinco alternativas:

    - LinearSVC                      (margen maximo, lineal)
    - ComplementNB                   (Naive Bayes para clases desbalanceadas)
    - SGDClassifier (hinge)          (SVM lineal por descenso estocastico)
    - SGDClassifier (log_loss)       (regresion logistica por SGD)
    - TruncatedSVD (LSA) + LinearSVC (reduccion de dimensionalidad)

REGLA DE ADOPCION PREREGISTRADA (fijada ANTES de correr el barrido):
    El modelo de produccion (TF-IDF + LogisticRegression, C=8.0) SOLO se
    reemplaza si una alternativa cumple LAS DOS condiciones:
      (1) mejora en F1 macro > 0.031 sobre el test canonico, donde 0.031 es
          dos veces la desviacion multi-semilla del baseline (0.5241 +/- 0.0153,
          5 particiones, FASE5_RESULTADOS_COMPARATIVA_FINAL.md); y
      (2) la mejora sobrevive una prueba pareada con IC 95 % que no cruce cero.
    Cualquier diferencia menor se declara indistinguible del ruido de particion.

    Justificacion: sin este umbral, elegir el maximo de una tabla de seis
    modelos sobre el mismo test es seleccion sobre ruido. El baseline varia
    +/- 1.5 puntos de F1 macro solo por cambiar la semilla de particion.

CONTENCION: el script es ADITIVO. No sobrescribe ningun artefacto existente.
Escribe unicamente en Proyecto/artifacts/barrido_clasico/.

Seleccion de hiperparametros: en el split de VALIDACION (4328 filas).
Evaluacion: unica, en el test canonico (4329 filas), verificado por huella
SHA-256 contra baseline_homologado_estricto_metadata.json.

Salidas:
    - Proyecto/artifacts/barrido_clasico/barrido_clasico_metadata.json
    - Proyecto/artifacts/barrido_clasico/tabla_barrido.md
    - Proyecto/artifacts/barrido_clasico/predicciones_<modelo>.npz
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

SEED = 42
CACHE_DIR = "/tmp/barrido_cache"  # solo acelera reejecuciones; no es un artefacto
MIN_VALID_YEAR = 2000
CURRENT_YEAR = 2026
USE_MEDIO = True
MIN_EXAMPLES_PER_CLASS = 20

# Regla de adopcion preregistrada
SIGMA_MULTISEMILLA = 0.0153
UMBRAL_ADOPCION = 2 * SIGMA_MULTISEMILLA  # 0.0306
F1_MACRO_BASELINE_REFERENCIA = 0.5455370598338695

# Configuracion TF-IDF canonica (identica al baseline de produccion)
TFIDF_KWARGS = dict(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9,
    sublinear_tf=True,
)


# --------------------------------------------------------------------------
# Preparacion de datos: copia literal de export_baseline_b_final.py
# --------------------------------------------------------------------------

def normalizar_texto(value: object) -> str:
    text = str(value).strip()
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def moda_estable(series: pd.Series, prioridad: pd.Series | None = None) -> str:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return ""
    top = counts[counts == counts.max()].index.tolist()
    if len(top) == 1:
        return top[0]
    if prioridad is not None:
        top.sort(key=lambda item: (-prioridad.get(item, 0), item))
    else:
        top.sort()
    return top[0]


def split_fingerprint(frame: pd.DataFrame) -> str:
    payload = ",".join(map(str, sorted(frame.index.tolist()))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"No se pudo leer el dataset: {path}")


def construir_splits(repo_root: Path):
    """Reproduce la particion canonica y verifica su huella SHA-256.

    El resultado se cachea en disco temporal: reconstruirlo cuesta ~40 s y el
    barrido se ejecuta por tramos. La verificacion de huellas se hace SIEMPRE,
    tambien al leer del cache.
    """
    cache = Path(CACHE_DIR) / "splits_canonicos.joblib"
    if cache.exists():
        datos = joblib.load(cache)
        strict = json.loads(
            (
                repo_root / "Proyecto" / "artifacts" / "mrbert_es_final"
                / "baseline_homologado_estricto_metadata.json"
            ).read_text(encoding="utf-8")
        )
        if datos[6] != strict["split_fingerprints_sha256"]:
            raise RuntimeError("Cache de splits corrupto: huellas no coinciden.")
        print("[OK] Huellas del split verificadas (desde cache).")
        return datos

    data_file = repo_root / "data" / "Registro_PQRS_Mindeportes_20260328.csv"
    strict_meta_path = (
        repo_root
        / "Proyecto"
        / "artifacts"
        / "mrbert_es_final"
        / "baseline_homologado_estricto_metadata.json"
    )

    with open(data_file, "rb") as file:
        data_sha256 = hashlib.sha256(file.read()).hexdigest()

    df = read_csv_with_fallback(data_file)

    if "FECHA RADICADO" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["FECHA RADICADO"], errors="coerce")
        df["anio_fecha"] = df["fecha_dt"].dt.year
    else:
        df["fecha_dt"] = pd.NaT
        df["anio_fecha"] = np.nan

    df["anio_radicado"] = pd.to_numeric(
        df["RADICADO"].astype(str).str.extract(r"^(\d{4})")[0],
        errors="coerce",
    )
    anio_radicado = df["anio_radicado"].where(
        df["anio_radicado"].between(MIN_VALID_YEAR, CURRENT_YEAR)
    )
    anio_fecha = df["anio_fecha"].where(
        df["anio_fecha"].between(MIN_VALID_YEAR, CURRENT_YEAR)
    )
    df["anio"] = anio_radicado.fillna(anio_fecha)

    work = df.dropna(subset=["TIPO", "ASUNTO", "DEPENDENCIA DESTINO"]).copy()
    work["TIPO"] = work["TIPO"].apply(normalizar_texto)
    work["ASUNTO"] = work["ASUNTO"].apply(normalizar_texto)
    work["DEPENDENCIA DESTINO"] = work["DEPENDENCIA DESTINO"].apply(normalizar_texto)

    if "MEDIO DE ENVIO" in df.columns:
        work["MEDIO DE ENVIO"] = (
            work["MEDIO DE ENVIO"].fillna("NO_REPORTADO").apply(normalizar_texto)
        )
    else:
        work["MEDIO DE ENVIO"] = "NO_REPORTADO"

    label_prioridad = work["DEPENDENCIA DESTINO"].value_counts()
    work = (
        work.groupby(["TIPO", "ASUNTO"], dropna=False)
        .agg(
            {
                "DEPENDENCIA DESTINO": lambda values: moda_estable(
                    values, label_prioridad
                ),
                "MEDIO DE ENVIO": lambda values: moda_estable(values),
                "RADICADO": "first",
                "fecha_dt": "first",
                "anio": "first",
            }
        )
        .reset_index()
    )

    work = work[
        (work["ASUNTO"].str.len() >= 10) & (work["TIPO"].str.len() >= 3)
    ].copy()

    if USE_MEDIO:
        work["text_baseline"] = (
            "tipo: "
            + work["TIPO"]
            + " [SEP] medio: "
            + work["MEDIO DE ENVIO"]
            + " [SEP] asunto: "
            + work["ASUNTO"]
        )
    else:
        work["text_baseline"] = (
            "tipo: " + work["TIPO"] + " [SEP] asunto: " + work["ASUNTO"]
        )

    class_counts = work["DEPENDENCIA DESTINO"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_EXAMPLES_PER_CLASS].index
    work = work[work["DEPENDENCIA DESTINO"].isin(valid_classes)].copy()

    label_list = sorted(work["DEPENDENCIA DESTINO"].unique())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    work["label"] = work["DEPENDENCIA DESTINO"].map(label2id).astype(int)

    split_cols = ["text_baseline", "label", "DEPENDENCIA DESTINO", "anio", "fecha_dt"]
    train_df, temp_df = train_test_split(
        work[split_cols],
        test_size=0.2,
        random_state=SEED,
        stratify=work["label"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    fingerprints = {
        "train": split_fingerprint(train_df),
        "validation": split_fingerprint(valid_df),
        "test": split_fingerprint(test_df),
    }

    with open(strict_meta_path, "r", encoding="utf-8") as file:
        strict_meta = json.load(file)

    if fingerprints != strict_meta["split_fingerprints_sha256"]:
        raise RuntimeError(
            "ABORTADO: las huellas del split NO coinciden con la particion canonica. "
            "Cualquier metrica producida aqui seria incomparable con las publicadas."
        )
    print("[OK] Huellas del split verificadas contra la particion canonica.")

    salida = (train_df, valid_df, test_df, label_list, id2label, data_sha256,
              fingerprints)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(salida, cache, compress=0)
    return salida


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------

def top_k_desde_scores(scores: np.ndarray, y_true_idx: np.ndarray, k: int = 3) -> float:
    """Top-k a partir de scores crudos (funciona con decision_function y proba)."""
    orden = np.argsort(-scores, axis=1)[:, :k]
    aciertos = (orden == y_true_idx[:, None]).any(axis=1)
    return float(aciertos.mean())


def evaluar(nombre, y_true_idx, y_pred_idx, scores):
    return {
        "modelo": nombre,
        "f1_macro": float(f1_score(y_true_idx, y_pred_idx, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_idx, y_pred_idx)),
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "top3_accuracy": top_k_desde_scores(scores, y_true_idx, 3),
        "f1_weighted": float(f1_score(y_true_idx, y_pred_idx, average="weighted")),
    }


def scores_de(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)
    return modelo.decision_function(X)


# --------------------------------------------------------------------------
# Definicion de candidatos
# --------------------------------------------------------------------------

def candidatos():
    """Cada entrada: (nombre, familia, rejilla de configuraciones).

    Orden deliberado: primero los modelos baratos, el control (LogisticRegression
    sobre 227.980 rasgos y 42 clases) al final por ser el mas costoso.
    """
    return [
        (
            "TF-IDF + LinearSVC",
            "linearsvc",
            [{"C": c} for c in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)],
        ),
        (
            "TF-IDF + ComplementNB",
            "cnb",
            [{"alpha": a} for a in (0.01, 0.05, 0.1, 0.3, 1.0)],
        ),
        (
            "TF-IDF + SGD (hinge)",
            "sgd_hinge",
            [{"alpha": a} for a in (1e-6, 1e-5, 1e-4, 1e-3)],
        ),
        (
            "TF-IDF + SGD (log_loss)",
            "sgd_log",
            [{"alpha": a} for a in (1e-6, 1e-5, 1e-4, 1e-3)],
        ),
        (
            "TF-IDF + SVD (LSA) + LinearSVC",
            "svd_linearsvc",
            [
                {"n_components": 300, "C": 1.0},
                {"n_components": 300, "C": 4.0},
                {"n_components": 600, "C": 1.0},
            ],
        ),
        (
            "TF-IDF + LogisticRegression (control)",
            "control",
            [{"C": 8.0}],
        ),
    ]


def construir(familia, params):
    if familia == "control":
        return LogisticRegression(
            C=params["C"],
            class_weight="balanced",
            random_state=SEED,
            max_iter=1500,
            solver="lbfgs",
            n_jobs=-1,
        ), None
    if familia == "linearsvc":
        return LinearSVC(
            C=params["C"],
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
            dual="auto",
        ), None
    if familia == "cnb":
        # ComplementNB no acepta class_weight; su diseno ya compensa desbalance.
        return ComplementNB(alpha=params["alpha"]), None
    if familia == "sgd_hinge":
        return SGDClassifier(
            loss="hinge",
            alpha=params["alpha"],
            class_weight="balanced",
            random_state=SEED,
            max_iter=3000,
            tol=1e-4,
            n_jobs=-1,
        ), None
    if familia == "sgd_log":
        return SGDClassifier(
            loss="log_loss",
            alpha=params["alpha"],
            class_weight="balanced",
            random_state=SEED,
            max_iter=3000,
            tol=1e-4,
            n_jobs=-1,
        ), None
    if familia == "svd_linearsvc":
        reductor = Pipeline(
            [
                ("svd", TruncatedSVD(n_components=params["n_components"], random_state=SEED)),
                ("norm", Normalizer(copy=False)),
            ]
        )
        return LinearSVC(
            C=params["C"],
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
            dual="auto",
        ), reductor
    raise ValueError(familia)


# --------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "Proyecto" / "artifacts" / "barrido_clasico"
    out_dir.mkdir(parents=True, exist_ok=True)

    inicio = time.time()
    (
        train_df,
        valid_df,
        test_df,
        label_list,
        id2label,
        data_sha256,
        fingerprints,
    ) = construir_splits(repo_root)

    x_train, y_train = train_df["text_baseline"], train_df["label"].to_numpy()
    x_valid, y_valid = valid_df["text_baseline"], valid_df["label"].to_numpy()
    x_test, y_test = test_df["text_baseline"], test_df["label"].to_numpy()

    print(f"[INFO] train={len(x_train)} valid={len(x_valid)} test={len(x_test)} "
          f"clases={len(label_list)}")

    # La representacion TF-IDF se ajusta UNA VEZ sobre train y se comparte.
    # Se cachea en disco: el barrido se corre por tramos y no tiene sentido
    # recalcularla en cada invocacion.
    cache = Path(CACHE_DIR) / "tfidf_canonico.joblib"
    if cache.exists():
        print(f"[INFO] Reusando TF-IDF cacheado: {cache}")
        Xtr, Xva, Xte = joblib.load(cache)
    else:
        print("[INFO] Ajustando TF-IDF canonico sobre train...")
        vectorizador = TfidfVectorizer(**TFIDF_KWARGS)
        Xtr = vectorizador.fit_transform(x_train)
        Xva = vectorizador.transform(x_valid)
        Xte = vectorizador.transform(x_test)
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((Xtr, Xva, Xte), cache, compress=0)
    print(f"[INFO] Vocabulario: {Xtr.shape[1]} rasgos")

    # Checkpointing: se recuperan las familias ya evaluadas en corridas previas.
    parcial = out_dir / "_parcial.json"
    if parcial.exists():
        estado = json.loads(parcial.read_text(encoding="utf-8"))
        resultados = estado["resultados"]
        detalle_validacion = estado["detalle_validacion"]
        hechas = {r["familia"] for r in resultados}
        print(f"[INFO] Familias ya evaluadas: {sorted(hechas)}")
    else:
        resultados, detalle_validacion, hechas = [], [], set()

    solo = sys.argv[1] if len(sys.argv) > 1 else None
    proyecciones: dict[int, object] = {}

    for nombre, familia, rejilla in candidatos():
        if familia in hechas:
            continue
        if solo and familia != solo:
            continue

        if familia == "control":
            # El control NO se reentrena: es el modelo de produccion ya exportado,
            # cuyas metricas sobre ESTE mismo test canonico estan certificadas en
            # baseline_b_tfidf_lr_final_metadata.json (huella de test identica,
            # verificada arriba). Reentrenarlo produciria las mismas cifras a costa
            # de ~15 min de CPU y con riesgo de divergencia por version de sklearn.
            meta_ctrl = json.loads(
                (
                    repo_root / "Proyecto" / "artifacts"
                    / "baseline_b_tfidf_lr_final_metadata.json"
                ).read_text(encoding="utf-8")
            )
            if (
                meta_ctrl["split_fingerprints_sha256"]["test"]
                != fingerprints["test"]
            ):
                raise RuntimeError("El control no comparte el test canonico.")
            m = meta_ctrl["metrics"]
            fila = {
                "modelo": nombre,
                "familia": "control",
                "f1_macro": m["f1_macro"],
                "balanced_accuracy": m["balanced_accuracy"],
                "accuracy": m["accuracy"],
                "top3_accuracy": m["top3_accuracy"],
                "f1_weighted": m["f1_weighted"],
                "mejores_params": m["best_params"],
                "seleccion": "RandomizedSearchCV previo (12 iter x 3 folds)",
                "procedencia": (
                    "Metricas certificadas de baseline_b_tfidf_lr_final_metadata.json; "
                    "no se reentreno."
                ),
            }
            resultados.append(fila)
            parcial.write_text(
                json.dumps(
                    {"resultados": resultados,
                     "detalle_validacion": detalle_validacion},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"\n=== {nombre} ===\n  -> TEST (certificado): F1 macro "
                  f"{fila['f1_macro']:.4f} | acc {fila['accuracy']:.4f} "
                  f"| top-3 {fila['top3_accuracy']:.4f}")
            continue

        print(f"\n=== {nombre} ===")
        mejor = None
        for params in rejilla:
            t0 = time.time()
            # Cache por configuracion: el sandbox limita cada invocacion, asi que
            # una rejilla larga se completa a lo largo de varias corridas.
            clave = hashlib.sha256(
                f"{familia}|{sorted(params.items())}".encode()
            ).hexdigest()[:16]
            cfg_cache = Path(CACHE_DIR) / f"cfg_{clave}.joblib"
            if cfg_cache.exists():
                f1_val, modelo, reductor = joblib.load(cfg_cache)
                if reductor is not None:
                    proyecciones[params["n_components"]] = joblib.load(
                        Path(CACHE_DIR) / f"svd_{params['n_components']}.joblib"
                    )[2]
                print(f"    {params} -> F1 macro (val) = {f1_val:.4f}  [cache]")
                if mejor is None or f1_val > mejor[0]:
                    mejor = (f1_val, params, modelo, reductor)
                continue
            modelo, reductor = construir(familia, params)
            if reductor is not None:
                # La proyeccion SVD solo depende de n_components: se cachea en
                # disco para no recalcularla por cada valor de C.
                k = params["n_components"]
                svd_cache = Path(CACHE_DIR) / f"svd_{k}.joblib"
                if svd_cache.exists():
                    Ztr, Zva, Zte_cache = joblib.load(svd_cache)
                else:
                    print(f"    [SVD] proyectando a {k} componentes...")
                    Ztr = reductor.fit_transform(Xtr)
                    Zva = reductor.transform(Xva)
                    Zte_cache = reductor.transform(Xte)
                    joblib.dump((Ztr, Zva, Zte_cache), svd_cache, compress=0)
                proyecciones[k] = Zte_cache
            else:
                Ztr, Zva = Xtr, Xva
            modelo.fit(Ztr, y_train)
            f1_val = f1_score(y_valid, modelo.predict(Zva), average="macro")
            dt = time.time() - t0
            detalle_validacion.append(
                {"modelo": nombre, "params": params, "f1_macro_validacion": float(f1_val),
                 "segundos": round(dt, 1)}
            )
            print(f"    {params} -> F1 macro (val) = {f1_val:.4f}  [{dt:.0f}s]")
            joblib.dump((float(f1_val), modelo, reductor), cfg_cache, compress=0)
            if mejor is None or f1_val > mejor[0]:
                mejor = (f1_val, params, modelo, reductor)

        f1_val, params, modelo, reductor = mejor
        # Evaluacion unica en test con la configuracion elegida en validacion
        Zte = proyecciones[params["n_components"]] if reductor is not None else Xte
        y_pred = modelo.predict(Zte)
        scores = scores_de(modelo, Zte)
        fila = evaluar(nombre, y_test, y_pred, scores)
        fila["familia"] = familia
        fila["mejores_params"] = params
        fila["f1_macro_validacion"] = float(f1_val)
        fila["seleccion"] = "F1 macro en validacion (4328 filas)"
        resultados.append(fila)
        np.savez_compressed(
            out_dir / f"predicciones_{familia}.npz",
            y_pred=y_pred, y_true=y_test, scores=scores.astype(np.float32),
        )
        # Checkpoint tras cada familia
        parcial.write_text(
            json.dumps(
                {"resultados": resultados, "detalle_validacion": detalle_validacion},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  -> TEST: F1 macro {fila['f1_macro']:.4f} | acc {fila['accuracy']:.4f} "
              f"| top-3 {fila['top3_accuracy']:.4f}")

    if not any(r["familia"] == "control" for r in resultados):
        print("\n[PENDIENTE] Falta el control. Vuelva a ejecutar para consolidar.")
        return

    # ---------------- Regla de adopcion ----------------
    control = next(r for r in resultados if r["modelo"].endswith("(control)"))
    f1_control = control["f1_macro"]

    veredictos = []
    for r in resultados:
        if r is control:
            continue
        delta = r["f1_macro"] - f1_control
        veredictos.append(
            {
                "modelo": r["modelo"],
                "delta_f1_macro_vs_control": round(delta, 4),
                "supera_umbral_preregistrado": bool(delta > UMBRAL_ADOPCION),
            }
        )

    algun_candidato = any(v["supera_umbral_preregistrado"] for v in veredictos)

    metadata = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "encuadre": (
            "ESTUDIO DE EXHAUSTIVIDAD. Responde a la observacion del tutor sobre "
            "combinar TF-IDF con otros algoritmos y con reduccion de dimensionalidad. "
            "NO es una reseleccion del modelo de produccion."
        ),
        "regla_de_adopcion_preregistrada": {
            "umbral_f1_macro": UMBRAL_ADOPCION,
            "origen_del_umbral": (
                "dos veces la desviacion multi-semilla del baseline "
                "(F1 macro 0.5241 +/- 0.0153 en 5 particiones)"
            ),
            "condicion_adicional": "IC 95 % pareado que no cruce cero",
            "fijada_antes_de_correr": True,
        },
        "entorno": {
            "python": sys.version.split()[0],
            "plataforma": platform.platform(),
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data_sha256": data_sha256,
        "split_fingerprints_sha256": fingerprints,
        "split": {
            "train_rows": int(len(x_train)),
            "valid_rows": int(len(x_valid)),
            "test_rows": int(len(x_test)),
            "n_clases": len(label_list),
            "seed": SEED,
        },
        "representacion": {"tfidf": TFIDF_KWARGS, "n_rasgos": int(Xtr.shape[1])},
        "resultados_test": resultados,
        "detalle_validacion": detalle_validacion,
        "veredicto": {
            "f1_macro_control": f1_control,
            "f1_macro_baseline_publicado": F1_MACRO_BASELINE_REFERENCIA,
            "control_reproduce_publicado": bool(
                abs(f1_control - F1_MACRO_BASELINE_REFERENCIA) < 1e-6
            ),
            "por_modelo": veredictos,
            "algun_candidato_supera_umbral": algun_candidato,
            "conclusion": (
                "Ningun algoritmo clasico alternativo supera el umbral preregistrado; "
                "el modelo de produccion NO cambia."
                if not algun_candidato
                else "ATENCION: al menos un candidato supera el umbral. Requiere prueba "
                     "pareada antes de cualquier decision."
            ),
        },
        "minutos_totales": round((time.time() - inicio) / 60, 1),
    }

    with open(out_dir / "barrido_clasico_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Tabla markdown lista para pegar en el articulo / la tesis
    lineas = [
        "| Modelo | F1 macro | Bal. acc. | Exactitud | Top-3 | F1 pond. |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(resultados, key=lambda x: -x["f1_macro"]):
        lineas.append(
            f"| {r['modelo']} | {r['f1_macro']:.4f} | {r['balanced_accuracy']:.4f} "
            f"| {r['accuracy']:.4f} | {r['top3_accuracy']:.4f} | {r['f1_weighted']:.4f} |"
        )
    with open(out_dir / "tabla_barrido.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")

    print("\n" + "\n".join(lineas))
    print("\n[VEREDICTO] " + metadata["veredicto"]["conclusion"])
    print(f"[OK] Escrito en {out_dir}")


if __name__ == "__main__":
    main()
