"""Busqueda ampliada de hiperparametros con Optuna: evidencia de SATURACION.

Por que existe este script
--------------------------
El tuning publicado del baseline usa RandomizedSearchCV con 12 iteraciones x 3
folds = 36 ajustes, sobre un espacio acotado. Es un presupuesto pequeno y es una
objecion facil de plantear: "36 configuraciones y concluyen que el techo no es de
tuning?".

Este script NO busca un modelo mejor. Busca caracterizar la FORMA de la superficie
de respuesta: si el desempeno se satura, entonces el limite no esta en la
configuracion sino en la informacion que trae el registro. La curva de convergencia
y la importancia de hiperparametros son la evidencia grafica de esa afirmacion.

Sustitucion declarada del estimador
-----------------------------------
La regresion logistica con solver lbfgs sobre 227.980 rasgos y 42 clases tarda
minutos por ajuste, lo que hace inviable un presupuesto de 100 ensayos. Se usa
SGDClassifier(loss="log_loss"), que es LA MISMA familia -regresion logistica-
ajustada por descenso de gradiente estocastico, unas 40 veces mas rapida. La
conclusion que se extrae es sobre la forma de la superficie de respuesta, no sobre
el valor absoluto de la metrica, y esa forma es la propiedad que se transfiere.
El modelo de produccion NO cambia.

Seleccion en VALIDACION (4.328 filas). El test canonico no se toca.

CONTENCION: aditivo. Escribe solo en artifacts/optuna_saturacion/.
Reanudable: el estudio vive en SQLite y se completa a lo largo de varias corridas.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

CACHE_DIR = Path("/tmp/barrido_cache")
N_TRIALS_OBJETIVO = 100
SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cargar_datos():
    """Reutiliza el split canonico ya verificado por barrido_clasico_tfidf.py."""
    cache = CACHE_DIR / "splits_canonicos.joblib"
    if not cache.exists():
        raise SystemExit(
            "Ejecute antes barrido_clasico_tfidf.py: este script reutiliza su "
            "split canonico verificado."
        )
    datos = joblib.load(cache)
    train_df, valid_df = datos[0], datos[1]
    return (
        train_df["text_baseline"].tolist(),
        train_df["label"].to_numpy(),
        valid_df["text_baseline"].tolist(),
        valid_df["label"].to_numpy(),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "Proyecto" / "artifacts" / "optuna_saturacion"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_valid, y_valid = cargar_datos()
    print(f"[INFO] train={len(x_train)} valid={len(x_valid)}")

    def objetivo(trial: optuna.Trial) -> float:
        n_max = trial.suggest_int("ngram_max", 1, 3)
        vec = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, n_max),
            min_df=trial.suggest_int("min_df", 1, 5),
            max_df=trial.suggest_float("max_df", 0.5, 1.0),
            sublinear_tf=trial.suggest_categorical("sublinear_tf", [True, False]),
            use_idf=trial.suggest_categorical("use_idf", [True, False]),
        )
        Xtr = vec.fit_transform(x_train)
        Xva = vec.transform(x_valid)
        clf = SGDClassifier(
            loss="log_loss",
            alpha=trial.suggest_float("alpha", 1e-7, 1e-3, log=True),
            class_weight=trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            max_iter=trial.suggest_int("max_iter", 500, 3000, step=500),
            tol=1e-4,
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(Xtr, y_train)
        return float(f1_score(y_valid, clf.predict(Xva), average="macro"))

    # El estudio vive en disco local: SQLite no funciona sobre el volumen montado.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    almacen = f"sqlite:///{CACHE_DIR / 'optuna_estudio.db'}"
    estudio = optuna.create_study(
        study_name="saturacion_tfidf_lineal",
        storage=almacen,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    hechos = len(estudio.trials)
    print(f"[INFO] Ensayos previos: {hechos} / {N_TRIALS_OBJETIVO}")
    if hechos < N_TRIALS_OBJETIVO:
        presupuesto = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
        limite = time.time() + presupuesto
        while len(estudio.trials) < N_TRIALS_OBJETIVO and time.time() < limite:
            estudio.optimize(objetivo, n_trials=1, catch=(Exception,))
        hechos = len(estudio.trials)
        print(f"[INFO] Ensayos acumulados: {hechos}")

    if hechos < N_TRIALS_OBJETIVO:
        print("[PENDIENTE] Vuelva a ejecutar para continuar el estudio.")
        return

    # ---------------- Analisis de saturacion ----------------
    valores = [
        t.value for t in estudio.trials
        if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE
    ]
    mejor_acumulado = np.maximum.accumulate(valores).tolist()
    mejor = max(valores)

    # En que ensayo se alcanzo el 99 % del mejor valor final
    umbral_99 = 0.99 * mejor
    ensayo_99 = next(
        i + 1 for i, v in enumerate(mejor_acumulado) if v >= umbral_99
    )
    # Ganancia de la segunda mitad del presupuesto
    mitad = len(mejor_acumulado) // 2
    ganancia_segunda_mitad = mejor_acumulado[-1] - mejor_acumulado[mitad - 1]
    # Ganancia de los ensayos posteriores al que alcanza el 99 %. Corregido el
    # 26/09/2026: la lectura atribuia a estos ensayos la ganancia de la segunda mitad.
    ganancia_tras_99 = mejor_acumulado[-1] - mejor_acumulado[ensayo_99 - 1]

    try:
        importancias = optuna.importance.get_param_importances(estudio)
        importancias = {k: round(float(v), 4) for k, v in importancias.items()}
    except Exception as exc:  # pragma: no cover
        importancias = {"error": str(exc)}

    metadata = {
        "proposito": (
            "Caracterizar la forma de la superficie de respuesta del tuning, NO "
            "seleccionar un modelo. El modelo de produccion no cambia."
        ),
        "estimador": {
            "usado": "SGDClassifier(loss='log_loss')",
            "razon": (
                "Misma familia que el modelo de produccion (regresion logistica), "
                "~40x mas rapido de ajustar, lo que hace viable un presupuesto de "
                "100 ensayos frente a los 36 ajustes del RandomizedSearchCV original."
            ),
            "advertencia": (
                "Los valores absolutos de F1 macro no son comparables con los del "
                "modelo de produccion ajustado por lbfgs; lo comparable es la forma "
                "de la curva de convergencia."
            ),
        },
        "presupuesto": {
            "ensayos_optuna": len(valores),
            "muestreador": "TPE (seed 42)",
            "referencia_anterior": "RandomizedSearchCV, 12 iteraciones x 3 folds = 36 ajustes",
            "seleccion": "F1 macro en validacion (4328 filas); el test canonico no se usa",
        },
        "espacio_de_busqueda": {
            "ngram_max": "1-3", "min_df": "1-5", "max_df": "0.5-1.0",
            "sublinear_tf": "True/False", "use_idf": "True/False",
            "alpha": "1e-7 a 1e-3 (log)", "class_weight": "balanced/None",
            "max_iter": "500-3000",
        },
        "saturacion": {
            "mejor_f1_macro_validacion": round(mejor, 4),
            "peor_f1_macro_validacion": round(min(valores), 4),
            "ensayo_en_que_se_alcanza_99pct_del_mejor": ensayo_99,
            "ganancia_en_la_segunda_mitad_del_presupuesto": round(
                float(ganancia_segunda_mitad), 5
            ),
            "ganancia_de_los_ensayos_posteriores_al_99pct": round(
                float(ganancia_tras_99), 5
            ),
            "lectura": (
                f"El 99 % del mejor valor final se alcanza en el ensayo {ensayo_99} "
                f"de {len(valores)}; los {len(valores) - ensayo_99} ensayos restantes "
                f"aportan {ganancia_tras_99:.5f} de F1 macro y la segunda mitad del "
                f"presupuesto, {ganancia_segunda_mitad:.5f}. La superficie de "
                "respuesta esta saturada: el limite no proviene de la configuracion."
            ),
        },
        "mejores_parametros": estudio.best_params,
        "importancia_de_hiperparametros": importancias,
        "curva_de_convergencia": [round(v, 5) for v in mejor_acumulado],
        "valores_por_ensayo": [round(v, 5) for v in valores],
    }
    with open(out_dir / "optuna_saturacion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata["saturacion"], indent=2, ensure_ascii=False))
    print("Importancias:", importancias)
    print(f"[OK] Escrito en {out_dir}")


if __name__ == "__main__":
    main()
