"""
Ensamble Baseline B + MrBERT-es por promedio de probabilidades.

POR QUE AHORA SI TIENE SENTIDO. El marco de proceso adoptado (Zolbanin y
Aubert, 2025) pide evaluar metodos de ensamble. Hasta el 23/08/2026 era un
tramite; la prueba pareada le dio motivo concreto: los dos modelos tienen
exactitud equivalente (IC95% de la diferencia [-0.0057 , 0.0407], McNemar
p = 0.414) pero en 662 de 4329 casos (15.3%) solo uno de los dos acierta. Exactitud comparable
con errores decorrelacionados es justamente la condicion en que promediar
probabilidades puede superar a ambos.

ENCUADRE: ESTE ENSAMBLE NO ES CANDIDATO A PRODUCCION.
Requiere ejecutar los dos modelos, incluido el encoder en GPU, lo que anula el
argumento por el que se selecciono el baseline (costo operativo ante desempeno
equivalente). Su valor es metodologico: mide cuanta senal adicional queda por
explotar combinando arquitecturas. Si el ensamble tampoco despega, es evidencia
fuerte de que el techo es del dataset y no del modelado, que es la tesis
central del trabajo. Anticipar la pregunta del jurado "y por que no
desplegaron el ensamble" con esta respuesta, no con silencio.

QUE CALCULA
  1. Probabilidades completas (42 clases) de ambos modelos sobre el test canonico.
  2. Cota oraculo: si un juez perfecto eligiera, caso por caso, el modelo que
     acierta, cuanto se lograria. Es el techo de CUALQUIER combinacion de estos
     dos modelos. Si esa cota es baja, ningun ensamble va a rescatar mucho.
  3. Ensamble por promedio ponderado, barriendo el peso del baseline.
  4. Ensamble por promedio de rangos, robusto a que los dos modelos tengan
     escalas de probabilidad distintas (el baseline usa class_weight=balanced y
     el encoder loss ponderada: no hay razon para que esten calibrados igual).
  5. Prueba pareada del mejor ensamble contra cada modelo individual. Sin esto
     estariamos repitiendo el error que la prueba pareada acaba de corregir:
     leer una diferencia de metricas puntuales como si fuera una mejora.

DONDE CORRE: GPUAI (necesita los pesos de MrBERT-es).
    python Proyecto\\Scripts\\ensamble_baseline_mrbert.py

ESTADO: parcialmente verificado.
  VERIFICADO por ejecucion (sin GPU): probas_baseline() reproduce las metricas
  publicadas del modelo de produccion; alinear() reordena por nombre de clase y
  aborta si los conjuntos de etiquetas difieren; metricas(), prueba_pareada(),
  promedio de rangos y cota oraculo corren y dan valores coherentes.
  NO VERIFICADO: probas_mrbert(), que necesita los pesos del encoder.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import chi2, rankdata
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = REPO_ROOT / "Proyecto"
ARTIFACTS = PROJECT / "artifacts"

BASELINE_JOBLIB = ARTIFACTS / "baseline_b_tfidf_lr_final.joblib"
MRBERT_MODEL_DIR = ARTIFACTS / "mrbert_es_final" / "model"
TEST_INDICES = ARTIFACTS / "mrbert_es_final" / "test_indices.parquet"

OUT_DIR = ARTIFACTS / "ensamble"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "ensamble_metadata.json"
OUT_PROBS = OUT_DIR / "probabilidades_test_4329.npz"

MAX_LEN = 256
BATCH = 32
N_BOOT = 2000
SEED = 42
PESOS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]  # peso del baseline

# Metricas publicadas, para verificar que las probabilidades reconstruidas
# corresponden a los modelos de los que habla la tesis.
CANONICO_BASELINE = {"f1_macro": 0.5455370598338695, "accuracy": 0.705936705936706}
CANONICO_MRBERT = {"f1_macro": 0.5287, "accuracy": 0.7117}


def metricas(y_true, proba, clases):
    pred = clases[proba.argmax(1)]
    top3 = np.argsort(-proba, axis=1)[:, :3]
    acierto3 = float(np.mean([y_true[i] in clases[top3[i]] for i in range(len(y_true))]))
    return {
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "top3_accuracy": acierto3,
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }


def prueba_pareada(y_true, proba_a, proba_b, clases, nombre_a, nombre_b, seed=SEED):
    """Bootstrap pareado de la diferencia de F1 macro + McNemar sobre top-1."""
    rng = np.random.default_rng(seed)
    etiquetas = np.unique(y_true)
    pred_a = clases[proba_a.argmax(1)]
    pred_b = clases[proba_b.argmax(1)]
    n = len(y_true)

    punto = (f1_score(y_true, pred_a, labels=etiquetas, average="macro", zero_division=0)
             - f1_score(y_true, pred_b, labels=etiquetas, average="macro", zero_division=0))
    difs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        difs[i] = (
            f1_score(y_true[idx], pred_a[idx], labels=etiquetas, average="macro", zero_division=0)
            - f1_score(y_true[idx], pred_b[idx], labels=etiquetas, average="macro", zero_division=0)
        )
    lo, hi = np.percentile(difs, [2.5, 97.5])

    ok_a = y_true == pred_a
    ok_b = y_true == pred_b
    b_ = int(np.sum(ok_a & ~ok_b))
    c_ = int(np.sum(~ok_a & ok_b))
    if b_ + c_ > 0:
        stat = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
        pval = float(1 - chi2.cdf(stat, df=1))
    else:
        stat, pval = 0.0, 1.0

    cruza_cero = lo <= 0 <= hi
    return {
        "comparacion": f"{nombre_a} vs {nombre_b}",
        "diferencia_f1_macro": round(float(punto), 6),
        "ci95_low": round(float(lo), 6),
        "ci95_high": round(float(hi), 6),
        "mcnemar_p_value": pval,
        # Corregido el 26/09/2026: antes se guardaban invertidas (a_gana recibia c_).
        "discordantes_a_gana": b_,  # solo acierta a
        "discordantes_b_gana": c_,  # solo acierta b
        "significativo": bool((not cruza_cero) and pval < 0.05),
    }


def probas_baseline(textos):
    modelo = joblib.load(BASELINE_JOBLIB)
    # softmax(decision_function) == predict_proba multinomial, y evita depender
    # del atributo multi_class, ausente en sklearn >= 1.8.
    P = softmax(modelo.decision_function(textos), axis=1)
    return P, np.array(modelo.classes_, dtype=object)


def probas_mrbert(textos):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  dispositivo: {device}")
    tok = AutoTokenizer.from_pretrained(str(MRBERT_MODEL_DIR))
    modelo = AutoModelForSequenceClassification.from_pretrained(str(MRBERT_MODEL_DIR)).to(device)
    modelo.eval()
    id2label = {int(k): v for k, v in modelo.config.id2label.items()}
    clases = np.array([id2label[i] for i in range(len(id2label))], dtype=object)

    salida = []
    with torch.no_grad():
        for ini in range(0, len(textos), BATCH):
            enc = tok(textos[ini:ini + BATCH], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            salida.append(torch.softmax(modelo(**enc).logits, dim=-1).cpu().numpy())
            if (ini // BATCH) % 20 == 0:
                print(f"    {min(ini + BATCH, len(textos))}/{len(textos)}")
    return np.vstack(salida), clases


def alinear(P, clases_origen, clases_destino):
    """Reordena columnas por NOMBRE de clase, no por posicion. Los dos modelos
    resultaron tener el mismo orden alfabetico, pero asumirlo en vez de
    verificarlo es como se producen errores silenciosos que invalidan todo."""
    assert set(clases_origen) == set(clases_destino), (
        "Los conjuntos de etiquetas difieren entre modelos: "
        f"solo en origen {sorted(set(clases_origen) - set(clases_destino))}, "
        f"solo en destino {sorted(set(clases_destino) - set(clases_origen))}"
    )
    pos = {c: i for i, c in enumerate(clases_origen)}
    return P[:, [pos[c] for c in clases_destino]]


def main():
    for ruta in (BASELINE_JOBLIB, MRBERT_MODEL_DIR, TEST_INDICES):
        if not ruta.exists():
            raise SystemExit(
                f"Falta {ruta}.\n"
                "Este script necesita los pesos de MrBERT-es: correr en GPUAI."
            )

    test = pd.read_parquet(TEST_INDICES)
    assert len(test) == 4329, f"Se esperaban 4329 filas, hay {len(test)}"
    y = test["DEPENDENCIA DESTINO"].astype(str).values

    print("Probabilidades del baseline (CPU)...")
    P_base, cls_base = probas_baseline(test["text_baseline"].tolist())
    print("Probabilidades de MrBERT-es...")
    P_mr, cls_mr = probas_mrbert(test["text_transformer"].tolist())

    # Orden canonico de clases: el del baseline (alfabetico).
    clases = cls_base
    P_mr = alinear(P_mr, cls_mr, clases)
    print(f"Etiquetas alineadas: {len(clases)} clases.\n")

    m_base = metricas(y, P_base, clases)
    m_mr = metricas(y, P_mr, clases)

    print("Verificacion contra metricas publicadas:")
    print(f"  baseline F1 {m_base['f1_macro']:.6f} vs {CANONICO_BASELINE['f1_macro']:.6f} "
          f"-> {'OK' if abs(m_base['f1_macro']-CANONICO_BASELINE['f1_macro'])<5e-3 else 'DIVERGE'}")
    print(f"  MrBERT   F1 {m_mr['f1_macro']:.6f} vs ~{CANONICO_MRBERT['f1_macro']:.4f} "
          f"-> {'OK' if abs(m_mr['f1_macro']-CANONICO_MRBERT['f1_macro'])<1e-2 else 'DIVERGE'}")
    if abs(m_base["f1_macro"] - CANONICO_BASELINE["f1_macro"]) >= 5e-3:
        raise SystemExit("El baseline no reproduce su metrica publicada. Detener.")
    print()

    # ---------------------------------------------------------------- oraculo
    ok_base = y == clases[P_base.argmax(1)]
    ok_mr = y == clases[P_mr.argmax(1)]
    oraculo_acc = float(np.mean(ok_base | ok_mr))
    ambos_fallan = float(np.mean(~ok_base & ~ok_mr))
    discordantes = int(np.sum(ok_base ^ ok_mr))

    print("=" * 70)
    print("COTA ORACULO (techo de cualquier combinacion de estos dos modelos)")
    print("=" * 70)
    print(f"  accuracy baseline:            {m_base['accuracy']:.4f}")
    print(f"  accuracy MrBERT-es:           {m_mr['accuracy']:.4f}")
    print(f"  accuracy oraculo (el mejor por caso): {oraculo_acc:.4f}")
    print(f"  margen que deja el oraculo:   +{oraculo_acc - max(m_base['accuracy'], m_mr['accuracy']):.4f}")
    print(f"  casos donde AMBOS fallan:     {ambos_fallan:.4f} ({int(ambos_fallan*len(y))} casos)")
    print(f"  casos discordantes:           {discordantes} ({100*discordantes/len(y):.1f}%)")
    print()
    print("  Lectura: el oraculo es un techo INALCANZABLE (requiere saber la")
    print("  respuesta para elegir el modelo). Si el margen es pequeno, ningun")
    print("  ensamble puede mejorar mucho. Si ambos fallan en una fraccion alta,")
    print("  esa fraccion es irreducible combinando SOLO estos dos modelos.")
    print()

    # ------------------------------------------------------------- ensambles
    print("=" * 70)
    print("ENSAMBLE POR PROMEDIO PONDERADO DE PROBABILIDADES")
    print("=" * 70)
    print(f"{'peso_base':>10} {'F1 macro':>10} {'BalAcc':>10} {'Accuracy':>10} {'Top-3':>10}")
    resultados = []
    for w in PESOS:
        P = w * P_base + (1 - w) * P_mr
        m = metricas(y, P, clases)
        m["peso_baseline"] = w
        resultados.append(m)
        print(f"{w:>10.2f} {m['f1_macro']:>10.4f} {m['balanced_accuracy']:>10.4f} "
              f"{m['accuracy']:>10.4f} {m['top3_accuracy']:>10.4f}")

    # Promedio de rangos: inmune a diferencias de calibracion entre modelos
    R = (rankdata(P_base, axis=1) + rankdata(P_mr, axis=1)) / 2.0
    m_rank = metricas(y, R, clases)
    print(f"{'rangos':>10} {m_rank['f1_macro']:>10.4f} {m_rank['balanced_accuracy']:>10.4f} "
          f"{m_rank['accuracy']:>10.4f} {m_rank['top3_accuracy']:>10.4f}")
    print()
    print("  Nota de calibracion: el baseline se entreno con class_weight=balanced")
    print("  y el encoder con loss ponderada; no hay razon para que sus")
    print("  probabilidades esten en la misma escala. Por eso se reporta tambien")
    print("  el promedio de rangos, que solo usa el orden.")
    print()

    intermedios = [r for r in resultados if 0 < r["peso_baseline"] < 1]
    mejor = max(intermedios + [dict(m_rank, peso_baseline="rangos")],
                key=lambda r: r["f1_macro"])
    w_mejor = mejor["peso_baseline"]
    P_mejor = R if w_mejor == "rangos" else w_mejor * P_base + (1 - w_mejor) * P_mr

    # --------------------------------------------------------------- pareado
    print("=" * 70)
    print(f"PRUEBA PAREADA DEL MEJOR ENSAMBLE (peso={w_mejor}) CONTRA CADA MODELO")
    print("=" * 70)
    pruebas = [
        prueba_pareada(y, P_mejor, P_base, clases, f"Ensamble({w_mejor})", "Baseline B"),
        prueba_pareada(y, P_mejor, P_mr, clases, f"Ensamble({w_mejor})", "MrBERT-es"),
    ]
    for pr in pruebas:
        print(f"\n  {pr['comparacion']}")
        print(f"    diferencia F1 macro : {pr['diferencia_f1_macro']:+.6f}")
        print(f"    IC95%               : [{pr['ci95_low']:+.6f} , {pr['ci95_high']:+.6f}]")
        print(f"    McNemar p           : {pr['mcnemar_p_value']:.6f}")
        print(f"    significativo       : {'SI' if pr['significativo'] else 'NO'}")

    alguno_significativo = any(p["significativo"] and p["diferencia_f1_macro"] > 0 for p in pruebas)
    if alguno_significativo:
        veredicto = (
            "El ensamble mejora de forma establecida sobre al menos uno de los "
            "modelos. Reportarlo en la tesis como evidencia de complementariedad "
            "entre arquitecturas. NO cambia el modelo de produccion: el ensamble "
            "exige GPU y anula el argumento de costo por el que se eligio el "
            "baseline. Documentar esa decision explicitamente."
        )
    else:
        veredicto = (
            "El ensamble NO mejora de forma establecida sobre los modelos "
            "individuales. Es el resultado mas informativo para la tesis: dos "
            "arquitecturas distintas, con errores parcialmente decorrelacionados, "
            "combinadas, siguen sin despegar. Refuerza que el techo (~0.55 de F1 "
            "macro) es del dataset y no del modelado. Cumple la exigencia del "
            "marco de evaluar ensambles, y se descarta CON argumento y evidencia."
        )
    print()
    print(f"VEREDICTO: {veredicto}")

    np.savez_compressed(
        OUT_PROBS,
        original_index=test["original_index"].values,
        y_true=y.astype(str),
        clases=clases.astype(str),
        proba_baseline=P_base.astype(np.float32),
        proba_mrbert=P_mr.astype(np.float32),
    )

    salida = {
        "encuadre": (
            "Ensamble INFORMATIVO, no candidato a produccion: requiere GPU y "
            "anula el argumento de costo por el que se selecciono el baseline."
        ),
        "motivacion": (
            "Exactitud equivalente (prueba pareada: IC95% [-0.0057, 0.0407], "
            f"McNemar p=0.414) con {discordantes} casos discordantes de {len(y)} "
            "-> errores decorrelacionados, condicion ideal para ensamblar."
        ),
        "modelos_individuales": {"baseline_b": m_base, "mrbert_es": m_mr},
        "cota_oraculo": {
            "accuracy_oraculo": round(oraculo_acc, 6),
            "margen_sobre_el_mejor": round(oraculo_acc - max(m_base["accuracy"], m_mr["accuracy"]), 6),
            "fraccion_ambos_fallan": round(ambos_fallan, 6),
            "casos_discordantes": discordantes,
            "nota": "Techo inalcanzable; acota lo que cualquier combinacion podria lograr.",
        },
        "barrido_pesos": resultados,
        "promedio_rangos": m_rank,
        "mejor_ensamble": {"peso_baseline": str(w_mejor), **{k: v for k, v in mejor.items() if k != "peso_baseline"}},
        "pruebas_pareadas": pruebas,
        "veredicto": veredicto,
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print()
    print(f"Guardado: {OUT_JSON}")
    print(f"Guardado: {OUT_PROBS}  (matrices completas de 42 clases, reutilizables)")


if __name__ == "__main__":
    main()
