"""
Selecciona el peso del ensamble en VALIDACION y lo evalua una sola vez en test.

POR QUE HACE FALTA ESTE SEGUNDO SCRIPT.
`ensamble_baseline_mrbert.py` barrio 8 configuraciones (7 pesos + rangos), eligio
la mejor POR F1 MACRO EN EL TEST, y despues midio la mejora contra los modelos
individuales EN ESE MISMO TEST. Eso infla la significancia: parte de la ganancia
de +0.0172 es haber elegido el maximo de 8 opciones correlacionadas sobre el
mismo conjunto con el que se evalua. Con una sd por particion de 0.0153, la
ganancia reportada equivale a 1.1 desviaciones, de modo que el sesgo no es
despreciable frente al efecto.

El procedimiento correcto, y el que sostiene una afirmacion en la tesis:
  1. Elegir el peso en el split de VALIDACION (4328 filas), que no se uso para
     evaluar a ninguno de los dos modelos.
  2. Evaluar ESE peso, una sola vez, en el test canonico.
  3. Prueba pareada del peso asi elegido contra cada modelo individual.

Asi la cifra de test es una medicion honesta y no el maximo de un barrido.

QUE NO CAMBIA. Las ganancias en accuracy y top-3 del primer script son amplias
y suaves a lo largo de todo el rango de pesos, no un pico aislado: eso ya
sugiere efecto real y no artefacto de seleccion. Este script lo confirma o lo
desmiente con el procedimiento correcto.

DONDE CORRE: GPUAI (necesita los pesos de MrBERT-es para inferir validacion).
    python Proyecto\\Scripts\\ensamble_peso_en_validacion.py

ESTADO: escrito, SIN EJECUTAR. Reutiliza funciones ya verificadas de
ensamble_baseline_mrbert.py (probas_baseline, probas_mrbert, alinear, metricas,
prueba_pareada).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = REPO_ROOT / "Proyecto"
sys.path.insert(0, str(PROJECT))

_spec = importlib.util.spec_from_file_location(
    "ens", str(PROJECT / "Scripts" / "ensamble_baseline_mrbert.py"))
ens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ens)

import canonical_split as cs  # noqa: E402

OUT_JSON = PROJECT / "artifacts" / "ensamble" / "ensamble_peso_validacion_metadata.json"
LABEL = "DEPENDENCIA DESTINO"
PESOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
METRICA_SELECCION = "f1_macro"


def combinar(P_base, P_mr, w):
    if w == "rangos":
        return (rankdata(P_base, axis=1) + rankdata(P_mr, axis=1)) / 2.0
    return w * P_base + (1 - w) * P_mr


def main():
    print("Reconstruyendo el marco canonico (verifica fingerprints)...")
    df = cs.build_canonical_frame(verify=True)
    valid = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    print(f"  validacion: {len(valid)} filas | test: {len(test)} filas\n")
    assert len(valid) == 4328, f"Se esperaban 4328 filas de validacion, hay {len(valid)}"
    assert len(test) == 4329, f"Se esperaban 4329 filas de test, hay {len(test)}"

    # --- probabilidades en VALIDACION -------------------------------------
    print("Validacion: probabilidades del baseline...")
    Pv_base, clases = ens.probas_baseline(valid["text_baseline"].tolist())
    print("Validacion: probabilidades de MrBERT-es...")
    # canonical_split no construye text_transformer; se arma igual que el
    # notebook 05: "tipo: X [SEP] medio: Y [SEP] asunto: Z".
    txt_v = ("tipo: " + valid["TIPO"].astype(str)
             + " [SEP] medio: " + valid["MEDIO DE ENVIO"].astype(str)
             + " [SEP] asunto: " + valid["ASUNTO"].astype(str)).tolist()
    Pv_mr, cls_mr = ens.probas_mrbert(txt_v)
    Pv_mr = ens.alinear(Pv_mr, cls_mr, clases)
    yv = valid[LABEL].astype(str).values

    print("\n" + "=" * 70)
    print("SELECCION DEL PESO EN VALIDACION (el test no participa)")
    print("=" * 70)
    print(f"{'peso':>10} {'F1 macro':>10} {'BalAcc':>10} {'Accuracy':>10} {'Top-3':>10}")
    filas_val = []
    for w in PESOS + ["rangos"]:
        m = ens.metricas(yv, combinar(Pv_base, Pv_mr, w), clases)
        m["peso"] = str(w)
        filas_val.append(m)
        print(f"{str(w):>10} {m['f1_macro']:>10.4f} {m['balanced_accuracy']:>10.4f} "
              f"{m['accuracy']:>10.4f} {m['top3_accuracy']:>10.4f}")

    mejor_val = max(filas_val, key=lambda r: r[METRICA_SELECCION])
    w_sel = mejor_val["peso"]
    w_sel = "rangos" if w_sel == "rangos" else float(w_sel)
    print(f"\n  Peso elegido por {METRICA_SELECCION} en validacion: {w_sel}")
    print(f"  (en validacion daba {mejor_val[METRICA_SELECCION]:.4f})")

    # --- evaluacion UNICA en TEST ----------------------------------------
    print("\nTest: probabilidades del baseline...")
    Pt_base, clases_t = ens.probas_baseline(test["text_baseline"].tolist())
    assert list(clases_t) == list(clases), "El orden de clases cambio entre splits"
    print("Test: probabilidades de MrBERT-es...")
    txt_t = ("tipo: " + test["TIPO"].astype(str)
             + " [SEP] medio: " + test["MEDIO DE ENVIO"].astype(str)
             + " [SEP] asunto: " + test["ASUNTO"].astype(str)).tolist()
    Pt_mr, cls_mr_t = ens.probas_mrbert(txt_t)
    Pt_mr = ens.alinear(Pt_mr, cls_mr_t, clases)
    yt = test[LABEL].astype(str).values

    m_base = ens.metricas(yt, Pt_base, clases)
    m_mr = ens.metricas(yt, Pt_mr, clases)
    P_ens = combinar(Pt_base, Pt_mr, w_sel)
    m_ens = ens.metricas(yt, P_ens, clases)

    print("\n" + "=" * 70)
    print(f"RESULTADO EN TEST DEL PESO ELEGIDO EN VALIDACION ({w_sel})")
    print("=" * 70)
    print(f"{'':<22}{'F1 macro':>10}{'BalAcc':>10}{'Accuracy':>10}{'Top-3':>10}")
    for nom, m in [("Baseline B", m_base), ("MrBERT-es", m_mr), (f"Ensamble({w_sel})", m_ens)]:
        print(f"{nom:<22}{m['f1_macro']:>10.4f}{m['balanced_accuracy']:>10.4f}"
              f"{m['accuracy']:>10.4f}{m['top3_accuracy']:>10.4f}")

    print("\n" + "=" * 70)
    print("PRUEBA PAREADA (sin sesgo de seleccion: el peso no se eligio aqui)")
    print("=" * 70)
    pruebas = [
        ens.prueba_pareada(yt, P_ens, Pt_base, clases, f"Ensamble({w_sel})", "Baseline B"),
        ens.prueba_pareada(yt, P_ens, Pt_mr, clases, f"Ensamble({w_sel})", "MrBERT-es"),
    ]
    for pr in pruebas:
        print(f"\n  {pr['comparacion']}")
        print(f"    diferencia F1 macro : {pr['diferencia_f1_macro']:+.6f}")
        print(f"    IC95%               : [{pr['ci95_low']:+.6f} , {pr['ci95_high']:+.6f}]")
        print(f"    McNemar p           : {pr['mcnemar_p_value']:.6g}")
        print(f"    significativo       : {'SI' if pr['significativo'] else 'NO'}")

    # El veredicto se emite POR METRICA. Un unico "significativo si y solo si el
    # IC no cruza cero Y McNemar < 0.05" colapsa resultados distintos en uno:
    # el IC evalua el F1 macro (dominado por clases raras y por tanto ruidoso) y
    # McNemar evalua la tasa de acierto top-1 (ponderada por volumen). Que uno
    # de significancia y el otro no es informacion, no contradiccion.
    vs_base = pruebas[0]
    ic_cruza_cero = vs_base["ci95_low"] <= 0 <= vs_base["ci95_high"]
    mcnemar_sig = vs_base["mcnemar_p_value"] < 0.05
    d_acc = m_ens["accuracy"] - m_base["accuracy"]
    d_top3 = m_ens["top3_accuracy"] - m_base["top3_accuracy"]

    lineas = []
    if ic_cruza_cero:
        lineas.append(
            f"F1 MACRO: mejora NO establecida. IC95% "
            f"[{vs_base['ci95_low']:+.6f} , {vs_base['ci95_high']:+.6f}] cruza cero. "
            "El F1 macro promedia las 42 clases por igual, de modo que lo dominan "
            "las clases raras, donde la evidencia es escasa y el estimador ruidoso."
        )
    else:
        lineas.append(
            f"F1 MACRO: mejora establecida. IC95% "
            f"[{vs_base['ci95_low']:+.6f} , {vs_base['ci95_high']:+.6f}] no cruza cero."
        )
    if mcnemar_sig:
        lineas.append(
            f"ACIERTO TOP-1: mejora establecida. McNemar p = "
            f"{vs_base['mcnemar_p_value']:.3g}. Accuracy {d_acc:+.4f}, "
            f"top-3 {d_top3:+.4f}. Son las metricas ponderadas por volumen, que es "
            "lo que el sistema hace en operacion."
        )
    else:
        lineas.append(f"ACIERTO TOP-1: sin evidencia. McNemar p = {vs_base['mcnemar_p_value']:.3g}.")

    if ic_cruza_cero and mcnemar_sig:
        lineas.append(
            "LECTURA CONJUNTA: el ensamble mejora de forma confiable donde hay "
            "volumen (top-1 y top-3) y no de forma demostrable en el promedio "
            "macro. Coherente con que el proyecto declare el top-3 como metrica "
            "operativa. Reportar ambas cosas: afirmar la ganancia en top-3 y "
            "accuracy, y NO afirmar ganancia en F1 macro."
        )
    veredicto = " | ".join(lineas)
    print("\nVEREDICTO POR METRICA:")
    for l in lineas:
        print(f"  - {l}")

    salida = {
        "motivacion": (
            "Corrige el sesgo de seleccion de ensamble_baseline_mrbert.py, que "
            "eligio el peso por F1 macro en el mismo test con el que midio la mejora."
        ),
        "procedimiento": (
            "Peso elegido en el split de validacion (4328 filas, no usado para "
            "evaluar ninguno de los dos modelos); evaluacion unica en el test "
            "canonico (4329 filas)."
        ),
        "metrica_de_seleccion": METRICA_SELECCION,
        "barrido_en_validacion": filas_val,
        "peso_elegido": str(w_sel),
        "test": {
            "baseline_b": m_base,
            "mrbert_es": m_mr,
            f"ensamble_{w_sel}": m_ens,
        },
        "pruebas_pareadas": pruebas,
        "veredicto": veredicto,
        "nota_produccion": (
            "Cualquiera sea el resultado, el ensamble no entra a produccion: "
            "exige correr el encoder en GPU y anula el argumento de costo por el "
            "que se selecciono el baseline. Su valor es de evidencia."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {OUT_JSON}")


if __name__ == "__main__":
    main()
