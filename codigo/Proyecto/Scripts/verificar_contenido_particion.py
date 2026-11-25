"""Verifica que el conjunto de prueba canónico tiene el mismo CONTENIDO en las dos copias
del archivo fuente, no solo los mismos índices.

Las huellas SHA-256 de la tesis (tabla de huellas, sección 7.1) se calculan sobre los
índices ordenados de cada partición. Una huella de índices garantiza las mismas posiciones
del conjunto consolidado, pero no por sí sola el mismo texto ni la misma etiqueta, y el
archivo fuente tiene dos huellas distintas según la copia (fe380fc5… en el servidor GPU y
17af0cd4… en los demás entornos). Este script cierra esa brecha con artefactos existentes:

1. Reconstruye la partición canónica desde la copia local del archivo fuente
   (``canonical_split.build_canonical_frame``) y compara, registro por registro, el texto
   de entrada del modelo lineal, la etiqueta y el año del conjunto de prueba con los que el
   servidor GPU guardó junto a MrBERT-es (``artifacts/mrbert_es_final/test_indices.parquet``).
2. Compara la etiqueta y el año por observación entre las predicciones del modelo lineal
   (entorno Linux), de MrBERT-es y de Qwen2.5-3B (servidor GPU).

No entrena ni reevalúa ningún modelo. Escribe el resultado en
``artifacts/verificacion_contenido/verificacion_contenido_test.json``.

Uso, desde ``Proyecto/`` (necesita sklearn y pyarrow; ``.venv-app`` tiene sklearn y pyarrow
puede instalarse aparte con ``pip install --target <carpeta>`` y ``PYTHONPATH``):

    ..\\.venv-app\\Scripts\\python.exe Scripts/verificar_contenido_particion.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

import canonical_split as cs  # noqa: E402

ART = PROYECTO / "artifacts"
SALIDA = ART / "verificacion_contenido" / "verificacion_contenido_test.json"
HUELLA_TEST = "9b363104e8d1621fcda0b80d30d0c1fa2545dcfce13566d703adaa0adaaa5a64"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def iguales(a: pd.Series, b: pd.Series) -> int:
    return int((a.astype(str).to_numpy() == b.astype(str).to_numpy()).sum())


def main() -> int:
    resultado: dict = {
        "fecha": date.today().isoformat(),
        "archivo_fuente_local": cs.DATA_PATH.name,
        "sha256_archivo_fuente_local": sha256_archivo(cs.DATA_PATH),
        "huella_indices_test_esperada": HUELLA_TEST,
    }

    # 1. Copia local reconstruida frente a la copia del servidor GPU.
    marco = cs.build_canonical_frame(verify=True)
    local = marco[marco["split"] == "test"].copy()
    local["original_index"] = local.index
    resultado["huella_indices_test_local"] = cs.split_fingerprint(local)
    local = local.set_index("original_index")

    servidor = pd.read_parquet(ART / "mrbert_es_final" / "test_indices.parquet").set_index("original_index")
    comunes = local.index.intersection(servidor.index)
    n = len(servidor)
    resultado["copia_local_vs_servidor_gpu"] = {
        "registros_test_servidor": n,
        "registros_test_local": len(local),
        "indices_en_comun": len(comunes),
        "texto_entrada_modelo_lineal_igual": iguales(local.loc[comunes, "text_baseline"], servidor.loc[comunes, "text_baseline"]),
        "etiqueta_igual": iguales(local.loc[comunes, "DEPENDENCIA DESTINO"], servidor.loc[comunes, "DEPENDENCIA DESTINO"]),
        "anio_igual": iguales(local.loc[comunes, "anio"], servidor.loc[comunes, "anio"]),
    }

    # 2. Etiquetas por observación entre los archivos de predicciones de cada modelo.
    lineal = pd.read_parquet(ART / "baseline_b_predicciones_test_4329.parquet").set_index("original_index")
    otros = {
        "mrbert_es_servidor_gpu": ART / "mrbert_es_final" / "mrbert_es_predicciones_test_4329.parquet",
        "qwen25_3b_servidor_gpu": ART / "qwen_slm_lora" / "predicciones_test_4329.parquet",
    }
    resultado["etiquetas_por_observacion_vs_lineal"] = {}
    for nombre, ruta in otros.items():
        d = pd.read_parquet(ruta).set_index("original_index")
        c = lineal.index.intersection(d.index)
        resultado["etiquetas_por_observacion_vs_lineal"][nombre] = {
            "indices_en_comun": len(c),
            "y_true_igual": iguales(lineal.loc[c, "y_true"], d.loc[c, "y_true"]),
            "anio_igual": iguales(lineal.loc[c, "anio"], d.loc[c, "anio"]),
        }

    cvs = resultado["copia_local_vs_servidor_gpu"]
    ok = (
        resultado["huella_indices_test_local"] == HUELLA_TEST
        and cvs["indices_en_comun"] == n == cvs["registros_test_local"]
        and cvs["texto_entrada_modelo_lineal_igual"] == n
        and cvs["etiqueta_igual"] == n
        and cvs["anio_igual"] == n
        and all(v["indices_en_comun"] == n and v["y_true_igual"] == n and v["anio_igual"] == n
                for v in resultado["etiquetas_por_observacion_vs_lineal"].values())
    )
    resultado["veredicto"] = (
        "El conjunto de prueba tiene el mismo contenido (texto de entrada, etiqueta y año) en las dos "
        "copias del archivo fuente, y las etiquetas por observación coinciden entre los tres modelos."
        if ok else "HAY DIFERENCIAS: revisar los conteos."
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
