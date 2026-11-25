"""Reconstruccion del split canonico del proyecto.

Este modulo reproduce **exactamente** el pipeline de preparacion de datos usado
en `05_mrbert_es_c6_final.ipynb` y en `Scripts/export_baseline_b_final.py`:

- limpieza y normalizacion de acentos;
- consolidacion de duplicados por `TIPO + ASUNTO` con `moda_estable`;
- filtro de longitud minima y de clases con al menos 20 ejemplos;
- split 80/10/10 estratificado con `SEED=42`.

A diferencia del script de exportacion, aqui no se entrena nada: el objetivo es
obtener el marco consolidado con una columna `split` que indique si cada caso
pertenece a `train`, `validation` o `test`.

Esto habilita dos garantias metodologicas para la fase HITL:

1. **Sin fuga de datos.** La cola de auditoria puede restringirse al split de
   test (4329 filas held-out), de modo que las tasas de aceptacion medidas con
   validadores humanos sean atribuibles a generalizacion y no a memorizacion.
2. **Sin duplicados.** El marco ya viene consolidado por `TIPO + ASUNTO`, de
   modo que un validador nunca revisa dos veces el mismo caso.

La correccion de la reconstruccion se verifica comparando los fingerprints
SHA256 de los tres splits contra la metadata del modelo de produccion. Si algun
fingerprint no coincide, el modulo falla de forma ruidosa en lugar de entregar
un split silenciosamente distinto al que se uso para entrenar.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_PATH = REPO_ROOT / "data" / "Registro_PQRS_Mindeportes_20260328.csv"
METADATA_PATH = (
    PROJECT_DIR / "artifacts" / "baseline_b_tfidf_lr_final_metadata.json"
)

SEED = 42
MIN_VALID_YEAR = 2000
USE_MEDIO = True
MIN_EXAMPLES_PER_CLASS = 20
LABEL_COLUMN = "DEPENDENCIA DESTINO"

_CACHE: pd.DataFrame | None = None


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


def _consolidar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza y consolidacion por TIPO+ASUNTO."""
    current_year = datetime.now().year

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
        df["anio_radicado"].between(MIN_VALID_YEAR, current_year)
    )
    anio_fecha = df["anio_fecha"].where(
        df["anio_fecha"].between(MIN_VALID_YEAR, current_year)
    )
    df["anio"] = anio_radicado.fillna(anio_fecha)

    work = df.dropna(subset=["TIPO", "ASUNTO", LABEL_COLUMN]).copy()
    work["TIPO"] = work["TIPO"].apply(normalizar_texto)
    work["ASUNTO"] = work["ASUNTO"].apply(normalizar_texto)
    work[LABEL_COLUMN] = work[LABEL_COLUMN].apply(normalizar_texto)

    if "MEDIO DE ENVIO" in df.columns:
        work["MEDIO DE ENVIO"] = (
            work["MEDIO DE ENVIO"].fillna("NO_REPORTADO").apply(normalizar_texto)
        )
    else:
        work["MEDIO DE ENVIO"] = "NO_REPORTADO"

    label_prioridad = work[LABEL_COLUMN].value_counts()
    work = (
        work.groupby(["TIPO", "ASUNTO"], dropna=False)
        .agg(
            {
                LABEL_COLUMN: lambda values: moda_estable(values, label_prioridad),
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

    class_counts = work[LABEL_COLUMN].value_counts()
    valid_classes = class_counts[class_counts >= MIN_EXAMPLES_PER_CLASS].index
    work = work[work[LABEL_COLUMN].isin(valid_classes)].copy()

    label_list = sorted(work[LABEL_COLUMN].unique())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    work["label"] = work[LABEL_COLUMN].map(label2id).astype(int)
    return work


def build_canonical_frame(verify: bool = True) -> pd.DataFrame:
    """Devuelve el marco consolidado con la columna `split`.

    Si `verify` es True (por defecto) valida los fingerprints de los tres splits
    contra `baseline_b_tfidf_lr_final_metadata.json` y lanza `RuntimeError` si
    alguno no coincide.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset base: {DATA_PATH}")

    work = _consolidar(read_csv_with_fallback(DATA_PATH))

    split_cols = ["text_baseline", "label", LABEL_COLUMN, "anio", "fecha_dt"]
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

    if verify:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"No existe la metadata final: {METADATA_PATH}")
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        expected = metadata.get("split_fingerprints_sha256", {})
        if fingerprints != expected:
            raise RuntimeError(
                "El split reconstruido NO coincide con el split canonico del "
                "modelo de produccion. No se puede garantizar que los casos "
                "marcados como 'test' sean held-out.\n"
                f"  reconstruido: {fingerprints}\n"
                f"  esperado    : {expected}"
            )

    work["split"] = "train"
    work.loc[valid_df.index, "split"] = "validation"
    work.loc[test_df.index, "split"] = "test"

    _CACHE = work
    return work


def split_counts() -> dict[str, int]:
    frame = build_canonical_frame()
    return frame["split"].value_counts().to_dict()


if __name__ == "__main__":
    frame = build_canonical_frame()
    print("Split canonico reconstruido y verificado contra la metadata final.")
    print(f"Filas consolidadas (sin duplicados TIPO+ASUNTO): {len(frame)}")
    for name, count in sorted(split_counts().items()):
        print(f"  {name:11s}: {count}")
    print(f"Clases: {frame['label'].nunique()}")
