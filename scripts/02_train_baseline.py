# -*- coding: utf-8 -*-
"""
Baseline ML: predizione del valore di compravendita OMI (€/m²).

- Legge da omi.db (vista valori_split, split temporale 60/20/20).
- Predice Compr_min e Compr_max (=> range) con due HistGradientBoostingRegressor
  su scala log; il valore puntuale è il punto medio delle due predizioni.
- Target encoding (calcolato solo sul train) per Comune e Zona OMI.
- Valuta su val e test: MAE, MAPE, R2 e copertura del range.

Uso:  python scripts/02_train_baseline.py
Output: models/hgb_compr_min.joblib, models/hgb_compr_max.joblib, models/target_encodings.joblib
"""
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "omi.db"
MODELS_DIR = ROOT / "models"

# "Zona" esclusa: cardinalità 389 > limite 255 di HGB; è già coperta da Fascia + te_LinkZona
CAT_COLS = ["Prov", "Fascia", "Cod_Tip", "Stato"]
TE_COLS = ["Comune_amm", "LinkZona"]  # alta cardinalità -> target encoding
NUM_COLS = ["anno", "semestre"]


def load_data() -> pd.DataFrame:
    cols = ", ".join(f'"{c}"' for c in CAT_COLS + TE_COLS)
    query = f"""
        SELECT {cols}, anno, semestre, "Compr_min", "Compr_max", split
        FROM valori_split
        WHERE "Compr_min" IS NOT NULL AND "Compr_max" IS NOT NULL
          AND "Compr_min" > 0 AND "Compr_max" >= "Compr_min"
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(query, conn)
    for c in CAT_COLS:
        df[c] = df[c].fillna("NA").astype("category")
    return df


def fit_target_encoding(train: pd.DataFrame, col: str, y_log: np.ndarray):
    te = pd.Series(y_log, index=train.index).groupby(train[col]).mean()
    return te, float(y_log.mean())


def apply_target_encoding(df: pd.DataFrame, col: str, te: pd.Series, default: float):
    return df[col].map(te).fillna(default).astype("float64")


def report(name: str, y_min, y_max, p_min, p_max) -> None:
    y_mid, p_mid = (y_min + y_max) / 2, (p_min + p_max) / 2
    coverage = float(np.mean((y_mid >= p_min) & (y_mid <= p_max)))
    print(f"  [{name}] valore puntuale (punto medio, €/m²): "
          f"MAE={mean_absolute_error(y_mid, p_mid):.0f}  "
          f"MAPE={mean_absolute_percentage_error(y_mid, p_mid):.1%}  "
          f"R2={r2_score(y_mid, p_mid):.3f}")
    print(f"  [{name}] range: copertura del valore reale nel range predetto = {coverage:.1%}  "
          f"(ampiezza media predetta {np.mean(p_max - p_min):.0f} €/m², "
          f"reale {np.mean(y_max - y_min):.0f} €/m²)")


def main() -> None:
    t0 = time.time()
    print("Caricamento dati da omi.db...")
    df = load_data()
    print(f"  {len(df):,} righe utilizzabili "
          f"(train {sum(df.split == 'train'):,} / val {sum(df.split == 'val'):,} / "
          f"test {sum(df.split == 'test'):,})")

    parts = {s: df[df.split == s].copy() for s in ("train", "val", "test")}
    train = parts["train"]

    # target encoding leakage-safe: statistiche solo dal train
    y_log_mid = np.log((train["Compr_min"] + train["Compr_max"]) / 2).to_numpy()
    encodings = {}
    for col in TE_COLS:
        te, default = fit_target_encoding(train, col, y_log_mid)
        encodings[col] = (te, default)
        for p in parts.values():
            p[f"te_{col}"] = apply_target_encoding(p, col, te, default)

    feat_cols = CAT_COLS + [f"te_{c}" for c in TE_COLS] + NUM_COLS
    cat_idx = [feat_cols.index(c) for c in CAT_COLS]

    models = {}
    for target in ("Compr_min", "Compr_max"):
        print(f"\nTraining HistGradientBoosting per {target} (scala log)...")
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.1, max_depth=None,
            categorical_features=cat_idx, early_stopping=False, random_state=42,
        )
        model.fit(train[feat_cols], np.log(train[target]))
        models[target] = model

    print("\n=== Valutazione ===")
    for split in ("val", "test"):
        p = parts[split]
        p_min = np.exp(models["Compr_min"].predict(p[feat_cols]))
        p_max = np.exp(models["Compr_max"].predict(p[feat_cols]))
        # coerenza del range predetto
        p_min, p_max = np.minimum(p_min, p_max), np.maximum(p_min, p_max)
        print(f"\n{split.upper()} ({p['anno'].min()}/{p['semestre'].iloc[0]} -> {p['anno'].max()}):")
        report(split, p["Compr_min"].to_numpy(), p["Compr_max"].to_numpy(), p_min, p_max)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(models["Compr_min"], MODELS_DIR / "hgb_compr_min.joblib")
    joblib.dump(models["Compr_max"], MODELS_DIR / "hgb_compr_max.joblib")
    joblib.dump(encodings, MODELS_DIR / "target_encodings.joblib")
    print(f"\nModelli salvati in {MODELS_DIR}  ({time.time() - t0:.0f}s totali)")


if __name__ == "__main__":
    main()
