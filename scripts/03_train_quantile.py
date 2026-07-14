# -*- coding: utf-8 -*-
"""
Modello quantile con lag features: stima del valore di compravendita (€/m²)
con incertezza esplicita (range calibrato), pensato per il nowcasting.

Idea: le quotazioni OMI cambiano lentamente -> il modello riceve l'ultima
quotazione nota della stessa (zona, tipologia, stato) e il trend provinciale,
e predice il semestre successivo. Tre modelli quantile che predicono il
DELTA log rispetto all'ultima quotazione (non il livello: HGB binnizza le
feature in 255 bin e sul livello satura nella coda alta dei prezzi):
  - q10 su delta di Compr_min   (estremo basso prudente)
  - q50 su delta del punto medio (valore puntuale)
  - q90 su delta di Compr_max   (estremo alto prudente)
La predizione finale è exp(lag + delta_predetto).

Uso:  python scripts/03_train_quantile.py
Output: models/quantile_bundle.joblib
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

KEY = ["LinkZona", "Cod_Tip", "Stato"]
CAT_COLS = ["Prov", "Fascia", "Cod_Tip", "Stato"]
NUM_COLS = ["semestre", "lag1_log_mid", "lag1_log_width", "delta1", "lag_gap", "delta_prov"]
FEAT_COLS = CAT_COLS + NUM_COLS
MAX_LAG_GAP = 4  # scarta righe la cui ultima quotazione nota è più vecchia di 4 semestri

# nome -> (quantile, colonna target, colonna lag da cui calcolare/riaggiungere il delta)
QUANTILES = {"q10_min": (0.10, "log_min", "lag1_log_min"),
             "q50_mid": (0.50, "log_mid", "lag1_log_mid"),
             "q90_max": (0.90, "log_max", "lag1_log_max")}


def load_data() -> pd.DataFrame:
    query = """
        SELECT periodo, semestre, "Prov", "Fascia", "Cod_Tip", "Stato",
               "LinkZona", "Compr_min", "Compr_max", split
        FROM valori_split
        WHERE "Compr_min" IS NOT NULL AND "Compr_max" IS NOT NULL
          AND "Compr_min" > 0 AND "Compr_max" >= "Compr_min"
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(query, conn)
    df["Stato"] = df["Stato"].fillna("NA")
    for c in ("Prov", "Fascia", "Cod_Tip"):
        df[c] = df[c].fillna("NA")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    periods = np.sort(df["periodo"].unique())
    pidx = {p: i for i, p in enumerate(periods)}
    df["period_idx"] = df["periodo"].map(pidx)

    df["log_min"] = np.log(df["Compr_min"])
    df["log_max"] = np.log(df["Compr_max"])
    df["log_mid"] = np.log((df["Compr_min"] + df["Compr_max"]) / 2)
    df["log_width"] = df["log_max"] - df["log_min"]

    # se la stessa chiave compare più volte nello stesso periodo, media (raro)
    df = (df.groupby(KEY + ["periodo", "period_idx", "semestre", "Prov", "Fascia", "split"],
                     as_index=False)
            .agg(log_min=("log_min", "mean"), log_mid=("log_mid", "mean"),
                 log_max=("log_max", "mean"), log_width=("log_width", "mean")))

    df = df.sort_values(KEY + ["period_idx"], kind="stable").reset_index(drop=True)
    g = df.groupby(KEY, sort=False)
    df["lag1_log_mid"] = g["log_mid"].shift(1)
    df["lag1_log_min"] = g["log_min"].shift(1)
    df["lag1_log_max"] = g["log_max"].shift(1)
    df["lag1_log_width"] = g["log_width"].shift(1)
    df["lag2_log_mid"] = g["log_mid"].shift(2)
    df["delta1"] = df["lag1_log_mid"] - df["lag2_log_mid"]
    df["lag_gap"] = df["period_idx"] - g["period_idx"].shift(1)

    # trend provinciale noto al momento della predizione: media(T-1) - media(T-2)
    prov = (df.groupby(["Prov", "period_idx"])["log_mid"].mean()
              .rename("prov_mean").reset_index())
    prov["delta_prov"] = prov.groupby("Prov")["prov_mean"].diff()
    prov["period_idx"] += 1  # il delta calcolato fino a T-1 si usa per predire T
    df = df.merge(prov[["Prov", "period_idx", "delta_prov"]],
                  on=["Prov", "period_idx"], how="left")

    df = df[df["lag1_log_mid"].notna() & (df["lag_gap"] <= MAX_LAG_GAP)]
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


def evaluate(name: str, part: pd.DataFrame, preds: dict) -> None:
    y_mid = np.exp(part["log_mid"].to_numpy())
    p_lo, p_mid, p_hi = (np.exp(preds[k]) for k in ("q10_min", "q50_mid", "q90_max"))
    p_lo, p_hi = np.minimum(p_lo, p_hi), np.maximum(p_lo, p_hi)
    cov = float(np.mean((y_mid >= p_lo) & (y_mid <= p_hi)))
    print(f"  [{name}] puntuale (q50 vs punto medio reale): "
          f"MAE={mean_absolute_error(y_mid, p_mid):.0f} €/m²  "
          f"MAPE={mean_absolute_percentage_error(y_mid, p_mid):.1%}  "
          f"R2={r2_score(y_mid, p_mid):.3f}")
    print(f"  [{name}] range [q10_min, q90_max]: copertura={cov:.1%}  "
          f"ampiezza media={np.mean(p_hi - p_lo):.0f} €/m²")


def main() -> None:
    t0 = time.time()
    print("Caricamento dati da omi.db...")
    df = load_data()
    print(f"  {len(df):,} quotazioni valide; costruzione lag features...")
    df = build_features(df)
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    print(f"  righe con lag disponibile: train {len(parts['train']):,} / "
          f"val {len(parts['val']):,} / test {len(parts['test']):,}")

    cat_idx = [FEAT_COLS.index(c) for c in CAT_COLS]
    models = {}
    for name, (q, target, lag_col) in QUANTILES.items():
        print(f"\nTraining {name}: quantile {q} sul delta {target} - {lag_col}...")
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.1,
            categorical_features=cat_idx, early_stopping=False, random_state=42,
        )
        train = parts["train"]
        m.fit(train[FEAT_COLS], train[target] - train[lag_col])
        models[name] = m

    print("\n=== Valutazione (nowcasting con lag features) ===")
    for split in ("val", "test"):
        p = parts[split]
        preds = {name: m.predict(p[FEAT_COLS]) + p[QUANTILES[name][2]].to_numpy()
                 for name, m in models.items()}
        print(f"\n{split.upper()} (periodi {p['periodo'].min()} -> {p['periodo'].max()}):")
        evaluate(split, p, preds)

    MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "models": models,
        "feat_cols": FEAT_COLS,
        "cat_cols": CAT_COLS,
        "categories": {c: list(df[c].cat.categories) for c in CAT_COLS},
        "max_lag_gap": MAX_LAG_GAP,
        # i modelli predicono delta log: riaggiungere il lag indicato
        "delta_lag_col": {name: lag_col for name, (_, _, lag_col) in QUANTILES.items()},
    }
    joblib.dump(bundle, MODELS_DIR / "quantile_bundle.joblib")
    print(f"\nBundle salvato in {MODELS_DIR / 'quantile_bundle.joblib'}  "
          f"({time.time() - t0:.0f}s totali)")


if __name__ == "__main__":
    main()
