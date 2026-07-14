# -*- coding: utf-8 -*-
"""
Genera i dati precalcolati per l'app Streamlit (app_data/).

Esegue in locale l'inferenza batch su TUTTE le combinazioni
(zona, tipologia, stato) con quotazioni recenti, e salva:
  - app_data/stime.parquet : stime q10/q50/q90 €/m² per il prossimo semestre
  - app_data/zone.parquet  : anagrafica zone (comune, zona, fascia, descrizione)
  - app_data/meta.json     : periodo stimato, ultimo semestre OMI, data di build

L'app deployata è così una pura consultazione: niente modelli né DB nel repo.
Flusso di aggiornamento: nuovo semestre OMI -> 01_import + 03_train (se si
vuole riaddestrare) -> questo script -> git push -> redeploy automatico.

Uso:  python scripts/06_build_app_data.py
"""
import json
import sqlite3
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from valutatore import prossimo_periodo, periodo_str

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "omi.db"
BUNDLE_PATH = ROOT / "models" / "quantile_bundle.joblib"
OUT_DIR = ROOT / "app_data"

KEY = ["LinkZona", "Cod_Tip", "Stato"]


def load_recent(conn: sqlite3.Connection, max_lag_gap: int) -> pd.DataFrame:
    """Quotazioni degli ultimi max_lag_gap+1 semestri (bastano per lag e delta)."""
    periods = [r[0] for r in conn.execute(
        "SELECT DISTINCT periodo FROM valori ORDER BY periodo DESC LIMIT ?",
        (max_lag_gap + 1,))]
    ph = ",".join("?" * len(periods))
    df = pd.read_sql(f"""
        SELECT periodo, "Prov", "Fascia", "Cod_Tip", "Descr_Tipologia",
               "Stato", "LinkZona", "Zona", "Comune_descrizione",
               "Compr_min", "Compr_max"
        FROM valori
        WHERE periodo IN ({ph})
          AND "Compr_min" > 0 AND "Compr_max" >= "Compr_min"
    """, conn, params=periods)
    df["Stato"] = df["Stato"].fillna("NA")
    for c in ("Prov", "Fascia", "Cod_Tip"):
        df[c] = df[c].fillna("NA")
    return df


def main() -> None:
    t0 = time.time()
    bundle = joblib.load(BUNDLE_PATH)
    conn = sqlite3.connect(DB_PATH)

    all_periods = [r[0] for r in conn.execute(
        "SELECT DISTINCT periodo FROM valori ORDER BY periodo")]
    pidx = {p: i for i, p in enumerate(all_periods)}
    ultimo, penultimo = all_periods[-1], all_periods[-2]
    target_periodo = prossimo_periodo(ultimo)
    target_idx = len(all_periods)

    print(f"Ultimo semestre OMI: {periodo_str(ultimo)} -> stimo {periodo_str(target_periodo)}")
    df = load_recent(conn, bundle["max_lag_gap"])
    print(f"  {len(df):,} quotazioni recenti caricate")

    # per ogni chiave: ultima quotazione (lag1) e penultima (per delta1)
    df["period_idx"] = df["periodo"].map(pidx)
    df = df.sort_values("period_idx").drop_duplicates(KEY + ["period_idx"], keep="last")
    g = df.sort_values(KEY + ["period_idx"], kind="stable").groupby(KEY, sort=False)
    last = g.tail(1).copy()
    prev = g.nth(-2)[KEY + ["Compr_min", "Compr_max"]].rename(
        columns={"Compr_min": "prev_min", "Compr_max": "prev_max"})
    last = last.merge(prev, on=KEY, how="left")

    last["lag1_log_min"] = np.log(last["Compr_min"])
    last["lag1_log_max"] = np.log(last["Compr_max"])
    last["lag1_log_mid"] = np.log((last["Compr_min"] + last["Compr_max"]) / 2)
    last["lag1_log_width"] = last["lag1_log_min"].rsub(last["lag1_log_max"])
    last["delta1"] = last["lag1_log_mid"] - np.log((last["prev_min"] + last["prev_max"]) / 2)
    last["lag_gap"] = (target_idx - last["period_idx"]).astype(float)
    last["semestre"] = target_periodo % 10

    # trend provinciale sugli ultimi due semestri
    dprov = pd.read_sql("""
        SELECT "Prov",
               AVG(CASE WHEN periodo = ? THEN LN(("Compr_min"+"Compr_max")/2.0) END)
             - AVG(CASE WHEN periodo = ? THEN LN(("Compr_min"+"Compr_max")/2.0) END)
               AS delta_prov
        FROM valori WHERE periodo IN (?, ?) AND "Compr_min" > 0 GROUP BY "Prov"
    """, conn, params=(ultimo, penultimo, ultimo, penultimo))
    last = last.merge(dprov, on="Prov", how="left")
    last["delta_prov"] = last["delta_prov"].fillna(0.0)

    last = last[last["lag_gap"] <= bundle["max_lag_gap"]]
    print(f"  {len(last):,} combinazioni (zona, tipologia, stato) stimabili")

    X = last[bundle["feat_cols"]].copy()
    for c in bundle["cat_cols"]:
        X[c] = pd.Categorical(X[c], categories=bundle["categories"][c])
    pred = {}
    for name, m in bundle["models"].items():
        lag_col = bundle["delta_lag_col"][name]
        pred[name] = np.exp(m.predict(X) + last[lag_col].to_numpy())
    lo = np.minimum(pred["q10_min"], pred["q90_max"])
    hi = np.maximum(pred["q10_min"], pred["q90_max"])

    stime = pd.DataFrame({
        "comune": last["Comune_descrizione"].str.strip(),
        "zona": last["Zona"].str.strip(),
        "linkzona": last["LinkZona"],
        "fascia": last["Fascia"],
        "cod_tip": last["Cod_Tip"],
        "tipologia": last["Descr_Tipologia"].str.strip(),
        "stato": last["Stato"],
        "mq_min": lo.round(0), "mq_puntuale": pred["q50_mid"].round(0),
        "mq_max": hi.round(0),
        "uq_periodo": last["periodo"].map(periodo_str),
        "uq_min": last["Compr_min"], "uq_max": last["Compr_max"],
    })

    zone = pd.read_sql("""
        SELECT DISTINCT "Comune_descrizione" AS comune, "Zona" AS zona,
               "Fascia" AS fascia, "Zona_Descr" AS descr, "LinkZona" AS linkzona
        FROM zone WHERE periodo = (SELECT MAX(periodo) FROM zone)
    """, conn)
    zone["comune"] = zone["comune"].str.strip()
    zone["descr"] = zone["descr"].str.strip("'").str.strip()

    OUT_DIR.mkdir(exist_ok=True)
    stime.to_parquet(OUT_DIR / "stime.parquet", index=False)
    zone.to_parquet(OUT_DIR / "zone.parquet", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "periodo_stimato": periodo_str(target_periodo),
        "ultimo_semestre_omi": periodo_str(ultimo),
        "build": date.today().isoformat(),
        "n_stime": len(stime),
    }, indent=2), encoding="utf-8")

    for f in OUT_DIR.iterdir():
        print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Fatto in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
