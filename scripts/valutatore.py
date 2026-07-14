# -*- coding: utf-8 -*-
"""
Modulo di valutazione: logica condivisa tra la CLI a argomenti (04_valuta.py)
e la CLI interattiva di test (05_cli_test.py).

Carica i modelli quantile (03_train_quantile.py) e omi.db, e stima il valore
di compravendita per il semestre successivo all'ultimo disponibile.
"""
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "omi.db"
BUNDLE_PATH = ROOT / "models" / "quantile_bundle.joblib"

STATI = ("OTTIMO", "NORMALE", "SCADENTE")


def periodo_str(p: int) -> str:
    return f"{p // 10}/{p % 10}"


def prossimo_periodo(p: int) -> int:
    return p + 1 if p % 10 == 1 else (p // 10 + 1) * 10 + 1


class Valutatore:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.bundle = joblib.load(BUNDLE_PATH)
        periods = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT periodo FROM valori ORDER BY periodo")]
        self.pidx = {p: i for i, p in enumerate(periods)}
        self.ultimo = periods[-1]
        self.penultimo = periods[-2]
        self.target_periodo = prossimo_periodo(self.ultimo)
        self.target_idx = len(periods)

    # ---------- esplorazione ----------

    def zone(self, comune: str) -> list[dict]:
        rows = self.conn.execute("""
            SELECT "Zona" AS zona, "Fascia" AS fascia, "LinkZona" AS linkzona,
                   MAX("Zona_Descr") AS descr
            FROM zone
            WHERE UPPER("Comune_descrizione") = ?
              AND periodo = (SELECT MAX(periodo) FROM zone z2
                             WHERE UPPER(z2."Comune_descrizione") = UPPER(?))
            GROUP BY "Zona", "Fascia", "LinkZona"
            ORDER BY "Fascia", "Zona"
        """, (comune.upper(), comune)).fetchall()
        return [dict(r) for r in rows]

    def tipologie(self, linkzona: str) -> list[dict]:
        rows = self.conn.execute("""
            SELECT "Cod_Tip" AS cod, MAX("Descr_Tipologia") AS descr,
                   GROUP_CONCAT(DISTINCT "Stato") AS stati
            FROM valori
            WHERE "LinkZona" = ?
              AND periodo = (SELECT MAX(periodo) FROM valori v2
                             WHERE v2."LinkZona" = valori."LinkZona")
            GROUP BY "Cod_Tip" ORDER BY descr
        """, (linkzona,)).fetchall()
        return [dict(r) for r in rows]

    # ---------- stima ----------

    def _storico(self, linkzona: str, cod_tip: str, stato: str) -> pd.DataFrame:
        return pd.read_sql("""
            SELECT periodo, "Prov", "Fascia", "Compr_min", "Compr_max"
            FROM valori
            WHERE "LinkZona" = ? AND "Cod_Tip" = ? AND COALESCE("Stato",'NA') = ?
              AND "Compr_min" > 0 AND "Compr_max" >= "Compr_min"
            ORDER BY periodo DESC LIMIT 4
        """, self.conn, params=(linkzona, cod_tip, stato))

    def _delta_prov(self, prov: str) -> float:
        row = self.conn.execute("""
            SELECT AVG(CASE WHEN periodo = ? THEN LN(("Compr_min"+"Compr_max")/2.0) END)
                 - AVG(CASE WHEN periodo = ? THEN LN(("Compr_min"+"Compr_max")/2.0) END)
            FROM valori
            WHERE "Prov" = ? AND periodo IN (?, ?) AND "Compr_min" > 0
        """, (self.ultimo, self.penultimo, prov, self.ultimo, self.penultimo)).fetchone()
        return row[0] if row and row[0] is not None else 0.0

    def stima(self, comune: str, zona: str, tipologia: str,
              stato: str = "NORMALE", superficie: float | None = None) -> dict:
        """Ritorna dict con la stima; solleva ValueError con messaggio utente se impossibile."""
        zrow = self.conn.execute("""
            SELECT DISTINCT "LinkZona", "Prov", "Fascia" FROM valori
            WHERE UPPER("Comune_descrizione") = ? AND "Zona" = ?
            ORDER BY periodo DESC LIMIT 1
        """, (comune.upper(), zona.upper())).fetchone()
        if not zrow:
            raise ValueError(f"Zona '{zona}' non trovata per il comune '{comune}'.")
        trow = self.conn.execute("""
            SELECT DISTINCT "Cod_Tip", "Descr_Tipologia" FROM valori
            WHERE "Cod_Tip" = ?1 OR UPPER("Descr_Tipologia") = UPPER(?1) LIMIT 1
        """, (str(tipologia),)).fetchone()
        if not trow:
            raise ValueError(f"Tipologia '{tipologia}' non riconosciuta.")

        stato_usato, hist = stato, self._storico(zrow["LinkZona"], trow["Cod_Tip"], stato)
        if hist.empty and stato != "NORMALE":
            stato_usato = "NORMALE"
            hist = self._storico(zrow["LinkZona"], trow["Cod_Tip"], "NORMALE")
        if hist.empty:
            raise ValueError(
                f"Nessuna quotazione storica per {comune}/{zona} "
                f"tipologia '{trow['Descr_Tipologia']}'.")

        lag1 = hist.iloc[0]
        lag_gap = self.target_idx - self.pidx[int(lag1["periodo"])]
        if lag_gap > self.bundle["max_lag_gap"]:
            raise ValueError(
                f"Ultima quotazione troppo vecchia ({periodo_str(int(lag1['periodo']))}): "
                f"oltre il limite di affidabilità del modello "
                f"({self.bundle['max_lag_gap']} semestri).")

        log_min1 = float(np.log(lag1["Compr_min"]))
        log_max1 = float(np.log(lag1["Compr_max"]))
        log_mid1 = float(np.log((lag1["Compr_min"] + lag1["Compr_max"]) / 2))
        delta1 = np.nan
        if len(hist) > 1:
            lag2 = hist.iloc[1]
            delta1 = log_mid1 - float(np.log((lag2["Compr_min"] + lag2["Compr_max"]) / 2))

        row = {
            "Prov": zrow["Prov"], "Fascia": zrow["Fascia"],
            "Cod_Tip": trow["Cod_Tip"], "Stato": stato_usato,
            "semestre": self.target_periodo % 10,
            "lag1_log_mid": log_mid1, "lag1_log_min": log_min1,
            "lag1_log_max": log_max1,
            "lag1_log_width": log_max1 - log_min1,
            "delta1": delta1, "lag_gap": float(lag_gap),
            "delta_prov": self._delta_prov(zrow["Prov"]),
        }
        X = pd.DataFrame([row], columns=self.bundle["feat_cols"])
        for c in self.bundle["cat_cols"]:
            X[c] = pd.Categorical([row[c]], categories=self.bundle["categories"][c])
        # i modelli predicono il delta log rispetto al lag corrispondente
        pred = {k: float(np.exp(m.predict(X)[0] + row[self.bundle["delta_lag_col"][k]]))
                for k, m in self.bundle["models"].items()}
        lo, mid, hi = pred["q10_min"], pred["q50_mid"], pred["q90_max"]
        lo, hi = min(lo, hi), max(lo, hi)

        return {
            "comune": comune.upper(), "zona": zona.upper(),
            "linkzona": zrow["LinkZona"], "fascia": zrow["Fascia"],
            "tipologia": trow["Descr_Tipologia"], "cod_tip": trow["Cod_Tip"],
            "stato_richiesto": stato, "stato_usato": stato_usato,
            "periodo_stimato": periodo_str(self.target_periodo),
            "ultima_quotazione": {
                "periodo": periodo_str(int(lag1["periodo"])),
                "compr_min": float(lag1["Compr_min"]),
                "compr_max": float(lag1["Compr_max"]),
            },
            "mq_min": lo, "mq_puntuale": mid, "mq_max": hi,
            "superficie": superficie,
            "valore_min": lo * superficie if superficie else None,
            "valore_puntuale": mid * superficie if superficie else None,
            "valore_max": hi * superficie if superficie else None,
        }


def stampa_stima(s: dict) -> None:
    print(f"\nStima per {s['comune']} zona {s['zona']} "
          f"({s['linkzona']}, fascia {s['fascia']})")
    extra = (f"  (richiesto: {s['stato_richiesto']}, non quotato)"
             if s["stato_usato"] != s["stato_richiesto"] else "")
    print(f"Tipologia: {s['tipologia']}  |  Stato: {s['stato_usato']}{extra}")
    uq = s["ultima_quotazione"]
    print(f"Periodo stimato: {s['periodo_stimato']} "
          f"(ultima quotazione OMI usata: {uq['periodo']}, "
          f"{uq['compr_min']:.0f}-{uq['compr_max']:.0f} €/m²)")
    print(f"\n  €/m² stimati:  min prudente {s['mq_min']:,.0f}  |  "
          f"puntuale {s['mq_puntuale']:,.0f}  |  max prudente {s['mq_max']:,.0f}")
    if s["superficie"]:
        print(f"  Valore immobile ({s['superficie']:.0f} m²):  "
              f"{s['valore_min']:,.0f} €  |  {s['valore_puntuale']:,.0f} €  |  "
              f"{s['valore_max']:,.0f} €")
    print("\n  Nota: range calibrato ~80% (q10-q90) su quotazioni di zona OMI; non tiene"
          "\n  conto delle caratteristiche specifiche dell'immobile (piano, taglio, ecc.).")
