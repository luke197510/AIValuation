# -*- coding: utf-8 -*-
"""
App Streamlit: valutazione immobili su dati OMI.

Pura consultazione delle stime precalcolate in app_data/ (generate in locale
da scripts/06_build_app_data.py): nessun modello né database nel deploy.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DATA = Path(__file__).parent / "app_data"

st.set_page_config(page_title="AIValuation — Stima valori immobiliari", page_icon="🏠", layout="centered")


@st.cache_data
def carica_dati():
    stime = pd.read_parquet(APP_DATA / "stime.parquet")
    zone = pd.read_parquet(APP_DATA / "zone.parquet")
    meta = json.loads((APP_DATA / "meta.json").read_text(encoding="utf-8"))
    return stime, zone, meta


stime, zone, meta = carica_dati()

st.title("🏠 Valutazione immobile")
st.caption(f"Stime riferite al semestre **{meta['periodo_stimato']}** · "
           f"basate su quotazioni OMI fino al {meta['ultimo_semestre_omi']} · "
           f"aggiornamento del {meta['build']}")

# --- selezione guidata ---
comuni = sorted(stime["comune"].unique())
comune = st.selectbox("Comune", comuni, index=None,
                      placeholder="Cerca il comune (es. MILANO)...")

if comune:
    z_stime = stime[stime["comune"] == comune]
    z_meta = zone[zone["comune"] == comune].set_index("linkzona")["descr"].to_dict()

    zone_disp = (z_stime[["zona", "fascia", "linkzona"]].drop_duplicates()
                 .sort_values(["fascia", "zona"]))
    def etichetta_zona(r):
        descr = z_meta.get(r["linkzona"], "")
        return f"{r['zona']} — fascia {r['fascia']}" + (f" — {descr}" if descr else "")
    opzioni_zona = {etichetta_zona(r): r["zona"] for _, r in zone_disp.iterrows()}
    scelta_zona = st.selectbox("Zona OMI", list(opzioni_zona))
    zona_sel = opzioni_zona[scelta_zona]

    t_stime = z_stime[z_stime["zona"] == zona_sel]
    tipologia = st.selectbox("Tipologia", sorted(t_stime["tipologia"].unique()))

    s_stime = t_stime[t_stime["tipologia"] == tipologia]
    stati = [s for s in ("OTTIMO", "NORMALE", "SCADENTE") if s in set(s_stime["stato"])]
    col1, col2 = st.columns(2)
    with col1:
        stato = st.selectbox("Stato conservativo", stati,
                             index=stati.index("NORMALE") if "NORMALE" in stati else 0)
    with col2:
        superficie = st.number_input("Superficie commerciale (m²)",
                                     min_value=0.0, value=0.0, step=5.0,
                                     help="Lascia 0 per vedere solo i valori al m²")

    riga = s_stime[s_stime["stato"] == stato]
    if riga.empty:
        st.warning("Combinazione non quotata: nessuna stima disponibile.")
    else:
        r = riga.iloc[0]
        st.divider()
        st.subheader("Stima al m²")
        c1, c2, c3 = st.columns(3)
        c1.metric("Minimo prudente", f"{r['mq_min']:,.0f} €/m²")
        c2.metric("Valore puntuale", f"{r['mq_puntuale']:,.0f} €/m²")
        c3.metric("Massimo prudente", f"{r['mq_max']:,.0f} €/m²")

        if superficie > 0:
            st.subheader(f"Valore immobile ({superficie:.0f} m²)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Minimo prudente", f"{r['mq_min'] * superficie:,.0f} €")
            c2.metric("Valore puntuale", f"{r['mq_puntuale'] * superficie:,.0f} €")
            c3.metric("Massimo prudente", f"{r['mq_max'] * superficie:,.0f} €")

        st.caption(
            f"Ultima quotazione OMI ({r['uq_periodo']}): "
            f"{r['uq_min']:,.0f}–{r['uq_max']:,.0f} €/m². "
            "Range calibrato ~80% (q10–q90) su quotazioni di zona OMI; non tiene "
            "conto delle caratteristiche specifiche dell'immobile (piano, taglio, "
            "esposizione, ecc.)."
        )

st.divider()
st.caption("Fonte dati: quotazioni OMI — Agenzia delle Entrate. "
           "Stime a scopo indicativo, non costituiscono perizia.")
