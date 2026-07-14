# -*- coding: utf-8 -*-
"""
Valutazione di un immobile (dati OMI) da riga di comando.

Esempi:
  python scripts/04_valuta.py --comune MILANO --zona B1 --tipologia "Abitazioni civili" --stato NORMALE --superficie 80
  python scripts/04_valuta.py --comune MILANO --lista-zone
  python scripts/04_valuta.py --comune MILANO --zona B1 --lista-tipologie

Per i test interattivi usa: python scripts/05_cli_test.py
"""
import argparse
import sys

from valutatore import STATI, Valutatore, stampa_stima


def main() -> None:
    ap = argparse.ArgumentParser(description="Stima del valore di un immobile (dati OMI)")
    ap.add_argument("--comune", required=True, help="Denominazione comune (es. MILANO)")
    ap.add_argument("--zona", help="Zona OMI (es. B1)")
    ap.add_argument("--tipologia", help="Descrizione o codice tipologia (es. 'Abitazioni civili' o 20)")
    ap.add_argument("--stato", default="NORMALE", choices=list(STATI),
                    help="Stato conservativo (default: NORMALE)")
    ap.add_argument("--superficie", type=float, help="Superficie commerciale in m²")
    ap.add_argument("--lista-zone", action="store_true", help="Elenca le zone OMI del comune")
    ap.add_argument("--lista-tipologie", action="store_true", help="Elenca le tipologie quotate nella zona")
    args = ap.parse_args()

    v = Valutatore()

    if args.lista_zone:
        zone = v.zone(args.comune)
        if not zone:
            sys.exit(f"Nessuna zona trovata per il comune '{args.comune}'.")
        print(f"Zone OMI di {args.comune.upper()}:")
        for z in zone:
            print(f"  {z['zona']:4s} (fascia {z['fascia']})  {z['descr'] or ''}")
        return

    if args.lista_tipologie:
        if not args.zona:
            sys.exit("--lista-tipologie richiede --zona.")
        zone = [z for z in v.zone(args.comune) if z["zona"] == args.zona.upper()]
        if not zone:
            sys.exit(f"Zona '{args.zona}' non trovata per '{args.comune}'.")
        print(f"Tipologie quotate per {args.comune.upper()} zona {args.zona.upper()}:")
        for t in v.tipologie(zone[0]["linkzona"]):
            print(f"  [{t['cod']}] {t['descr']}  (stati: {t['stati']})")
        return

    if not (args.zona and args.tipologia):
        sys.exit("Servono --zona e --tipologia (oppure usa --lista-zone / --lista-tipologie).")

    try:
        stampa_stima(v.stima(args.comune, args.zona, args.tipologia,
                             args.stato, args.superficie))
    except ValueError as e:
        sys.exit(f"ERRORE: {e}")


if __name__ == "__main__":
    main()
