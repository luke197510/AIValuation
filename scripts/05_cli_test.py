# -*- coding: utf-8 -*-
"""
CLI interattiva per testare la valutazione degli immobili.

Guida passo passo: comune -> zona OMI -> tipologia -> stato -> superficie,
poi mostra la stima (range + valore puntuale). Invio a vuoto per i default,
'q' in qualsiasi momento per uscire.

Uso:  python scripts/05_cli_test.py
"""
import sys

from valutatore import STATI, Valutatore, stampa_stima


def chiedi(prompt: str, default: str | None = None) -> str:
    suff = f" [{default}]" if default else ""
    try:
        risposta = input(f"{prompt}{suff}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCiao!")
        sys.exit(0)
    if risposta.lower() == "q":
        print("Ciao!")
        sys.exit(0)
    return risposta or (default or "")


def scegli(prompt: str, opzioni: list[str], default: int | None = None) -> int:
    """Mostra opzioni numerate e ritorna l'indice scelto."""
    for i, o in enumerate(opzioni, 1):
        print(f"  {i:2d}) {o}")
    while True:
        r = chiedi(prompt, str(default) if default else None)
        if r.isdigit() and 1 <= int(r) <= len(opzioni):
            return int(r) - 1
        print(f"  Scelta non valida (1-{len(opzioni)}, 'q' per uscire).")


def una_valutazione(v: Valutatore) -> None:
    # 1. comune
    while True:
        comune = chiedi("\nComune")
        if not comune:
            continue
        zone = v.zone(comune)
        if zone:
            break
        print(f"  Nessuna zona OMI trovata per '{comune}'. Controlla il nome (es. MILANO).")

    # 2. zona
    print(f"\nZone OMI di {comune.upper()}:")
    etichette = [f"{z['zona']:4s} fascia {z['fascia']}  {(z['descr'] or '').strip(chr(39))}"
                 for z in zone]
    zona = zone[scegli("Zona (numero)", etichette)]

    # 3. tipologia
    tipi = v.tipologie(zona["linkzona"])
    if not tipi:
        print("  Nessuna tipologia quotata nell'ultimo semestre per questa zona.")
        return
    print(f"\nTipologie quotate in zona {zona['zona']}:")
    tipo = tipi[scegli("Tipologia (numero)", [t["descr"] for t in tipi])]

    # 4. stato
    print("\nStato conservativo:")
    stato = STATI[scegli("Stato (numero)", list(STATI), default=2)]

    # 5. superficie
    superficie = None
    r = chiedi("Superficie commerciale in m² (invio per saltare)", "")
    if r:
        try:
            superficie = float(r.replace(",", "."))
        except ValueError:
            print("  Superficie non valida, la salto.")

    # 6. stima
    try:
        stampa_stima(v.stima(comune, zona["zona"], tipo["cod"], stato, superficie))
    except ValueError as e:
        print(f"\nImpossibile stimare: {e}")


def main() -> None:
    print("=== AIValuation — test valutazione immobili (dati OMI) ===")
    print("('q' in qualsiasi momento per uscire)")
    print("\nCaricamento modelli...")
    v = Valutatore()
    print(f"Pronto. Ultimo semestre in banca dati: {v.ultimo // 10}/{v.ultimo % 10}; "
          f"le stime si riferiscono al semestre successivo.")
    while True:
        una_valutazione(v)
        if chiedi("\nAltra valutazione? (s/n)", "s").lower() not in ("s", "si", "sì"):
            print("Ciao!")
            return


if __name__ == "__main__":
    main()
