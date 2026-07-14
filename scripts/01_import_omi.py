# -*- coding: utf-8 -*-
"""
Importa i CSV OMI (VALORI + ZONE) in un database SQLite.

- Legge il periodo dalla prima riga di ogni CSV ("Semestre YYYY/S") e
  verifica che la serie 2004/1 -> 2025/2 sia completa e senza gap.
- Converte i decimali con virgola in REAL (Compr_min/max, Loc_min/max).
- Assegna lo split temporale 60/20/20 (train/val/test) per semestre.

Uso:  python scripts/01_import_omi.py
Output: omi.db nella root del progetto.
"""
import csv
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DatasetOMI"
DB_PATH = ROOT / "omi.db"
ENCODING = "latin-1"

PERIOD_RE = re.compile(r"Semestre (\d{4})/(\d)")

NUMERIC_COLS = {"Compr_min", "Compr_max", "Loc_min", "Loc_max"}


def read_period(path: Path) -> tuple[int, int]:
    with open(path, encoding=ENCODING) as f:
        first = f.readline()
    m = PERIOD_RE.search(first)
    if not m:
        sys.exit(f"ERRORE: periodo non trovato nella prima riga di {path.name}")
    return int(m.group(1)), int(m.group(2))


def check_gaps(periods: list[tuple[int, int]]) -> None:
    periods = sorted(periods)
    expected = []
    y, s = periods[0]
    while (y, s) <= periods[-1]:
        expected.append((y, s))
        y, s = (y, s + 1) if s == 1 else (y + 1, 1)
    missing = sorted(set(expected) - set(periods))
    dupes = sorted({p for p in periods if periods.count(p) > 1})
    if missing:
        sys.exit(f"ERRORE: semestri mancanti: {missing}")
    if dupes:
        sys.exit(f"ERRORE: semestri duplicati: {dupes}")
    print(f"OK: {len(periods)} semestri contigui da {periods[0][0]}/{periods[0][1]} "
          f"a {periods[-1][0]}/{periods[-1][1]}, nessun gap.")


def parse_value(col: str, raw: str):
    v = raw.strip()
    if not v:
        return None
    if col in NUMERIC_COLS:
        try:
            return float(v.replace(".", "").replace(",", "."))
        except ValueError:
            return None
    return v


def import_file(conn: sqlite3.Connection, path: Path, table: str,
                anno: int, semestre: int) -> int:
    periodo = anno * 10 + semestre  # es. 20041
    with open(path, encoding=ENCODING, newline="") as f:
        f.readline()  # riga 1: metadati periodo
        reader = csv.reader(f, delimiter=";")
        header = [c.strip() for c in next(reader) if c.strip()]
        ncols = len(header)

        cols_sql = ", ".join(
            f'"{c}" REAL' if c in NUMERIC_COLS else f'"{c}" TEXT' for c in header
        )
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {table} '
            f'(anno INTEGER, semestre INTEGER, periodo INTEGER, {cols_sql})'
        )
        placeholders = ", ".join("?" * (ncols + 3))
        insert = f"INSERT INTO {table} VALUES ({placeholders})"

        rows, n = [], 0
        for rec in reader:
            if not any(x.strip() for x in rec):
                continue
            vals = [parse_value(header[i], rec[i]) if i < len(rec) else None
                    for i in range(ncols)]
            rows.append((anno, semestre, periodo, *vals))
            n += 1
            if len(rows) >= 50_000:
                conn.executemany(insert, rows)
                rows = []
        if rows:
            conn.executemany(insert, rows)
    return n


def assign_splits(conn: sqlite3.Connection) -> None:
    """Split temporale ~60/20/20 calcolato sul numero di righe cumulate di 'valori'."""
    per = conn.execute(
        "SELECT periodo, COUNT(*) FROM valori GROUP BY periodo ORDER BY periodo"
    ).fetchall()
    total = sum(c for _, c in per)
    cum, split_of = 0, {}
    for p, c in per:
        frac = cum / total  # frazione righe precedenti a questo periodo
        split_of[p] = "train" if frac < 0.60 else ("val" if frac < 0.80 else "test")
        cum += c

    conn.execute("DROP TABLE IF EXISTS split_periodi")
    conn.execute("CREATE TABLE split_periodi (periodo INTEGER PRIMARY KEY, split TEXT)")
    conn.executemany("INSERT INTO split_periodi VALUES (?, ?)", split_of.items())

    print("\nSplit temporale (per semestre):")
    for name in ("train", "val", "test"):
        ps = sorted(p for p, s in split_of.items() if s == name)
        rows = sum(c for p, c in per if split_of[p] == name)
        print(f"  {name:5s}: {ps[0]} -> {ps[-1]}  ({len(ps)} semestri, "
              f"{rows:,} righe = {rows / total:.1%})")


def main() -> None:
    valori = sorted(DATA_DIR.glob("*_VALORI.csv"))
    zone = sorted(DATA_DIR.glob("*_ZONE.csv"))
    if not valori:
        sys.exit(f"ERRORE: nessun *_VALORI.csv in {DATA_DIR}")

    print("Verifica gap sui periodi (prima riga di ogni CSV)...")
    periods = [read_period(p) for p in valori]
    check_gaps(periods)

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")

    tot_v = tot_z = 0
    for path, (anno, sem) in zip(valori, periods):
        tot_v += import_file(conn, path, "valori", anno, sem)
    for path in zone:
        anno, sem = read_period(path)
        tot_z += import_file(conn, path, "zone", anno, sem)
    conn.commit()
    print(f"\nImportate {tot_v:,} righe in 'valori' e {tot_z:,} in 'zone'.")

    assign_splits(conn)

    print("\nCreazione indici...")
    conn.execute("CREATE INDEX idx_valori_periodo ON valori(periodo)")
    conn.execute('CREATE INDEX idx_valori_linkzona ON valori("LinkZona")')
    conn.execute('CREATE INDEX idx_zone_periodo ON zone(periodo)')
    conn.execute('CREATE INDEX idx_zone_linkzona ON zone("LinkZona")')
    conn.execute("""
        CREATE VIEW IF NOT EXISTS valori_split AS
        SELECT v.*, s.split
        FROM valori v JOIN split_periodi s USING (periodo)
    """)
    conn.commit()

    conn.execute("PRAGMA optimize")
    conn.close()
    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\nFatto: {DB_PATH} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
