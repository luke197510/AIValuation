# AIValuation — Predizione valori immobiliari da quotazioni OMI

Progetto ML per la predizione del valore di compravendita (€/m²) a partire dalle
quotazioni OMI (Agenzia delle Entrate), come **range** (min–max) o **valore puntuale**
(punto medio del range predetto).

## Dati

- `DatasetOMI/`: 44 semestri dal **2004/1 al 2025/2** (serie verificata: nessun gap,
  nessun duplicato), 2 CSV per semestre:
  - `*_VALORI.csv`: quotazioni min/max di compravendita e locazione per zona OMI,
    tipologia e stato conservativo (~7,5 M righe totali)
  - `*_ZONE.csv`: anagrafica delle zone OMI (~1,3 M righe totali)
- Formato CSV: prima riga = metadati con il periodo ("Semestre YYYY/S"),
  separatore `;`, decimali con virgola, encoding latin-1.

## Database

`omi.db` (SQLite, ~1,4 GB) generato da `scripts/01_import_omi.py`:

| Oggetto | Contenuto |
|---|---|
| `valori` | tutte le quotazioni, con colonne aggiunte `anno`, `semestre`, `periodo` (es. 20041) |
| `zone` | anagrafica zone OMI per periodo |
| `split_periodi` | assegnazione periodo → train/val/test |
| `valori_split` (vista) | `valori` + colonna `split` |

Indici su `periodo` e `LinkZona` in entrambe le tabelle.

## Split 60/20/20

Split **temporale** per semestre (calcolato sulle righe cumulate, così le
percentuali effettive restano vicine al 60/20/20):

| Split | Periodi | Semestri | Righe |
|---|---|---|---|
| train | 2004/1 → 2016/2 | 26 | 4.627.359 (61,6%) |
| val | 2017/1 → 2021/1 | 9 | 1.450.398 (19,3%) |
| test | 2021/2 → 2025/2 | 9 | 1.438.738 (19,1%) |

Nota: lo split temporale evita leakage tra periodi ma richiede al modello di
estrapolare nel futuro. Per uno split casuale per riga basta sostituire la
logica in `assign_splits()` di `01_import_omi.py`.

## App Streamlit

`streamlit_app.py` è una pura consultazione delle stime **precalcolate** in
`app_data/` (2,5 MB: `stime.parquet` con q10/q50/q90 €/m² per 172k combinazioni
zona×tipologia×stato, `zone.parquet` con l'anagrafica, `meta.json`). Nessun
modello né database nel deploy: il repo resta leggero e Streamlit Community
Cloud lo esegue con le sole dipendenze di `requirements.txt`.

```
.venv\Scripts\python -m streamlit run streamlit_app.py   # esecuzione locale
```

### Deploy (Streamlit Community Cloud)

- Repo GitHub: **https://github.com/luke197510/AIValuation** — branch
  `master`, main file `streamlit_app.py`. Ogni push su `master` ri-deploya
  automaticamente l'app.
- Il deploy da repo privato richiede che l'app GitHub di Streamlit abbia accesso
  al repo: github.com → Settings → Applications → Installed GitHub Apps →
  Streamlit → Configure → Repository access. Se nel form di deploy compare
  "This repository does not exist", è questo il problema.
- Da repo privato l'app nasce **privata** (visibile solo al proprietario): per
  esporla, su share.streamlit.io → ⋮ dell'app → Settings → **Sharing** →
  "This app is public and searchable" (oppure invitare singole email).
  Il codice del repo resta privato in ogni caso.
- Il piano gratuito consente **una sola app da repo privato**.

**Aggiornamento a ogni nuovo semestre OMI** (tutto in locale, poi push):

1. copiare i nuovi CSV in `DatasetOMI/`
2. `python scripts/01_import_omi.py` (ricrea omi.db)
3. facoltativo: `python scripts/03_train_quantile.py` (riaddestra)
4. `python scripts/06_build_app_data.py` (rigenera `app_data/`)
5. commit + push → Streamlit Community Cloud si ri-deploya da solo

## Ambiente

Usare **sempre** il venv del progetto (i pickle di scikit-learn non sono
compatibili tra versioni: il Python di Anaconda con sklearn 1.5 non può
caricare modelli addestrati con sklearn 1.9):

```
python -m venv .venv                                    # solo la prima volta
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python scripts\05_cli_test.py             # esecuzione
```

`requirements.txt` contiene le sole dipendenze dell'app (usato da Streamlit
Community Cloud); `requirements-dev.txt` aggiunge le versioni pinnate di
training (pandas 3.0.3, scikit-learn 1.9.0). Se si aggiorna scikit-learn vanno
riaddestrati i modelli.

## Script

```
python scripts/01_import_omi.py       # verifica gap + import in omi.db + split
python scripts/02_train_baseline.py   # baseline HistGradientBoosting min/max
python scripts/03_train_quantile.py   # modello quantile + lag features (nowcasting)
python scripts/04_valuta.py --comune MILANO --zona B1 --tipologia "Abitazioni civili" --superficie 80
python scripts/05_cli_test.py         # CLI interattiva di test (menu guidato)
python scripts/06_build_app_data.py   # inferenza batch -> app_data/ per l'app
python -m streamlit run streamlit_app.py   # app web in locale
```

- **Baseline** (`02`): due `HistGradientBoostingRegressor` (uno per `Compr_min`,
  uno per `Compr_max`, su scala log) con feature categoriche native + target
  encoding — calcolato solo sul train — per Comune e zona OMI (`LinkZona`).
- **Modello quantile** (`03`, quello usato in produzione): tre HGB quantile
  (q10 su min, q50 sul punto medio, q90 su max) con lag features (ultima
  quotazione nota della chiave zona×tipologia×stato, trend di zona e
  provinciale). Predicono il **delta log** rispetto all'ultima quotazione, non
  il livello (il livello satura sulla coda alta dei prezzi — vedi CLAUDE.md).
  Output: range calibrato ~80% + valore puntuale, riferiti al semestre
  successivo all'ultimo in banca dati. Metriche sul test (2021/2→2025/2):
  MAE 11 €/m², MAPE 1,2%, R² 0,998, copertura del range 99,6%.
- **Valutazione immobile** (`04` CLI a argomenti, `05` interattiva): la logica
  condivisa è in `scripts/valutatore.py` (classe `Valutatore`).
- **Export per l'app** (`06`): inferenza batch su tutte le combinazioni
  stimabili (~173k) → `app_data/*.parquet` consultati da `streamlit_app.py`.

Modelli salvati in `models/` (fuori dal repo, come `omi.db` e `DatasetOMI/`).
