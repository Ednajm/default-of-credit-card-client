from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "dataset" / "default of credit card clients.xls"
CLEAN_PATH = BASE_DIR / "dataset" / "credit_default_clean.csv"
FIG_DIR = BASE_DIR / "figures"
TARGET = "default payment next month"
ID_COL = "ID"
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERIC_COLS = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS
CONTINUOUS_COLS = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS
RANDOM_STATE = 42  
TEST_SIZE = 0.20    
CV_FOLDS = 5        
DOCUMENTED_CODES = {
    "SEX": {1, 2},                    
    "EDUCATION": {1, 2, 3, 4},        
    "MARRIAGE": {1, 2, 3},            
}
SEX_LABELS = {1: "Maschio", 2: "Femmina"}
EDUCATION_LABELS = {
    0: "0 (non doc.)",
    1: "Laurea magistrale",
    2: "Laurea",
    3: "Scuola superiore",
    4: "Altro",
    5: "5 (non doc.)",
    6: "6 (non doc.)",
}
MARRIAGE_LABELS = {
    0: "0 (non doc.)",
    1: "Sposato/a",
    2: "Single",
    3: "Altro",
}
PAY_LABELS = {
    -2: "-2\nnon usa",
    -1: "-1\nsaldato",
    0: "0\nregola",
}

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)
C_NO_DEFAULT = "#2a78d6"
C_DEFAULT = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#992d2c"],
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 130,
    }
)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, linewidth=0.8, color=GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def label_bars(ax, bars, counts) -> None:
    for bar, n in zip(bars, counts):
        x = bar.get_x() + bar.get_width() / 2
        ax.annotate(
            f"{bar.get_height():.0f}%",
            (x, bar.get_height()),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        ax.annotate(
            f"n={n:,}",
            (x, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=MUTED,
        )


def save(fig, name: str) -> None:
    """Salva la figura in figures/ e la chiude (backend Agg: niente finestre)."""
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grafico salvato] figures/{name}")


def load_raw() -> pd.DataFrame:
    """Legge l'.xls originale: l'intestazione vera e' alla seconda riga (header=1)."""
    df = pd.read_excel(RAW_PATH, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    return df
def explore_shape_and_types(df: pd.DataFrame) -> None:
    section("1. DIMENSIONI DEL DATASET E TIPI DI DATO")

    n_rows, n_cols = df.shape
    print(f"Righe (osservazioni) : {n_rows:,}")
    print(f"Colonne (feature)    : {n_cols}  ({n_cols - 2} predittori + ID + target)")
    print(f"Memoria occupata     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(
        "\n-> Con ~30.000 osservazioni il dataset e' ampiamente sufficiente per uno "
        "\n   split train/test stratificato e per una k-fold cross-validation (k=5 o 10):"
        "\n   anche la classe minoritaria resta ben rappresentata in ogni fold."
    )

    print("\n--- Tipi di dato per colonna ---")
    types_report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "valori_unici": df.nunique(),
            "esempio": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
        }
    )
    print(types_report.to_string())

    print("\n--- Composizione ---")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols]
    print(f"Colonne numeriche : {len(numeric_cols)}")
    print(f"Colonne testuali  : {len(other_cols)}  -> {other_cols if other_cols else 'nessuna'}")

    print(
        "\nATTENZIONE: tutte le colonne sono lette come numeriche, ma NON tutte lo sono"
        "\ndavvero dal punto di vista semantico. Le seguenti sono categorie codificate"
        "\ncon numeri interi (etichette, non quantita'):"
    )
    for col in CATEGORICAL_COLS:
        print(f"  - {col:<10} valori presenti: {sorted(df[col].unique().tolist())}")
    print(
        f"  - {'PAY_*':<10} valori presenti: "
        f"{sorted(pd.unique(df[PAY_COLS].to_numpy().ravel()).tolist())}"
        "\n               (stato del pagamento: -2 = carta non utilizzata, -1 = saldo pagato per"
        "\n                intero, 0 = pagamento minimo effettuato senza ritardo, 1..9 = mesi di"
        "\n                ritardo. ATTENZIONE: il paper documenta solo -1 e 1..9, quindi -2 e 0"
        "\n                sono codici da interpretare pur essendo i piu' frequenti: vedi sez. 11;"
        "\n                ordinale, quindi l'ordine ha un senso ma le distanze non sono uniformi)"
    )
    print(
        "\n-> Se trattate come numeriche, un modello lineare interpreterebbe EDUCATION=3"
        "\n   come 'tre volte' EDUCATION=1, che non ha alcun significato. Queste colonne"
        "\n   vanno quindi ricodificate (one-hot / raggruppamento) nel task 4."
    )
    print("\n--- Codici presenti ma NON documentati nel paper originale ---")
    found_any = False
    for col, valid in DOCUMENTED_CODES.items():
        undocumented = sorted(set(df[col].unique()) - valid)
        if undocumented:
            found_any = True
            counts = df[col].value_counts()
            detail = ", ".join(f"{code} (n={counts[code]})" for code in undocumented)
            print(f"  - {col:<10} codici anomali: {detail}")
    if not found_any:
        print("  Nessuno: tutti i codici corrispondono alla documentazione.")
    else:
        print(
            "\n-> Non sono valori mancanti in senso tecnico (la cella non e' vuota), ma sono"
            "\n   categorie fuori specifica. Vanno annotati ora e accorpati alla categoria"
            "\n   'altro' nel task 4, non rimossi a caso in questa fase."
        )
def check_missing(df: pd.DataFrame) -> None:
    section("2a. CONTROLLO VALORI MANCANTI")

    missing = df.isna().sum()
    total_missing = int(missing.sum())

    if total_missing == 0:
        print("Verifica eseguita su tutte le colonne: 0 valori mancanti (NaN / celle vuote).")
        print(
            "\n-> Risultato da riportare esplicitamente in relazione: nessuna imputazione"
            "\n   (media/mediana/moda) e nessuna rimozione di righe e' necessaria. La verifica"
            "\n   e' stata fatta, non e' stata data per scontata."
        )
    else:
        print("Colonne con valori mancanti:")
        print(missing[missing > 0].to_string())
        print(f"\nTotale celle mancanti: {total_missing:,}")
    print("\n--- Nota: zeri presenti (possibili 'mancanti mascherati') ---")
    inspected = [c for c in df.columns if c not in (ID_COL, TARGET)]
    zero_counts = (df[inspected] == 0).sum()
    print(zero_counts[zero_counts > 0].to_string())
    print(
        "\n-> Negli importi (BILL_AMT*, PAY_AMT*) lo zero e' un valore legittimo"
        "\n   (nessun saldo / nessun pagamento) e nei PAY_* indica un pagamento in regola."
        "\n   In EDUCATION e MARRIAGE, invece, lo zero non e' un codice previsto dalla"
        "\n   documentazione: e' di fatto un 'non dichiarato' mascherato da numero."
        "\n   (ID e target sono esclusi dal conteggio: nel target lo 0 e' semplicemente"
        "\n   la classe 'non insolvente'.)"
    )
def check_duplicates(df: pd.DataFrame) -> None:
    section("2b. CONTROLLO RIGHE DUPLICATE")

    dup_with_id = int(df.duplicated().sum())
    print(f"Duplicati sull'intero dataset (ID incluso) : {dup_with_id}")
    print(f"ID univoci                                 : {df[ID_COL].nunique():,} su {len(df):,}")

    feature_cols = [c for c in df.columns if c != ID_COL]
    dup_no_id = int(df.duplicated(subset=feature_cols).sum())
    print(f"Duplicati sulle sole feature (ID escluso)  : {dup_no_id}")

    print(
        "\n-> IMPORTANTE, l'ordine dei controlli conta: l'ID e' un progressivo univoco,"
        "\n   quindi finche' resta tra le colonne NESSUNA riga potra' mai risultare"
        "\n   duplicata. Il controllo sensato e' quello sulle sole feature: due clienti"
        "\n   con ID diverso ma identici in tutte le 24 variabili sono, per il modello,"
        "\n   la stessa identica osservazione ripetuta."
    )

    if dup_no_id > 0:
        dup_mask = df.duplicated(subset=feature_cols, keep=False)
        n_groups = df.loc[dup_mask, feature_cols].drop_duplicates().shape[0]
        print(
            f"\nTrovate {dup_mask.sum()} righe coinvolte in {n_groups} gruppi di duplicati "
            f"({dup_no_id} righe in eccesso da rimuovere)."
        )
        print("\nEsempio (primi 2 gruppi, colonne principali):")
        preview_cols = [ID_COL, "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", TARGET]
        example = df.loc[dup_mask].sort_values(feature_cols).head(4)
        print(example[preview_cols].to_string(index=False))
        print(
            "\n-> Le righe duplicate vanno rimosse: se restassero, quei casi peserebbero"
            "\n   il doppio in fase di addestramento senza alcuna giustificazione, e le"
            "\n   stesse osservazioni potrebbero finire sia in train sia in test,"
            "\n   gonfiando artificialmente le metriche di valutazione."
        )
    else:
        print("\n-> Nessun duplicato: nessuna rimozione necessaria.")
def clean(df: pd.DataFrame) -> pd.DataFrame:
    section("3. RIMOZIONE COLONNA ID E DEDUPLICAZIONE")

    print(f"Shape iniziale: {df.shape}")

    df_clean = df.drop(columns=[ID_COL])
    print(f"Rimossa colonna '{ID_COL}'          -> {df_clean.shape}")
    print(
        "\n-> Perche': l'ID e' un semplice progressivo assegnato in fase di raccolta dati,"
        "\n   non ha alcun legame causale con la probabilita' di default. Lasciandolo tra"
        "\n   le feature, un modello ad alta capacita' (Random Forest, Gradient Boosting)"
        "\n   puo' agganciarsi a pattern casuali legati all'ORDINE di inserimento dei"
        "\n   record e memorizzare rumore: overfitting mascherato, che non si generalizza"
        "\n   su dati nuovi (dove gli ID saranno del tutto diversi)."
    )

    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed = n_before - len(df_clean)
    print(f"\nRimosse {removed} righe duplicate  -> {df_clean.shape}")

    return df_clean
def summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    section("4. RIEPILOGO")

    print(f"Dataset originale : {df_raw.shape[0]:,} righe x {df_raw.shape[1]} colonne")
    print(f"Dataset pulito    : {df_clean.shape[0]:,} righe x {df_clean.shape[1]} colonne")
    print(f"Variabile target  : '{TARGET}' (0 = non insolvente, 1 = insolvente)")

    print("\nDistribuzione del target nel dataset pulito:")
    dist = df_clean[TARGET].value_counts().sort_index()
    for label, count in dist.items():
        print(f"  classe {label}: {count:>6,}  ({count / len(df_clean):6.2%})")
    print(
        "\n-> Classi sbilanciate: lo split train/test e la cross-validation dovranno essere"
        "\n   STRATIFICATI per mantenere questa proporzione in ogni sottoinsieme."
    )

    print("\nAzioni eseguite in questo task:")
    print("  [x] Dimensioni e tipi di dato ispezionati")
    print("  [x] Variabili categoriche codificate come interi individuate (SEX, EDUCATION, MARRIAGE)")
    print("  [x] Codici fuori specifica annotati per il task 4")
    print("  [x] Valori mancanti verificati")
    print("  [x] Righe duplicate verificate e rimosse")
    print("  [x] Colonna ID rimossa")

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nDataset pulito salvato in: {CLEAN_PATH.relative_to(BASE_DIR)}")
def target_distribution(df: pd.DataFrame) -> None:
    section("5. DISTRIBUZIONE DEL TARGET (sbilanciamento delle classi)")

    counts = df[TARGET].value_counts().sort_index()
    perc = counts / len(df) * 100
    ratio = counts[0] / counts[1]

    print(f"Osservazioni totali: {len(df):,}\n")
    for label in (0, 1):
        nome = "non insolvente (0)" if label == 0 else "insolvente (1)"
        print(f"  {nome:<20} {counts[label]:>7,}  ({perc[label]:5.2f}%)")
    print(f"\nRapporto tra le classi: {ratio:.2f} : 1  (maggioritaria : minoritaria)")

    print(
        "\n-> Lo sbilanciamento e' MODERATO, non estremo: circa 1 cliente su 5 va in default."
        "\n   Conseguenze pratiche per i task successivi:"
        f"\n   * un modello banale che predice sempre 'non insolvente' otterrebbe {perc[0]:.2f}%"
        "\n     di accuratezza pur non individuando NESSUN insolvente: l'accuracy da sola"
        "\n     e' quindi una metrica fuorviante e va affiancata da precision, recall,"
        "\n     F1 sulla classe 1 e ROC-AUC;"
        "\n   * train/test split e cross-validation devono essere STRATIFICATI;"
        "\n   * ha senso valutare class_weight='balanced' o tecniche di ricampionamento."
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    bars = ax.bar(
        ["Non insolvente\n(0)", "Insolvente\n(1)"],
        counts.values,
        width=0.5,
        color=[C_NO_DEFAULT, C_DEFAULT],
    )
    for bar, n, p in zip(bars, counts.values, perc.values):
        ax.annotate(
            f"{n:,}\n{p:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_title("Distribuzione della classe target")
    ax.set_ylabel("Numero di clienti")
    ax.set_ylim(0, counts.max() * 1.18)
    style_axes(ax)
    save(fig, "01_distribuzione_target.png")
def numeric_distributions(df: pd.DataFrame) -> None:
    section("6. VARIABILI NUMERICHE CHIAVE (confronto tra le due classi)")

    print("--- Statistiche descrittive (dataset intero) ---")
    print(df[["LIMIT_BAL", "AGE"] + BILL_COLS[:2] + PAY_AMT_COLS[:2]].describe().round(1).to_string())

    print("\n--- Mediana per classe (la mediana, non la media: le distribuzioni sono asimmetriche) ---")
    med = df.groupby(TARGET)[NUMERIC_COLS].median().T
    med.columns = ["classe 0 (no default)", "classe 1 (default)"]
    med["differenza %"] = np.where(
        med["classe 0 (no default)"] != 0,
        (med["classe 1 (default)"] - med["classe 0 (no default)"])
        / med["classe 0 (no default)"].abs()
        * 100,
        np.nan,
    )
    print(med.round(1).to_string())

    lim0 = df.loc[df[TARGET] == 0, "LIMIT_BAL"].median()
    lim1 = df.loc[df[TARGET] == 1, "LIMIT_BAL"].median()
    age0 = df.loc[df[TARGET] == 0, "AGE"].median()
    age1 = df.loc[df[TARGET] == 1, "AGE"].median()
    print(
        f"\n-> LIMIT_BAL: mediana {lim0:,.0f} NT$ per i non insolventi contro {lim1:,.0f} NT$ per gli"
        f"\n   insolventi ({(lim1 - lim0) / lim0:+.0%}). Il fido concesso e' gia' una sintesi del"
        "\n   merito creditizio valutato dalla banca: chi ha un limite basso e' considerato piu'"
        "\n   rischioso a priori, e in effetti va in default piu' spesso. E' una delle poche"
        "\n   variabili anagrafico-contrattuali con un segnale chiaro."
        f"\n-> AGE: mediana {age0:.0f} contro {age1:.0f} anni, distribuzioni quasi sovrapposte."
        "\n   L'eta' da sola separa poco le due classi."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    for ax, col, bins, xlabel in (
        (axes[0], "LIMIT_BAL", np.arange(0, 800_001, 50_000), "Limite di credito (NT$)"),
        (axes[1], "AGE", np.arange(20, 81, 2), "Eta' (anni)"),
    ):
        for label, color, nome in ((0, C_NO_DEFAULT, "Non insolvente"), (1, C_DEFAULT, "Insolvente")):
            serie = df.loc[df[TARGET] == label, col]
            ax.hist(
                serie,
                bins=bins,
                weights=np.full(len(serie), 100 / len(serie)),
                histtype="step",
                linewidth=2,
                color=color,
                label=nome,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("% dei clienti della classe")
        style_axes(ax)

    axes[0].set_title("Limite di credito per classe")
    axes[0].xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    axes[1].set_title("Eta' per classe")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "02_limitbal_age_per_classe.png")
    mesi = ["Set", "Ago", "Lug", "Giu", "Mag", "Apr"]
    x = np.arange(6)
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, cols, titolo, ylab, fmt in (
        (axes[0], BILL_COLS, "Estratto conto mensile (mediana)", "BILL_AMT mediano (NT$)", "k"),
        (axes[1], PAY_AMT_COLS, "Pagamento effettuato (mediana)", "PAY_AMT mediano (NT$)", "unita"),
    ):
        for i, (label, color, nome) in enumerate(
            ((0, C_NO_DEFAULT, "Non insolvente"), (1, C_DEFAULT, "Insolvente"))
        ):
            vals = df.loc[df[TARGET] == label, cols].median().values
            # offset +-w/2 con barre larghe w*0.95: resta un piccolo stacco tra le due serie
            ax.bar(x + (i - 0.5) * w, vals, width=w * 0.95, color=color, label=nome)
        ax.set_xticks(x, mesi)
        ax.set_xlabel("Mese (da settembre a aprile 2005)")
        ax.set_ylabel(ylab)
        ax.set_title(titolo)
        if fmt == "k":
            ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        else:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        style_axes(ax)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "03_bill_pay_amt_per_classe.png")

    bill_med = df.groupby(TARGET)[BILL_COLS].median()
    pay_med = df.groupby(TARGET)[PAY_AMT_COLS].median()
    print(
        "\n-> BILL_AMT* (quanto e' esposto il cliente): le mediane delle due classi sono"
        f"\n   simili (es. BILL_AMT1: {bill_med.loc[0, 'BILL_AMT1']:,.0f} vs"
        f" {bill_med.loc[1, 'BILL_AMT1']:,.0f} NT$)."
        "\n   Il saldo in se' non distingue: e' normale avere un estratto conto alto."
        "\n-> PAY_AMT* (quanto il cliente PAGA davvero): qui la differenza e' netta, gli"
        f"\n   insolventi pagano circa la meta' (es. PAY_AMT1: {pay_med.loc[0, 'PAY_AMT1']:,.0f} vs"
        f" {pay_med.loc[1, 'PAY_AMT1']:,.0f} NT$)."
        "\n   Non conta quanto devi, conta quanto stai rimborsando rispetto a quanto devi:"
        "\n   nel task 4 vale la pena costruire feature derivate come il rapporto"
        "\n   PAY_AMT / BILL_AMT e l'utilizzo del fido BILL_AMT / LIMIT_BAL."
    )
def categorical_counts(df: pd.DataFrame) -> None:
    section("7. VARIABILI CATEGORICHE (SEX, EDUCATION, MARRIAGE)")

    specs = [("SEX", SEX_LABELS), ("EDUCATION", EDUCATION_LABELS), ("MARRIAGE", MARRIAGE_LABELS)]

    for col, labels in specs:
        tab = pd.DataFrame(
            {
                "n": df[col].value_counts().sort_index(),
                "% dataset": df[col].value_counts(normalize=True).sort_index() * 100,
                "% default": df.groupby(col)[TARGET].mean() * 100,
            }
        )
        tab.index = [f"{code} = {labels.get(code, '?')}".replace("\n", " ") for code in tab.index]
        print(f"\n--- {col} ---")
        print(tab.round(2).to_string())

    base = df[TARGET].mean() * 100
    print(f"\n(tasso di default medio del dataset: {base:.2f}%)")

    print(
        "\n-> Le categorie sono molto squilibrate nella numerosita': EDUCATION e MARRIAGE"
        "\n   hanno codici non documentati (0, 5, 6) con pochissime osservazioni. Stimare un"
        "\n   tasso di default su poche decine di casi non e' affidabile: nel task 4 questi"
        "\n   codici vanno accorpati alla categoria 'altro' invece di restare classi a se'."
        "\n-> I tassi di default si muovono poco attorno alla media generale: queste tre"
        "\n   variabili anagrafiche sono predittori DEBOLI. L'unico andamento leggibile e'"
        "\n   su EDUCATION, dove il titolo di studio piu' basso mostra il tasso piu' alto."
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), height_ratios=[1, 2.4, 1.5])
    for ax, (col, labels) in zip(axes, specs):
        vc = df[col].value_counts().sort_index()
        rate = df.groupby(col)[TARGET].mean() * 100
        ypos = np.arange(len(vc))[::-1]  
        names = [labels.get(code, str(code)) for code in vc.index]
        ax.barh(ypos, 100 - rate.values - 0.5, height=0.6, color=C_NO_DEFAULT, label="Non insolvente")
        ax.barh(
            ypos,
            rate.values,
            left=100 - rate.values,
            height=0.6,
            color=C_DEFAULT,
            label="Insolvente",
        )
        for y, r, n in zip(ypos, rate.values, vc.values):
            ax.annotate(
                f"{r:.1f}% default",
                (100 - r, y),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8.5,
                color="white",
            )
            ax.annotate(
                f"n = {n:,}",
                (100.8, y),
                va="center",
                fontsize=8.5,
                color=INK_2,
                annotation_clip=False,
            )
        ax.axvline(100 - base, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
        ax.set_yticks(ypos, names)
        ax.set_xlim(0, 100)
        ax.set_xlabel("")
        ax.set_title(col, loc="left")
        ax.set_axisbelow(True)
        ax.grid(axis="x", linewidth=0.8, color=GRID)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)

    axes[-1].set_xlabel(f"% dei clienti della categoria  (linea tratteggiata = media dataset, {base:.1f}%)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.28), ncols=2)
    fig.suptitle("Composizione di ogni categoria: quota di insolventi e numerosita'", y=1.0)
    fig.tight_layout()
    save(fig, "04_categoriche_composizione.png")
def payment_status(df: pd.DataFrame) -> None:
    section("8. STORICO DEI PAGAMENTI (PAY_0 ... PAY_6) E RELAZIONE CON IL DEFAULT")

    base = df[TARGET].mean() * 100
    tab = pd.DataFrame(
        {
            "n": df["PAY_0"].value_counts().sort_index(),
            "% default": df.groupby("PAY_0")[TARGET].mean() * 100,
        }
    )
    print("--- PAY_0 (stato del pagamento di settembre, il mese piu' recente) ---")
    print(tab.round(2).to_string())
    print("\n(-2 = nessun utilizzo, -1 = saldato per intero, 0 = pagamento minimo in regola,")
    print(" 1..8 = mesi di ritardo accumulati)")

    in_regola = df.loc[df["PAY_0"] <= 0, TARGET].mean() * 100
    in_ritardo = df.loc[df["PAY_0"] >= 1, TARGET].mean() * 100
    quota_ritardo = (df["PAY_0"] >= 1).mean() * 100
    print(
        f"\nClienti in regola a settembre (PAY_0 <= 0): {100 - quota_ritardo:.1f}% del dataset,"
        f" tasso di default {in_regola:.1f}%"
        f"\nClienti in ritardo a settembre (PAY_0 >= 1): {quota_ritardo:.1f}% del dataset,"
        f" tasso di default {in_ritardo:.1f}%"
        f"\n-> Il rischio passa da {in_regola:.1f}% a {in_ritardo:.1f}%: piu' di"
        f" {in_ritardo / in_regola:.1f} volte."
    )

    n_ritardi = (df[PAY_COLS] >= 1).sum(axis=1)
    per_n = df.groupby(n_ritardi)[TARGET].agg(["size", "mean"])
    per_n["mean"] *= 100
    per_n.columns = ["n clienti", "% default"]
    per_n.index.name = "mesi in ritardo (su 6)"
    print("\n--- Numero di mesi in ritardo negli ultimi 6 mesi ---")
    print(per_n.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), width_ratios=[1.35, 1])

    # Pannello A: tasso di default per valore di PAY_0
    vals = tab.index.tolist()
    ax = axes[0]
    bars = ax.bar(
        range(len(vals)),
        tab["% default"].values,
        width=0.62,
        color=[C_DEFAULT if v >= 1 else C_NO_DEFAULT for v in vals],
    )
    label_bars(ax, bars, tab["n"].values)
    ax.axhline(
        base,
        color=INK,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        zorder=3,
        label=f"media dataset ({base:.1f}%)",
    )
    ax.legend(loc="upper left")
    ax.set_xticks(range(len(vals)), [PAY_LABELS.get(v, str(v)) for v in vals], fontsize=8)
    ax.set_title("Tasso di default per stato del pagamento piu' recente (PAY_0)")
    ax.set_xlabel("PAY_0  (da 1 in poi: mesi di ritardo accumulati)")
    ax.set_ylabel("% di insolventi nella categoria")
    ax.set_ylim(0, 105)
    style_axes(ax)

    # Pannello B: tasso di default per numero di mesi in ritardo
    ax = axes[1]
    bars = ax.bar(
        per_n.index.astype(str),
        per_n["% default"].values,
        width=0.62,
        color=C_DEFAULT,
    )
    label_bars(ax, bars, per_n["n clienti"].values)
    ax.axhline(
        base,
        color=INK,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        zorder=3,
        label=f"media dataset ({base:.1f}%)",
    )
    ax.legend(loc="upper left")
    ax.set_title("Tasso di default per numero di mesi in ritardo")
    ax.set_xlabel("Mesi con ritardo negli ultimi 6")
    ax.set_ylabel("% di insolventi")
    ax.set_ylim(0, 105)
    style_axes(ax)

    fig.tight_layout()
    save(fig, "05_pay_status_vs_default.png")

    print(
        "\n-> E' la relazione piu' forte di tutta l'analisi ed e' anche la piu' intuitiva:"
        "\n   il comportamento di pagamento recente predice il comportamento futuro."
        "\n   Il tasso di default cresce in modo monotono con i mesi di ritardo: chi a"
        "\n   settembre ha gia' due mesi di arretrato va in default nel 69% dei casi."
        "\n   Attenzione pero' alla numerosita': i"
        "\n   ritardi lunghi (5+ mesi) riguardano poche decine di clienti, quindi quelle"
        "\n   percentuali sono stime instabili."
        "\n-> Nota sui codici: -2 (nessun utilizzo della carta) e -1 (saldo pagato per"
        "\n   intero) hanno un tasso di default piu' BASSO di 0 (pagamento minimo in"
        "\n   regola). L'ordine numerico -2 < -1 < 0 non riflette quindi un ordine di"
        "\n   rischio crescente: trattare PAY_* come variabile puramente numerica e'"
        "\n   un'approssimazione, e nel task 4 conviene valutare un raggruppamento"
        "\n   (es. 'non utilizza / in regola / in ritardo di k mesi')."
    )
def correlation_matrix(df: pd.DataFrame) -> None:
    section("9. MATRICE DI CORRELAZIONE TRA LE VARIABILI NUMERICHE")

    cols = NUMERIC_COLS + [TARGET]
    corr = df[cols].corr(method="pearson")

    print("--- Correlazione di ogni feature con il target (ordinata per valore assoluto) ---")
    with_target = corr[TARGET].drop(TARGET)
    ordered = with_target.reindex(with_target.abs().sort_values(ascending=False).index)
    print(ordered.round(3).to_string())

    bill_corr = corr.loc[BILL_COLS, BILL_COLS].values
    off_diag = bill_corr[~np.eye(6, dtype=bool)]
    pay_corr = corr.loc[PAY_COLS, PAY_COLS].values
    pay_off = pay_corr[~np.eye(6, dtype=bool)]
    payamt_corr = corr.loc[PAY_AMT_COLS, PAY_AMT_COLS].values
    payamt_off = payamt_corr[~np.eye(6, dtype=bool)]

    print("\n--- Ridondanza interna ai blocchi (correlazione media fuori diagonale) ---")
    print(f"  BILL_AMT1..6 : {off_diag.mean():.3f}   (min {off_diag.min():.3f}, max {off_diag.max():.3f})")
    print(f"  PAY_0..PAY_6 : {pay_off.mean():.3f}   (min {pay_off.min():.3f}, max {pay_off.max():.3f})")
    print(f"  PAY_AMT1..6  : {payamt_off.mean():.3f}   (min {payamt_off.min():.3f}, max {payamt_off.max():.3f})")

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(corr.values, cmap=CORR_CMAP, vmin=-1, vmax=1)

    labels = [c.replace(TARGET, "DEFAULT (target)") for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), labels, fontsize=8)
    # Separatori tra i blocchi tematici (anagrafica | PAY_* | BILL_AMT* | PAY_AMT* | target)
    for pos in (1.5, 7.5, 13.5, 19.5):
        ax.axhline(pos, color=SURFACE, linewidth=2)
        ax.axvline(pos, color=SURFACE, linewidth=2)
    ax.set_title("Correlazione di Pearson tra le variabili numeriche", pad=12)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.72, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("coefficiente di correlazione", color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=MUTED, labelcolor=INK_2)
    fig.tight_layout()
    save(fig, "06_matrice_correlazione.png")

    print(
        "\n-> I sei BILL_AMT* sono quasi la stessa variabile ripetuta"
        f" (correlazione media {off_diag.mean():.2f}):"
        "\n   il saldo di un mese e' quasi identico a quello del mese precedente. E'"
        "\n   ridondanza (multicollinearita'): per una regressione logistica i coefficienti"
        "\n   diventano instabili e non interpretabili. Possibili rimedi nel task 4: tenerne"
        "\n   uno solo, sostituirli con una sintesi (media, trend, utilizzo del fido) oppure"
        "\n   applicare una PCA al blocco."
        f"\n-> Anche i PAY_* sono correlati tra loro ({pay_off.mean():.2f}) ma meno: ogni mese"
        "\n   aggiunge informazione sull'evoluzione del ritardo."
        f"\n-> I PAY_AMT* sono invece poco correlati tra loro ({payamt_off.mean():.2f}): quanto si"
        "\n   paga varia molto di mese in mese."
        "\n-> Rispetto al target la correlazione lineare piu' alta e' quella dei PAY_*"
        f" (PAY_0: {corr.loc['PAY_0', TARGET]:.3f});"
        f"\n   LIMIT_BAL e' negativa ({corr.loc['LIMIT_BAL', TARGET]:.3f}), i BILL_AMT* sono"
        "\n   praticamente nulli. Sono valori bassi in assoluto, ma la correlazione di Pearson"
        "\n   misura solo legami LINEARI con un target binario: come si e' visto alla sezione 8"
        "\n   la relazione tra ritardi e default e' forte, quindi modelli non lineari (alberi,"
        "\n   gradient boosting) possono sfruttare molto piu' di quanto questa matrice suggerisca."
    )
def eda_conclusions(df: pd.DataFrame) -> None:
    section("10. OSSERVAZIONI CONCLUSIVE DELL'EDA")

    base = df[TARGET].mean() * 100
    in_regola = df.loc[df["PAY_0"] <= 0, TARGET].mean() * 100
    in_ritardo = df.loc[df["PAY_0"] >= 1, TARGET].mean() * 100

    print(
        f"1. Classi sbilanciate ma gestibili: {base:.1f}% di insolventi. Serve stratificazione"
        "\n   e metriche oltre l'accuracy (recall e F1 sulla classe 1, ROC-AUC)."
        "\n"
        "\n2. Lo storico dei pagamenti e' il segnale dominante. Un cliente in regola a"
        f"\n   settembre ha un rischio del {in_regola:.1f}%, uno gia' in ritardo del {in_ritardo:.1f}%."
        "\n   Il rischio cresce in modo monotono con il numero di mesi di ritardo."
        "\n"
        "\n3. Quanto si deve (BILL_AMT*) non distingue le due classi; quanto si PAGA"
        "\n   (PAY_AMT*) si'. Le feature interessanti sono i rapporti, non i valori assoluti:"
        "\n   PAY_AMT/BILL_AMT (capacita' di rimborso) e BILL_AMT/LIMIT_BAL (utilizzo del fido)."
        "\n"
        "\n4. Il limite di credito e' l'unica variabile 'contrattuale' con un segnale chiaro:"
        "\n   fido piu' basso -> piu' default, perche' riflette una valutazione di rischio"
        "\n   gia' fatta dalla banca al momento della concessione."
        "\n"
        "\n5. Le variabili anagrafiche (SEX, AGE, EDUCATION, MARRIAGE) separano poco. Vanno"
        "\n   comunque trattate correttamente (one-hot per le categoriche, accorpamento dei"
        "\n   codici non documentati), ma non ci si aspetta che siano i predittori principali."
        "\n   Vale anche la pena ricordare che usare SEX come predittore in un modello di"
        "\n   credito ha implicazioni di equita' da valutare esplicitamente."
        "\n"
        "\n6. Forte ridondanza tra i sei BILL_AMT*: da ridurre prima di usare modelli lineari."
        "\n"
        "\n7. Le distribuzioni degli importi sono molto asimmetriche e su scale diverse da"
        "\n   quelle di AGE o dei PAY_*: per i modelli sensibili alla scala (regressione"
        "\n   logistica, KNN, SVM) servira' una standardizzazione nel task 4."
    )
def data_cleaning(df: pd.DataFrame, n_removed_before: int = 0) -> pd.DataFrame:
    section("11. DATA CLEANING: CATEGORIE ANOMALE E CODICI PAY_*")

    df_out = df.copy()
    n_rows = len(df_out)

    # --- 11a. EDUCATION e MARRIAGE: codici fuori documentazione -------------
    print("--- Codici non documentati: quanti sono e come si comportano ---")
    anomalies = {}
    for col, valid, fallback in (("EDUCATION", DOCUMENTED_CODES["EDUCATION"], 4),
                                 ("MARRIAGE", DOCUMENTED_CODES["MARRIAGE"], 3)):
        codes = sorted(int(v) for v in set(df_out[col].unique()) - valid)
        anomalies[col] = (codes, fallback)
        tab = pd.DataFrame(
            {
                "n": df_out[col].value_counts(),
                "% dataset": df_out[col].value_counts(normalize=True) * 100,
                "% default": df_out.groupby(col)[TARGET].mean() * 100,
            }
        ).loc[codes]
        print(f"\n{col}: codici anomali {codes} -> verranno accorpati nel codice {fallback} ('altro')")
        print(tab.round(2).to_string())

    mask_edu = df_out["EDUCATION"].isin(anomalies["EDUCATION"][0])
    mask_mar = df_out["MARRIAGE"].isin(anomalies["MARRIAGE"][0])
    n_edu, n_mar = int(mask_edu.sum()), int(mask_mar.sum())
    n_touched = int((mask_edu | mask_mar).sum())

    print(
        f"\nRighe con EDUCATION anomala : {n_edu:>4}  ({n_edu / n_rows:.2%} del dataset)"
        f"\nRighe con MARRIAGE anomala  : {n_mar:>4}  ({n_mar / n_rows:.2%} del dataset)"
        f"\nRighe coinvolte in totale   : {n_touched:>4}  ({n_touched / n_rows:.2%} del dataset)"
    )

    print(
        "\n-> DECISIONE: accorpare, non rimuovere. Le motivazioni, in ordine di peso:"
        f"\n   1) sono poche ({n_touched / n_rows:.2%}) ma non pochissime: buttarle via significa"
        "\n      perdere qualche centinaio di osservazioni valide, in cui le altre 22 variabili"
        "\n      (limite di credito, storico pagamenti, importi) sono perfettamente utilizzabili;"
        "\n   2) rimuovere righe in base al valore di una feature introduce un bias di selezione:"
        "\n      il campione di training smetterebbe di rappresentare la popolazione reale;"
        "\n   3) soprattutto, il problema non sparirebbe: in produzione arriverebbero comunque"
        "\n      clienti con quei codici, e un modello che non sa gestirli sarebbe inutilizzabile"
        "\n      proprio su quei casi. Accorpandoli in 'altro' il modello impara a trattarli;"
        "\n   4) semanticamente e' coerente: EDUCATION=4 e MARRIAGE=3 sono gia' la categoria"
        "\n      'altro' prevista dalla documentazione, quindi 'non dichiarato' ci sta dentro."
        "\n   Se invece fossero state migliaia di righe con un tasso di default molto diverso,"
        "\n   avrebbe avuto senso tenerle come categoria separata anziche' diluirle in 'altro'."
    )

    for col, (codes, fallback) in anomalies.items():
        df_out[col] = df_out[col].replace({code: fallback for code in codes})
        print(f"\n{col} dopo l'accorpamento: {sorted(int(v) for v in df_out[col].unique())}")
    print("\n--- Verifica dei valori nelle colonne PAY_* ---")
    pay_values = pd.DataFrame(
        {col: df_out[col].value_counts() for col in PAY_COLS}
    ).sort_index().fillna(0).astype(int)
    print(pay_values.to_string())
    documented_pay = {-1} | set(range(1, 10))
    present = set(pd.unique(df_out[PAY_COLS].to_numpy().ravel()).tolist())
    extra = sorted(int(v) for v in present - documented_pay)

    n_cells = df_out[PAY_COLS].size
    n_extra = int(df_out[PAY_COLS].isin(extra).to_numpy().sum())
    print(f"\nValori presenti ma non descritti nel paper: {extra}")
    print(
        f"   ({n_extra:,} celle su {n_cells:,}, il {n_extra / n_cells:.1%} del totale)"
        "\n-> Sono i due codici PIU' frequenti del dataset, quindi non errori di battitura ma"
        "\n   stati reali del conto che il paper semplicemente non enumera. La letteratura sul"
        "\n   dataset li interpreta cosi': -2 = 'nessun consumo, carta non utilizzata nel mese',"
        "\n   0 = 'pagamento minimo effettuato, nessun ritardo'. Vanno quindi interpretati e"
        "\n   tenuti, non trattati come dati sporchi: eliminarli o azzerarli significherebbe"
        f"\n   buttare via il {n_extra / n_cells:.0%} dell'informazione sullo storico dei pagamenti,"
        "\n   che la sezione 8 ha mostrato essere il predittore piu' forte del dataset."
    )

    rates = {v: df_out.loc[df_out["PAY_0"] == v, TARGET].mean() * 100 for v in (-2, -1, 0)}
    print(
        "\n-> DECISIONE sui PAY_*: tenerli come variabili ORDINALI NUMERICHE, senza modifiche."
        "\n   Motivazione: da 1 in poi la scala e' un ordine di rischio pulito e monotono"
        "\n   (piu' mesi di ritardo -> piu' default, come mostrato nella sezione 8), ed e'"
        "\n   esattamente l'informazione che serve al modello. La parte non monotona e' solo"
        f"\n   in fondo alla scala (-2: {rates[-2]:.1f}%, -1: {rates[-1]:.1f}%, 0: {rates[0]:.1f}% di default):"
        "\n   valori vicini tra loro, quindi l'errore che si commette trattandoli come numeri"
        "\n   e' piccolo. In cambio si evita di creare ~60 colonne dummy (6 mesi x ~10 codici),"
        "\n   che gonfierebbero lo spazio delle feature con categorie da poche decine di casi."
        "\n   Gli alberi, inoltre, possono comunque isolare i singoli valori con dei split."
        "\n   -> 0 righe modificate su questa famiglia di colonne."
    )

    # --- 11c. Riepilogo per la relazione ------------------------------------
    print("\n--- RIEPILOGO MODIFICHE (da riportare in relazione) ---")
    recap = pd.DataFrame(
        [
            ("EDUCATION: codici 0/5/6 -> 4 ('altro')", n_edu, 0),
            ("MARRIAGE: codice 0 -> 3 ('altro')", n_mar, 0),
            ("PAY_0..PAY_6: nessuna modifica", 0, 0),
        ],
        columns=["intervento", "righe modificate", "righe rimosse"],
    )
    print(recap.to_string(index=False))
    print(
        f"\nRighe modificate almeno una volta : {n_touched:,} su {n_rows:,} ({n_touched / n_rows:.2%})"
        f"\nRighe rimosse in questo task      : 0"
        f"\nRighe rimosse nel task 1 (duplicati): {n_removed_before}"
        f"\nDimensione del dataset: invariata, {df_out.shape[0]:,} righe x {df_out.shape[1]} colonne"
    )

    return df_out
def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    section("12. ENCODING DELLE VARIABILI CATEGORICHE")

    y = df[TARGET]
    X_raw = df.drop(columns=[TARGET])

    print("--- Variabili nominali da codificare ---")
    for col in CATEGORICAL_COLS:
        codici = sorted(int(v) for v in df[col].unique())
        print(f"  {col:<10} {df[col].nunique()} categorie: {codici}")

    print(
        "\n-> One-hot encoding con drop_first=True (k categorie -> k-1 colonne 0/1)."
        "\n   Perche' one-hot: SEX, EDUCATION e MARRIAGE sono NOMINALI, i numeri sono solo"
        "\n   etichette. Lasciandole come interi un modello lineare leggerebbe"
        "\n   'EDUCATION=3' come il triplo di 'EDUCATION=1' e stimerebbe un unico coefficiente"
        "\n   lungo un ordine inventato."
        "\n   Perche' drop_first: la categoria omessa diventa il riferimento e si evita la"
        "\n   'dummy variable trap' (le k colonne sommano sempre a 1, quindi sono linearmente"
        "\n   dipendenti dall'intercetta: per la regressione logistica i coefficienti"
        "\n   diventerebbero indeterminati). L'informazione non si perde: 'tutte le dummy a 0'"
        "\n   identifica la categoria di riferimento."
    )

    X = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    n_num = len(NUMERIC_COLS)
    n_dummy_atteso = sum(df[col].nunique() - 1 for col in CATEGORICAL_COLS)
    n_atteso = n_num + n_dummy_atteso
    dummy_cols = [c for c in X.columns if c not in NUMERIC_COLS]

    print("\n--- Verifica di coerenza del numero di feature ---")
    print(f"  numeriche/ordinali mantenute      : {n_num:>3}  (LIMIT_BAL, AGE, 6 PAY_*, 6 BILL_AMT*, 6 PAY_AMT*)")
    for col in CATEGORICAL_COLS:
        k = df[col].nunique()
        etichetta = f"dummy da {col}"
        print(f"  {etichetta:<34}: {k - 1:>3}  ({k} categorie - 1 di riferimento)")
    print(f"  {'TOTALE atteso':<34}: {n_atteso:>3}")
    print(f"  {'TOTALE effettivo':<34}: {X.shape[1]:>3}  -> {'OK' if X.shape[1] == n_atteso else 'INCOERENTE'}")
    print(f"\n  Colonne dummy create: {dummy_cols}")
    print(f"  Matrice delle feature X: {X.shape[0]:,} righe x {X.shape[1]} colonne")
    print(f"  Vettore target y: {y.shape[0]:,} valori, {y.mean():.2%} di classe 1")

    assert X.shape[1] == n_atteso, "numero di feature incoerente dopo l'encoding"
    assert X.isna().sum().sum() == 0, "valori mancanti introdotti dall'encoding"
    print("\n-> Nessun valore mancante introdotto e conteggio delle colonne coerente.")

    return X, y


# =========================================================================== #
# TASK 5 - SPLIT TRAIN / TEST
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 13. Divisione in training set e test set
# --------------------------------------------------------------------------- #
def split_data(X: pd.DataFrame, y: pd.Series):
    section("13. SPLIT TRAIN / TEST")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,          # mantiene la proporzione di insolventi nei due insiemi
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print(
        f"Split scelto: {100 * (1 - TEST_SIZE):.0f}% training / {100 * TEST_SIZE:.0f}% test"
        f"\n  training : {len(X_train):>7,} righe"
        f"\n  test     : {len(X_test):>7,} righe"
    )
    print(
        "\n-> Perche' 80/20: con ~30.000 osservazioni il 20% lascia circa 6.000 casi di test,"
        f"\n   di cui ~{int(len(X_test) * y.mean()):,} insolventi. E' abbastanza per stimare"
        "\n   precision e recall con un errore standard contenuto (~0.5-1 punto percentuale),"
        "\n   e allo stesso tempo lascia al training la massima quantita' di dati possibile."
        "\n   Con dataset piccoli si sarebbe scelto 70/30 (test piu' affidabile) o una nested"
        "\n   CV; qui non serve."
    )

    print("\n--- Verifica della stratificazione ---")
    check = pd.DataFrame(
        {
            "n": [len(y), len(y_train), len(y_test)],
            "% classe 1": [y.mean() * 100, y_train.mean() * 100, y_test.mean() * 100],
        },
        index=["dataset completo", "training set", "test set"],
    )
    print(check.round(3).to_string())
    print(
        "\n-> stratify=y e' obbligatorio con classi sbilanciate: senza, uno split casuale"
        "\n   potrebbe assegnare al test una percentuale di insolventi diversa da quella reale"
        "\n   e le metriche misurerebbero anche quella distorsione, non solo il modello."
        f"\n-> random_state={RANDOM_STATE}: seed fisso, quindi lo split (e tutti i risultati"
        "\n   che ne derivano) e' riproducibile identico ad ogni esecuzione. Da dichiarare in"
        "\n   relazione: senza seed i numeri cambierebbero ad ogni run."
        "\n-> Il test set da qui in poi non viene piu' toccato fino alla sezione 16: tutte le"
        "\n   scelte (iperparametri, scaler, modello) si fanno solo sul training set."
    )

    return X_train, X_test, y_train, y_test

def make_scaler() -> ColumnTransformer:
    return ColumnTransformer(
        [("scale", StandardScaler(), CONTINUOUS_COLS)],
        remainder="passthrough",
    )
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    section("14. SCALAMENTO DELLE FEATURE")

    print("--- Feature da scalare ---")
    print(f"  continue (standardizzate) : {CONTINUOUS_COLS}")
    print(f"  ordinali (lasciate come sono): {PAY_COLS}")
    print(f"  dummy 0/1 (gia' sulla stessa scala): {[c for c in X_train.columns if c not in NUMERIC_COLS]}")

    print("\n--- Perche' servono scale confrontabili: ordini di grandezza attuali ---")
    ranges = pd.DataFrame(
        {
            "min": X_train[NUMERIC_COLS].min(),
            "max": X_train[NUMERIC_COLS].max(),
            "std": X_train[NUMERIC_COLS].std(),
        }
    )
    print(ranges.round(1).to_string())

    print(
        "\n-> SCALER SCELTO: StandardScaler (z-score: (x - media) / deviazione standard)."
        "\n   Perche' serve: BILL_AMT arriva a ~1.000.000 mentre AGE sta sotto 80. La"
        "\n   regressione logistica regolarizzata (L2) penalizza i coefficienti grandi, quindi"
        "\n   senza standardizzazione la penalita' colpirebbe di piu' le variabili con valori"
        "\n   piccoli, solo per una questione di unita' di misura; l'SVM, che si basa su"
        "\n   distanze, verrebbe di fatto deciso dai soli importi. Standardizzando, tutte le"
        "\n   feature partono con lo stesso peso e i coefficienti diventano confrontabili."
        "\n   Perche' StandardScaler e non MinMaxScaler: gli importi hanno outlier estremi"
        "\n   (fino a 1,7 milioni di NT$); MinMax schiaccerebbe il 99% dei dati in una"
        "\n   fascia strettissima vicino a 0. (RobustScaler, basato su mediana e IQR, sarebbe"
        "\n   l'alternativa piu' difendibile ed e' un possibile miglioramento.)"
        "\n-> Alberi, Random Forest e gradient boosting NON ne hanno bisogno: tagliano su"
        "\n   soglie del singolo attributo, e una trasformazione monotona non cambia gli"
        "\n   split. Per questo lo scaler viaggia dentro una Pipeline ed e' attivo solo per"
        "\n   i modelli che ne beneficiano."
    )
    scaler = make_scaler()
    X_train_scaled = scaler.fit_transform(X_train)   
    X_test_scaled = scaler.transform(X_test)         

    print("\n--- Fit sul solo training set, transform su entrambi ---")
    n_cont = len(CONTINUOUS_COLS)
    verifica = pd.DataFrame(
        {
            "media train (dopo)": X_train_scaled[:, :n_cont].mean(axis=0),
            "std train (dopo)": X_train_scaled[:, :n_cont].std(axis=0),
            "media test (dopo)": X_test_scaled[:, :n_cont].mean(axis=0),
            "std test (dopo)": X_test_scaled[:, :n_cont].std(axis=0),
        },
        index=CONTINUOUS_COLS,
    )
    print(verifica.round(4).to_string())
    print(
        "\n-> Il training set ha ora media 0 e deviazione standard 1 su ogni colonna continua."
        "\n   Il test set NON e' esattamente centrato su 0: e' la prova che lo scaler e' stato"
        "\n   calcolato solo sul training. Se avessimo fatto fit sull'intero dataset, media e"
        "\n   deviazione standard del test sarebbero entrate nella trasformazione: e' data"
        "\n   leakage, e le metriche finali risulterebbero ottimisticamente gonfiate."
        "\n-> ATTENZIONE a come viene usato tutto questo: le due matrici calcolate qui sopra"
        "\n   servono SOLO a produrre questa verifica. Ai modelli non vengono passate: nella"
        "\n   sezione 15 lo scaler sta dentro la Pipeline e viene rifittato sui soli dati di"
        "\n   addestramento di ciascun fold della cross-validation, quindi k volte anziche'"
        "\n   una. E' la versione corretta dello stesso principio: uno scaler fittato una"
        "\n   volta sola sull'intero training set farebbe entrare, in ogni fold, media e"
        "\n   deviazione standard calcolate anche sulla porzione usata per validare."
    )

def build_models() -> dict:
    return {
        "Regressione logistica": (
            Pipeline(
                [
                    ("prep", make_scaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__C": [0.01, 0.1, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
            True,
        ),
        "SVM lineare": (
            Pipeline(
                [
                    ("prep", make_scaler()),
                    ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__C": [0.01, 0.1, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
            True,
        ),
        "Albero decisionale": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [3, 5, 10],
                "min_samples_leaf": [20, 100],
                "class_weight": [None, "balanced"],
            },
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            {
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 10],
                "class_weight": [None, "balanced"],
            },
            False,
        ),
        "Gradient boosting": (
            HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_STATE),
            {
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "class_weight": [None, "balanced"],
            },
            False,
        ),
    }
def model_selection(X_train: pd.DataFrame, y_train: pd.Series):
    section("15. MODEL SELECTION CON CROSS-VALIDATION")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall"}

    print(
        f"Schema: StratifiedKFold(k={CV_FOLDS}, shuffle=True, random_state={RANDOM_STATE})"
        f" su {len(X_train):,} righe di training."
        "\n\n-> METRICA DI SELEZIONE: F1 sulla classe 1 (insolventi)."
        "\n   Perche' non l'accuracy: con il 22% di positivi, il modello 'predici sempre non"
        "\n   insolvente' raggiunge il 78% di accuracy senza individuare un solo insolvente."
        "\n   Perche' F1: e' la media armonica di precision e recall calcolate sulla classe"
        "\n   minoritaria, quindi premia solo i modelli che ne trovano tanti (recall) senza"
        "\n   inondare la banca di falsi allarmi (precision)."
        "\n   Viene comunque riportata anche la ROC-AUC, che misura la capacita' di ordinare"
        "\n   i clienti per rischio indipendentemente dalla soglia: e' utile perche' F1"
        "\n   dipende dalla soglia fissa 0.5, l'AUC no."
        "\n\n--- Griglia di iperparametri testata ---"
    )
    models = build_models()
    for nome, (_, grid, usa_scaler) in models.items():
        n_comb = int(np.prod([len(v) for v in grid.values()]))
        print(f"\n  {nome}  ({'con' if usa_scaler else 'senza'} standardizzazione)")
        for par, valori in grid.items():
            print(f"    {par:<22} {valori}")
        print(f"    -> {n_comb} combinazioni x {CV_FOLDS} fold = {n_comb * CV_FOLDS} addestramenti")

    print("\n--- Esecuzione della ricerca (puo' richiedere qualche minuto) ---")
    searches, rows = {}, []
    for nome, (estimator, grid, _) in models.items():
        t0 = perf_counter()
        gs = GridSearchCV(
            estimator,
            grid,
            scoring=scoring,
            refit="f1",          # il modello finale viene riaddestrato su tutto il training
            cv=cv,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        elapsed = perf_counter() - t0
        searches[nome] = gs

        i = gs.best_index_
        res = gs.cv_results_
        rows.append(
            {
                "modello": nome,
                "F1 (CV)": res["mean_test_f1"][i],
                "F1 std": res["std_test_f1"][i],
                "AUC (CV)": res["mean_test_roc_auc"][i],
                "precision (CV)": res["mean_test_precision"][i],
                "recall (CV)": res["mean_test_recall"][i],
                "tempo (s)": elapsed,
            }
        )
        print(f"  {nome:<22} F1={res['mean_test_f1'][i]:.4f}  AUC={res['mean_test_roc_auc'][i]:.4f}"
              f"  ({elapsed:.1f}s)")

    results = pd.DataFrame(rows).set_index("modello").sort_values("F1 (CV)", ascending=False)

    print("\n--- TABELLA RIASSUNTIVA DELLA CROSS-VALIDATION (da mettere in relazione) ---")
    print(results.round(4).to_string())

    print("\n--- Iperparametri migliori per ciascun modello ---")
    for nome in results.index:
        print(f"  {nome:<22} {searches[nome].best_params_}")

    winner = results.index[0]
    print(
        f"\n-> MODELLO VINCITORE: {winner}, F1 medio in CV = {results.loc[winner, 'F1 (CV)']:.4f}"
        f" (deviazione standard tra i fold {results.loc[winner, 'F1 std']:.4f})."
        "\n   La deviazione standard bassa tra i fold indica che il risultato non dipende da"
        "\n   quale porzione di dati capita in validazione: la stima e' stabile."
        "\n-> Nota sul class_weight='balanced': dove viene scelto, aumenta la recall a scapito"
        "\n   della precision. E' esattamente il compromesso che l'F1 arbitra."
    )

    best_auc = results["AUC (CV)"].idxmax()
    if best_auc != winner:
        print(
            f"-> Nota: {best_auc} ha l'AUC piu' alta ({results.loc[best_auc, 'AUC (CV)']:.4f} contro"
            f" {results.loc[winner, 'AUC (CV)']:.4f}),\n   quindi ordina i clienti per rischio un"
            " filo meglio, ma alla soglia fissa 0.5 il compromesso\n   precision/recall premia"
            f" {winner}. Se in seguito si decidesse di tarare la soglia sui\n   costi aziendali,"
            f" {best_auc} sarebbe il candidato da riconsiderare."
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ordine = results.index[::-1]
    for ax, colonna, titolo, errore in (
        (axes[0], "F1 (CV)", f"F1 medio in {CV_FOLDS}-fold CV (metrica di selezione)", results.loc[ordine, "F1 std"]),
        (axes[1], "AUC (CV)", f"ROC-AUC media in {CV_FOLDS}-fold CV", None),
    ):
        colori = [C_DEFAULT if m == winner else C_NO_DEFAULT for m in ordine]
        bars = ax.barh(range(len(ordine)), results.loc[ordine, colonna], height=0.6, color=colori)
        if errore is not None:
            ax.errorbar(
                results.loc[ordine, colonna],
                range(len(ordine)),
                xerr=errore,
                fmt="none",
                ecolor=INK_2,
                elinewidth=1.2,
                capsize=3,
            )
        for bar, v in zip(bars, results.loc[ordine, colonna]):
            ax.annotate(
                f"{v:.3f}",
                (v, bar.get_y() + bar.get_height() / 2),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=INK,
            )
        ax.set_yticks(range(len(ordine)), ordine)
        ax.set_xlim(0, max(results[colonna]) * 1.25)
        ax.set_title(titolo, loc="left", fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", linewidth=0.8, color=GRID)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    fig.suptitle(
        f"Confronto dei modelli in cross-validation - in arancio il vincitore ({winner})",
        y=1.04,
        fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, "07_confronto_modelli_cv.png")

    return results, searches, winner
def _scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
def final_evaluation(results, searches, winner, X_test, y_test) -> None:
    section("16. VALUTAZIONE FINALE SUL TEST SET")

    gs = searches[winner]
    model = gs.best_estimator_   # gia' riaddestrato su tutto il training set (refit='f1')

    print(f"Modello applicato : {winner}")
    print(f"Iperparametri     : {gs.best_params_}")
    print(f"Test set          : {len(X_test):,} clienti mai visti, {y_test.mean():.2%} insolventi")
    print(
        "\n-> Il test set viene toccato UNA sola volta, ora, con il modello e gli"
        "\n   iperparametri gia' congelati: e' l'unica stima non ottimistica delle prestazioni."
    )

    y_pred = model.predict(X_test)
    y_score = _scores(model, X_test)

    metriche = pd.Series(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision (classe 1)": precision_score(y_test, y_pred),
            "recall (classe 1)": recall_score(y_test, y_pred),
            "F1 (classe 1)": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_score),
        }
    )
    print("\n--- Metriche sul test set ---")
    print(metriche.round(4).to_string())

    baseline_acc = 1 - y_test.mean()
    print(
        f"\n(riferimento: il modello banale 'nessuno va in default' avrebbe accuracy"
        f" {baseline_acc:.4f} e F1 = 0.0000 sulla classe 1)"
    )

    print("\n--- Report per classe ---")
    print(
        classification_report(
            y_test, y_pred, target_names=["non insolvente (0)", "insolvente (1)"], digits=3
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("--- Matrice di confusione ---")
    print(
        pd.DataFrame(
            cm,
            index=["reale: non insolvente", "reale: insolvente"],
            columns=["predetto: non insolvente", "predetto: insolvente"],
        ).to_string()
    )
    print(
        f"\n  Veri negativi  (TN) = {tn:>5}  clienti sani correttamente lasciati stare"
        f"\n  Falsi positivi (FP) = {fp:>5}  clienti sani segnalati per errore -> costo commerciale"
        f"\n  Falsi negativi (FN) = {fn:>5}  insolventi NON individuati -> costo di credito, il piu' caro"
        f"\n  Veri positivi  (TP) = {tp:>5}  insolventi individuati in anticipo"
        f"\n\n-> Il modello individua {tp / (tp + fn):.1%} degli insolventi reali; quando lancia un"
        f"\n   allarme ha ragione nel {tp / (tp + fp):.1%} dei casi. Restano {fn:,} insolventi non"
        "\n   intercettati: in un uso reale la soglia di decisione (qui il 0.5 di default) andrebbe"
        "\n   spostata in base al costo relativo di FP e FN, cosa che cambia precision e recall"
        "\n   senza richiedere un nuovo addestramento."
    )

    # --- Coerenza validation / test ---
    f1_cv = results.loc[winner, "F1 (CV)"]
    auc_cv = results.loc[winner, "AUC (CV)"]
    delta_f1 = metriche["F1 (classe 1)"] - f1_cv
    delta_auc = metriche["ROC-AUC"] - auc_cv

    print("\n--- Coerenza tra cross-validation e test ---")
    confronto = pd.DataFrame(
        {
            "cross-validation": [f1_cv, auc_cv],
            "test set": [metriche["F1 (classe 1)"], metriche["ROC-AUC"]],
            "differenza": [delta_f1, delta_auc],
        },
        index=["F1 (classe 1)", "ROC-AUC"],
    )
    print(confronto.round(4).to_string())

    soglia = 0.03
    if abs(delta_f1) <= soglia:
        giudizio = (
            f"Scarto di {abs(delta_f1):.4f} in F1 ({abs(delta_f1) / f1_cv:.1%} del valore in CV),"
            f" sotto la soglia di {soglia}\n   che ci si e' dati: nessun segnale di overfitting."
            " La differenza e' dello stesso ordine della\n   deviazione standard tra i fold"
            f" ({results.loc[winner, 'F1 std']:.4f}), quindi e' normale variabilita'"
            "\n   campionaria. Il modello generalizza a dati mai visti come faceva in"
            "\n   validazione: la scelta degli iperparametri non ha 'inseguito' il rumore dei fold."
        )
    elif delta_f1 < -soglia:
        giudizio = (
            "Il test e' sensibilmente peggiore della CV: possibile overfitting alla griglia di"
            "\n   iperparametri, oppure semplice variabilita' campionaria. Da approfondire."
        )
    else:
        giudizio = (
            "Il test e' migliore della CV: nessun overfitting, ma il test set potrebbe essere"
            "\n   'facile' per caso. Con questa numerosita' e' comunque una differenza plausibile."
        )
    print(f"\n-> {giudizio}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    seq = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#184f95"])
    ax.imshow(cm, cmap=seq, vmin=0, vmax=cm.max())
    etichette = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.annotate(
                f"{etichette[i][j]}\n{cm[i, j]:,}\n{cm[i, j] / cm.sum():.1%}",
                (j, i),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if cm[i, j] > cm.max() * 0.55 else INK,
            )
    ax.set_xticks([0, 1], ["predetto\nnon insolvente", "predetto\ninsolvente"])
    ax.set_yticks([0, 1], ["reale\nnon insolvente", "reale\ninsolvente"])
    ax.set_title(f"Matrice di confusione sul test set - {winner}", fontsize=10, loc="left")
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    ax.plot(fpr, tpr, color=C_NO_DEFAULT, linewidth=2, label=f"{winner} (AUC = {metriche['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="modello casuale (AUC = 0.5)")
    ax.set_xlabel("Falsi positivi / totale non insolventi")
    ax.set_ylabel("Veri positivi / totale insolventi")
    ax.set_title("Curva ROC sul test set", fontsize=10, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="both")

    fig.tight_layout()
    save(fig, "08_valutazione_finale.png")


def main() -> None:
    df_raw = load_raw()
    explore_shape_and_types(df_raw)
    check_missing(df_raw)
    check_duplicates(df_raw)
    df_clean = clean(df_raw)
    summary(df_raw, df_clean)
    FIG_DIR.mkdir(exist_ok=True)
    print(f"\nGrafici salvati in: {FIG_DIR.relative_to(BASE_DIR)}/")

    target_distribution(df_clean)
    numeric_distributions(df_clean)
    categorical_counts(df_clean)
    payment_status(df_clean)
    correlation_matrix(df_clean)
    eda_conclusions(df_clean)
    df_model = data_cleaning(df_clean, n_removed_before=len(df_raw) - len(df_clean))
    X, y = encode(df_model)
    X_train, X_test, y_train, y_test = split_data(X, y)
    scale_features(X_train, X_test)
    results, searches, winner = model_selection(X_train, y_train)
    final_evaluation(results, searches, winner, X_test, y_test)


if __name__ == "__main__":
    main()
