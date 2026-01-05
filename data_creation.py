import pandas as pd  

def load_raw_dataset(path: str) -> pd.DataFrame:
    # Carica il dataset CSV dal percorso indicato
    df = pd.read_csv(path)
    return df


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Elenco delle colonne considerate utili per l'analisi
    colonneUtili = [
        "original_title",
        "genres",
        "popularity",
        "vote_average",
        "vote_count",
        "runtime",
        "release_year"
    ]
    
    # Seleziona solo le colonne rilevanti
    df = df[colonneUtili]
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Crea una copia del DataFrame originale
    dfNew = df.copy()
    
    # Rimuove i duplicati
    dfNew = dfNew.drop_duplicates()
    
    # Rimuove le righe con valori mancanti nelle colonne principali
    dfNew = dfNew.dropna(subset=["popularity", "vote_average", "vote_count"])
    
    return dfNew


def save_clean_dataset(df: pd.DataFrame, path: str) -> None:
    # Salva il DataFrame pulito su file CSV senza l'indice
    df.to_csv(path, index=False)


def maindc():
    # Carica il dataset originale
    df = load_raw_dataset("data/movies_data.csv")
    
    # Seleziona solo le colonne rilevanti
    df = select_relevant_columns(df)
    
    # Pulisce il dataset
    dfNew = clean_dataset(df)
    
    # Salva il dataset pulito
    save_clean_dataset(dfNew, "data/movies_clean.csv")
