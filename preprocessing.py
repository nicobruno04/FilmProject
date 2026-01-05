import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_clean_dataset(path: str) -> pd.DataFrame:
    # Carica il dataset pulito da file CSV
    df = pd.read_csv(path)
    return df


def encode_genres(df: pd.DataFrame, topk: int = 8) -> pd.DataFrame:
    # Funzione interna per trasformare la stringa dei generi in una lista
    def parse_genres(st):
        if pd.isna(st):
            return []
        parts = st.split('|')      # I generi sono separati da "|"
        genres = []
        for genre in parts:
            g = genre.strip()
            if g != "":
                genres.append(g)
        return genres

    # Crea una nuova colonna con la lista dei generi
    df['lista_generi'] = df['genres'].apply(parse_genres)

    # Conta la frequenza di ogni genere
    all_genres = {}
    for generi in df['lista_generi']:
        for genere in generi:
            if genere in all_genres:
                all_genres[genere] += 1
            else:
                all_genres[genere] = 1

    # Ordina i generi per frequenza decrescente
    lista = list(all_genres.items())
    lista.sort(key=lambda item: item[1], reverse=True)

    # Seleziona i top-k generi più frequenti
    count = 0
    listafinale = []
    keys = [key for key, item in lista]
    while count < topk:
        listafinale.append(keys[count])
        count = count + 1

    # One-hot encoding dei generi selezionati
    for genere in listafinale:
        colonna = "is_" + genere.lower().replace(" ", "_")
        df[colonna] = df['lista_generi'].apply(
            lambda lista: 1 if genere in lista else 0
        )

    return df


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Seleziona solo colonne numeriche o booleane
    features = []
    for colonne in df.columns:
        if df[colonne].dtype in ['int64', 'float64'] or df[colonne].dtype == bool:
            features.append(colonne)

    # Crea un nuovo DataFrame con solo le feature selezionate
    dfNew = df[features].copy()
    return dfNew


def scale_numeric_features(df: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):  # type: ignore
    # Seleziona le colonne da scalare (numeriche o booleane, escluse alcune)
    scalare = []
    for colonne in df.columns:
        if (df[colonne].dtype in ['int64', 'float64'] or df[colonne].dtype == bool) and colonne != "id" and colonne != "budget":
            scalare.append(colonne)

    # Applica Min-Max scaling
    scaler = MinMaxScaler()
    sub_df = df[scalare]
    valori_scalati = scaler.fit_transform(sub_df)

    # Sostituisce i valori originali con quelli scalati
    df.loc[:, scalare] = valori_scalati
    return df, scaler


def save_preprocessed_dataset(df: pd.DataFrame, path: str) -> None:
    # Salva il dataset preprocessato su file CSV
    df.to_csv(path, index=False)


def main_preprocessing():
    # Carica il dataset pulito
    df = load_clean_dataset("data/movies_clean.csv")

    # Codifica i generi (one-hot encoding dei top 8)
    df = encode_genres(df, 8)

    # Seleziona solo le feature numeriche
    dfNew = select_feature_columns(df)

    # Normalizza le feature numeriche
    df_scalato, scaler = scale_numeric_features(dfNew)

    # Salva il dataset preprocessato
    save_preprocessed_dataset(df_scalato, "data/movies_preprocessed.csv")