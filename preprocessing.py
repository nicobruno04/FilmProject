import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_clean_dataset(path:str) -> pd.DataFrame:
    df= pd.read_csv(path)
    return df

def encode_genres(df: pd.DataFrame, topk: int = 8) -> pd.DataFrame:
    def parse_genres(st):
        if pd.isna(st):
            return []
        parts = st.split('|')
        genres = []
        for genre in parts:
            g = genre.strip()
            if g != "":
                genres.append(g) 
        return genres
    df['lista_generi'] = df['genres'].apply(parse_genres)
    all_genres = {}
    for generi in df['lista_generi']:
        for genere in generi:
            if genere in all_genres:
                all_genres[genere] += 1
            else:
                all_genres[genere] = 1
    lista = list(all_genres.items())
    lista.sort(key=lambda item: item[1], reverse=True)
    count = 0
    listafinale = []
    keys=[key for key, item in lista]
    while count<topk:
        listafinale.append(keys[count])
        count = count +1
    for genere in listafinale:
        colonna = "is_" + genere.lower().replace(" ", "_")
        df[colonna] = df['lista_generi'].apply(
            lambda lista:1 if genere in lista else 0
        )
    
    return df

def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    features = []
    for colonne in df.columns:
        if df[colonne].dtype in ['int64', 'float64'] or df[colonne].dtype == bool:
            features.append(colonne)
    dfNew = df[features].copy()
    return dfNew

def scale_numeric_features(df: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler): #type: ignore
    scalare = []
    for colonne in df.columns:
        if (df[colonne].dtype in ['int64', 'float64'] or df[colonne].dtype == bool) and colonne != "id" and colonne != "budget":
            scalare.append(colonne)
    scaler = MinMaxScaler()
    sub_df = df[scalare]
    valori_scalati = scaler.fit_transform(sub_df)
    df.loc[:, scalare] = valori_scalati
    return df, scaler

def save_preprocessed_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def main_preprocessing():
    df = load_clean_dataset("data/movies_clean.csv")
    df = encode_genres(df, 8)
    dfNew = select_feature_columns(df)
    df_scalato, scaler = scale_numeric_features(dfNew)
    save_preprocessed_dataset(df_scalato, "data/movies_preprocessed.csv")


