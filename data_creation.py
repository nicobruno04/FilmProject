import pandas as pd

def load_raw_dataset(path: str) -> pd.DataFrame:
    df= pd.read_csv(path)
    return df

def select_relevant_columns(df:  pd.DataFrame) -> pd.DataFrame:
    colonneUtili = [
        "original_title",
        "genres",
        "popularity",
        "vote_average",
        "vote_count",
        "runtime",
        "release_year"
    ]
    df = df[colonneUtili]
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    dfNew= df.copy()
    dfNew = dfNew.drop_duplicates()    
    dfNew = dfNew.dropna(subset=["popularity", "vote_average", "vote_count"])
    return dfNew

def save_clean_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def maindc():
    df = load_raw_dataset("data/movies_data.csv")
    df = select_relevant_columns(df)
    dfNew = clean_dataset(df)
    save_clean_dataset(dfNew, "data/movies_clean.csv")

