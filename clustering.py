import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def load_preprocessed_dataset(path: str) -> pd.DataFrame:
    # Carica il dataset preprocessato da file CSV
    df = pd.read_csv(path)
    return df

def compute_inertia_for_k(df: pd.DataFrame, k_min: int = 1, k_max: int = 10, n_init: int = 5) -> list:
    # Calcola l'inertia per diversi valori di k (metodo del gomito)
    inertias = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        km.fit(df)
        inertias.append(km.inertia_)     # Salva l'inertia del modello
    return inertias

def plot_elbow_curve(k_values: list, inertias: list, save_path: str) -> None:
    # Crea la cartella di destinazione se non esiste
    cartella = os.path.dirname(save_path)
    if cartella != "":
        os.makedirs(cartella, exist_ok=True)

    plt.figure()                         # Crea una nuova figura
    plt.plot(k_values, inertias, marker="o")  # Disegna la curva del gomito
    plt.xlabel("Numero cluter k")        # Asse X
    plt.ylabel("Inertia")                # Asse Y
    plt.title("Metodo del gomito per KMeans")  # Titolo del grafico
    plt.grid(True)                       # Migliora la leggibilità

    plt.savefig(save_path, bbox_inches="tight")  # Salva il grafico su file
    plt.close()                          # Chiude la figura

def run_kmeans(df: pd.DataFrame, k: int, n_init: int = 5) -> (pd.Series, pd.DataFrame):  # type: ignore
   # Esegue K-Means con k cluster
   km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
   labels = pd.Series(km.fit_predict(df))            # Etichette dei cluster
   centroids = pd.DataFrame(km.cluster_centers_, columns=df.columns)  # Centroidi
   return labels, centroids

def attach_cluster_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    # Aggiunge la colonna cluster_id al DataFrame
    df['cluster_id'] = labels.values
    return df

def save_dataset_with_clusters(df: pd.DataFrame, path: str) -> None:
    # Salva il dataset con i cluster su file CSV
    df.to_csv(path, index=False)

def analyze_clusters(df: pd.DataFrame) -> None:
    # Conta quanti film ci sono per ogni cluster
    value = df['cluster_id'].value_counts()

    # Grafico della distribuzione dei cluster
    plt.bar(value.index, value.values)
    plt.xlabel("Cluster")
    plt.ylabel("Numero di film")
    plt.title("Analyze Clusters")
    plt.savefig("plots/cluster_distribution.png", bbox_inches="tight")
    plt.close()

    # Calcola la media delle feature per ogni cluster
    df_means = df.groupby('cluster_id').mean(numeric_only=True)
    df_means.to_csv("data/cluster_means.csv")

def main_clustering():
    # Carica il dataset preprocessato (solo feature numeriche)
    df = load_preprocessed_dataset("data/movies_preprocessed.csv")

    # Carica il dataset pulito con i titoli
    df_meta = pd.read_csv("data/movies_clean.csv")  # qui c'è original_title

    # Controllo che i due dataset abbiano lo stesso numero di righe
    if len(df) != len(df_meta):
        raise ValueError(f"Righe diverse: preprocessed={len(df)} clean={len(df_meta)}. "
                         "Devi avere stessi filtri/drop e stesso ordine.")

    # Calcolo dell'inertia per il metodo del gomito
    k_values = range(1, 11)
    inertias = compute_inertia_for_k(df, 1, 10, 5)
    plot_elbow_curve(k_values, inertias, "plots/elbow.png")

    # Numero di cluster scelto
    k = 4
    labels, centroids = run_kmeans(df, k, 5)

    # FILE PER MACHINE LEARNING (senza titoli)
    df_ml = df.copy()
    df_ml["cluster_id"] = labels.values
    save_dataset_with_clusters(df_ml, "data/movies_with_clusters.csv")
    analyze_clusters(df_ml)

    # FILE PER PROLOG (con titoli)
    df_prolog = df_ml.copy()
    df_prolog["original_title"] = df_meta["original_title"].values
    save_dataset_with_clusters(df_prolog, "data/movies_with_clusters_prolog.csv")