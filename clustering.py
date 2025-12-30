import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def load_preprocessed_dataset(path: str) -> pd.DataFrame:
    df= pd.read_csv(path)
    return df

def compute_inertia_for_k(df: pd.DataFrame, k_min: int = 1, k_max: int = 10, n_init: int = 5) -> list:
    inertias = []
    for k in range (k_min, k_max +1):
        km=KMeans(n_clusters = k, n_init=n_init, random_state=42)
        km.fit(df)
        inertias.append(km.inertia_)
    return inertias


def plot_elbow_curve(k_values: list, inertias: list, save_path: str) -> None:
    cartella = os.path.dirname(save_path)
    if cartella != "":
        os.makedirs(cartella, exist_ok=True)

    plt.figure() #crea immagine
    plt.plot(k_values, inertias, marker="o") #inserisce i valori che andranno nel grafico
    plt.xlabel("Numero cluter k") #nome dell'asse x
    plt.ylabel("Inertia") #nome dell'asse y
    plt.title("Metodo del gomito per KMeans") #nome del grafico
    plt.grid(True) #migliora la leggibilità

    plt.savefig(save_path, bbox_inches="tight") #funzione per salvare il grafico su file
    #bbox_inches taglia i margini perfettamente
    plt.close()

def run_kmeans(df: pd.DataFrame, k: int, n_init: int = 5) -> (pd.Series, pd.DataFrame): # type: ignore
   km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
   labels = pd.Series(km.fit_predict(df))
   centroids = pd.DataFrame(km.cluster_centers_, columns=df.columns)
   return labels, centroids

def attach_cluster_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    df['cluster_id'] = labels.values
    return df

def save_dataset_with_clusters(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def analyze_clusters(df: pd.DataFrame) -> None:
    value = df['cluster_id'].value_counts()
    plt.bar(value.index, value.values)
    plt.xlabel("Cluster")
    plt.ylabel("Numero di film")
    plt.title("Analyze Clusters")
    plt.savefig("plots/cluster_distribution.png", bbox_inches="tight") #funzione per salvare il grafico su file
    plt.close()
    df_means= df.groupby('cluster_id').mean(numeric_only=True)
    df_means.to_csv("data/cluster_means.csv")

def main_clustering():
    df = load_preprocessed_dataset("data/movies_preprocessed.csv")
    df_meta = pd.read_csv("data/movies_clean.csv")  # qui c'è original_title

    # controllo importantissimo
    if len(df) != len(df_meta):
        raise ValueError(f"Righe diverse: preprocessed={len(df)} clean={len(df_meta)}. "
                         "Devi avere stessi filtri/drop e stesso ordine.")

    k_values = range(1,11)
    inertias = compute_inertia_for_k(df, 1, 10, 5)
    plot_elbow_curve(k_values, inertias, "plots/elbow.png")

    k = 4
    labels, centroids = run_kmeans(df, k, 5)

    # FILE PER ML (NO TITOLI)
    df_ml = df.copy()
    df_ml["cluster_id"] = labels.values
    save_dataset_with_clusters(df_ml, "data/movies_with_clusters.csv")
    analyze_clusters(df_ml)

    # FILE PER PROLOG (CON TITOLI) 
    df_prolog = df_ml.copy()
    df_prolog["original_title"] = df_meta["original_title"].values
    save_dataset_with_clusters(df_prolog, "data/movies_with_clusters_prolog.csv")


