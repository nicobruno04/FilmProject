import pandas as pd
import os

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

def load_dataset(path: str) -> pd.DataFrame:
    # Carica il dataset da file CSV
    df = pd.read_csv(path)
    return df


def discretize_features(df: pd.DataFrame):
    # Discretizza (trasforma in categorie) alcune feature numeriche
    # pd.cut divide i valori in 3 intervalli (bins=3) e assegna etichette
    df['popularity_disc'] = pd.cut(df['popularity'], bins=3, labels=["low", "medium", "high"])
    df['voteAvg_disc'] = pd.cut(df['vote_average'], bins=3, labels=["low", "medium", "high"])
    df['runtime_disc'] = pd.cut(df['runtime'], bins=3, labels=["short", "medium", "long"])
    
    # Restituisce solo le colonne discrete + cluster_id
    return df[["popularity_disc", "voteAvg_disc", "runtime_disc", "cluster_id"]]


def define_structure():
    # Definisce la struttura del grafo (archi) della rete bayesiana
    # cluster_id è il "genitore" e influenza le 3 variabili discrete
    edges = [
        ("cluster_id", "popularity_disc"),
         ("cluster_id", "voteAvg_disc"),
        ("cluster_id", "runtime_disc")
    ]
    return edges


def build_model(edges):
    # Crea la rete bayesiana a partire dalla lista di archi
    model = DiscreteBayesianNetwork(edges)
    return model


def fit_cpd(df: pd.DataFrame, model: DiscreteBayesianNetwork):
    # Stima le CPD (tabelle di probabilità condizionata) dai dati
    model.fit(df)


def run_inference(model: DiscreteBayesianNetwork, evidence, query):
    # Crea l'oggetto di inferenza e calcola la distribuzione richiesta
    infer = VariableElimination(model)
    res = infer.query(variables=query, evidence=evidence)
    return res


def main_BSN():
    # Crea la cartella di output per salvare i risultati
    os.makedirs("results/bayes", exist_ok=True)

    # Carica il dataset con i cluster
    df = load_dataset("data/movies_with_clusters.csv")

    # Discretizza feature numeriche e tiene solo colonne utili
    dfDis = discretize_features(df)

    # Definisce struttura della rete (archi)
    edges = define_structure()

    # Costruisce il modello
    bn = build_model(edges)

    # Fit delle CPD (apprendimento parametri)
    fit_cpd(dfDis, bn)

    # Stampa nodi e archi a video
    print("Nodi del modello:", bn.nodes())
    print("Archi del modello:", bn.edges())

    # Salva struttura del modello su file
    with open("results/bayes/model_structure.txt", "w") as f:
        f.write("Nodi:\n")
        for n in bn.nodes():
            f.write(f"- {n}\n")
        f.write("\nArchi:\n")
        for e in bn.edges():
            f.write(f"- {e}\n")

    # Esempio di inferenza:
    # "Dato popularity=low e runtime=short, qual è la probabilità dei diversi cluster?"
    evidence = {"popularity_disc": "low", "runtime_disc": "short"}
    query = ["cluster_id"]

    # Esegue inferenza
    result = run_inference(bn, evidence, query)

    # Stampa a video il risultato
    print("\nRisultato inferenza:")
    print(result)

    # Salva il risultato dell'inferenza su file
    with open("results/bayes/inference_result.txt", "w") as f:
        f.write("Inferenza: P(cluster_id | popularity_disc=low, runtime_disc=short)\n\n")
        f.write(str(result))