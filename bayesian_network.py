import pandas as pd
import os

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

def load_dataset(path:str) -> pd.DataFrame:
    df= pd.read_csv(path)
    return df

def discretize_features(df: pd.DataFrame):
    df['popularity_disc'] = pd.cut(df['popularity'], bins=3, labels=["low", "medium", "high"])
    df['voteAvg_disc'] = pd.cut(df['vote_average'], bins=3, labels=["low", "medium", "high"])
    df['runtime_disc'] = pd.cut(df['runtime'], bins=3, labels=["short", "medium", "long"])
    return df[["popularity_disc", "voteAvg_disc", "runtime_disc", "cluster_id"]]

def define_structure():
    edges = [
        ("cluster_id", "popularity_disc"),
         ("cluster_id", "voteAvg_disc"),
        ("cluster_id", "runtime_disc")
    ]
    return edges

def build_model(edges):
    model = DiscreteBayesianNetwork(edges)
    return model

def fit_cpd(df: pd.DataFrame, model: DiscreteBayesianNetwork):
    model.fit(df)

def run_inference(model: DiscreteBayesianNetwork, evidence, query):
    infer = VariableElimination(model)
    res = infer.query(variables= query, evidence=evidence)
    return res

def main_BSN():
    os.makedirs("results/bayes", exist_ok=True)
    df= load_dataset("data/movies_with_clusters.csv")
    dfDis = discretize_features(df)
    edges = define_structure()
    bn= build_model(edges)
    fit_cpd(dfDis, bn)

    print("Nodi del modello:", bn.nodes())
    print("Archi del modello:", bn.edges())

    # Salva struttura su file
    with open("results/bayes/model_structure.txt", "w") as f:
        f.write("Nodi:\n")
        for n in bn.nodes():
            f.write(f"- {n}\n")
        f.write("\nArchi:\n")
        for e in bn.edges():
            f.write(f"- {e}\n")

    # 7) Esempio di inferenza
    evidence = {"popularity_disc": "low", "runtime_disc": "short"}
    query = ["cluster_id"]

    result = run_inference(bn, evidence, query)

    # Stampa a video
    print("\nRisultato inferenza:")
    print(result)

    # 8) Salva inferenza su file
    with open("results/bayes/inference_result.txt", "w") as f:
        f.write("Inferenza: P(cluster_id | popularity_disc=low, runtime_disc=short)\n\n")
        f.write(str(result))