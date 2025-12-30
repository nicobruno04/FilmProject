from data_creation import maindc
from preprocessing import main_preprocessing
from clustering import main_clustering
from supervised_learning import main_supervised
from bayesian_network import main_BSN
from filmProlog import main_prolog

def main():
    # 1) Creazione del dataset pulito (da eseguire solo se cambi il raw dataset)
    print("\n[1/5] Creazione dataset pulito...")
    maindc()

    # 2) Preprocessing (encoding generi, scaling, selezione feature)
    print("\n[2/5] Preprocessing...")
    main_preprocessing()

    # 3) Clustering (KMeans + salvataggio movies_with_clusters.csv)
    print("\n[3/5] Clustering...")
    main_clustering()

    # 4) Apprendimento supervisionato (DecisionTree, RandomForest, LogisticRegression)
    print("\n[4/5] Apprendimento supervisionato...")
    main_supervised()

    # 5) Rete bayesiana (inferenza su cluster_id dato alcune feature)
    print("\n[5/5] Rete Bayesiana...")
    main_BSN()

    # 6) Ragionamento logico (Prolog)
    print("\n[6/6] Ragionamento logico (Prolog)...")
    main_prolog()

if __name__ == "__main__":
    main()