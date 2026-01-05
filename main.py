from data_creation import maindc
from preprocessing import main_preprocessing
from clustering import main_clustering
from supervised_learning import main_supervised
from bayesian_network import main_BSN
from filmProlog import main_prolog

def main():
    # 1) Creazione del dataset pulito 
    print("\n Creazione dataset pulito")
    maindc()

    # 2) Preprocessing (encoding generi, scaling, selezione feature)
    print("\n[2 Preprocessing")
    main_preprocessing()

    # 3) Clustering (KMeans + salvataggio movies_with_clusters.csv)
    print("\n Clustering...")
    main_clustering()

    # 4) Apprendimento supervisionato (DecisionTree, RandomForest, LogisticRegression)
    print("\n Apprendimento supervisionato")
    main_supervised()

    # 5) Rete bayesiana (inferenza su cluster_id dato alcune feature)
    print("\n Rete Bayesiana")
    main_BSN()

    # 6) Ragionamento logico (Prolog)
    print("\n Ragionamento logico (Prolog)")
    main_prolog()

if __name__ == "__main__":
    main()