import pandas as pd                      
import matplotlib.pyplot as plt         
import seaborn as sns                   
import os 
import numpy as np


from sklearn.model_selection import train_test_split          
from sklearn import tree                                       
from sklearn.ensemble import RandomForestClassifier            
from sklearn.linear_model import LogisticRegression            
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.model_selection import learning_curve
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_dataset(path: str) -> pd.DataFrame:
    # Carica il dataset dal file CSV
    df = pd.read_csv(path)
    return df

def split_features_target(df: pd.DataFrame) -> (pd.DataFrame, pd.Series): # type: ignore
    # Separa feature (X) e target (y)
    X = df.drop(columns=["cluster_id"])  # Tutte le colonne tranne cluster_id
    Y = df["cluster_id"]                # Target da predire
    return X, Y

def train_test_split_ds(X, y, test_size: int = 0.2, random_state: int=42):
    # Divide in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def run_decision_tree(X_train, X_test, y_train):
    # Allena un Decision Tree e produce le predizioni sul test set
    clf = tree.DecisionTreeClassifier(random_state=42)
    modello = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return modello, y_pred

def run_random_forest(X_train, X_test, y_train):
    # Allena un Random Forest e produce le predizioni sul test set
    rf = RandomForestClassifier(random_state=42)
    modello = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return modello, y_pred

#“La Logistic Regression non è molto adatta a questo dataset perché i cluster non sono linearmente separabili.”
def run_logistic_regression(X_train, X_test, y_train):
    # Allena una Logistic Regression e produce le predizioni sul test set
    lr = LogisticRegression(random_state=42, max_iter=5000)
    LogisticRegression()
    modello = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return modello, y_pred

def evaluate_model(y_true, y_pred):
    # Calcola metriche principali: accuracy, confusion matrix e report completo
    accur = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    return accur, f1_macro, f1_weighted, cm, cr


def save_confusion_matrix_plot(cm, model_name):
    # Salva la confusion matrix come immagine (heatmap)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_confusion_matrix.png")
    plt.close()

def save_accuracy_barplot(results_dict):
    # Salva un grafico a barre con le accuracy dei modelli
    labels = list(results_dict.keys())
    accuracies = list(results_dict.values())

    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=accuracies, palette="viridis")
    plt.title("Accuracy Comparison Between Models")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("plots/accuracy_comparison.png")
    plt.close()

def generate_learning_curve(estimator, X, y, title: str, out_path: str,
                            train_sizes=None, cv: int = 5, random_state: int = 42):
    if train_sizes is None:
        # se vuoi esattamente come gli screenshot, lascia questi
        train_sizes = [50, 150, 250, 350, 450]

    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        shuffle=True,
        random_state=random_state,
        n_jobs=-1
    )

    train_err = 1.0 - train_scores.mean(axis=1)
    val_err = 1.0 - val_scores.mean(axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, train_err, label="Train Error")
    plt.plot(sizes, val_err, label="Test Error")  # in realtà validation error (CV)
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_learning_curves_all_models(X, y):
    os.makedirs("plots", exist_ok=True)

    generate_learning_curve(
        tree.DecisionTreeClassifier(random_state=42),
        X, y,
        title="Learning Curve for DecisionTree",
        out_path="plots/learning_curve_decision_tree.png"
    )

    generate_learning_curve(
        RandomForestClassifier(random_state=42),
        X, y,
        title="Learning Curve for RandomForest",
        out_path="plots/learning_curve_random_forest.png"
    )

    generate_learning_curve(
        LogisticRegression(random_state=42, max_iter=5000),
        X, y,
        title="Learning Curve for LogisticRegression",
        out_path="plots/learning_curve_logistic_regression.png"
    )

def main_supervised():
    # Crea cartella output per salvare i risultati
    os.makedirs("results/supervised", exist_ok=True)

    # Crea cartella output per salvare i grafici
    os.makedirs("plots", exist_ok=True)

    # Carica dataset con feature + cluster_id
    df = load_dataset("data/movies_with_clusters.csv")

    # Divide feature e target
    X, y = split_features_target(df)

    # Numero di run per ripetere la valutazione al fine di calcolare media e deviazione standard
    n_runs = 20

    # Lista per salvare le metriche di ogni run e modelli
    all_runs_metrics = []

    # Dizionario con le accuracy medie per il grafico finale
    results = {}

    # Ripete train/test split + training + evaluation su più run
    for run in range(n_runs):

        # Imposta un random_state diverso ad ogni run per cambiare lo split
        rs = 42 + run

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split_ds(
            X, y, test_size=0.2, random_state=rs
        )

        #     DECISION TREE
        modello1, y_pred1 = run_decision_tree(X_train, X_test, y_train)
        accur1, f1m1, f1w1, cm1, cr1 = evaluate_model(y_test, y_pred1)

        # Salvo le metriche di questa run
        all_runs_metrics.append({
            "model": "DecisionTree",
            "run": run,
            "accuracy": accur1,
            "f1_macro": f1m1,
            "f1_weighted": f1w1
        })

        #     RANDOM FOREST
        modello2, y_pred2 = run_random_forest(X_train, X_test, y_train)
        accur2, f1m2, f1w2, cm2, cr2 = evaluate_model(y_test, y_pred2)

        # Salvo le metriche di questa run
        all_runs_metrics.append({
            "model": "RandomForest",
            "run": run,
            "accuracy": accur2,
            "f1_macro": f1m2,
            "f1_weighted": f1w2
        })

        #   LOGISTIC REGRESSION
        modello3, y_pred3 = run_logistic_regression(X_train, X_test, y_train)
        accur3, f1m3, f1w3, cm3, cr3 = evaluate_model(y_test, y_pred3)

        # Salvo le metriche di questa run
        all_runs_metrics.append({
            "model": "LogisticRegression",
            "run": run,
            "accuracy": accur3,
            "f1_macro": f1m3,
            "f1_weighted": f1w3
        })

    #learning curves 
    generate_learning_curves_all_models(X, y)

    # Converto le metriche di tutte le run in DataFrame
    runs_df = pd.DataFrame(all_runs_metrics)

    # Salvo su file tutte le metriche per run
    runs_df.to_csv("results/supervised/all_runs_metrics.csv", index=False)

    # Calcolo media e deviazione standard per ciascun modello
    agg = runs_df.groupby("model")[["accuracy", "f1_macro", "f1_weighted"]].agg(["mean", "std"])
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    # Salvo la tabella finale mean/std
    agg.to_csv("results/supervised/summary_mean_std.csv", index=False)

    # Dizionario con le accuracy medie per il grafico finale
    for _, row in agg.iterrows():
        results[row["model"]] = row["accuracy_mean"]

    # Salvo un grafico a barre con le accuracy medie dei modelli
    save_accuracy_barplot(results)

    # Stampo tabella finale con accuracy e F1
    print("\n=== RISULTATI FINALI (media ± dev. std su più run) ===")
    print(agg)

    # Split train/test (esempio fisso per salvare una confusion matrix rappresentativa)
    X_train, X_test, y_train, y_test = train_test_split_ds(
        X, y, test_size=0.2, random_state=42
    )

    # RANDOM FOREST esempio
    modello_ex, y_pred_ex = run_random_forest(X_train, X_test, y_train)

    # Calcolo le metriche sullo split esemplificativo
    accur_ex, f1m_ex, f1w_ex, cm_ex, cr_ex = evaluate_model(y_test, y_pred_ex)

    # Salva accuracy + f1 dello split esemplificativo
    pd.DataFrame({"accuracy": [accur_ex], "f1_macro": [f1m_ex], "f1_weighted": [f1w_ex]}).to_csv(
        "results/supervised/example_rf_metrics.csv", index=False
    )

    # Salvo confusion matrix dello split esemplificativo
    pd.DataFrame(cm_ex).to_csv("results/supervised/example_rf_confusion_matrix.csv", index=False)

    # Salvo report dello split esemplificativo
    with open("results/supervised/example_rf_report.txt", "w") as f:
        f.write(cr_ex)

    # Salvo grafico confusion matrix
    save_confusion_matrix_plot(cm_ex, "random_forest_example")
