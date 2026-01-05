import pandas as pd                      
import matplotlib.pyplot as plt         
import seaborn as sns                   
import os                               

from sklearn.model_selection import train_test_split          
from sklearn import tree                                       
from sklearn.ensemble import RandomForestClassifier            
from sklearn.linear_model import LogisticRegression            
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

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
    return accur, cm, cr

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


def main_supervised():
    # Crea cartella output per salvare i risultati
    os.makedirs("results/supervised", exist_ok=True)

    # Carica dataset con feature + cluster_id
    df = load_dataset("data/movies_with_clusters.csv")

    # Divide feature e target
    X, y = split_features_target(df)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split_ds(
        X, y, test_size=0.2, random_state=42
    )

    results = {}  # Dizionario per salvare le accuracy finali

    #     DECISION TREE
    modello1, y_pred1 = run_decision_tree(X_train, X_test, y_train)
    accur1, cm1, cr1 = evaluate_model(y_test, y_pred1)
    results["DecisionTree"] = accur1

    # Salva accuracy, confusion matrix e report su file
    pd.DataFrame({"accuracy": [accur1]}).to_csv("results/supervised/dt_accuracy.csv", index=False)
    pd.DataFrame(cm1).to_csv("results/supervised/dt_confusion_matrix.csv", index=False)
    with open("results/supervised/dt_report.txt", "w") as f:
        f.write(cr1)

    # Salva grafico confusion matrix
    save_confusion_matrix_plot(cm1, "decision_tree")

    # Stampa risultati
    print("\n=== DECISION TREE ===")
    print("Accuracy:", accur1)
    print(cr1)

    #     RANDOM FOREST
    modello2, y_pred2 = run_random_forest(X_train, X_test, y_train)
    accur2, cm2, cr2 = evaluate_model(y_test, y_pred2)
    results["RandomForest"] = accur2

    # Salva accuracy, confusion matrix e report su file
    pd.DataFrame({"accuracy": [accur2]}).to_csv("results/supervised/rf_accuracy.csv", index=False)
    pd.DataFrame(cm2).to_csv("results/supervised/rf_confusion_matrix.csv", index=False)
    with open("results/supervised/rf_report.txt", "w") as f:
        f.write(cr2)

    # Salva grafico confusion matrix
    save_confusion_matrix_plot(cm2, "random_forest")

    # Stampa risultati
    print("\n=== RANDOM FOREST ===")
    print("Accuracy:", accur2)
    print(cr2)

    #   LOGISTIC REGRESSION
    modello3, y_pred3 = run_logistic_regression(X_train, X_test, y_train)
    accur3, cm3, cr3 = evaluate_model(y_test, y_pred3)
    results["LogisticRegression"] = accur3

    # Salva accuracy, confusion matrix e report su file
    pd.DataFrame({"accuracy": [accur3]}).to_csv("results/supervised/lr_accuracy.csv", index=False)
    pd.DataFrame(cm3).to_csv("results/supervised/lr_confusion_matrix.csv", index=False)
    with open("results/supervised/lr_report.txt", "w") as f:
        f.write(cr3)

    # Salva grafico confusion matrix
    save_confusion_matrix_plot(cm3, "logistic_regression")

    # Stampa risultati
    print("\n=== LOGISTIC REGRESSION ===")
    print("Accuracy:", accur3)
    print(cr3)

    #   GRAFICO ACCURACY FINALI
    save_accuracy_barplot(results)

    # Stampa tabella finale delle accuracy
    print("\n=== RISULTATI FINALI ===")
    print(pd.DataFrame.from_dict(results, orient="index", columns=["accuracy"]))
