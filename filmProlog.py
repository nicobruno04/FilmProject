import csv
import os
import re
from typing import Dict, List, Tuple

from pyswip import Prolog


def safe_atom(s: str) -> str:
    """
    Converte una stringa (titolo) in un atomo Prolog sicuro:
    - lowercase
    - caratteri non alfanumerici -> _
    - non deve iniziare con cifra
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = f"m_{s}"
    return s


def prolog_quote(s: str) -> str:
    """Quote Prolog sicure con apici singoli, escapando eventuali apici interni."""
    s = (s or "").replace("'", "\\'")
    return f"'{s}'"


def write_kb_from_csv(csv_path: str, kb_path: str, max_rows: int | None = None,) -> Tuple[int, List[str]]:
    """
    Crea KB Prolog con:
    - title_atom(Atom, 'Titolo originale').
    - cluster(Atom, ClusterId).
    - feature(Atom, popularity, Val).
    - feature(Atom, vote_average, Val).
    - feature(Atom, vote_count, Val).
    - feature(Atom, runtime, Val).
    - feature(Atom, release_year, Val).
    - has_genre(Atom, Genre). per ogni is_<genere> = 1
    """
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        genre_cols = [c for c in fieldnames if c.startswith("is_")]  # da preprocessing.py :contentReference[oaicite:3]{index=3}
        numeric_cols = ["popularity", "vote_average", "vote_count", "runtime", "release_year"]

        n = 0
        with open(kb_path, "w", encoding="utf-8") as kb:
            kb.write("% === Knowledge Base (auto-generata) ===\n\n")

            # Etichette semantiche dei cluster (coerenti con la tua interpretazione in doc)
            kb.write("cluster_label(0, thriller_drama).\n")
            kb.write("cluster_label(1, action_thriller).\n")
            kb.write("cluster_label(2, comedy).\n")
            kb.write("cluster_label(3, drama_high_rating).\n\n")

            # Regole utili
            kb.write("% movie_info('Titolo', ClusterId, Label).\n")
            kb.write("movie_info(Title, Cluster, Label) :- title_atom(A, Title), cluster(A, Cluster), cluster_label(Cluster, Label).\n\n")

            kb.write("% movie_in_cluster('Titolo', Label).\n")
            kb.write("movie_in_cluster(Title, Label) :- movie_info(Title, _C, Label).\n\n")

            kb.write("% movie_has_genre('Titolo', Genre).\n")
            kb.write("movie_has_genre(Title, Genre) :- title_atom(A, Title), has_genre(A, Genre).\n\n")

            kb.write("% movie_in_cluster_with_genre('Titolo', Label, Genre).\n")
            kb.write("movie_in_cluster_with_genre(Title, Label, Genre) :- title_atom(A, Title), cluster(A, C), cluster_label(C, Label), has_genre(A, Genre).\n\n")

            kb.write("% === Fatti ===\n")

            for row in reader:
                if max_rows is not None and n >= max_rows:
                    break

                title = row.get("original_title", "")  # da data_creation.py :contentReference[oaicite:4]{index=4}
                atom = safe_atom(title)

                # title mapping
                kb.write(f"title_atom({atom}, {prolog_quote(title)}).\n")

                # cluster
                cluster_id = row.get("cluster_id", None)
                if cluster_id is not None and cluster_id != "":
                    kb.write(f"cluster({atom}, {int(float(cluster_id))}).\n")

                # numeric features (round per evitare problemi di floating)
                for col in numeric_cols:
                    v = row.get(col, None)
                    if v is None or v == "":
                        continue
                    try:
                        fv = float(v)
                        kb.write(f"feature({atom}, {col}, {fv:.5f}).\n")
                    except ValueError:
                        pass

                # genres come fatti separati
                for gc in genre_cols:
                    gv = row.get(gc, "0")
                    try:
                        if int(float(gv)) == 1:
                            genre_name = gc.replace("is_", "")
                            kb.write(f"has_genre({atom}, {genre_name}).\n")
                    except ValueError:
                        continue

                kb.write("\n")
                n += 1

    return n, genre_cols


def run_sample_queries(kb_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    prolog = Prolog()
    prolog.consult(kb_path)

    queries: List[Tuple[str, str]] = [
        ("Tutti i cluster semantici disponibili", "cluster_label(C, L)"),
        ("Esempio: 10 film nel cluster comedy", "movie_in_cluster(Title, comedy)"),
        ("Esempio: film action nel cluster action_thriller", "movie_in_cluster_with_genre(Title, action_thriller, action)"),
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        for header, q in queries:
            f.write(f"### {header}\n")
            f.write(f"Query: {q}\n\n")

            count = 0
            for sol in prolog.query(q):
                f.write(str(sol) + "\n")
                count += 1
                if "movie_" in q and count >= 10:
                    break

            if count == 0:
                f.write("(nessun risultato)\n")
            f.write("\n\n")


def main_prolog():
    kb_path = "results/prolog/kb_movies.pl"
    csv_path = "data/movies_with_clusters_prolog.csv"
    out_path = "results/prolog/query_results.txt"

    n, genre_cols = write_kb_from_csv(csv_path, kb_path)
    print(f"[Prolog] KB generata: {kb_path} (righe lette: {n}, colonne genere: {len(genre_cols)})")

    run_sample_queries(kb_path, out_path)
    print(f"[Prolog] Query salvate in: {out_path}")


if __name__ == "__main__":
    main_prolog()
