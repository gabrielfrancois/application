"""
Fonctions pour le projet
"""

def convert_csv_parquet(MY_BUCKET:str = "gfrancois", CHEMIN_FICHIER:str = "ensae-reproductibilite/data/raw/data.csv"):
    import os
    import duckdb

    con = duckdb.connect(database=":memory:")
    query_definition = f"SELECT * FROM read_csv('s3://{MY_BUCKET}/{CHEMIN_FICHIER}')"
    con.sql(
        f"""
            COPY (
                SELECT * 
                FROM read_csv_auto('s3://{MY_BUCKET}/{CHEMIN_FICHIER}')
            )
            TO 's3://{MY_BUCKET}/{CHEMIN_FICHIER.replace("csv", "parquet")}'
            (FORMAT PARQUET);
        """
    )