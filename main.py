import os
import argparse
import duckdb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

load_dotenv()
JETON_API = os.environ["JETON_API"]
parser = argparse.ArgumentParser(description="Hyperparrameters")
parser.add_argument("--n_tree", type=int, default=20, help="number of tree for random forest")
parser.add_argument("--max_depth", type=int, default=None, help="maximum depth for random forest")
parser.add_argument("--max_features", type=str, default="sqrt")

args = parser.parse_args()
n_tree = args.n_tree
max_depth = args.max_depth
max_features = args.max_features
print(f"number of tree: {n_tree}")

logger.debug(f'number of tree: {n_tree}')
logger.info(f'max depth : {max_depth}')
logger.warning(f'max features : {max_features}')

os.chdir("/home/onyxia/work/application/data/raw")
titanic = pd.read_csv("data.csv")

con = duckdb.connect(database=":memory:")

# Check la structure de Name "Nom, Prénom"
bad = con.sql("""
    SELECT COUNT(*) AS n_bad
    FROM titanic
    WHERE list_count(string_split(Name, ',')) <> 2
""").fetchone()[0]

if bad == 0:
    print("Test 'Name' OK se découpe toujours en 2 parties avec ','")
else:
    print(f"Problème dans la colonne Name: {bad} ne se décomposent pas en 2 parties.")

numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20)),
    ]
)


# splitting samples
y = titanic["Survived"]
X = titanic.drop("Survived", axis="columns")

# split train dataset: -> validation croisée une partie apprendre l'autre score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# check que pas de problème de data leakage
if set(X_train["Embarked"].dropna().unique()) - set(
    X_test["Embarked"].dropna().unique()
):
    message = "Problème de data leakage pour la variable Embarked"
else:
    message = "Pas de problème de data leakage pour la variable Embarked"

print(message)

if set(X_train["Sex"].dropna().unique()) - set(X_test["Sex"].dropna().unique()):
    message = "Problème de data leakage pour la variable Sex"
else:
    message = "Pas de problème de data leakage pour la variable Embarked"

print(message)

# Vérifie les valeurs manquantes
# TODO: généraliser à toutes les variables
n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM titanic
    WHERE Survived IS NULL
""").fetchone()[0]

message_ok = "Pas de valeur manquante pour la variable Survived"
message_warn = f"{n_missing} valeurs manquantes pour la variable Survived"
message = message_ok if n_missing == 0 else message_warn
print(message)

n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM titanic
    WHERE Age IS NULL
""").fetchone()[0]

message_ok = "Pas de valeur manquante pour la variable Age"
message_warn = f"{n_missing} valeurs manquantes pour la variable Age"
message = message_ok if n_missing == 0 else message_warn
print(message)

# Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)


# calculons du score ur le test et le train (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
