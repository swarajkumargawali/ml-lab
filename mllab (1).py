import json
import os
import re
import sys

PROGRAMS = {
    1: 'Gaussian Naive Bayes - Iris and credit-card fraud',
    2: 'Multinomial Naive Bayes - 20 Newsgroups',
    3: 'Sentiment analysis - Naive Bayes vs Logistic Regression',
    4: 'Pipeline + GridSearchCV - Scaler, PCA, KNN on Iris',
    5: 'Random Forest on Iris',
    6: 'XGBoost on Iris with feature importance',
    7: 'Customer churn - Random Forest vs XGBoost with cross-validation',
}

STAGES = {}

STAGES[1] = [
r"""from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd""",
r"""iris = load_iris()
x = iris.data
y = iris.target""",
r"""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)""",
r"""model = GaussianNB()
model.fit(x_train, y_train)""",
r"""y_pred = model.predict(x_test)""",
r"""print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))""",
r"""df = pd.read_csv(r"")
df.head()""",
r"""x = df.drop('Class', axis=1)
y = df['Class']""",
r"""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)""",
r"""model = GaussianNB()
model.fit(x_train, y_train)""",
r"""y_pred = model.predict(x_test)""",
r"""print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred))""",
r"""cm = confusion_matrix(y_test, y_pred)
cm""",
]

STAGES[2] = [
r"""from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report""",
r"""newsgroups = fetch_20newsgroups(subset='all')
x, y = newsgroups.data, newsgroups.target""",
r"""xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)""",
r"""vectorizer = TfidfVectorizer(stop_words="english", max_df=0.5)
x_train_tfdif = vectorizer.fit_transform(xtrain)
x_test_tfdif = vectorizer.transform(xtest)""",
r"""nb_classifier = MultinomialNB()
nb_classifier.fit(x_train_tfdif, ytrain)""",
r"""y_pred = nb_classifier.predict(x_test_tfdif)""",
r"""accuracy = accuracy_score(ytest, y_pred)
class_report = classification_report(ytest, y_pred, target_names=newsgroups.target_names)
print(f"Accuracy: {accuracy*100:.2f}%\n")
print(f"Classification Report:\n{class_report}")""",
]

STAGES[3] = [
r"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score""",
r"""data = {
    'text': [
        "I absolutely love this product, it works perfectly!",
        "This is the worst purchase I have ever made.",
        "It's okay, nothing special but does the job.",
        "Fantastic customer service and beautiful design.",
        "Horrible quality, it broke on the first day.",
        "I am highly satisfied with the fast shipping.",
        "Waste of money. Do not buy this product.",
        "Pretty good value for the price, I recommend it.",
        "item arrived late and the product was damaged, very disappointed."
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}
df = pd.DataFrame(data)
df""",
r"""x = df["text"]
y = df["sentiment"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)""",
r"""vectorize = TfidfVectorizer(stop_words="english", lowercase=True)
x_train_vec = vectorize.fit_transform(xtrain)
x_test_vec = vectorize.transform(xtest)""",
r"""lr_model = LogisticRegression()
lr_model.fit(x_train_vec, ytrain)
nb_model = MultinomialNB()
nb_model.fit(x_train_vec, ytrain)""",
r"""lr_preds = lr_model.predict(x_test_vec)
nb_preds = nb_model.predict(x_test_vec)""",
r"""print("---Model Accuracy Comparison---")
print(f"Logistic regression accuracy: {accuracy_score(ytest, lr_preds)*100:.2f}%")
print(f"Multinomial Naive bayes accuracy: {accuracy_score(ytest, nb_preds)*100:.2f}%")""",
r"""def predict_sentiment(new_texts):
    new_texts_vec = vectorize.transform(new_texts)
    lr_predictions = lr_model.predict(new_texts_vec)
    nb_predictions = nb_model.predict(new_texts_vec)
    print("--- Testing Custom Input ---")
    for text, lr_pred, nb_pred in zip(new_texts, lr_predictions, nb_predictions):
        print(f"Text: '{text}'")
        print(f" -> LR Prediction: **{lr_pred.upper()}**")
        print(f" -> NB Prediction: **{nb_pred.upper()}**\n")


custom_samples = ["The item arrived broken and late.", "it is an amazing product", "It works incredibly well, love it!", "not worht the money", "looks good"]
predict_sentiment(custom_samples)""",
]

STAGES[4] = [
r"""import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt""",
r"""data = load_iris()
x, y = data.data, data.target
labels = data.target_names""",
r"""pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])""",
r"""xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)""",
r"""pipeline.fit(xtrain, ytrain)
test_score = pipeline.score(xtest, ytest)
print(f"pipeline accuracy\n-----------------\ntest score: {test_score:.3f}")""",
r"""ypred = pipeline.predict(xtest)""",
r"""conf_matrix = confusion_matrix(ytest, ypred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("classification pipeline confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()""",
r"""pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("knn", KNeighborsClassifier())
])
param_grid = {
    "pca__n_components": [2, 3],
    "knn__n_neighbors": [3, 5, 7]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring="accuracy", verbose=2)""",
r"""best_model.fit(xtrain, ytrain)""",
r"""test_score = best_model.score(xtest, ytest)
print(f"test score: {test_score:.3f}")
best_model.best_params_""",
r"""y_pred = best_model.predict(xtest)""",
]

STAGES[5] = [
r"""from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report""",
r"""data = load_iris()
x = data.data
y = data.target""",
r"""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)""",
r"""rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)""",
r"""y_pred = rf_classifier.predict(x_test)""",
r"""accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"accuracy: {accuracy*100:.2f}%")
print("classification report:")
print(classification_rep)""",
]

STAGES[6] = [
r"""import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance""",
r"""iris = load_iris()
x = iris.data
y = iris.target
feature_names = iris.feature_names
target_name = iris.target_names""",
r"""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)""",
r"""model = XGBClassifier(objective="multi:softprob", num_class=3, random_state=42, eval_metric="mlogloss")
model.fit(x_train, y_train)""",
r"""y_pred = model.predict(x_test)""",
r"""print(f"XGBoost accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print("Classifiction report:\n", classification_report(y_test, y_pred, target_names=target_name))""",
r"""cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_name, yticklabels=target_name, cbar=False, annot_kws={"size": 14, "weight": "bold"})
plt.title("xgboost classification confution matrix")
plt.xlabel("predicted species group")
plt.ylabel("actual species group")
plt.tight_layout()
plt.show()""",
r"""plt.figure(figsize=(8, 4))
plot_importance(model, importance_type="weight", grid=False, color="skyblue")
plt.title("XGBoost feature importance - iris dataset", fontsize=14, pad=15)
plt.tight_layout()
plt.show()""",
]

STAGES[7] = [
r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier""",
r"""df = pd.read_excel(r"")
df.head()""",
r"""target_col = 'churn'
if target_col not in df.columns:
    target_col = df.columns[-1]
    print(f"'churn' not found. Using the last column '{target_col}' as target.")
df = df.dropna()
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])""",
r"""X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)""",
r"""models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)""",
r"""for name, model in models.items():
    print("\n" + "=" * 50)
    print(f"MODEL: {name}")
    print("=" * 50)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))

    print("[Confusion Matrix]")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    print("\n[5-Fold Cross-Validation Accuracy]")
    print(f"Scores per fold: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    print("\n[Top 5 important features]")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(5, len(X.columns))):
        print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")""",
]

_state = {"program": None, "step": None}
_hooked = [False]


def _norm(text):
    text = re.sub(r'r?"[^"\n]*"', '""', text)
    text = re.sub(r"r?'[^'\n]*'", "''", text)
    return "".join(text.split())


def _ipython():
    if "google.colab" in sys.modules:
        return None
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None


def _after_cell(result):
    n, step = _state["program"], _state["step"]
    if n is None:
        return
    if not getattr(result, "success", True):
        return
    raw = getattr(getattr(result, "info", None), "raw_cell", "") or ""
    if _norm(raw) != _norm(STAGES[n][step]):
        return
    nxt = step + 1
    if nxt >= len(STAGES[n]):
        _state["program"], _state["step"] = None, None
        return
    _state["step"] = nxt
    _ipython().set_next_input(STAGES[n][nxt], replace=False)


def code(n):
    return "\n\n".join(STAGES[n]) + "\n"


def show(n):
    print(code(n))


def list_programs():
    for n, name in PROGRAMS.items():
        print(f"{n}. {name} ({len(STAGES[n])} stages)")


def get(n, step=1):
    if n not in STAGES:
        print("Choose a number from 1 to 7")
        return list_programs()
    if not 1 <= step <= len(STAGES[n]):
        print(f"Program {n} has stages 1 to {len(STAGES[n])}")
        return
    ip = _ipython()
    if ip is None:
        for i, s in enumerate(STAGES[n][step - 1:], start=step):
            print(f"----- stage {i} -----")
            print(s)
            print()
        return
    if not _hooked[0]:
        ip.events.register("post_run_cell", _after_cell)
        _hooked[0] = True
    _state["program"], _state["step"] = n, step - 1
    ip.set_next_input(STAGES[n][step - 1], replace=False)


def stop():
    _state["program"], _state["step"] = None, None


def nb(n, path=None):
    if n not in STAGES:
        print("Choose a number from 1 to 7")
        return list_programs()
    cells = []
    for s in STAGES[n]:
        lines = s.split("\n")
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        })
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = path or f"program{n}.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    try:
        from IPython.display import FileLink, display
        display(FileLink(path))
    except ImportError:
        print(f"Created {os.path.abspath(path)}")
