import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

file_path = "heart-disease.csv"
df = pd.read_csv(file_path)
df.head(10)
df.target.value_counts()
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);
df.info()
X = df.drop("target", axis=1)

y = df.target.values
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2)

models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(),
          "Random Forest": RandomForestClassifier()}


def fit_and_score(models, X_train, X_test, y_train, y_test):

    np.random.seed(42)

    model_scores = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores

model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();''

