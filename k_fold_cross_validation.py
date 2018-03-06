from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# k fold cross validation for parameter tuning and model selection
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Parameter tuning using k fold cross validation starts
    k_range = range(1, 31)
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # 10 fold cross validation
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Cross-Validation Accuracy")
    plt.show()

    # Parameter tuning using k fold cross validation ends
    # ----------------------------------------------------

    # Model Selection using k fold cross validation starts

    knn = KNeighborsClassifier(n_neighbors=20)
    print("10 fold cross validation with best KNN model: ", end=" ")
    print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

    logreg = LogisticRegression()
    print("10 fold cross validation with best Logistic Regression: ", end=" ")
    print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

    # Model Selection using k fold cross validation ends
    # ----------------------------------------------------

    # Feature Selection using k fold cross validation starts
    data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)
    lm = LinearRegression()
    feature_cols = ['TV', 'radio', 'newspaper']
    X = data[feature_cols]
    y = data['sales']

    print(feature_cols, end=': ')
    print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

    feature_cols = ['TV', 'radio']
    X = data[feature_cols]
    y = data['sales']

    print(feature_cols, end=': ')
    print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

    # Feature Selection using k fold cross validation ends
