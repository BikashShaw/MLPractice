from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# k fold cross validation for parameter tuning and model selection
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

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

    knn = KNeighborsClassifier(n_neighbors=20)
    print("10 fold cross validation with best KNN model: ", end=" ")
    print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

    logreg = LogisticRegression()
    print("10 fold cross validation with best Logistic Regression: ", end=" ")
    print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

