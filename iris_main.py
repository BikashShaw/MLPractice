from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def predict(classifier):
    y_predict = classifier.predict(X_new)
    print(y_predict)
    print("Prediction Accuracy: ", end=' ')
    y_predict = classifier.predict(X)
    print(metrics.accuracy_score(y, y_predict))


def k_neighbors():
    print("::KNeighbors Classifier::")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)

    print("K Nearest Neighbors with neighbors 1:", end=' ')
    predict(knn)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    print("K Nearest Neighbors with neighbors 5:", end=' ')
    predict(knn)


def logistic_regression():
    print("::LogisticRegression Classifier::")
    logreg = LogisticRegression()
    logreg.fit(X, y)
    predict(logreg)


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
    k_neighbors()
    logistic_regression()
