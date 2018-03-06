from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    k_range = range(1, 31)


    param_grid = dict(n_neighbors=k_range)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    grid_mean_score = [result for result in grid.cv_results_['mean_test_score']]

    plt.plot(k_range, grid_mean_score)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Cross Validation Accuracy")
    plt.show()

    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)

    print(grid.predict([[3, 5, 4, 2]]))

    weight_options = ['uniform', 'distance']
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    r_grid = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
    r_grid.fit(X, y)

    print(r_grid.best_score_)
    print(r_grid.best_params_)
    print(r_grid.best_estimator_)

    print(r_grid.predict([[3, 5, 4, 2]]))

    for _ in range(20):
        r_grid = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10)
        r_grid.fit(X, y)
        print(r_grid.best_score_)
        print(grid.predict([[3, 5, 4, 2]]))

