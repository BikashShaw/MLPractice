import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def linear_regression_feature_selection(feature_cols):
    print(feature_cols)

    X = data[feature_cols]
    y = data['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_predict = linreg.predict(X_test)

    # MAE: Mean Absolute Error
    print("Mean Absolute Error (MAE): ", end=" ")
    print(metrics.mean_absolute_error(y_test, y_predict))
    # MSE: Mean Squared Error
    print("Mean Squared Error (MSE): ", end=" ")
    print(metrics.mean_squared_error(y_test, y_predict))
    # RMSE: Root Mean Squared Error
    print("Root Mean Squared Error (RMSE): ", end=" ")
    print(np.sqrt(metrics.mean_squared_error(y_test, y_predict)))


if __name__ == '__main__':
    data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)
    print(data.head())
    print(data.tail())

    linear_regression_feature_selection(['TV', 'radio', 'newspaper'])
    linear_regression_feature_selection(['TV', 'radio'])
    linear_regression_feature_selection(['TV', 'newspaper'])
    linear_regression_feature_selection(['radio', 'newspaper'])
    linear_regression_feature_selection(['TV'])
    linear_regression_feature_selection(['radio'])
    linear_regression_feature_selection(['newspaper'])

    sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')
    plt.show()
