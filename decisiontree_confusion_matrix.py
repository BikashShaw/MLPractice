import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/yew1eb/machine-learning/master/Naive-bayes/pima-indians-diabetes.data.csv'
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv(url, header=None, names=col_names)

    feature_col = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'pedigree', 'age']
    X = pima[feature_col]
    y = pima.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    decisiontree = DecisionTreeClassifier(random_state=1)
    decisiontree.fit(X_train, y_train)
    y_predict = decisiontree.predict(X_test)

    print('Classification Accuracy Score:', metrics.accuracy_score(y_test, y_predict))
    # calculate null accuracy (for multi-class classification problems)
    print('Null Accuracy Score:', y_test.value_counts().head(1) / len(y_test))

    # print the first 25 true and predicted responses
    print('True:', y_test.values[0:25])
    print('Pred:', y_predict[0:25])

    confusion = metrics.confusion_matrix(y_test, y_predict)

    print('Confusion Matrix:\n', confusion)
    print('Classification Accuracy:', metrics.accuracy_score(y_test, y_predict))
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_predict))
