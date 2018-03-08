import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/yew1eb/machine-learning/master/Naive-bayes/pima-indians-diabetes.data.csv'
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv(url, header=None, names=col_names)
    print(pima.head())

    feature_col = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'pedigree', 'age']
    X = pima[feature_col]
    y = pima.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_predict = logreg.predict(X_test)

    print('Classification Accuracy Score:', metrics.accuracy_score(y_test, y_predict))

    # examine the class distribution of the testing set (using a Pandas Series method)
    print(y_test.value_counts())
    # calculate the percentage of 1s (ones)
    print(y_test.mean())
    # calculate the percentage of 0s (zeros)
    print(1 - y_test.mean())

    # calculate null accuracy (for multi-class classification problems)
    print('Null Accuracy Score:', y_test.value_counts().head(1) / len(y_test))

    # print the first 25 true and predicted responses
    print('True:', y_test.values[0:25])
    print('Pred:', y_predict[0:25])
    confusion = metrics.confusion_matrix(y_test, y_predict)
    print('Confusion Matrix:\n', confusion)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    print('Classification Accuracy: ', (TP + TN) / float(TP + TN + FP + FN))
    print('Classification Accuracy:', metrics.accuracy_score(y_test, y_predict))

    print('Classification Error: ', (FP + FN) / float(TP + TN + FP + FN))
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_predict))

    print('Sensitivity (True Positive Rate):', TP / float(TP + FN))
    print('Sensitivity (True Positive Rate):', metrics.recall_score(y_test, y_predict))

    print('Specificity (True Negative Rate):', TN / float(TN + FP))

    print('False Positive Rate:', FP / float(FP + TN))

    print('Precision (Rate of correct positive prediction):', (TP / float(TP + FP)))
    print('Precision (Rate of correct positive prediction):', metrics.precision_score(y_test, y_predict))

    print(logreg.predict(X_test)[0:20])
    y_predict_proba = logreg.predict_proba(X_test)[:, 1]
    print(y_predict_proba)

    plt.rcParams['font.size'] = 14
    plt.hist(y_predict_proba, bins=8)
    plt.xlim(0, 1)
    plt.title("Histogram of predicted probabilities")
    plt.xlabel("Predicted probability of diabetes")
    plt.ylabel("Frequency")
    plt.show()

    y_predict = binarize([y_predict_proba], 0.3)[0]

    print('True:', y_test.values[0:25])
    print('Pred:', y_predict[0:25])
    confusion = metrics.confusion_matrix(y_test, y_predict)
    print('Confusion Matrix:\n', confusion)

    print('Classification Accuracy:', metrics.accuracy_score(y_test, y_predict))
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_predict))
    print('Sensitivity (True Positive Rate):', metrics.recall_score(y_test, y_predict))
    print('Precision (Rate of correct positive prediction):', metrics.precision_score(y_test, y_predict))
