import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

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

    y_predict_proba = decisiontree.predict_proba(X_test)[:, 1]
    print(y_predict_proba)

    plt.rcParams['font.size'] = 14
    plt.hist(y_predict_proba, bins=8)
    plt.xlim(0, 1)
    plt.title("Histogram of predicted probabilities")
    plt.xlabel("Predicted probability of diabetes")
    plt.ylabel("Frequency")
    plt.show()

    falsePositiveRate, truePositiveRate, threshold = metrics.roc_curve(y_test, y_predict_proba)

    plt.plot(falsePositiveRate, truePositiveRate)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()
