import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data['sales']

print(feature_cols, end=': ')

print('X', X)

# Normalizing the data - subtract the mean of the feature and divide by the standard deviation

mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

print('X', X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


def build_model():
    sequential_model = models.Sequential()

    sequential_model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    sequential_model.add(layers.Dense(64, activation='relu'))
    sequential_model.add(layers.Dense(1))
    sequential_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return sequential_model


k = 2
num_val_samples = len(X_train) // k
num_epochs = 2
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)

    # Prepares the validation data: data from partition #k
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepares the training data: data from all other partitions
    partial_X = np.concatenate([X_train[:i * num_val_samples], X_train[(i + 1) * num_val_samples:]],
                               axis=0)
    partial_y = np.concatenate(
        [y_train[:i * num_val_samples], y_train[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()

    history = model.fit(partial_X, partial_y, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=20, verbose=1)

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

# Building the history of successive mean K-fold validation scores
average_mae_history = [np.mean([X_train[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

test_mse_score, test_mae_score = model.evaluate(X_test, y_test)

print('test_mse_score', test_mse_score)
print('test_mae_score', test_mae_score)
