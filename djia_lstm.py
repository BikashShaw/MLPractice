import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../Datasets/djia/all_stocks_2006-01-01_to_2018-01-01.csv', parse_dates=['Date'])
df.Date = pd.to_datetime(df.Date)

df.set_index('Date', inplace=True)

# Backfill `Open` column
values = np.where(df['2017-07-31']['Open'].isnull(), df['2017-07-28']['Open'], df['2017-07-31']['Open'])
df['2017-07-31'] = df['2017-07-31'].assign(Open=values.tolist())

values = np.where(df['2017-07-31']['Close'].isnull(), df['2017-07-28']['Close'], df['2017-07-31']['Close'])
df['2017-07-31'] = df['2017-07-31'].assign(Close=values.tolist())

values = np.where(df['2017-07-31']['High'].isnull(), df['2017-07-28']['High'], df['2017-07-31']['High'])
df['2017-07-31'] = df['2017-07-31'].assign(High=values.tolist())

values = np.where(df['2017-07-31']['Low'].isnull(), df['2017-07-28']['Low'], df['2017-07-31']['Low'])
df['2017-07-31'] = df['2017-07-31'].assign(Low=values.tolist())

df.reset_index(inplace=True)

missing_data_stocks = ['CSCO', 'AMZN', 'INTC', 'AAPL', 'MSFT', 'MRK', 'GOOGL', 'AABA']

for stock in missing_data_stocks:
    tdf = df[(df.Name == stock) & (df.Date == '2014-03-28')].copy()
    tdf.Date = '2014-04-01'
    pd.concat([df, tdf])
print("Complete")

df = df[~((df.Date == '2012-08-01') & (df.Name == 'DIS'))]

google_df = df[df.Name == 'AABA']

gdf = google_df[['Date', 'Close']].sort_values('Date')

gdf.head()

training_set = gdf[gdf.Date.dt.year != 2017].Close.values

test_set = gdf[gdf.Date.dt.year == 2017].Close.values

print("Training set size: ", training_set.size)
print("Test set size: ", test_set.size)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

training_set_scaled = scaler.fit_transform(training_set.reshape(-1, 1))


def create_train_data(training_set_scaled):
    X_train, y_train = [], []
    for i in range(5, training_set_scaled.size):
        X_train.append(training_set_scaled[i - 5: i])
        y_train.append(training_set_scaled[i])
    # Converting list to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


X_train, y_train = create_train_data(training_set_scaled)


def create_test_data():
    X_test = []
    inputs = gdf[len(gdf) - len(test_set) - 5:].Close.values
    inputs = scaler.transform(inputs.reshape(-1, 1))
    for i in range(5, test_set.size + 5):  # Range of the number of values in the training dataset
        X_test.append(inputs[i - 5: i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test


X_test = create_test_data()

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM


def create_dl_model():
    model = Sequential()

    # Adding the first LSTM layer
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))

    # Adding a second LSTM layer
    model.add(LSTM(units=100, return_sequences=True))

    # Adding a third LSTM layer
    model.add(LSTM(units=100, return_sequences=True))

    # Adding a fourth LSTM layer
    model.add(LSTM(units=100, return_sequences=True))

    # Adding a fifth LSTM layer
    model.add(LSTM(units=100))

    # Adding the output layer
    model.add(Dense(units=1))
    return model


def compile_and_run(model, epochs=50, batch_size=64):
    model.compile(metrics=['accuracy'], optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=3)
    return history


def plot_metrics(history):
    metrics_df = pd.DataFrame(data={"loss": history.history['loss']})
    plt.plot(metrics_df)
    plt.legend()
    plt.show()


def make_predictions(X_test, model):
    y_pred = model.predict(X_test)
    final_predictions = scaler.inverse_transform(y_pred)
    fp = np.ndarray.flatten(final_predictions)
    ap = np.ndarray.flatten(test_set)
    pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
    plt.plot(pdf)
    plt.legend()
    plt.show()


dl_model = create_dl_model()

history = compile_and_run(dl_model, epochs=100)

plot_metrics(history)

make_predictions(X_test, dl_model)
