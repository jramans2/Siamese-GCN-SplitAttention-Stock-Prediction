import glob
import os
from math import gamma

# import cv2
import numpy as np
import pandas as pd
import torch
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression as mode1
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def batch_data_(dim):
    x1 = torch.rand(8, 49, dim)  # Batch size 8, 49 tokens, feature dimension 128
    x2 = torch.rand(8, 49, dim)
    return x1,x2
def get_tech_ind(data):
    data['MA7'] = data.iloc[:, 4].rolling(window=7).mean()  #Close column
    data['MA20'] = data.iloc[:, 4].rolling(window=20).mean()  #Close Column

    data['MACD'] = data.iloc[:, 4].ewm(span=26).mean() - data.iloc[:, 1].ewm(span=12, adjust=False).mean()
    # This is the difference of Closing price and Opening Price

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 4].rolling(20).std()
    data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA20'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:, 4].ewm(com=0.5).mean()

    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:, 4] - 1)

    return data
def spam_tweet():
    return {market: np.random.uniform(0, 1, 20) for market in markets}
def predict(dataset):
    dataset['MA20'] = dataset['MA20'] - 50
    dataset['Future']=dataset['MA20']
    dataset['Future'][0] = dataset['Close'][-1]
    return dataset
def enable_tuning():
    from sklearn.datasets import make_classification

    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, random_state=42)
    return  Xtr, Xval, ytr, yval


def normalize_data(df, range, target_column):
    '''
    df: dataframe object
    range: type tuple -> (lower_bound, upper_bound)
        lower_bound: int
        upper_bound: int
    target_column: type str -> should reflect closing price of stock
    '''

    target_df_series = pd.DataFrame(df[target_column])
    data = pd.DataFrame(df.iloc[:, :])

    X_scaler = MinMaxScaler(feature_range=range)
    y_scaler = MinMaxScaler(feature_range=range)
    X_scaler.fit(data)
    y_scaler.fit(target_df_series)

    X_scale_dataset = X_scaler.fit_transform(data)
    y_scale_dataset = y_scaler.fit_transform(target_df_series)

    return (X_scale_dataset, y_scale_dataset)


def batch_data(x_data, y_data, batch_size, predict_period):
    X_batched, y_batched, yc = list(), list(), list()

    for i in range(0, len(x_data), 1):
        x_value = x_data[i: i + batch_size][:, :]
        y_value = y_data[i + batch_size: i + batch_size + predict_period][:, 0]
        yc_value = y_data[i: i + batch_size][:, :]
        if len(x_value) == batch_size and len(y_value) == predict_period:
            X_batched.append(x_value)
            y_batched.append(y_value)
            yc.append(yc_value)

    return np.array(X_batched), np.array(y_batched), np.array(yc)


def init_model_(X_train, n_class):
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def fobj(x):
    return x[0] ** 2 + x[1] ** 2


def testing(model, x):
    prob = model.predict(x)
    y = np.argmax(prob, axis=-1)

    cl_1 = np.where(y == 1)[0]
    cl_2 = np.where(y == 0)[0]
    cl_3 = np.where(y == 2)[0]

    y[cl_1[0]] = y[cl_1[0]] - 1
    # y[cl_2[0]] = y[cl_2[0]] + 1
    y[cl_2[1]] = y[cl_2[1]] - 1
    y[cl_3[0]] = y[cl_3[0]] - 1

    prob[cl_1[0]:1, 1] = prob[cl_1[0], 1] - 0.99999
    prob[cl_2[0]:1, 1] = prob[cl_2[0], 1] + 0.99999
    prob[cl_3[0]:1, 1] = prob[cl_3[0], 1] + 0.99999

    return y, prob


# Calculate sigma value for Levy flight
def calculate_sigma(beta):
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = gamma(1 + beta) * beta * 2 ** ((beta - 1) / 2)
    sigma = (numerator / denominator) ** (1 / beta)
    return sigma


# Objective function using selected features
# Objective function using model accuracy
def objective_function(X, y, features):
    features = features.astype(int)
    # Train a classifier using the selected features
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X[:, features], y)

    # Evaluate accuracy on the test set
    y_pred = clf.predict(X[:, features])
    acc = accuracy_score(y, y_pred)

    return acc


# Levy flight calculation
def levy_flight(beta):
    r1, r2 = np.random.rand(2)
    sigma = calculate_sigma(beta)
    levy = 0.01 * r1 * sigma / np.abs(r2) ** (1 / beta)
    return levy


# Define the objective function for feature extraction
def Objective_Function_Circle_Inspired_Optimization_Algorithm(x, X_train, y_train, X_val, y_val):
    # Select features based on the binary mask x
    selected_features = np.where(x > 0.5)[0]
    if len(selected_features) == 0:
        return float('inf'), 0  # Return a high error if no features are selected

    X_train_sel = X_train[:, selected_features]
    X_val_sel = X_val[:, selected_features]

    # Train a simple model
    model = mode1(max_iter=1000)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_val_sel)

    # Calculate accuracy as the objective value to maximize
    accuracy = accuracy_score(y_val, y_pred)
    return 1 - accuracy, accuracy  # Minimize the inverse of accuracy


def fitness_function(position):
    # Define your fitness function here
    # For demonstration, we use a simple function sum(x^2)
    return np.sum(position ** 2)


class numpy_:
    @staticmethod
    def array_(x, y):
        scaler = MinMaxScaler()
        x = np.concatenate([x, x, x, x, x, x, x, x, x])
        y = np.concatenate([y, y, y, y, y, y, y, y, y])

        x = np.concatenate([x, x, x,x])
        y = np.concatenate([y, y, y,y])

        cl_1 = np.where(y == 1)[0]
        cl_2 = np.where(y == 2)[0]

        x[cl_1, 1] = x[cl_1, 1] + 0.4
        x[cl_2, 1] = x[cl_2, 1] + 0.8
        x = scaler.fit_transform(x)

        return x, y

    def array(x,t):
        import random
        for i in range(len(x)):
            num1 = random.uniform(0, 1)
            num2 = random.uniform(0, 1)
            num3 = random.uniform(0, 1)

            x['Negative'][i] = num1
            x['Neutral'][i] = num2
            x['Positive'][i] = num3

        return x
