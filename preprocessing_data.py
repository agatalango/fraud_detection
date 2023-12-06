import pandas as pd
import profiler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def preparing_data(df, target_column_name):
    print(df.info())
    X = df.drop([target_column_name], axis=1)
    print(X.shape)
    y = df[target_column_name]
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test