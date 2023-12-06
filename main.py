import pandas as pd

import prediction
import preprocessing_data
import evaluation
import profiler
import train_model


def fraud_detection():
    df = pd.read_csv('fraud_detection_bank_dataset.csv')
    #profiler.create_profile_report(df, "Fraud data from bank", "fraud_report")
    X_train, X_test, y_train, y_test = preprocessing_data.preparing_data(df,'targets')
    model = train_model.model_training("Logistic Regression", X_train, y_train)
    y_pred = prediction.predict(model, X_test)
    evaluation.evaluate_prediction(y_test, y_pred)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fraud_detection()
