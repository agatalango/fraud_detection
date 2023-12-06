from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def model_training(model_name, X_train, y_train):
    model = LogisticRegression()
    #model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=10)
    model.fit(X_train, y_train)
    return model