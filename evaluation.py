from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_prediction(y_test, y_pred):
    roc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(roc)
    print(precision)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
