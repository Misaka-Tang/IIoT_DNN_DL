import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MaxAbsScaler
# csv --> dataframe
def readCsv(path):
    return pd.read_csv(path, index_col=False, dtype='unicode')

def getXY(df):
    # x = df[features].values
    x = df[df.columns[:-1]].values
    # Y = df['Label'].values
    y = df[df.columns[-1]].values

    scaler = MaxAbsScaler()
    x = scaler.fit_transform(x)
    return x, y

def evaluation(y_test, y_pred):
    # if label == 7 -> normal
    # else -> anomaly
    y_test = np.where(y_test == '7', True, False)
    # print how many true item in y_test
    print('Number of normal data: ', np.count_nonzero(y_test==True))
    
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred, labels=[True, False])
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_yticklabels([ 'Normal', 'Anomaly'])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)
    precision = precision_score(y_test, y_pred, average='weighted')
    print('Precision: ', precision)
    recall = recall_score(y_test, y_pred, average='weighted')
    print('Recall: ', recall)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('F1 score: ', f1)
    

def saveModel(model, path):
    model.save(path)
    
#result
# Accuracy:  0.9333615400888513
# Precision:  0.9386744013573938
# Recall:  0.9333615400888513
# F1 score:  0.9291232845681952