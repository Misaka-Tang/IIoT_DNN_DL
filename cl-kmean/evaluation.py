from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Evaluation:
    def plot_confusion_matrix(self, y_true, y_pred):
        matrix = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    def get_basic_evaluation_score(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy: ', accuracy)
        precision = precision_score(y_test, y_pred, average='weighted')
        print('Precision: ', precision)
        recall = recall_score(y_test, y_pred, average='weighted')
        print('Recall: ', recall)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print('F1 score: ', f1)
        return accuracy, precision, recall, f1_score
    
    def evalution(self, y_test, y_pred):
        y_test = np.where(y_test == '7', True, False)
        # self.plot_confusion_matrix(y_test, y_pred)
        self.get_basic_evaluation_score(y_test, y_pred)