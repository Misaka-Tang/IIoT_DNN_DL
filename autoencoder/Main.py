import tensorflow
import keras

from AE import AutoEncoder as AE
from Utils import readCsv, getXY, evaluation

trainCsv = "datas/normal_train_AE.csv"
validationCsv = 'datas/anomaly_info_AE.csv'

trainDf = readCsv(trainCsv)
validationDf = readCsv(validationCsv)

x_train, y_train = getXY(trainDf)
x_validation, y_validation = getXY(validationDf)

# print("x_train = " + str(x_train.shape))
# print("y_train = " + str(y_train.shape))

# print("x_validation = " + str(x_validation.shape))
# print("y_validation = " + str(y_validation.shape))

# print("x_train = ", len(x_train))
# print("y_train = ", len(y_train))

# print("x_validation = ", len(x_validation))
# print("y_validation = " ,len(y_validation))
input_dim = x_train.shape[1]

ae = AE(input_dim = input_dim)
ae.summary()
threshold = ae.train(x_train, x_train)
anomalies = ae.detect_anomalies(x_validation, threshold)
# ae.plot_reconstruction_error(x_validation, y_validation)

evaluation(y_validation, anomalies)
ae.polt_threshhold_split_NAndA(x_validation, threshold)
