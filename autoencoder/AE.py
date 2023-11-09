import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras import regularizers, initializers
from keras.callbacks import EarlyStopping

tf.random.set_seed(123)

class AutoEncoder():
    
        def __init__(self, input_dim):
    
            input_layer = Input(shape=(input_dim,))
    
            layer = Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal())(input_layer)
            layer = Dense(16, activation='relu', kernel_initializer=initializers.RandomNormal())(layer)
            layer = Dense(8,  activation='relu', kernel_initializer=initializers.RandomNormal())(layer)
            layer = Dense(16, activation='relu', kernel_initializer=initializers.RandomNormal())(layer)
            layer = Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal())(layer)
    
            output_layer = Dense(input_dim, activation='tanh', kernel_initializer=initializers.RandomNormal())(layer) 
    
            self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
    
        def summary(self, ):
            self.autoencoder.summary()
    
        

        def train(self, x, y):

            epochs = 90
            batch_size = 1024
            validation_split = 0.1  

            self.autoencoder.compile(optimizer='Nadam', loss='mean_squared_error')

            es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
            history = self.autoencoder.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                                        shuffle=True, verbose=2, callbacks=[es])

            # plt.plot(history.history['loss'])
            # plt.plot(history.history['val_loss'])
            # plt.title('model loss')
            # plt.ylabel('loss')
            # plt.xlabel('epoch')
            # plt.legend(['training', 'validation'], loc='upper right')
            # plt.show()

            #   Computation of the detection threshold with a percentage
            #       of the training set equal to 'validation_split'
            loss = history.history['loss']
            threshold = np.mean(loss) + 3 * np.std(loss)
            print("Detection threshold: " + str(threshold))
            return threshold
    
        def predict(self, x):
            outcome = self.autoencoder.predict(x)
            return outcome
        
        def detect_anomalies(self, x, threshold):
        
            y_pred = self.autoencoder.predict(x)
            print('y_pred: ', y_pred)
            mse = np.mean(np.power(x - y_pred, 2), axis=1)
            outcome = mse<=threshold
            return outcome
        
        def plot_reconstruction_error(self, x, y):
            outcome = self.autoencoder.predict(x)
            loss = np.mean(np.power(x - outcome, 2), axis=1)
            plt.hist(loss, bins=50)
            plt.xlabel("Reconstruction error")
            plt.ylabel("Frequency")
            plt.show()

        def polt_threshhold_split_NAndA(self, x, threshold):
            y_pred = self.autoencoder.predict(x)
            mse = np.mean(np.power(x - y_pred, 2), axis=1)
            normal_list = []
            anomaly_list = []
            for i in mse:
                if i <= threshold:
                    normal_list.append(i)
                else:
                    anomaly_list.append(i)
            plt.scatter(range(len(normal_list)), normal_list, label='Normal', s=1, c='b')
            plt.scatter(range(len(anomaly_list)), anomaly_list, label='Anomaly', s=1, c='r')
            plt.hlines(threshold, xmin=0, xmax=len(mse), colors='g', linestyles='dashed', label='Threshold')
            # plt.text(0.5, 1.05, "Scatter plot of reconstruction errors", ha='center', va='center', transform=plt.gca().transAxes)
            plt.xlabel("Data")
            plt.ylabel("Reconstruction error")
            plt.legend()
            plt.show()
                