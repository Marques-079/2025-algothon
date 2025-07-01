from collections import deque
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # or '0' to disable
import tensorflow as tf
import numpy as np

class RegressionNetWork():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(1)
        ])

        optimzer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimzer, loss='huber')
    def loss_epoch(self,loss):
        base = 200
        loss_factor = (10/loss)
        return min(base - int(loss_factor),120)
    
    def predict(self,t):
        wrapped_t = np.array([t])
        return self.model.predict(wrapped_t,verbose=1)

    def train_1(self, X_train, y_train,i):
        model = self.model
        x_sample = X_train
        y_sample = y_train
        loss = model.train_on_batch(x_sample, y_sample)
        for j in range(self.loss_epoch(loss)):
            x_sample = X_train[j:]
            y_sample = y_train[j:]
            localloss = model.train_on_batch(x_sample, y_sample)
            loss = localloss
        
        print(f" Step {i}, Loss: {loss:.5f}")
    
    def multiTrain(self,X,Y,time):
        pass