from collections import deque
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # or '0' to disable
import tensorflow as tf
import numpy as np

class RegressionNetWork():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(256, activation='swish'),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Dense(256, activation='swish'),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Dense(64, activation='swish'),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Dense(8, activation='swish'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse')
    def loss_epoch(self,loss):
        base = 100
        loss_factor = (10/loss)
        return min(base - int(loss_factor),70)
    
    def predict(self,t):
        wrapped_t = np.array([t])
        return self.model.predict(wrapped_t,verbose=0)
    
    
    def train_1(self, X_train, y_train, i):
        model = self.model
        x_sample = X_train[i:i+1]
        y_sample = y_train[i:i+1]
        loss = model.train_on_batch(x_sample, y_sample)
        for j in range(self.loss_epoch(loss)):
            x_sample = X_train[j:i+1]
            y_sample = y_train[j:i+1]
            loss = model.train_on_batch(x_sample, y_sample)
        
        # print(f" Step {i}, Loss: {loss:.5f}")
    
    

