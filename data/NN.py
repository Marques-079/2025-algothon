from collections import deque
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # or '0' to disable
import tensorflow as tf
import numpy as np

class RegressionNetWork():
    def __init__(self):
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Input(shape=(30,)),
            tf.keras.layers.LSTM(16, return_sequences=False,input_shape=(30,1)),

            tf.keras.layers.Dense(8, activation='relu'),

            tf.keras.layers.Dense(1)
        ])

        optimzer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimzer, loss='mse')
    def loss_epoch(self,loss):
        base = 100
        loss_factor = (10/loss)
        return min(base - int(loss_factor),80)
    
    def predict(self,t):
        wrapped_t = np.array([t])
        return self.model.predict(wrapped_t,verbose=1)

    def predict_2(self,prices):
        return self.model.predict(prices.reshape(1,30,1),verbose=1)
    def train_1(self, X_train, y_train,i):
        model = self.model
        x_sample = X_train
        y_sample = y_train
        loss = model.train_on_batch(x_sample, y_sample)
        for j in range(self.loss_epoch(loss)):
            x_sample = X_train[j:]
            y_sample = y_train[j:]
            loss = model.train_on_batch(x_sample, y_sample)
        
        print(f" Step {i}, Loss: {loss:.5f}")
    
    def train_2(self, prices, t):
        if t < 31:
            return  # Not enough data for input and delta

        x_t = prices[t-30:t].reshape(1, 30, 1)         # shape (1, 30, 1)
        y_t = (prices[t] - prices[t-1]).reshape(1, 1)  # delta as target

        model = self.model
        loss = model.train_on_batch(x_t, y_t)

        for _ in range(self.loss_epoch(loss)):
            loss = model.train_on_batch(x_t, y_t)

        print(f"Step {t}, Loss: {loss:.5f}")
    
    

