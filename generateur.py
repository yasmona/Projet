from keras.layers import Dense, LSTM, Input, TimeDistributed, RepeatVector, Lambda, Concatenate
from keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf

class Generateur():
    def modele_generateur(batch_size):  # max_length nombre d'element das une tweet

        noise = Input(shape=(batch_size, 1))

        label = Input(shape=(batch_size, 1))

        x0 = Concatenate(axis=2)([noise, label])

        x1 = LSTM(15, activation='relu', input_dim=batch_size)(x0) # 15 nombre de neurone ,batch_size=n_batch=20
        # repeat vector
        x2 = RepeatVector(batch_size)(x1)
        # decoder layer
        x3 = LSTM(15, activation='relu', return_sequences=True)(x2)
        x4 = TimeDistributed(Dense(1))(x3)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=2))(x4)
        dense_outputs = [Dense(batch_size)(x) for x in unstacked]
        merged = Lambda(lambda x: K.stack(x, axis=2))(dense_outputs)
        model = Model([noise, label], merged)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('generator')
        print(model.summary())
        return model