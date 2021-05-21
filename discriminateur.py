from keras.layers import Dense, LSTM, Input, TimeDistributed, Flatten, RepeatVector, Lambda, Concatenate
from keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf

class Discriminateur():

    def modele_discriminateur(batch_size):
        tweet = Input(shape=(batch_size, 1))

        label = Input(shape=(batch_size, 1))

        x0 = Concatenate(axis=1)([tweet, label])

        x2 = LSTM(15, activation='relu', input_dim=batch_size)(x0)
        x3 = RepeatVector(batch_size)(x2)
        x4 = LSTM(15, activation='relu', return_sequences=True)(x3)
        x5 = TimeDistributed(Dense(1))(x4)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=2))(x5)
        dense_outputs = [Dense(batch_size)(x) for x in unstacked]
        merged = Lambda(lambda x: K.stack(x, axis=2))(dense_outputs)
        x1 = Flatten()(merged)
        x6 = Dense(1, activation='sigmoid')(x1)
        model = Model([tweet, label], x6)
        print(model.summary())
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model