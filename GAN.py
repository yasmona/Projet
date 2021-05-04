#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import LSTM,Dense,Input,ConvLSTM2D,Reshape,Activation,Lambda,Softmax
from keras.layers import Dense, LSTM, Input, Embedding, TimeDistributed, Flatten, Dropout,RepeatVector,Reshape,Activation,Lambda,Softmax
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from keras import backend as K
import keras.backend as kerback
from keras.models import Model
from keras.preprocessing.text import one_hot
import tensorflow 
pd.__version__


# In[2]:


def prepare_data(f):
    data = pd.read_csv(f, sep="\n",header=None)
    list_listword=[]#liste des listes des mots de chaque tweet
    list_tweets=data[0].values.tolist()#liste des tweets
    l=[]
    for text in list_tweets:
        text = re.sub(r'http\S+', '', text)   # Remove URLs
        text = re.sub(r'—', ' ', text) 
        text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove @ mentions
        text = text.strip(" ")   # Remove whitespace resulting from above
        text = re.sub(r' +', ' ', text)   # Remove redundant spaces
        l.append(text)
    
    return l


# In[3]:


list_tweets=prepare_data('C:/tweet.txt')
print(list_tweets)


# In[15]:


def tokenizer_data(list_tweets):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list_tweets)
    sequences = tokenizer.texts_to_sequences(list_tweets)
    vocab_size=len(tokenizer.word_index)
    return sequences, tokenizer

def max_tweet(sequences):
    max_length=0
    for i in range(len(sequences)):
        if(len(sequences[i])>max_length):
            max_length=len(sequences[i])
        #print("max",max_length)
        #print("seq=",sequences)
        #print(vocab_size)
    return max_length

sequences, tokenizer=tokenizer_data(list_tweets)

print(len(tokenizer.word_index))
max_length=max_tweet(sequences)
print(max_length)


# In[34]:


def generate(list_tweets,g_model):
    print(list_tweets)
    sequences, tokenizer=tokenizer_data(list_tweets)
    sequences= pad_sequences(sequences, maxlen=70, padding='post')
    result = g_model.predict(sequences, batch_size=3, verbose=0)
    print(result)
    res=[]
    for i in range(len(result)):
        predicted_word = ''
        for j in range(len(result[i])):
            for word, index in tokenizer.word_index.items():
                if index == int(result[i][j][0]):
                    predicted_word = predicted_word+' '+word
                    break
        res.append(predicted_word)
    #print(res)
    for i in res:
        print(i)
    return res


# In[35]:


def disc_text(model_disc, seed_text):
    print(seed_text)
    sequences, tokenizer=tokenizer_data(seed_text)
    sequences= pad_sequences(sequences, maxlen=70, padding='post')
    #y=model_disc.predict([encoded])
    y = model_disc.predict([sequences], batch_size=3, verbose=0)
    print(y)
    for i in y:
        if(i<0.6):
            print("fake")
        else:
            print("real")
   


# In[36]:


# train a generative adversarial network on a one-dimensional function
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
#from matplotlib import pyplot
 
# define the standalone discriminator model
def define_discriminator(n_inputs=70):
    model = Sequential()
    x0=Input(shape=(70, 1))
    #x1=Flatten()(x0)
    x2=LSTM(15, activation='relu', input_shape=(70, 1))(x0)
    x3=RepeatVector(70)(x2)
    x4=LSTM(15, activation='relu', return_sequences=True)(x3)
    x5=TimeDistributed(Dense(1))(x4)
    unstacked = Lambda(lambda x: tensorflow.unstack(x, axis=2))(x5)
    dense_outputs = [Dense(70)(x) for x in unstacked]
    merged = Lambda(lambda x: K.stack(x, axis=2))(dense_outputs)
    x1=Flatten()(merged)
    x6=Dense(1, activation='sigmoid')(x1)
    model = Model(x0, x6)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(model.summary())
    return model
 
# define the standalone generator model
def define_generator(latent_dim=70, n_outputs=70):
    #model = Sequential()
    #model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Dense(n_outputs, activation='linear'))
    model = Sequential()
    x0=Input(shape=(n_outputs, 1))
    x1=LSTM(15, activation='relu', input_shape=(n_outputs,1))(x0)
    # repeat vector
    x2=RepeatVector(n_outputs)(x1)
    # decoder layer
    x3=LSTM(15, activation='relu', return_sequences=True)(x2)
    x4=TimeDistributed(Dense(1))(x3)
    unstacked = Lambda(lambda x: tensorflow.unstack(x, axis=2))(x4)
    dense_outputs = [Dense(70)(x) for x in unstacked]

    merged = Lambda(lambda x: K.stack(x, axis=2))(dense_outputs)

    model = Model(x0, merged)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print(model.summary())
    return model
 
# generate n real samples with class labels
def generate_real_samples():
    list_real_tweets=prepare_data('c:/real_tweets.txt')
    sequences, vocab_size=tokenizer_data(list_real_tweets)
    max_length=max_tweet(sequences)
    sequences = pad_sequences(sequences, maxlen=70, padding='post')
    #labels_real=[1 for i in range(len(list_real_tweets))]
    #labels_real=np.array(labels_real)
    # generate inputs in [-0.5, 0.5]
    labels_real = ones((len(list_real_tweets), 1))
    
    return sequences,labels_real

# generate points in latent space as input for the generator
def generate_latent_points():
    list_tweets=prepare_data('C:/tweet.txt')
    sequences, tokenizer=tokenizer_data(list_tweets)
    max_length=max_tweet(sequences)
    #labels_fake=[0 for i in range(len(list_tweets))]
    # generate inputs in [-0.5, 0.5]
    X =sequences[:33]
    X = pad_sequences(X, maxlen=70, padding='post')
    #print("X=",X)
    #print(X.shape)
    y =sequences[33:]
    y = pad_sequences(y, maxlen=70, padding='post')
    #print("y=",y)
    #print(y.shape)
    X=X.reshape(33, 70, 1)
    y=y.reshape(33, 70, 1)
    #print(X)
    #print(y)
    #print('train_x shape:', X.shape)
    #print('train_y shape:', y.shape)
    X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.33)
    #print(X_test.shape)
    #print(Y_test.shape)
    
    return X_train,Y_train,X_test,Y_test

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator):
    # generate points in latent space
    X_train,X_test,Y_train,Y_test = generate_latent_points()
    # predict outputs
    X = generator.predict(X_train)
    # create class labels
    y = zeros((22, 1))
    return X, y

 
# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = generate_real_samples()
    #print("real")
    #print(x_real.shape, y_real.shape)
    # evaluate discriminator on real examples
    acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator)
    # evaluate discriminator on fake examples
    acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    # scatter plot real and fake data points
    #pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    #pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    #pyplot.show()
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=1, n_batch=128, n_eval=2):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    x_real, y_real = generate_real_samples()
    #print(x_real)
    #print("real")
    #print(x_real)
    #print(y_real)
    #print(x_real.shape)
    #print(y_real.shape)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model)
    #print("fake")
    #print(x_fake, y_fake)
    #print(x_fake.shape,y_fake.shape)
    X_train,X_test,Y_train,Y_test = generate_latent_points()
    y_gan = ones((22, 1))
    #print("gan")
    #print(X_train,y_gan)
    #print(X_train.shape,y_gan.shape)
    for i in range(n_epochs):
        # prepare real samples
        #x_real, y_real = generate_real_samples()
        #print(x_real)
        #print("real")
        #print(x_real)
        #print(y_real)
        #print(x_real.shape)
        #print(y_real.shape)
        # prepare fake examples
        #x_fake, y_fake = generate_fake_samples(g_model)
        #print("fake")
        #print(x_fake, y_fake)
        #print(x_fake.shape,y_fake.shape)
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        #X_train,X_test,Y_train,Y_test = generate_latent_points()
        # create inverted labels for the fake samples
        #y_gan = ones((22, 1))
        #print("gan")
        #print(X_train,y_gan)
        #print(x_gan.shape,y_gan.shape)
        # update the generator via the discriminator's error
        gan_model.train_on_batch(X_train, y_gan)
        generate_tweets=generate(list_tweets[3:5],g_model)
        print("list_tweets[3:5]")
        print(list_tweets[3:5])
        print("generate_tweets")
        print(generate_tweets)
        disc_text(d_model,generate_tweets)
        # evaluate the model every n_eval epochs
        print(i)
        print((i+1)% n_eval)
        #if (i+1) % n_eval == 0:
        summarize_performance(i, g_model, d_model, latent_dim)


# size of the latent space
latent_dim = 5
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)


# In[ ]:





# In[ ]:




