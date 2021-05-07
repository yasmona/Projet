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


# train a generative adversarial network on a one-dimensional function
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
#from matplotlib import pyplot


# In[3]:


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


# In[4]:


list_tweets=prepare_data('C:/Users/takwa/OneDrive/Bureau/tweet.txt')
print(list_tweets)


# In[5]:


# define the standalone generator model
def build_generator(n_outputs=70):
    #model = Sequential()
    #model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Dense(n_outputs, activation='linear'))
    model = Sequential()
    x0=Input(shape=(n_outputs, 1))
    x1=LSTM(15, activation='relu', input_shape=(n_outputs,1),batch_size=20)(x0)
    #15 nombre de neurone ,batch_size=n_batch=20 
    # repeat vector
    x2=RepeatVector(n_outputs)(x1)
    # decoder layer
    x3=LSTM(15, activation='relu', return_sequences=True)(x2)
    x4=TimeDistributed(Dense(1))(x3)
    unstacked = Lambda(lambda x: tensorflow.unstack(x, axis=2))(x4)
    dense_outputs = [Dense(70)(x) for x in unstacked]

    merged = Lambda(lambda x: K.stack(x, axis=2))(dense_outputs)

    model = Model(x0, merged)
    #model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    print('generator')
    print(model.summary())
    return model


# In[31]:


# define the standalone discriminator model
def build_discriminator(tweet_size):
    model = Sequential()
    x0=Input(shape=(tweet_size, 1))
    #x1=Flatten()(x0)
    x2=LSTM(15, activation='relu', input_shape=(tweet_size, 1))(x0)
    x3=RepeatVector(tweet_size)(x2)
    x4=LSTM(15, activation='relu', return_sequences=True)(x3)
    x5=TimeDistributed(Dense(1))(x4)
    unstacked = Lambda(lambda x: tensorflow.unstack(x, axis=2))(x5)
    dense_outputs = [Dense(tweet_size)(x) for x in unstacked]
    merged = Lambda(lambda x: K.stack(x, axis=2))(dense_outputs)
    x1=Flatten()(merged)
    x6=Dense(1, activation='sigmoid')(x1)
    model = Model(x0, x6)
    model.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['accuracy'])
    print('discriminator')
    print(model.summary())
    return model


# In[7]:


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


# In[32]:


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


# In[9]:


def generateur(list_tweets,g_model):
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


# In[10]:


from numpy import random
#latent_dim =70 dimension
seed=random.normal(loc=0.0, scale=1.0,size=(70, 1))
seed


# In[33]:


# Compile both models in preparation for training

# Build and compile the discriminator
discriminator = build_discriminator(70)
discriminator.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['accuracy'])

# Build and compile the combined model
generator = build_generator(70)


# In[34]:


generator.summary()


# In[35]:


discriminator.summary()


# In[36]:


# Pass noise through generator to get an tweet
text = generator(seed)


# In[37]:


text


# In[38]:


text.shape


# In[39]:


discriminator.trainable = False


# In[40]:


# The true output is fake, but we label them real!
# Passing the output of Generator to the Discriminator
validity = discriminator(text)
print(validity)
#x=validity.reshape(-1,70)


# In[41]:


seed.shape


# In[42]:


# Create the combined (model) object
import sys, os
combined_generator = Model((seed,validity))


# In[43]:


combined_generator.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[44]:


# Config
batch_size = 30 #The batch size defines the number of samples that will be propagated through the network.
epochs = 1000
#sample_period = 100 # apres 200 step ==> creer text every `sample_period` steps generate and save some data"


# In[45]:


# Create batch labels to use when calling train_on_batch
y_real = ones((batch_size,1))
y_fake = zeros((batch_size,1))

# Create a folder to store generated tweets
#if not os.path.exists('gan_tweet'):
  #os.makedirs('gan_tweet')


# In[46]:


#def fake_samples(epoch):
   # nois=random.randn(70,1)
   # tweet= generator.predict(nois)
    #print (tweet)
    #save simple
   


# In[47]:


#def generate_real_samples():
    #list_real_tweets=prepare_data('C:/Users/takwa/OneDrive/Bureau/tweet.txt')
    #sequences, vocab_size=tokenizer_data(list_real_tweets)
    #max_length=max_tweet(sequences)
    #sequences = pad_sequences(sequences, maxlen=70, padding='post')
   # return sequences


# In[48]:


def generate_real_samples():
    list_tweets=prepare_data('C:/Users/takwa/OneDrive/Bureau/tweet.txt')
    sequences, tokenizer=tokenizer_data(list_tweets)
    max_length=max_tweet(sequences)
    X =sequences[:30]#on a prix les 30 1er tweets
    X = pad_sequences(X, maxlen=70, padding='post')
    X=X.reshape(30, 70, 1) #33 array chaque array contient 70 array chaque array contient 1 element
    print(X)
    return X


# In[49]:


x_real1=generate_real_samples()


# In[50]:


print(x_real1.shape)
print(x_real1[0].shape)


# In[ ]:





# In[51]:


y_real.shape


# In[58]:


#print(x_real)
for epoch in range(1):
    idx=random.randint(0,(len(x_real1)-1))
    #print("idx=")
    print(idx)
    x_real=x_real1[idx]
    x=x_real.reshape(1,70,1)
    print(x.shape)
    print(len(x_real))
    
    seed=random.randn(70,1)
    print(seed.shape)
    x_fake=generator.predict(seed)#shape(70,70,1)
    #print(x_fake)
    print(x_fake.shape)
   # print(len(x_fake))
   
    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
    #discriminator_metric_genereted = discriminator.train_on_batch(x_fake,y_fake)


# In[54]:


print(seed)
print(len(seed))


# In[ ]:




