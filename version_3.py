#!/usr/bin/env python
# coding: utf-8

# In[18]:


from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import LSTM,Dense,Input,ConvLSTM2D,Reshape,Activation,Lambda,Softmax
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split


# In[19]:


import pandas as pd
pd.__version__
df = pd.read_csv('C:/tweet.txt', sep="\n",header=None)
df


# In[20]:


list_listword=[]#liste des listes des mots de chaque tweet
list_tweets=df[0].values.tolist()#liste des tweets
list_tweets


# In[21]:


print(list_tweets[0])
for i in range(0,len(list_tweets)):
  list_listword.append(list_tweets[i].split())
list_listword


# In[22]:



tokenizer = Tokenizer()
tokenizer.fit_on_texts(list_tweets)
sequences = tokenizer.texts_to_sequences(list_tweets)
vocab_size=len(tokenizer.word_index)
max_length=0
for i in range(len(sequences)):
  if(len(sequences[i])>max_length):
    max_length=len(sequences[i])
print("max",max_length)
print("seq=",sequences)
print(vocab_size)


# In[23]:



X =sequences[:33]
X = pad_sequences(X, maxlen=max_length, padding='post')
print("X=",X)
print(X.shape)
y =sequences[33:]
y = pad_sequences(y, maxlen=max_length, padding='post')
print("y=",y)
print(y.shape)
X=X.reshape(33, max_length, 1)
y=y.reshape(33, max_length, 1)
print(X)
print(y)
print('train_x shape:', X.shape)
print('train_y shape:', y.shape)
X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.33)
print(X_test.shape)
print(Y_test.shape)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


# In[51]:


from keras import backend as K
import keras.backend as kerback
from keras.models import Model
import tensorflow 
model = Sequential()
x0=Input(shape=(57, 1))
# encoder layer
#model.add(LSTM(10, activation='relu', input_shape=(57, 1)))
x1=LSTM(15, activation='relu', input_shape=(57, 1))(x0)
#x1=LSTM(100, activation='relu', input_shape=(3, 1))
# repeat vector
#model.add(RepeatVector(57))
x2=RepeatVector(57)(x1)
# decoder layer
#model.add(LSTM(100, activation='relu', return_sequences=True))
x3=LSTM(15, activation='relu', return_sequences=True)(x2)
#model.add(LSTM(10, activation='relu', return_sequences=True))
#model.add(TimeDistributed(Dense(1)))
x4=TimeDistributed(Dense(1))(x3)
#x4=TimeDistributed(Dense(1))
unstacked = Lambda(lambda x: tensorflow.unstack(x, axis=2))(x4)
dense_outputs = [Dense(57)(x) for x in unstacked]
#print(unstacked)
merged = Lambda(lambda x: K.stack(x, axis=2))(dense_outputs)
#y6=Softmax(axis=-1)(merged)

model = Model(x0, merged)
#model.compile(optimizer='adam', loss='categorical_crossentropy')
model.compile(optimizer='adam', loss='binary_crossentropy')
print(model.summary())


# In[54]:


#model.fit(X, y, epochs=10, batch_size=1, verbose=2)
#model.fit(X, y, epochs=200, validation_split=0.2, verbose=1, batch_size=3)
#model.fit(X, y, epochs=200, verbose=0)
model.fit(X_train,Y_train,epochs=10, batch_size=15,validation_data=(X_test, Y_test))
#model.fit(X,y, epochs=20, batch_size=32, validation_split=3/9)
#loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
#print('Accuracy: %f' % (accuracy*100))


# In[53]:


#test_input = np.array([300, 305, 310])
test_input=X[5:9]
print(y[5:9])
print(test_input)
encoded = pad_sequences([test_input], maxlen = 3, truncating='pre')

result = model.predict(test_input, batch_size=3, verbose=0)
#print(test_output)
print(result)


# In[43]:


res=[]
for i in range(len(result)):
    predicted_word = ''
    for j in range(len(result[i])):
     for word, index in tokenizer.word_index.items():
       
        if index == int(result[i][j][0]):
          predicted_word = predicted_word+' '+word
          break
    res.append(predicted_word)
print(res)
for i in res:
    print(i)


# In[ ]:





# In[ ]:




