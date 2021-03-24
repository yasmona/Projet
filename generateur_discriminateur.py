#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import numpy as np
from tensorflow.keras.utils import to_categorical


# In[2]:


import pandas as pd
pd.__version__
df = pd.read_csv('C:/tweet.txt', sep="\n",header=None)
df


# In[3]:



list_listword=[]#liste des listes des mots de chaque tweet
list_tweets=df.values.tolist()#liste des tweets
list_tweets


# In[4]:


print(list_tweets[0])
for i in range(0,len(list_tweets)):
  list_listword.append(list_tweets[i][0].split())
list_listword


# In[5]:


length=20+1
lines=[]#les tweets decale par 11 mots
for j in range(len(list_tweets)):
    for i in range(length,len(list_listword[j])):
        seq=list_listword[j][i-length:i]
        line=' '.join(seq)
        lines.append(line)
print(lines[0])
print(lines[1])


# In[6]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
print(sequences)
#sequences=np.array(sequences)


# In[7]:


#y = sequences.reshape(-1, length, 1)
print(len(sequences[0]))
vocab_size=len(tokenizer.word_index)


# In[8]:



sequences = np.array(sequences[:50])
X, y = sequences[:, :-1], sequences[:,-1]

#y =sequences[:,20:21]


print(y)


# In[9]:


vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
seq_length


# In[10]:


model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
#model.add(TimeDistributed(Dense(1)))
model.add(Dense(vocab_size, activation='softmax'))


# In[11]:


model.summary()


# In[12]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[13]:


model.fit(X, y, batch_size = 256, epochs = 1000)


# In[14]:


seed_text=lines[27]
seed_text


# In[15]:


#n_words= nb mta3 les mots eli bech predictihom 
#seed_text = jomla ta3tihelo bech ikmel 3liha w tajem ta3tihelo fer4a ya3ni yasna3lek howa jomla jdida 
def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
  text = []

  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = model.predict_classes(encoded)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  #print(seed_text)
  return ' '.join(text)


# In[16]:


text=generate_text_seq(model, tokenizer, seq_length, "", 10)


# In[17]:


fake_tweets = pd.read_csv('c:/fake_tweets.txt', sep="\n",header=None)
fake_tweets


# In[18]:


list_fake_tweets=fake_tweets[0].values.tolist()#liste des tweets
list_fake_tweets
labels_fake=[0 for i in range(len(list_fake_tweets))]
print(len(labels_fake))


# In[19]:


real_tweets = pd.read_csv('c:/real_tweets.txt', sep="\n",header=None)
real_tweets


# In[20]:


list_real_tweets=real_tweets[0].values.tolist()#liste des tweets
labels_real=[1 for i in range(len(list_real_tweets))]


# In[108]:



labels=[0 for i in list_fake_tweets ]
for i in list_real_tweets:
    labels.append(1)

print(len(labels))
labels=np.array(labels)
print(labels)


# In[109]:


list_real_fake_tweets=[i for i in list_fake_tweets]
for i in list_real_tweets:
    list_real_fake_tweets.append(i)

print(len(list_real_fake_tweets))
#print(list_real_fake_tweets)


# In[110]:



vocab_size = 1000
encoded_docs = [one_hot(d, vocab_size) for d in list_real_fake_tweets]
print(encoded_docs)


# In[111]:


max_length=0
for i in range(len(encoded_docs)):
  if(len(encoded_docs[i])>max_length):
    max_length=len(encoded_docs[i])
print(max_length)


# In[112]:



padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
print(padded_docs.shape, padded_docs.dtype)
print(labels.shape, labels.dtype)


# In[113]:


model_disc = Sequential()
model_disc.add(Embedding(vocab_size, 8, input_length=max_length))
model_disc.add(Flatten())
model_disc.add(Dense(1, activation='sigmoid'))
# compile the model
model_disc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model_disc.summary())


# In[129]:


from sklearn.model_selection import train_test_split
padded_docs_train,padded_docs_test,labels_train,labels_test = train_test_split(padded_docs, labels, test_size=0.33)
print(padded_docs_train.shape, padded_docs_test.shape, labels_train.shape, labels_test.shape)
# fit the model
model_disc.fit(padded_docs_train, labels_train, epochs=200, verbose=0)
# evaluate the model
print(padded_docs_test)
print(labels)
print(labels_train)
print(labels_test)
loss, accuracy = model_disc.evaluate(padded_docs_test, labels_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))


# In[130]:


def discriminateur_text(model_disc, text_seq_length, seed_text):

  encoded=[one_hot(d, vocab_size) for d in seed_text]
  print(encoded)
  encoded = pad_sequences(encoded, maxlen = text_seq_length, truncating='pre')
  y=model_disc.predict([encoded])
  print(y)
  for i in y:
    if(i<0.6):
        print("it's fake")
    else:
        print("it's real")
   


# In[131]:


print(labels[42:50])
text=list_real_fake_tweets[42:50]
discriminateur_text(model_disc, max_length, text)


# In[ ]:





# In[90]:





# In[ ]:




