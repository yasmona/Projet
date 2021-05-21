from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import re
import string
class data_loader():
    def nettoyer_data(data):
         list_listword=[]#liste des listes des mots de chaque tweet
         list_tweets=data.split('\n')
         l=[]
         for text in list_tweets:
             text = text.translate(str.maketrans('', '', string.punctuation))
             text = re.sub(r'http\S+', '', text)   # Remove URLs
             text = re.sub(r'—', ' ', text)
             text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove @ mentions
             text = text.strip(" ")   # Remove whitespace resulting from above
             text = re.sub(r' +', ' ', text)   # Remove redundant spaces
             l.append(text)
         return l
    def tokenizer_data(list_tweets, tokenizer):
        tokenizer.fit_on_texts(list_tweets)
        sequences = tokenizer.texts_to_sequences(list_tweets)
        vocab_size=len(tokenizer.word_index)
        return sequences, vocab_size

    def max_tweet(sequences):
        max_length=0
        for i in range(len(sequences)):
            if(len(sequences[i])>max_length):
                max_length=len(sequences[i])
        return max_length
    def preparer_data(sequences, vocab_size, size_batch):
        sequences = np.array([xi+[0]*(size_batch-len(xi)) for xi in sequences])
        y = sequences[:, -1]
        y_train = y.reshape(-1, 1)
        X = sequences
        X_train = np.expand_dims(X, axis=2)
        nb_tweet = len(sequences)
        return X_train, y_train, nb_tweet